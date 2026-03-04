"""Min-snap trajectory planner.

Algorithm
---------
For N waypoints we have N-1 polynomial segments.  Each segment uses a
degree-7 polynomial (8 coefficients per axis) to allow minimisation of the
4th derivative (snap) while satisfying position, velocity, acceleration and
jerk boundary conditions.

The system is fully determined:
  unknowns  = 8 * (N-1)  per axis
  constraints:
    - start:  pos + vel + accel + jerk = 0         → 4
    - end:    pos + vel + accel + jerk = 0         → 4
    - per internal waypoint (N-2 of them):
        position (both sides)                      → 2
        derivative continuity d=1..5               → 5 (leaving 1 free per pair)
        → wait, actually d=1..6 gives 8 total
    Total internal: 8*(N-2)
  Grand total: 4 + 4 + 8*(N-2) = 8*(N-1)  ✓

Segment times are allocated proportional to the Euclidean distance between
consecutive waypoints, scaled so the drone travels at roughly `avg_speed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryState:
    position: np.ndarray      # [x, y, z]  metres
    velocity: np.ndarray      # [x_dot, y_dot, z_dot]  m/s
    acceleration: np.ndarray  # [x_ddot, y_ddot, z_ddot]  m/s²
    yaw: float = 0.0          # desired heading (radians)


class Trajectory:
    """Piecewise degree-7 polynomial trajectory."""

    def __init__(
        self,
        coeffs: np.ndarray,          # shape (n_seg, 3_axes, 8_coeffs)
        segment_times: np.ndarray,   # shape (n_seg,)  seconds per segment
    ) -> None:
        self.coeffs = coeffs
        self.segment_times = segment_times
        self._cumT = np.concatenate([[0.0], np.cumsum(segment_times)])

    @property
    def duration(self) -> float:
        return float(self._cumT[-1])

    def sample(self, t: float) -> TrajectoryState:
        t = float(np.clip(t, 0.0, self.duration))

        # Locate segment
        seg = int(np.searchsorted(self._cumT[1:], t, side="right"))
        seg = min(seg, len(self.segment_times) - 1)
        tau = t - self._cumT[seg]

        pos = np.array([_poly_eval(self.coeffs[seg, ax], tau, 0) for ax in range(3)])
        vel = np.array([_poly_eval(self.coeffs[seg, ax], tau, 1) for ax in range(3)])
        acc = np.array([_poly_eval(self.coeffs[seg, ax], tau, 2) for ax in range(3)])

        # Yaw tracks the velocity direction in the XY plane
        yaw = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-3 else 0.0

        return TrajectoryState(position=pos, velocity=vel, acceleration=acc, yaw=yaw)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TrajectoryPlanner:
    """Compute a min-snap trajectory through a sequence of waypoints."""

    #: Polynomial degree (7 = min-snap, 5 = min-jerk).
    DEGREE = 7

    def __init__(self, avg_speed: float = 3.0) -> None:
        """
        Parameters
        ----------
        avg_speed:
            Heuristic average speed (m/s) used to allocate segment times.
        """
        self.avg_speed = avg_speed

    def plan(self, start: np.ndarray, gates: list) -> Trajectory:
        """Compute the trajectory.

        Parameters
        ----------
        start:
            Starting position [x, y, z].
        gates:
            Ordered list of Gate objects (must have a .position attribute).

        Returns
        -------
        Trajectory
        """
        waypoints = np.vstack([start] + [g.position for g in gates])
        n_wp = len(waypoints)

        if n_wp < 2:
            raise ValueError("Need at least start + one gate.")

        segment_times = _allocate_times(waypoints, self.avg_speed)
        n_seg = len(segment_times)
        n = self.DEGREE + 1  # coefficients per segment per axis

        coeffs = np.zeros((n_seg, 3, n))
        for ax in range(3):
            coeffs[:, ax, :] = _solve_min_snap_axis(
                waypoints[:, ax], segment_times, n
            )

        return Trajectory(coeffs=coeffs, segment_times=segment_times)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _allocate_times(waypoints: np.ndarray, avg_speed: float) -> np.ndarray:
    """Allocate segment durations proportional to inter-waypoint distance."""
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    times = distances / avg_speed
    times = np.maximum(times, 0.3)   # minimum 0.3 s per segment
    return times


def _basis_deriv(tau: float, deriv: int, n: int) -> np.ndarray:
    """Row vector for evaluating the `deriv`-th derivative of the polynomial
    [1, t, t², …, t^(n-1)] at time `tau`.

    Element k:  prod_{i=0}^{deriv-1}(k-i) * tau^{k-deriv}  for k >= deriv.
    """
    v = np.zeros(n)
    for k in range(deriv, n):
        coeff = 1.0
        for i in range(deriv):
            coeff *= k - i
        v[k] = coeff * (tau ** (k - deriv))
    return v


def _solve_min_snap_axis(
    positions: np.ndarray,
    times: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return polynomial coefficients for one axis.

    Parameters
    ----------
    positions : (n_wp,) waypoint values for this axis.
    times     : (n_seg,) duration of each segment.
    n         : number of polynomial coefficients (degree + 1).

    Returns
    -------
    coeffs : (n_seg, n) polynomial coefficients.
    """
    n_seg = len(times)
    n_vars = n_seg * n

    A = np.zeros((n_vars, n_vars))
    b = np.zeros(n_vars)
    row = 0

    # --- Start boundary (4 constraints: pos, vel, accel, jerk = 0) ---
    for d in range(4):
        A[row, :n] = _basis_deriv(0.0, d, n)
        b[row] = positions[0] if d == 0 else 0.0
        row += 1

    # --- End boundary (4 constraints) ---
    for d in range(4):
        A[row, (n_seg - 1) * n : n_seg * n] = _basis_deriv(times[-1], d, n)
        b[row] = positions[-1] if d == 0 else 0.0
        row += 1

    # --- Internal waypoints ---
    for j in range(1, n_seg):
        sp = j - 1   # previous segment index
        sn = j       # next segment index

        # Position at end of previous segment
        A[row, sp * n : (sp + 1) * n] = _basis_deriv(times[sp], 0, n)
        b[row] = positions[j]
        row += 1

        # Position at start of next segment
        A[row, sn * n : (sn + 1) * n] = _basis_deriv(0.0, 0, n)
        b[row] = positions[j]
        row += 1

        # Continuity of derivatives d = 1 … n-2  (fills remaining rows)
        for d in range(1, n - 1):
            if row >= n_vars:
                break
            v_prev = _basis_deriv(times[sp], d, n)
            v_next = _basis_deriv(0.0, d, n)
            A[row, sp * n : (sp + 1) * n] = v_prev
            A[row, sn * n : (sn + 1) * n] = -v_next
            b[row] = 0.0
            row += 1

    try:
        flat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Min-snap linear system is singular.") from exc

    return flat.reshape(n_seg, n)


def _poly_eval(c: np.ndarray, tau: float, deriv: int) -> float:
    """Evaluate the `deriv`-th derivative of polynomial c at tau."""
    n = len(c)
    result = 0.0
    for k in range(deriv, n):
        coeff = c[k]
        for i in range(deriv):
            coeff *= k - i
        result += coeff * tau ** (k - deriv)
    return float(result)
