"""Geometric controller on SE(3).

Reference: Mellinger & Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors", ICRA 2011.

The controller computes angular-rate + throttle commands suitable for
cosysairsim's moveByAngleRatesThrottleAsync interface.

Pipeline
--------
1. Position error  e_p = p̂ - p_d
2. Velocity error  e_v = v̂ - v_d
3. Desired acceleration (feedback + feedforward):
       a_des = a_d - Kd·e_v - Kp·e_p
4. Required thrust vector in world frame:
       F_des = m·(a_des + g·ê₃)
5. Desired body-z axis (thrust direction):
       b₃_des = F_des / ‖F_des‖
6. Throttle: normalised thrust in [0, 1]
7. Attitude error (cross-product on S²):
       e_R = b₃_cur × b₃_des
8. Angular-rate commands proportional to attitude error,
   projected into body frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_estimator import Pose
from .trajectory import TrajectoryState

# Gravity (m/s²).  NED convention: +z is down, so gravity adds to z-accel.
_GRAVITY = 9.81


@dataclass
class ControlOutput:
    roll_rate: float    # rad/s (body x-axis)
    pitch_rate: float   # rad/s (body y-axis)
    yaw_rate: float     # rad/s (body z-axis)
    throttle: float     # [0, 1]


class GeometricController:
    """Converts trajectory tracking error into angular-rate + throttle commands.

    Parameters
    ----------
    mass:
        Vehicle mass in kg.
    Kp:
        Proportional position gain.
    Kd:
        Derivative (velocity) gain.
    Kp_att:
        Attitude proportional gain (maps attitude error to rate command).
    max_thrust_ratio:
        Ratio of maximum thrust to hover thrust (default 2).  Used to
        normalise the throttle output to [0, 1].
    """

    def __init__(
        self,
        mass: float,
        Kp: float,
        Kd: float,
        Kp_att: float = 10.0,
        max_thrust_ratio: float = 2.0,
    ) -> None:
        self.mass = mass
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_att = Kp_att
        # Maximum thrust expressed in world-frame acceleration units
        self._max_accel = max_thrust_ratio * _GRAVITY

    def compute(self, desired: TrajectoryState, estimated: Pose) -> ControlOutput:
        """Compute control output.

        Parameters
        ----------
        desired:
            Reference state from Trajectory.sample().
        estimated:
            Current pose estimate from StateEstimator.pose.

        Returns
        -------
        ControlOutput
        """
        # ------------------------------------------------------------------
        # 1. Tracking errors
        # ------------------------------------------------------------------
        e_p = estimated.position - desired.position   # position error
        e_v = estimated.velocity - desired.velocity   # velocity error

        # ------------------------------------------------------------------
        # 2. Desired acceleration in world frame
        #    Gravity vector: in NED, gravity is +9.81 in z → [0, 0, +g]
        # ------------------------------------------------------------------
        g_vec = np.array([0.0, 0.0, _GRAVITY])
        a_des = desired.acceleration - self.Kd * e_v - self.Kp * e_p + g_vec

        # ------------------------------------------------------------------
        # 3 & 4. Thrust magnitude and normalised throttle
        # ------------------------------------------------------------------
        thrust_accel = float(np.linalg.norm(a_des))   # in m/s² units
        throttle = float(np.clip(thrust_accel / self._max_accel, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 5. Desired body z-axis (thrust direction)
        # ------------------------------------------------------------------
        b3_des = a_des / (thrust_accel + 1e-9)

        # ------------------------------------------------------------------
        # 6. Current body z-axis from rotation matrix (third column, world frame)
        # ------------------------------------------------------------------
        R = estimated.orientation          # 3×3, world ← body
        b3_cur = R[:, 2]

        # ------------------------------------------------------------------
        # 7. Attitude error on S² (axis-angle, skew-symmetric direction)
        # ------------------------------------------------------------------
        e_att = np.cross(b3_cur, b3_des)   # in world frame

        # ------------------------------------------------------------------
        # 8. Map attitude error to body-frame angular-rate commands
        # ------------------------------------------------------------------
        # Project world-frame error onto body axes.
        rate_world = self.Kp_att * e_att
        roll_rate  = float(R[:, 0] @ rate_world)   # body x
        pitch_rate = float(R[:, 1] @ rate_world)   # body y

        # Yaw: track the desired heading from the trajectory
        yaw_error = _wrap_angle(desired.yaw - _rotation_to_yaw(R))
        yaw_rate  = float(np.clip(self.Kp_att * yaw_error, -2.0, 2.0))

        return ControlOutput(
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            throttle=throttle,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_to_yaw(R: np.ndarray) -> float:
    """Extract yaw (rotation about world z-axis) from a rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)
