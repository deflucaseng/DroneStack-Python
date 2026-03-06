"""15-state Extended Kalman Filter — JAX implementation.

State: [position(3), velocity(3), quaternion(4), gyro_bias(3), accel_bias(3)]

Design notes:
    - The heavy linear-algebra path (predict / update) is JIT-compiled.
    - State (x) and covariance (P) are stored as plain JAX arrays on the
      class instance.  Mutation happens only through array replacement
      (x = new_x), which is safe because JAX arrays are immutable values.
    - get_pose() / set_pose() bridge to the rest of the drone stack via
      NumPy copies, keeping the JAX boundary clean.

Config keys (all optional):
    sigma_accel       – accelerometer noise density  (m/s²/√Hz)   default 0.1
    sigma_gyro        – gyroscope noise density      (rad/s/√Hz)  default 0.01
    sigma_accel_bias  – accel bias random walk       (m/s³/√Hz)   default 1e-4
    sigma_gyro_bias   – gyro bias random walk        (rad/s²/√Hz) default 1e-5
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drone_stack.estimation.ekf.imu_propagation import (
    propagate_state,
    state_jacobian,
    quaternion_to_rotation_matrix,
)


# ---------------------------------------------------------------------------
# JIT-compiled prediction kernel
# ---------------------------------------------------------------------------
@jax.jit
def _predict_kernel(
    x: jnp.ndarray,
    P: jnp.ndarray,
    Q: jnp.ndarray,
    gyro: jnp.ndarray,
    accel: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-function prediction step (JIT-friendly).

    Returns:
        (x_new, P_new)
    """
    F = state_jacobian(x, gyro, accel, dt)
    x_new = propagate_state(x, gyro, accel, dt)
    P_new = F @ P @ F.T + Q
    return x_new, P_new


# ---------------------------------------------------------------------------
# JIT-compiled process-noise builder
# ---------------------------------------------------------------------------
@jax.jit
def _build_process_noise(
    sa: float,
    sg: float,
    sbg: float,
    sba: float,
    dt: float,
) -> jnp.ndarray:
    """Diagonal discrete process-noise matrix Q_d (15×15)."""
    diag = jnp.concatenate([
        jnp.zeros(3),                          # position — no direct noise
        jnp.full(3, sa ** 2 * dt),              # velocity
        jnp.full(4, sg ** 2 * 0.25 * dt),       # quaternion
        jnp.full(3, sbg ** 2 * dt),             # gyro bias
        jnp.full(3, sba ** 2 * dt),             # accel bias
    ])
    return jnp.diag(diag)


# ---------------------------------------------------------------------------
# EKF class
# ---------------------------------------------------------------------------
class ExtendedKalmanFilter:

    def __init__(self, config: dict) -> None:
        self.n_states = 16

        self.x = jnp.zeros(self.n_states).at[6].set(1.0)
        self.P = jnp.eye(self.n_states) * 0.1

        self._sa  = float(config.get("sigma_accel",      0.1))
        self._sg  = float(config.get("sigma_gyro",       0.01))
        self._sba = float(config.get("sigma_accel_bias", 1e-4))
        self._sbg = float(config.get("sigma_gyro_bias",  1e-5))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """IMU prediction step.

        Accepts plain NumPy arrays from the IMU driver; conversion to JAX
        arrays happens here so the caller doesn't need to care.
        """
        Q = _build_process_noise(self._sa, self._sg, self._sbg, self._sba, dt)
        self.x, self.P = _predict_kernel(
            self.x,
            self.P,
            Q,
            jnp.asarray(gyro),
            jnp.asarray(accel),
            dt,
        )

    def update_vision(self, measurement) -> None:
        """Vision/VIO measurement update — to be implemented."""
        pass

    def get_pose(self):
        """Extract a Pose from the current state.

        Copies data back to NumPy so the rest of the stack stays
        framework-agnostic.
        """
        from drone_stack.core.pose import Pose

        x_np = np.asarray(self.x)
        return Pose(
            position=x_np[0:3].copy(),
            velocity=x_np[3:6].copy(),
            orientation=np.asarray(quaternion_to_rotation_matrix(self.x[6:10])),
        )

    def set_pose(self, pose) -> None:
        """Overwrite position / velocity / orientation from a Pose."""
        from drone_stack.utils.math_utils import rotation_matrix_to_quaternion

        q = rotation_matrix_to_quaternion(pose.orientation)
        self.x = (
            self.x
            .at[0:3].set(jnp.asarray(pose.position))
            .at[3:6].set(jnp.asarray(pose.velocity))
            .at[6:10].set(jnp.asarray(q))
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_noise(self, dt: float) -> jnp.ndarray:
        return _build_process_noise(self._sa, self._sg, self._sbg, self._sba, dt)