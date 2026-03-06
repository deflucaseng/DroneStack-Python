"""Thread-safe state estimator wrapping the JAX EKF.

Intended usage:
    - IMU thread  → update_imu()   at ~400 Hz
    - Control thread reads .pose    at ~100 Hz
    - Vision thread → update_vision() when implemented

Threading notes:
    JAX arrays are immutable values, so a stale reader will simply see the
    previous snapshot — no torn reads.  The lock protects the *reference*
    swap of self._pose so that the control thread always gets a consistent
    Pose object.
"""

from __future__ import annotations

import threading

import jax.numpy as jnp
import numpy as np

from drone_stack.core.pose import Pose
from drone_stack.estimation.ekf.ekf_core import ExtendedKalmanFilter


class StateEstimator:

    def __init__(self, initial_pose: Pose, config: dict) -> None:
        self._ekf = ExtendedKalmanFilter(config)
        self._ekf.set_pose(initial_pose)

        self._pose: Pose = initial_pose
        self._last_time: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # IMU path (high frequency)
    # ------------------------------------------------------------------

    def update_imu(self, imu_data: dict) -> None:
        """Propagate the EKF with one IMU sample.

        Args:
            imu_data: Dict with keys
                'timestamp' – float, seconds
                'gyro'      – array-like (3,), rad/s, body frame
                'accel'     – array-like (3,), m/s², body frame
        """
        t = float(imu_data["timestamp"])

        if self._last_time is None:
            self._last_time = t
            return

        dt = t - self._last_time
        if dt <= 0.0:
            return
        self._last_time = t

        self._ekf.predict(
            np.asarray(imu_data["gyro"], dtype=np.float64),
            np.asarray(imu_data["accel"], dtype=np.float64),
            dt,
        )

        with self._lock:
            self._pose = self._ekf.get_pose()

    # ------------------------------------------------------------------
    # Vision path (low frequency)
    # ------------------------------------------------------------------

    def update_vision(self, measurement) -> None:
        """Forward a vision measurement to the EKF update step."""
        self._ekf.update_vision(measurement)
        with self._lock:
            self._pose = self._ekf.get_pose()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def pose(self) -> Pose:
        with self._lock:
            return self._pose