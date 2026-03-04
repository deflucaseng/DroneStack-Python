from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pose:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Rotation matrix R: columns are body x/y/z axes expressed in world frame.
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))


class StateEstimator:
    """Fuses camera + IMU to estimate pose. Thread-safe; update_* may be
    called from any thread, pose is safe to read from any thread.

    Implementation notes
    --------------------
    update_imu  – high-rate (~400 Hz): integrate gyro to propagate orientation,
                  integrate accelerometer (gravity-subtracted) for velocity/position.
    update_camera – low-rate (~30 Hz): correct accumulated drift with a VIO fix.
    Both updates are protected by a single lock so the reader always gets a
    consistent snapshot.
    """

    def __init__(self) -> None:
        self._pose = Pose()
        self._lock = threading.Lock()

        # Gravity vector in world frame (NED: positive z is down, so g = +9.81 * e3)
        self._g_world = np.array([0.0, 0.0, 9.81])

        # Last IMU timestamp for integration (seconds); None until first sample.
        self._last_imu_t: float | None = None

    # ------------------------------------------------------------------
    # Public update methods (called from sensor threads)
    # ------------------------------------------------------------------

    def update_imu(self, imu_data) -> None:
        """Integrate an IMU sample to propagate the pose estimate.

        Parameters
        ----------
        imu_data:
            cosysairsim ImuData object with fields:
            - time_stamp   (uint64, nanoseconds)
            - angular_velocity   (Vector3r: x, y, z rad/s in body frame)
            - linear_acceleration (Vector3r: x, y, z m/s² in body frame,
                                   includes gravity)
        """
        import time

        t_now = time.monotonic()

        with self._lock:
            if self._last_imu_t is None:
                self._last_imu_t = t_now
                return
            dt = t_now - self._last_imu_t
            self._last_imu_t = t_now

            R = self._pose.orientation   # world ← body

            # --- Attitude integration (gyro) ---
            omega = np.array([
                imu_data.angular_velocity.x_val,
                imu_data.angular_velocity.y_val,
                imu_data.angular_velocity.z_val,
            ])
            # Skew-symmetric matrix for cross-product: R_dot ≈ R @ skew(omega)
            skew = np.array([
                [ 0.0,     -omega[2],  omega[1]],
                [ omega[2],  0.0,     -omega[0]],
                [-omega[1],  omega[0],  0.0    ],
            ])
            R = R + R @ skew * dt
            # Re-orthonormalise to fight numerical drift (Gram-Schmidt on columns)
            R = _orthonormalise(R)
            self._pose.orientation = R

            # --- Velocity/position integration (accel) ---
            a_body = np.array([
                imu_data.linear_acceleration.x_val,
                imu_data.linear_acceleration.y_val,
                imu_data.linear_acceleration.z_val,
            ])
            a_world = R @ a_body - self._g_world   # subtract gravity
            self._pose.velocity += a_world * dt
            self._pose.position += self._pose.velocity * dt

    def update_camera(self, frame) -> None:
        """Correct the pose estimate with a VIO fix derived from a camera frame.

        In a real implementation this would run ORB-SLAM3, OpenVINS, or a
        similar visual-inertial odometry pipeline.  Here we accept the frame
        but leave the correction as a stub so the plumbing compiles.

        Parameters
        ----------
        frame:
            cosysairsim ImageResponse (or raw numpy array from cv2).
        """
        # TODO: run VIO, get corrected position/orientation, apply with EKF update.
        pass

    # ------------------------------------------------------------------
    # Public read property (called from the control loop)
    # ------------------------------------------------------------------

    @property
    def pose(self) -> Pose:
        """Return a consistent snapshot of the latest estimated pose."""
        with self._lock:
            return Pose(
                position=self._pose.position.copy(),
                velocity=self._pose.velocity.copy(),
                orientation=self._pose.orientation.copy(),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orthonormalise(R: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalisation of a 3×3 rotation matrix."""
    c0 = R[:, 0]
    c1 = R[:, 1]
    c2 = R[:, 2]
    c0 = c0 / (np.linalg.norm(c0) + 1e-12)
    c1 = c1 - np.dot(c1, c0) * c0
    c1 = c1 / (np.linalg.norm(c1) + 1e-12)
    c2 = np.cross(c0, c1)
    return np.column_stack([c0, c1, c2])
