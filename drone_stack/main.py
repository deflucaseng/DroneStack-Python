"""Orchestrator + control loop.

Run this file directly::

    python -m drone_stack.main

Or from the repo root::

    python drone_stack/main.py
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    import cosysairsim as airsim
except ImportError as exc:
    raise SystemExit(
        "cosysairsim is not installed.  "
        "See https://github.com/Cosys-Lab/Cosys-AirSim for setup instructions."
    ) from exc

from .controller import GeometricController
from .gate_map import GateMap
from .state_estimator import StateEstimator
from .trajectory import TrajectoryPlanner

# ---------------------------------------------------------------------------
# Rate constants
# ---------------------------------------------------------------------------
CONTROL_HZ = 100
IMU_HZ     = 400
CAMERA_HZ  = 30


def main() -> None:
    # ------------------------------------------------------------------
    # Connect to simulator
    # ------------------------------------------------------------------
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------
    gate_map   = GateMap("gate_course.yaml")
    estimator  = StateEstimator()
    planner    = TrajectoryPlanner(avg_speed=3.0)
    controller = GeometricController(mass=0.5, Kp=4.0, Kd=2.0)

    trajectory = planner.plan(
        start=np.array([0.0, 0.0, -1.5]),
        gates=gate_map.gates(),
    )

    # ------------------------------------------------------------------
    # Sensor threads
    # ------------------------------------------------------------------
    running = threading.Event()
    running.set()

    def imu_loop() -> None:
        dt = 1.0 / IMU_HZ
        while running.is_set():
            imu = client.getImuData()
            estimator.update_imu(imu)
            time.sleep(dt)

    def camera_loop() -> None:
        dt = 1.0 / CAMERA_HZ
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ]
        while running.is_set():
            responses = client.simGetImages(requests)
            if responses:
                estimator.update_camera(responses[0])
            time.sleep(dt)

    threading.Thread(target=imu_loop,    daemon=True, name="imu").start()
    threading.Thread(target=camera_loop, daemon=True, name="camera").start()

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    client.takeoffAsync().join()
    t_start = time.monotonic()

    try:
        while True:
            t = time.monotonic() - t_start
            desired   = trajectory.sample(t)
            estimated = estimator.pose
            cmd       = controller.compute(desired, estimated)

            client.moveByAngleRatesThrottleAsync(
                cmd.roll_rate,
                cmd.pitch_rate,
                cmd.yaw_rate,
                cmd.throttle,
                duration=1.0 / CONTROL_HZ,
            )

            if t >= trajectory.duration:
                break

            time.sleep(1.0 / CONTROL_HZ)
    finally:
        running.clear()
        client.hoverAsync().join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)


if __name__ == "__main__":
    main()
