"""test_estimation.py

Flies the four-gate course while running the EKF estimator driven by IMU data.
Prints a side-by-side comparison of ground-truth vs. estimated pose.

Threading model:
    - Main thread owns ALL cosysairsim calls (avoids tornado/asyncio conflicts).
      It polls IMU at ~400 Hz, updates the estimator, and prints comparisons.
    - A single background thread runs move_on_path (blocking flight command).
      It gets its own asyncio event loop to satisfy tornado.
"""

import asyncio
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from drone_stack.core.gate import GateMap
from drone_stack.core.pose import Pose
from drone_stack.estimation.state_estimator import StateEstimator
from drone_stack.simulation.airsim_drone import AirSimDrone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COURSE_YAML = 'config/gates/four_gate_course.yaml'
DRONE_IP    = '192.168.56.1'
IMU_HZ      = 400
PRINT_HZ    = 2      # comparison table refresh rate

EKF_CONFIG = {
    'sigma_accel':      0.1,
    'sigma_gyro':       0.01,
    'sigma_accel_bias': 1e-4,
    'sigma_gyro_bias':  1e-5,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_euler_deg(R_mat: np.ndarray) -> np.ndarray:
    """Rotation matrix → [roll, pitch, yaw] in degrees (ZYX / aerospace)."""
    return Rotation.from_matrix(R_mat).as_euler('ZYX', degrees=True)[::-1]


def print_comparison(true_pose: Pose, est_pose: Pose) -> None:
    tp, ep = true_pose.position,    est_pose.position
    tv, ev = true_pose.velocity,    est_pose.velocity
    te, ee = _to_euler_deg(true_pose.orientation), _to_euler_deg(est_pose.orientation)

    pos_err = np.linalg.norm(tp - ep)
    vel_err = np.linalg.norm(tv - ev)

    print(
        f"\n{'─'*62}\n"
        f"  {'':20s} {'True':>16s}   {'Estimated':>16s}\n"
        f"{'─'*62}\n"
        f"  {'Position x (m)':20s} {tp[0]:>16.3f}   {ep[0]:>16.3f}\n"
        f"  {'Position y (m)':20s} {tp[1]:>16.3f}   {ep[1]:>16.3f}\n"
        f"  {'Position z (m)':20s} {tp[2]:>16.3f}   {ep[2]:>16.3f}\n"
        f"  {'|pos error| (m)':20s} {'':>16s}   {pos_err:>16.3f}\n"
        f"{'─'*62}\n"
        f"  {'Velocity x (m/s)':20s} {tv[0]:>16.3f}   {ev[0]:>16.3f}\n"
        f"  {'Velocity y (m/s)':20s} {tv[1]:>16.3f}   {ev[1]:>16.3f}\n"
        f"  {'Velocity z (m/s)':20s} {tv[2]:>16.3f}   {ev[2]:>16.3f}\n"
        f"  {'|vel error| (m/s)':20s} {'':>16s}   {vel_err:>16.3f}\n"
        f"{'─'*62}\n"
        f"  {'Roll  (deg)':20s} {te[0]:>16.2f}   {ee[0]:>16.2f}\n"
        f"  {'Pitch (deg)':20s} {te[1]:>16.2f}   {ee[1]:>16.2f}\n"
        f"  {'Yaw   (deg)':20s} {te[2]:>16.2f}   {ee[2]:>16.2f}\n"
        f"{'─'*62}"
    )


# ---------------------------------------------------------------------------
# Flight thread
# ---------------------------------------------------------------------------

def flight_thread_fn(drone: AirSimDrone, path: list, speed: float,
                     done: threading.Event) -> None:
    # tornado/msgpackrpc needs its own event loop in non-main threads
    asyncio.set_event_loop(asyncio.new_event_loop())
    drone.move_on_path(path, speed)
    done.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Estimation Test — Gate Course ===")

    # Load course
    gate_map = GateMap(COURSE_YAML)
    gates    = gate_map.gates()
    speed    = gate_map.trajectory.flight_speed
    print(f"Loaded {len(gates)} gates — flight speed {speed} m/s")

    path = []
    for gate in gates:
        path.append(gate.approach_point)
        path.append(gate.position)
        path.append(gate.exit_point)

    # Connect (main thread owns the client)
    drone = AirSimDrone(DRONE_IP)
    drone.connect()

    # Seed estimator from ground-truth pose at rest
    initial_pose = drone.get_state()
    if initial_pose is None:
        raise RuntimeError("Could not get initial pose from simulator")
    estimator = StateEstimator(initial_pose, EKF_CONFIG)

    # Takeoff (main thread)
    print("Taking off...")
    drone.takeoff(altitude=2.0)
    time.sleep(2.0)

    # Launch flight in background; main thread keeps polling IMU
    flight_done = threading.Event()
    ft = threading.Thread(
        target=flight_thread_fn,
        args=(drone, path, speed, flight_done),
        daemon=True,
    )
    ft.start()
    print(f"Flying {len(path)} waypoints through {len(gates)} gates...")

    # Main loop: poll IMU at ~400 Hz, print comparison at PRINT_HZ
    imu_period   = 1.0 / IMU_HZ
    print_every  = IMU_HZ // PRINT_HZ   # print every N IMU samples
    sample_count = 0

    while not flight_done.is_set():
        t0 = time.monotonic()

        # IMU update
        try:
            estimator.update_imu(drone.get_imu())
        except Exception as e:
            print(f"[IMU] {e}")

        # Periodic comparison print
        sample_count += 1
        if sample_count >= print_every:
            sample_count = 0
            try:
                true_pose = drone.get_state()
                if true_pose is not None:
                    print_comparison(true_pose, estimator.pose)
            except Exception as e:
                print(f"[PRINT] {e}")

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, imu_period - elapsed))

    ft.join()

    # Land (main thread)
    print("Landing...")
    drone.land()
    print("Done!")


if __name__ == "__main__":
    main()
