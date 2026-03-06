import time
from drone_stack.simulation.airsim_drone import AirSimDrone
from drone_stack.core.gate import GateMap

COURSE_YAML = 'config/gates/four_gate_course.yaml'


def main():
    print("=== Basic Gate Flight Test ===")

    # Load course
    gate_map = GateMap(COURSE_YAML)
    gates = gate_map.gates()
    speed = gate_map.trajectory.flight_speed
    print(f"Loaded {len(gates)} gates — flight speed {speed} m/s")

    # Build full path: approach → centre → exit for every gate
    path = []
    for gate in gates:
        path.append(gate.approach_point)
        path.append(gate.position)
        path.append(gate.exit_point)

    # Connect
    drone = AirSimDrone('192.168.56.1')
    drone.connect()

    # Takeoff
    print("Taking off...")
    drone.takeoff(altitude=2.0)
    time.sleep(2.0)

    # Fly the course
    print(f"Flying {len(path)} waypoints through {len(gates)} gates...")
    drone.move_on_path(path, speed)

    # Land
    print("Landing...")
    drone.land()
    print("Done!")


if __name__ == "__main__":
    main()
