import time
import numpy as np
from drone_stack.simulation.airsim_drone import AirSimDrone
from drone_stack.control.pid_controller import SimplePIDController


# Moves the drone around but does NOT fly correctly, close enough. 

def main():
    print("=== Basic Flight Test ===")
    
    # Connect to drone
    drone = AirSimDrone('192.168.56.1')
    drone.connect()
    
    # Create controller
    controller = SimplePIDController(Kp_pos=1.5, Kd_pos=1.0)
    
    # Takeoff
    print("Taking off...")
    drone.takeoff(altitude=2.0)
    time.sleep(2.0)
    
    # Simple waypoint following
    waypoints = [
        np.array([0.0, 0.0, 4.0]),
        np.array([5.0, 0.0, 4.0]),
        np.array([5.0, 5.0, 4.0]),
        np.array([0.0, 5.0, 4.0]),
        np.array([0.0, 0.0, 4.0]),
    ]
    
    print("Following waypoints...")
    for wp in waypoints:
        print(f"Going to {wp}")
        
        # Hover at waypoint for 5 seconds
        for _ in range(500):  # 5 seconds at 100 Hz
            state = drone.get_state()
            if(state):
                control = controller.compute(wp, state.position, state.velocity)
                drone.send_control(control)
            time.sleep(0.01)  # 100 Hz
    
    # Land
    print("Landing...")
    drone.land()
    print("Done!")

if __name__ == "__main__":
    main()