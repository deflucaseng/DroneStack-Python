import numpy as np
from drone_stack.core.pose import ControlOutput

class SimplePIDController:
    def __init__(self, Kp_pos=1.0, Kd_pos=0.5, Kp_yaw=1.0):
        self.Kp_pos = Kp_pos
        self.Kd_pos = Kd_pos
        self.Kp_yaw = Kp_yaw
        self.prev_error = np.zeros(3)
    
    def compute(self, desired_pos, current_pos, current_vel, dt=0.01):
        """Simple PID position control."""
        # Position error
        pos_error = desired_pos - current_pos
        
        # Velocity error (derivative term)
        vel_error = -current_vel
        
        # Desired acceleration
        desired_accel = self.Kp_pos * pos_error + self.Kd_pos * vel_error
        
        # Convert to roll/pitch/throttle (simplified)
        # This is a hack - just to get flying!
        throttle = 0.59 + desired_accel[2] / 20.0  # Z-acceleration
        pitch_rate = -desired_accel[0] * 0.5       # X-acceleration
        roll_rate = desired_accel[1] * 0.5         # Y-acceleration
        yaw_rate = 0.0
        
        # Clamp values
        throttle = np.clip(throttle, 0.0, 1.0)
        pitch_rate = np.clip(pitch_rate, -1.0, 1.0)
        roll_rate = np.clip(roll_rate, -1.0, 1.0)
        
        return ControlOutput(roll_rate, pitch_rate, yaw_rate, throttle)