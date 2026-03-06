import warnings
import cosysairsim as airsim
import numpy as np
from drone_stack.interfaces.drone_interface import DroneInterface
from drone_stack.core.pose import Pose, ControlOutput
from drone_stack.utils.math_utils import quaternion_to_rotation_matrix, normalize_quaternion

class AirSimDrone(DroneInterface):
    def __init__(self, ip: str = ""):
        self.client = None
        self._ip = ip

    def connect(self):
        self.client = airsim.MultirotorClient(ip=self._ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        return True
    
    def get_state(self):
        """Get ground truth state from AirSim."""
        if self.client is None:
            warnings.warn("get_state called before connect()", RuntimeWarning, stacklevel=2)
            return None
        state = self.client.getMultirotorState()
        
        # Extract position
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Extract velocity
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        # Extract orientation (quaternion to rotation matrix)
        q = state.kinematics_estimated.orientation
        quat = normalize_quaternion(np.array([q.w_val, q.x_val, q.y_val, q.z_val]))
        R = quaternion_to_rotation_matrix(quat)
        
        return Pose(position, velocity, R)
    
    def get_imu(self):
        self._require_connected("get_imu")
        imu = self.client.getImuData()  # type: ignore[union-attr]
        return {
            'gyro':  np.array([imu.angular_velocity.x_val,
                               imu.angular_velocity.y_val,
                               imu.angular_velocity.z_val]),
            'accel': np.array([imu.linear_acceleration.x_val,
                               imu.linear_acceleration.y_val,
                               imu.linear_acceleration.z_val]),
            'timestamp': imu.time_stamp * 1e-9,
        }
    
    def _require_connected(self, method: str):
        if self.client is None:
            raise RuntimeError(f"{method} called before connect()")

    def send_control(self, control: ControlOutput):
        self._require_connected("send_control")
        self.client.moveByAngleRatesThrottleAsync(  # type: ignore[union-attr]
            control.roll_rate,
            control.pitch_rate,
            control.yaw_rate,
            control.throttle,
            duration=0.01
        )

    def takeoff(self, altitude=1.5):
        self._require_connected("takeoff")
        self.client.takeoffAsync().join()           # type: ignore[union-attr]
        self.client.moveToZAsync(-altitude, 1.0).join()  # type: ignore[union-attr]

    def land(self):
        self._require_connected("land")
        self.client.landAsync().join()              # type: ignore[union-attr]

    def move_on_path(self, waypoints, speed: float):
        self._require_connected("move_on_path")
        path = [airsim.Vector3r(float(wp[0]), float(wp[1]), float(wp[2])) for wp in waypoints]
        self.client.moveOnPathAsync(path, speed).join()  # type: ignore[union-attr]