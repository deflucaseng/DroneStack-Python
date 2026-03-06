from dataclasses import dataclass
import numpy as np

@dataclass
class Pose:
    position: np.ndarray     # (3,) [x, y, z]
    velocity: np.ndarray     # (3,) [vx, vy, vz]
    orientation: np.ndarray  # (3, 3) rotation matrix
    timestamp: float = 0.0

@dataclass
class ControlOutput:
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    throttle: float