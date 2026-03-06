from abc import ABC, abstractmethod

class DroneInterface(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def get_state(self):
        """Get current state from simulator."""
        pass
    
    @abstractmethod
    def send_control(self, control):
        pass
    
    @abstractmethod
    def takeoff(self, altitude: float):
        pass
    
    @abstractmethod
    def land(self):
        pass

    @abstractmethod
    def move_on_path(self, waypoints, speed: float):
        """Fly through an ordered list of (3,) NED positions at the given speed."""
        pass

    @abstractmethod
    def get_imu(self) -> dict:
        """Return {'gyro': (3,), 'accel': (3,), 'timestamp': float}."""
        pass