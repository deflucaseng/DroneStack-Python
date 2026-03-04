# DroneStack-Python

Autonomous drone racing stack for [CosysAirSim](https://github.com/Cosys-Lab/Cosys-AirSim).

## Module Architecture

```
drone_stack/
├── __init__.py          # re-exports all public classes
├── gate_map.py          # Gate positions loaded from YAML
├── state_estimator.py   # Thread-safe IMU/camera pose estimator
├── trajectory.py        # Min-snap trajectory planner
├── controller.py        # Geometric controller (SE3)
└── main.py              # Orchestrator + control loop
```

## Data Flow

```
Camera ──┐
         ├──► StateEstimator ──► Pose ──────────────────────────┐
IMU ─────┘                                                      ▼
GateMap ──► TrajectoryPlanner ──► Trajectory.sample(t) ──► GeometricController ──► moveByAngleRatesThrottleAsync
             (run once at start)      (100 Hz)
```

## Class Interfaces

### `gate_map.py`

```python
@dataclass
class Gate:
    name: str
    position: np.ndarray   # [x, y, z] metres, world frame
    heading: float = 0.0   # gate normal direction, radians

class GateMap:
    def __init__(self, yaml_path: str): ...
    def gates(self) -> list[Gate]: ...
    def nearest_gate(self, pos: np.ndarray) -> Gate: ...
```

Expected YAML schema (`gate_course.yaml`):

```yaml
gates:
  - name: gate_1
    position: [5.0, 0.0, -1.5]
    heading: 0.0
  - name: gate_2
    position: [10.0, 3.0, -1.5]
    heading: 0.785
```

### `state_estimator.py`

```python
@dataclass
class Pose:
    position: np.ndarray     # [x, y, z] metres
    velocity: np.ndarray     # [x_dot, y_dot, z_dot] m/s
    orientation: np.ndarray  # 3×3 rotation matrix (world ← body)

class StateEstimator:
    """Fuses camera + IMU to estimate pose. Thread-safe."""

    def update_imu(self, imu_data): ...    # ~400 Hz; integrates gyro + accel
    def update_camera(self, frame): ...    # ~30 Hz;  VIO correction (stub)

    @property
    def pose(self) -> Pose: ...            # consistent snapshot, safe from any thread
```

### `trajectory.py`

```python
@dataclass
class TrajectoryState:
    position: np.ndarray      # [x, y, z] metres
    velocity: np.ndarray      # m/s
    acceleration: np.ndarray  # m/s²
    yaw: float                # desired heading, radians

class TrajectoryPlanner:
    """Min-snap planner: degree-7 polynomial per segment, solved as a linear system."""

    def __init__(self, avg_speed: float = 3.0): ...
    def plan(self, start: np.ndarray, gates: list[Gate]) -> Trajectory: ...

class Trajectory:
    duration: float

    def sample(self, t: float) -> TrajectoryState: ...
```

### `controller.py`

```python
@dataclass
class ControlOutput:
    roll_rate: float   # rad/s, body x
    pitch_rate: float  # rad/s, body y
    yaw_rate: float    # rad/s, body z
    throttle: float    # [0, 1]

class GeometricController:
    """SE3 geometric controller.
    Pipeline: position error → desired thrust vector → attitude error → angular rates.
    """

    def __init__(self, mass: float, Kp: float, Kd: float,
                 Kp_att: float = 10.0, max_thrust_ratio: float = 2.0): ...

    def compute(self, desired: TrajectoryState, estimated: Pose) -> ControlOutput: ...
```

## Main Loop

```python
# main.py
import threading, time
import cosysairsim as airsim

CONTROL_HZ = 100
IMU_HZ     = 400
CAMERA_HZ  = 30

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # ── One-time setup ────────────────────────────────────────────────
    gate_map   = GateMap("gate_course.yaml")
    estimator  = StateEstimator()
    planner    = TrajectoryPlanner(avg_speed=3.0)
    controller = GeometricController(mass=0.5, Kp=4.0, Kd=2.0)

    trajectory = planner.plan(
        start=np.array([0.0, 0.0, -1.5]),
        gates=gate_map.gates(),
    )

    # ── Sensor threads ────────────────────────────────────────────────
    running = threading.Event()
    running.set()

    threading.Thread(target=imu_loop,    daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()

    # ── Control loop ──────────────────────────────────────────────────
    client.takeoffAsync().join()
    t_start = time.monotonic()

    try:
        while True:
            t         = time.monotonic() - t_start
            desired   = trajectory.sample(t)
            estimated = estimator.pose
            cmd       = controller.compute(desired, estimated)

            client.moveByAngleRatesThrottleAsync(
                cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate,
                cmd.throttle, duration=1 / CONTROL_HZ,
            )

            if t >= trajectory.duration:
                break
            time.sleep(1 / CONTROL_HZ)
    finally:
        running.clear()
        client.hoverAsync().join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
```

## Running

```bash
python -m drone_stack.main
```
