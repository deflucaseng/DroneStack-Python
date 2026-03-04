# DroneStack-Python

 Module Architecture

  drone_stack/
  ├── state_estimator.py   # VIO: camera + IMU → pose estimate
  ├── trajectory.py        # Min-snap planner (what you have in trajectory.ipynb)
  ├── controller.py        # Geometric controller: pose error → rates + throttle
  ├── gate_map.py          # Known gate positions, coordinate transforms
  └── main.py              # Orchestrator + control loop

  Data Flow

  Camera ──┐
           ├──► StateEstimator ──► estimated_pose ──┐
  IMU ─────┘                                        │
                                                    ▼
  GateMap ──► TrajectoryPlanner ──► desired_state ──► Controller ──► moveByAngleRatesThrottleAsync
                 (run once at start)

  Class Interfaces

  # state_estimator.py
  class StateEstimator:
      """Fuses camera + IMU to estimate pose. Runs in its own thread."""

      def update_imu(self, imu_data): ...        # called at ~400Hz
      def update_camera(self, frame): ...         # called at ~30Hz

      @property
      def pose(self) -> Pose:
          """Latest estimated position + orientation + velocity."""
          ...

  # gate_map.py
  class GateMap:
      """Known gate positions in world frame."""

      def __init__(self, yaml_path: str): ...
      def gates(self) -> list[Gate]: ...
      def nearest_gate(self, pos: np.ndarray) -> Gate: ...

  # trajectory.py  (refactored from your notebook)
  class TrajectoryPlanner:
      """Computes min-snap trajectory given gate positions + start pose."""

      def plan(self, start: np.ndarray, gates: list[Gate]) -> Trajectory: ...

  class Trajectory:
      def sample(self, t: float) -> TrajectoryState:
          """Returns desired pos, vel, accel at time t."""
          ...

  # controller.py
  class GeometricController:
      """Converts trajectory error into angular rate + throttle commands."""

      def __init__(self, mass: float, Kp: float, Kd: float): ...

      def compute(self,
                  desired: TrajectoryState,
                  estimated: Pose) -> ControlOutput:
          # 1. Position + velocity error
          # 2. Desired acceleration (feedforward from trajectory + feedback from error)
          # 3. Thrust vector decomposition → desired attitude
          # 4. Attitude error → angular rates
          ...

  Main Loop

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
      planner    = TrajectoryPlanner()
      controller = GeometricController(mass=0.5, Kp=4.0, Kd=2.0)

      trajectory = planner.plan(
          start=np.array([0, 0, -1.5]),
          gates=gate_map.gates()
      )

      # ── Sensor threads ────────────────────────────────────────────────
      def imu_loop():
          while running:
              imu = client.getImuData()
              estimator.update_imu(imu)
              time.sleep(1 / IMU_HZ)

      def camera_loop():
          while running:
              frame = client.simGetImages([...])[0]
              estimator.update_camera(frame)
              time.sleep(1 / CAMERA_HZ)

      running = True
      threading.Thread(target=imu_loop,    daemon=True).start()
      threading.Thread(target=camera_loop, daemon=True).start()

      # ── Control loop ──────────────────────────────────────────────────
      client.takeoffAsync().join()
      t_start = time.time()

      while True:
          t = time.time() - t_start
          desired   = trajectory.sample(t)
          estimated = estimator.pose
          cmd       = controller.compute(desired, estimated)

          client.moveByAngleRatesThrottleAsync(
              cmd.roll_rate,
              cmd.pitch_rate,
              cmd.yaw_rate,
              cmd.throttle,
              duration=1 / CONTROL_HZ,
          )

          if t >= trajectory.duration:
              break

          time.sleep(1 / CONTROL_HZ)

  main()
