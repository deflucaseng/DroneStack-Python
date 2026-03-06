from dataclasses import dataclass
import math
import numpy as np
import yaml

@dataclass
class Gate:
    id: int
    position: np.ndarray        # (3,) NED metres
    yaw: float                  # degrees; 0 → normal points North (+X)
    width: float                # metres
    height: float               # metres
    approach_distance: float    # metres before gate centre
    exit_distance: float        # metres after gate centre

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw)

    @property
    def normal(self) -> np.ndarray:
        """Unit vector pointing through the gate (NED x-y plane)."""
        return np.array([math.cos(self.yaw_rad), math.sin(self.yaw_rad), 0.0])

    @property
    def approach_point(self) -> np.ndarray:
        return self.position - self.normal * self.approach_distance

    @property
    def exit_point(self) -> np.ndarray:
        return self.position + self.normal * self.exit_distance


@dataclass
class TrajectoryConfig:
    max_velocity: float
    max_acceleration: float
    flight_speed: float
    approach_distance: float
    exit_distance: float


class GateMap:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        traj = data['trajectory']
        self.trajectory = TrajectoryConfig(
            max_velocity=traj['max_velocity'],
            max_acceleration=traj['max_acceleration'],
            flight_speed=traj['flight_speed'],
            approach_distance=traj['approach_distance'],
            exit_distance=traj['exit_distance'],
        )

        self._gates = []
        for g in data['gates']:
            pos = g['position']
            # Per-gate overrides fall back to trajectory-level defaults
            self._gates.append(Gate(
                id=g['id'],
                position=np.array([pos['x'], pos['y'], pos['z']]),
                yaw=g['yaw'],
                width=g['width'],
                height=g['height'],
                approach_distance=g.get('approach_distance', self.trajectory.approach_distance),
                exit_distance=g.get('exit_distance', self.trajectory.exit_distance),
            ))

    def gates(self) -> list[Gate]:
        return self._gates