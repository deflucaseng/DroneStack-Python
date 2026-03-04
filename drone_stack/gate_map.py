from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Gate:
    name: str
    position: np.ndarray   # [x, y, z] in world frame (metres)
    heading: float = 0.0   # yaw in radians (gate normal direction)


class GateMap:
    """Known gate positions in world frame, loaded from a YAML file.

    Expected YAML schema::

        gates:
          - name: gate_1
            position: [5.0, 0.0, -1.5]
            heading: 0.0
          - name: gate_2
            position: [10.0, 3.0, -1.5]
            heading: 0.785
    """

    def __init__(self, yaml_path: str) -> None:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        self._gates: list[Gate] = [
            Gate(
                name=g["name"],
                position=np.array(g["position"], dtype=float),
                heading=float(g.get("heading", 0.0)),
            )
            for g in data["gates"]
        ]

    def gates(self) -> list[Gate]:
        return self._gates

    def nearest_gate(self, pos: np.ndarray) -> Gate:
        return min(self._gates, key=lambda g: float(np.linalg.norm(g.position - pos)))
