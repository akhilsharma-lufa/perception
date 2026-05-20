import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TablePlane:
    normal_world: List[float]
    origin_world: List[float]
    inlier_ratio: float
    mean_abs_residual_m: float

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = np.asarray(self.normal_world, dtype=np.float64).reshape(3)
        o = np.asarray(self.origin_world, dtype=np.float64).reshape(3)
        return n, o


@dataclass
class CalibrationProfile:
    schema_version: str
    created_at_utc: str
    origin_tag_id: int
    tag_family: str
    tag_size_m: float
    world_tag_transforms: Dict[str, List[List[float]]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    table_plane: Optional[TablePlane] = None
    robot_world_transform: Optional[List[List[float]]] = None

    @classmethod
    def new(
        cls,
        origin_tag_id: int,
        tag_family: str,
        tag_size_m: float,
    ) -> "CalibrationProfile":
        return cls(
            schema_version="perception.calibration.v2",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            origin_tag_id=int(origin_tag_id),
            tag_family=tag_family,
            tag_size_m=float(tag_size_m),
        )

    def set_world_tag_transform(self, tag_id: int, t_world_tag: np.ndarray):
        self.world_tag_transforms[str(int(tag_id))] = np.asarray(
            t_world_tag, dtype=np.float64
        ).tolist()

    def get_world_tag_transform(self, tag_id: int) -> np.ndarray | None:
        key = str(int(tag_id))
        if key not in self.world_tag_transforms:
            return None
        return np.asarray(self.world_tag_transforms[key], dtype=np.float64)

    def set_robot_world_transform(self, t_robot_world: np.ndarray) -> None:
        self.robot_world_transform = np.asarray(
            t_robot_world, dtype=np.float64
        ).reshape(4, 4).tolist()

    def get_robot_world_transform(self) -> Optional[np.ndarray]:
        if self.robot_world_transform is None:
            return None
        return np.asarray(self.robot_world_transform, dtype=np.float64).reshape(4, 4)

    def set_table_plane(
        self,
        normal_world: np.ndarray,
        origin_world: np.ndarray,
        inlier_ratio: float,
        mean_abs_residual_m: float,
    ) -> None:
        self.table_plane = TablePlane(
            normal_world=np.asarray(normal_world, dtype=np.float64).reshape(3).tolist(),
            origin_world=np.asarray(origin_world, dtype=np.float64).reshape(3).tolist(),
            inlier_ratio=float(inlier_ratio),
            mean_abs_residual_m=float(mean_abs_residual_m),
        )


class CalibrationProfileIO:
    @staticmethod
    def save(profile: CalibrationProfile, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(asdict(profile), indent=2))

    @staticmethod
    def load(path: str) -> CalibrationProfile:
        data = json.loads(Path(path).read_text())
        table_plane_data = data.pop("table_plane", None)
        profile = CalibrationProfile(**data)
        if table_plane_data is not None:
            profile.table_plane = TablePlane(**table_plane_data)
        return profile
