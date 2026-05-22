import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
class CharucoBoardSpec:
    """Persisted ChArUco board geometry. Mirrors charuco_board.CharucoBoardConfig
    but lives here so profiles.py has no opencv import dependency."""
    squares_x: int = 7
    squares_y: int = 10
    square_length_m: float = 0.02
    marker_length_m: float = 0.015
    dictionary_name: str = "DICT_4X4_50"


@dataclass
class CalibrationProfile:
    schema_version: str
    created_at_utc: str
    origin_tag_id: int = 0
    tag_family: str = ""
    tag_size_m: float = 0.0
    world_tag_transforms: Dict[str, List[List[float]]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    table_plane: Optional[TablePlane] = None
    robot_world_transform: Optional[List[List[float]]] = None
    charuco_board: Optional[CharucoBoardSpec] = None

    @classmethod
    def new(
        cls,
        origin_tag_id: int = 0,
        tag_family: str = "",
        tag_size_m: float = 0.0,
    ) -> "CalibrationProfile":
        return cls(
            schema_version="perception.calibration.v2",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            origin_tag_id=int(origin_tag_id),
            tag_family=tag_family,
            tag_size_m=float(tag_size_m),
        )

    @classmethod
    def new_charuco(cls, board: CharucoBoardSpec) -> "CalibrationProfile":
        p = cls(
            schema_version="perception.calibration.v3-charuco",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        p.charuco_board = board
        return p

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

    def set_charuco_board(self, spec: CharucoBoardSpec) -> None:
        # Accept either a CharucoBoardSpec or anything dict-like / dataclass-like.
        if isinstance(spec, CharucoBoardSpec):
            self.charuco_board = spec
            return
        if hasattr(spec, "__dict__"):
            data: Dict[str, Any] = {
                k: getattr(spec, k) for k in (
                    "squares_x", "squares_y", "square_length_m",
                    "marker_length_m", "dictionary_name",
                )
            }
        else:
            data = dict(spec)
        self.charuco_board = CharucoBoardSpec(**data)


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
        charuco_data = data.pop("charuco_board", None)
        # Drop unknown keys so old / forward-compatible profiles still load.
        known = {
            "schema_version", "created_at_utc", "origin_tag_id", "tag_family",
            "tag_size_m", "world_tag_transforms", "metrics",
            "robot_world_transform",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        profile = CalibrationProfile(**filtered)
        if table_plane_data is not None:
            profile.table_plane = TablePlane(**table_plane_data)
        if charuco_data is not None:
            profile.charuco_board = CharucoBoardSpec(**charuco_data)
        return profile
