import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class CalibrationProfile:
    schema_version: str
    created_at_utc: str
    origin_tag_id: int
    tag_family: str
    tag_size_m: float
    world_tag_transforms: Dict[str, List[List[float]]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        origin_tag_id: int,
        tag_family: str,
        tag_size_m: float,
    ) -> "CalibrationProfile":
        return cls(
            schema_version="perception.calibration.v1",
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


class CalibrationProfileIO:
    @staticmethod
    def save(profile: CalibrationProfile, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(asdict(profile), indent=2))

    @staticmethod
    def load(path: str) -> CalibrationProfile:
        data = json.loads(Path(path).read_text())
        return CalibrationProfile(**data)
