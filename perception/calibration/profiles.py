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
class LensCalibration:
    """Camera-intrinsic lens model: refined K and distortion coefficients,
    estimated by `perception.demos.calibrate_lens`.

    Backward compatibility note: `CalibrationProfile.lens_calibration` is
    optional. Downstream code (PnP, depth unprojection) treats `None` as
    "assume pinhole / zero distortion" — i.e. the prior behaviour. So
    existing profiles, existing callers, and existing pipelines that don't
    know about this field continue to work unchanged.
    """
    k_refined: List[List[float]]            # 3x3, at image_size_wh resolution
    dist_coeffs: List[float]                # 5: (k1, k2, p1, p2, k3)
    image_size_wh: List[int]                # (W, H) the K is at
    rms_reproj_px: float
    n_captures: int
    captured_at: str

    def k_array(self) -> np.ndarray:
        return np.asarray(self.k_refined, dtype=np.float64).reshape(3, 3)

    def dist_array(self) -> np.ndarray:
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)

    def k_for_resolution(self, w: int, h: int) -> np.ndarray:
        """Return K_refined scaled to the requested (W, H) image resolution.
        `dist_coeffs` are dimensionless (operate on normalised image coords)
        and remain valid at any resolution as long as K is scaled to match."""
        k = self.k_array()
        calib_w, calib_h = int(self.image_size_wh[0]), int(self.image_size_wh[1])
        if (int(w), int(h)) == (calib_w, calib_h):
            return k
        sx = float(w) / float(calib_w)
        sy = float(h) / float(calib_h)
        out = np.array(k, dtype=np.float64, copy=True)
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out


@dataclass
class CharucoBoardSpec:
    """Persisted ChArUco board geometry. Mirrors charuco_board.CharucoBoardConfig
    but lives here so profiles.py has no opencv import dependency."""
    squares_x: int = 7
    squares_y: int = 10
    square_length_m: float = 0.02
    marker_length_m: float = 0.015
    dictionary_name: str = "DICT_4X4_50"
    legacy_pattern: bool = True


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
    lens_calibration: Optional[LensCalibration] = None

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

    def set_lens_calibration(self, lc: LensCalibration) -> None:
        """Attach a lens calibration. Downstream code that opts in (by passing
        `lens_calibration=` to `localize_objects_rgbd` or `dist_coeffs=` to
        `detect_board_pose`) will use it; legacy callers keep their pinhole
        behaviour."""
        self.lens_calibration = lc

    def set_charuco_board(self, spec: CharucoBoardSpec) -> None:
        # Accept either a CharucoBoardSpec or anything dict-like / dataclass-like.
        if isinstance(spec, CharucoBoardSpec):
            self.charuco_board = spec
            return
        if hasattr(spec, "__dict__"):
            data: Dict[str, Any] = {
                k: getattr(spec, k) for k in (
                    "squares_x", "squares_y", "square_length_m",
                    "marker_length_m", "dictionary_name", "legacy_pattern",
                ) if hasattr(spec, k)
            }
        else:
            data = dict(spec)
        self.charuco_board = CharucoBoardSpec(**data)


class CalibrationProfileIO:
    @staticmethod
    def save(profile: CalibrationProfile, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(profile)
        # Backward compatibility: do not emit `lens_calibration` at all when
        # it is unset. The other team's profile loaders (which we cannot
        # inspect) saw a fixed set of top-level keys before this field was
        # added; profiles produced by code paths that never set lens
        # calibration must stay byte-identical to the prior format. The new
        # field appears in the JSON only on profiles that actively carry a
        # lens calibration.
        if data.get("lens_calibration") is None:
            data.pop("lens_calibration", None)
        path_obj.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: str) -> CalibrationProfile:
        data = json.loads(Path(path).read_text())
        table_plane_data = data.pop("table_plane", None)
        charuco_data = data.pop("charuco_board", None)
        lens_data = data.pop("lens_calibration", None)
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
        if lens_data is not None:
            # Forward-compat: ignore unknown fields if a future calibrator adds
            # them (e.g. higher-order distortion). Keep the loader tolerant.
            allowed = {f for f in (
                "k_refined", "dist_coeffs", "image_size_wh",
                "rms_reproj_px", "n_captures", "captured_at",
            )}
            profile.lens_calibration = LensCalibration(
                **{k: v for k, v in lens_data.items() if k in allowed}
            )
        return profile
