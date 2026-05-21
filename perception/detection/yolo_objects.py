from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .orientation import RotationCode, detect_orientation, rotate_to_upright, unrotate_mask, unrotate_point

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perception.io.frame_packet import CameraPose

SYMMETRIC_CLASSES = frozenset({"cup", "bowl", "wine glass", "vase"})


@dataclass
class YoloDetectorSettings:
    model_path: str = "yolo26n-seg.pt"
    min_confidence: float = 0.30
    class_whitelist: Optional[Sequence[str]] = None
    inference_every_n_frames: int = 2
    min_area_px: int = 64
    hold_frames: int = 8
    track_iou_threshold: float = 0.15
    track_anchor_dist_px: float = 80.0
    anchor_ema_alpha: float = 0.40
    symmetric_classes: frozenset[str] = field(default_factory=lambda: SYMMETRIC_CLASSES)


@dataclass
class YoloDetection:
    label: str
    confidence: float
    class_id: int
    mask_rgb: np.ndarray
    anchor_x: int
    anchor_y: int
    yaw_hint_rad: Optional[float] = None


class YoloObjectDetector:
    def __init__(self, settings: Optional[YoloDetectorSettings] = None):
        self.settings = settings or YoloDetectorSettings()
        self._model = None
        self._last: list[YoloDetection] = []
        self._last_fresh_frame_id: int = -1000
        self._infer_times_ms: list[float] = []
        self._infer_count: int = 0
        self._logged_input_shape: bool = False

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        if inter <= 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def _stabilize_detections(self, fresh: list[YoloDetection]) -> list[YoloDetection]:
        """Per-object temporal persistence + anchor smoothing.

        This avoids flicker when one object drops out for a few frames and keeps
        object ordering stable for downstream consumers.
        """
        hold_limit = int(max(1, self.settings.hold_frames))
        if not hasattr(self, "_held"):
            self._held: list[dict] = []

        used_held: set[int] = set()
        new_held: list[dict] = []
        iou_thr = float(np.clip(self.settings.track_iou_threshold, 0.0, 1.0))
        dist_thr = float(max(1.0, self.settings.track_anchor_dist_px))
        alpha = float(np.clip(self.settings.anchor_ema_alpha, 0.05, 1.0))

        for det in fresh:
            best_idx = -1
            best_score = -1.0
            for i, h in enumerate(self._held):
                if i in used_held:
                    continue
                prev = h["det"]
                if prev.label != det.label:
                    continue
                iou = self._mask_iou(prev.mask_rgb, det.mask_rgb)
                dx = float(prev.anchor_x - det.anchor_x)
                dy = float(prev.anchor_y - det.anchor_y)
                dist = float(np.hypot(dx, dy))
                if iou < iou_thr and dist > dist_thr:
                    continue
                score = iou - (dist / max(1.0, dist_thr)) * 0.05
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                used_held.add(best_idx)
                prev: YoloDetection = self._held[best_idx]["det"]
                smoothed = YoloDetection(
                    label=det.label,
                    confidence=float((1.0 - alpha) * prev.confidence + alpha * det.confidence),
                    class_id=det.class_id,
                    mask_rgb=det.mask_rgb,
                    anchor_x=int(round((1.0 - alpha) * prev.anchor_x + alpha * det.anchor_x)),
                    anchor_y=int(round((1.0 - alpha) * prev.anchor_y + alpha * det.anchor_y)),
                    yaw_hint_rad=det.yaw_hint_rad if det.yaw_hint_rad is not None else prev.yaw_hint_rad,
                )
                new_held.append({"det": smoothed, "missed": 0})
            else:
                new_held.append({"det": det, "missed": 0})

        for i, h in enumerate(self._held):
            if i in used_held:
                continue
            missed = int(h["missed"]) + 1
            if missed <= hold_limit:
                new_held.append({"det": h["det"], "missed": missed})

        self._held = new_held
        stable = [h["det"] for h in self._held]
        # Keep deterministic order so object_id=f"{label}_{idx}" is stable.
        stable.sort(key=lambda d: (d.label, d.anchor_x, d.anchor_y))
        return stable

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.settings.model_path)
        try:
            dev = getattr(self._model, "device", None)
            dtype = next(self._model.model.parameters()).dtype
            print(
                f"[yolo] loaded model={self.settings.model_path} device={dev} "
                f"dtype={dtype}"
            )
        except Exception as exc:
            print(f"[yolo] could not introspect model device/dtype: {exc}")

    def _label_allowed(self, label: str) -> bool:
        if not self.settings.class_whitelist:
            return True
        label_l = label.lower().strip()
        return any(label_l == str(v).lower().strip() for v in self.settings.class_whitelist)

    def _is_symmetric(self, label: str) -> bool:
        return label.lower().strip() in self.settings.symmetric_classes

    @staticmethod
    def _to_image_mask(mask_data: np.ndarray, width: int, height: int) -> np.ndarray:
        mask = np.asarray(mask_data, dtype=np.float32)
        if mask.shape[:2] != (height, width):
            mask = np.array(
                cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )
        return mask > 0.5

    @staticmethod
    def _mask_anchor(mask: np.ndarray) -> tuple[int, int]:
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return 0, 0
        return int(np.round(xs.mean())), int(np.round(ys.mean()))

    @staticmethod
    def _yaw_from_mask(mask: np.ndarray) -> Optional[float]:
        ys, xs = np.where(mask)
        if xs.size < 40:
            return None
        points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        c = np.mean(points, axis=0, keepdims=True)
        centered = points - c
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        ratio = float(eigvals[-1]) / max(float(eigvals[0]), 1e-9)
        if ratio < 1.4:
            return None
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return float(np.arctan2(axis[1], axis[0]))

    def _infer_seg(self, rgb: np.ndarray, rot_code: RotationCode = RotationCode.NONE) -> list[YoloDetection]:
        orig_h, orig_w = rgb.shape[:2]
        infer_rgb = rotate_to_upright(rgb, rot_code)

        if not self._logged_input_shape:
            print(
                f"[yolo] first inference input shape (HxWxC) = "
                f"{infer_rgb.shape[0]}x{infer_rgb.shape[1]}x{infer_rgb.shape[2]}"
            )
            self._logged_input_shape = True

        _t0 = time.perf_counter()
        results = self._model(infer_rgb, verbose=False)
        _dt_ms = (time.perf_counter() - _t0) * 1000.0
        self._infer_times_ms.append(_dt_ms)
        self._infer_count += 1
        if len(self._infer_times_ms) >= 10:
            arr = np.asarray(self._infer_times_ms, dtype=np.float64)
            print(
                f"[yolo] infer #{self._infer_count - 9}..{self._infer_count}  "
                f"mean={arr.mean():.0f}ms  p95={np.percentile(arr, 95):.0f}ms  "
                f"last={arr[-1]:.0f}ms"
            )
            self._infer_times_ms.clear()
        if not results:
            return []
        res = results[0]
        if res.boxes is None or res.masks is None:
            return []

        rot_h, rot_w = infer_rgb.shape[:2]
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        masks = res.masks.data.cpu().numpy()
        out: list[YoloDetection] = []

        for i in range(len(confs)):
            conf = float(confs[i])
            if conf < self.settings.min_confidence:
                continue
            class_id = int(classes[i])
            label = str(res.names.get(class_id, class_id))
            if not self._label_allowed(label):
                continue

            mask_rot = self._to_image_mask(masks[i], width=rot_w, height=rot_h)
            if int(mask_rot.sum()) < int(self.settings.min_area_px):
                continue

            mask_rgb = unrotate_mask(mask_rot.astype(np.uint8), rot_code)
            mask_rgb = mask_rgb > 0

            if mask_rgb.shape[:2] != (orig_h, orig_w):
                mask_rgb = cv2.resize(
                    mask_rgb.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                ) > 0

            rot_ax, rot_ay = self._mask_anchor(mask_rot)
            ax, ay = unrotate_point(rot_ax, rot_ay, rot_code, orig_h, orig_w)

            if self._is_symmetric(label):
                yaw = None
            else:
                yaw = self._yaw_from_mask(mask_rgb)

            out.append(
                YoloDetection(
                    label=label,
                    confidence=conf,
                    class_id=class_id,
                    mask_rgb=mask_rgb,
                    anchor_x=int(ax),
                    anchor_y=int(ay),
                    yaw_hint_rad=yaw,
                )
            )
        return out

    def infer(
        self,
        rgb: np.ndarray,
        frame_id: int,
        camera_pose: CameraPose | None = None,
    ) -> list[YoloDetection]:
        self._ensure_model()
        n = max(1, int(self.settings.inference_every_n_frames))
        if frame_id % n != 0:
            stale = frame_id - self._last_fresh_frame_id
            if stale > int(self.settings.hold_frames):
                return []
            return self._last

        rot_code = detect_orientation(rgb, camera_pose)
        fresh = self._infer_seg(rgb, rot_code)

        stable = self._stabilize_detections(fresh)
        if stable:
            self._last = stable
            self._last_fresh_frame_id = frame_id
        else:
            stale = frame_id - self._last_fresh_frame_id
            if stale > int(self.settings.hold_frames):
                self._last = []

        return self._last
