from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..models import DetectionOverlay, FramePacket


@dataclass
class YoloSettings:
    model_path: str = "yolo26n-seg.pt"
    min_confidence: float = 0.35
    inference_every_n_frames: int = 2
    min_mask_area_px: int = 64
    confidence_floor: int = 1


class YoloDistanceProcessor:
    """
    Packet processor:
    - runs YOLO on RGB
    - estimates robust object distance from segmentation mask pixels
    - writes detection overlays into packet.overlays
    """

    def __init__(self, settings: Optional[YoloSettings] = None):
        self.settings = settings or YoloSettings()
        self._model = None
        self._last_overlays: List[DetectionOverlay] = []

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

    @staticmethod
    def _to_image_mask(mask_data: np.ndarray, width: int, height: int) -> np.ndarray:
        mask = np.asarray(mask_data, dtype=np.float32)
        if mask.shape[:2] != (height, width):
            mask = np.array(
                cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )
        return mask > 0.5

    def _estimate_distance_m(
        self, depth: np.ndarray, confidence: np.ndarray, mask_depth: np.ndarray
    ):
        if mask_depth.size == 0:
            return None

        valid = mask_depth & np.isfinite(depth) & (depth > 0)
        if confidence is not None and confidence.size != 0:
            if confidence.shape == depth.shape:
                valid = valid & (confidence >= self.settings.confidence_floor)

        vals = depth[valid]
        if vals.size < 20:
            return None
        return float(np.median(vals))

    def _infer(self, packet: FramePacket) -> List[DetectionOverlay]:
        self._ensure_model()
        results = self._model(packet.rgb, verbose=False)
        if not results:
            return []

        res = results[0]
        if res.boxes is None or res.masks is None:
            return []

        rgb_h, rgb_w = packet.rgb.shape[:2]
        depth_h, depth_w = packet.depth.shape[:2]
        overlays: List[DetectionOverlay] = []
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        masks = res.masks.data.cpu().numpy()

        for i in range(len(xyxy)):
            conf = float(confs[i])
            if conf < self.settings.min_confidence:
                continue

            x1, y1, x2, y2 = xyxy[i].tolist()
            x1 = max(0, min(rgb_w - 1, int(x1)))
            y1 = max(0, min(rgb_h - 1, int(y1)))
            x2 = max(0, min(rgb_w - 1, int(x2)))
            y2 = max(0, min(rgb_h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            mask_rgb = self._to_image_mask(masks[i], width=rgb_w, height=rgb_h)
            if int(mask_rgb.sum()) < self.settings.min_mask_area_px:
                continue
            mask_depth = self._to_image_mask(masks[i], width=depth_w, height=depth_h)
            if int(mask_depth.sum()) < self.settings.min_mask_area_px:
                continue

            class_id = int(classes[i])
            label = str(res.names.get(class_id, class_id))
            distance_m = self._estimate_distance_m(
                packet.depth, packet.confidence, mask_depth
            )
            anchor_x = x1
            anchor_y = y1
            ys, xs = np.where(mask_rgb)
            if ys.size > 0 and xs.size > 0:
                anchor_x = int(xs.min())
                anchor_y = int(ys.min())

            overlays.append(
                DetectionOverlay(
                    label=label,
                    confidence=conf,
                    mask=mask_rgb,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    distance_m=distance_m,
                )
            )
        return overlays

    def __call__(self, packet: FramePacket) -> FramePacket:
        n = max(1, int(self.settings.inference_every_n_frames))
        if packet.frame_id % n != 0:
            packet.overlays = self._last_overlays
            return packet

        self._last_overlays = self._infer(packet)
        packet.overlays = self._last_overlays
        return packet
