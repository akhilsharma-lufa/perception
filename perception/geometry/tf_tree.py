from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .transforms import invert_transform


@dataclass
class TimedTransform:
    source_frame: str
    target_frame: str
    t_source_target: np.ndarray
    ts_monotonic: float
    quality: float = 1.0


class TransformTree:
    """Minimal transform graph with direct/inverse lookup."""

    def __init__(self):
        self._edges: Dict[Tuple[str, str], TimedTransform] = {}

    def set_transform(self, tf: TimedTransform):
        self._edges[(tf.source_frame, tf.target_frame)] = tf

    def get_transform(self, source_frame: str, target_frame: str) -> Optional[TimedTransform]:
        if source_frame == target_frame:
            return TimedTransform(
                source_frame=source_frame,
                target_frame=target_frame,
                t_source_target=np.eye(4, dtype=np.float64),
                ts_monotonic=0.0,
                quality=1.0,
            )
        key = (source_frame, target_frame)
        if key in self._edges:
            return self._edges[key]
        inv_key = (target_frame, source_frame)
        if inv_key not in self._edges:
            return None
        tf = self._edges[inv_key]
        return TimedTransform(
            source_frame=source_frame,
            target_frame=target_frame,
            t_source_target=invert_transform(tf.t_source_target),
            ts_monotonic=tf.ts_monotonic,
            quality=tf.quality,
        )
