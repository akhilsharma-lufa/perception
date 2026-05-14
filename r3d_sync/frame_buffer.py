from collections import deque
from threading import Lock
from typing import Optional

from .models import FramePacket, StreamStats


class LatestFrameBuffer:
    def __init__(self, maxlen: int):
        self._queue = deque(maxlen=maxlen)
        self._lock = Lock()
        self.stats = StreamStats()

    def push(self, packet: FramePacket):
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self.stats.frames_dropped += 1
            self._queue.append(packet)
            self.stats.frames_enqueued += 1

    def pop_latest(self) -> Optional[FramePacket]:
        with self._lock:
            if not self._queue:
                return None
            latest = self._queue[-1]
            self._queue.clear()
            return latest
