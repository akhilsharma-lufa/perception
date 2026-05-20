from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .mycobot_driver import MyCobotDriver


@dataclass
class GripperSettings:
    # AG gripper TCP offset along tool0-Z (meters). Refined in v2 via touch-cal.
    tip_offset_z_m: float = 0.095
    default_speed: int = 50
    # pymycobot value range for AG: 0 = fully open, 100 = fully closed.
    open_value: int = 0
    close_value_default: int = 80
    # Close value tuned for the small plastic shot cup (~4 cm dia). Tune live.
    close_value_shot_cup: int = 40
    # Time to wait when pymycobot does not expose is_gripper_moving cleanly.
    blocking_wait_s: float = 1.0
    # Gripper mode: 0 = transparent (drive cleanly), 1 = io protocol on the AG.
    mode: int = 0


class Gripper:
    """AG (Adaptive Gripper) helpers wrapping pymycobot's gripper API.

    pymycobot's gripper API has varied across firmware revisions; this class
    centralizes that surface so the rest of the codebase calls only
    open(), close(), set_width(0..100) and is_moving().
    """

    def __init__(self, driver: "MyCobotDriver", settings: Optional[GripperSettings] = None):
        self.driver = driver
        self.settings = settings or GripperSettings()
        self._mode_set = False

    def _mc(self):
        # Reaches through the driver to the pymycobot instance.
        return self.driver._require_connected()  # noqa: SLF001 — internal sibling access

    def ensure_mode(self) -> None:
        if self._mode_set:
            return
        mc = self._mc()
        try:
            mc.set_gripper_mode(int(self.settings.mode))
        except AttributeError:
            # Older firmware: no set_gripper_mode. Continue.
            pass
        self._mode_set = True

    def set_width(self, value_0_100: int, speed: Optional[int] = None, wait: bool = True) -> None:
        """value: 0 = fully open, 100 = fully closed. pymycobot's convention."""
        self.ensure_mode()
        mc = self._mc()
        v = int(max(0, min(100, value_0_100)))
        s = int(speed if speed is not None else self.settings.default_speed)
        s = max(1, min(100, s))
        mc.set_gripper_value(v, s)
        if wait:
            self.wait_until_done()

    def open(self, speed: Optional[int] = None, wait: bool = True) -> None:
        self.set_width(self.settings.open_value, speed=speed, wait=wait)

    def close(
        self,
        speed: Optional[int] = None,
        wait: bool = True,
        value: Optional[int] = None,
    ) -> None:
        v = int(value if value is not None else self.settings.close_value_default)
        self.set_width(v, speed=speed, wait=wait)

    def close_on_shot_cup(self, speed: Optional[int] = None, wait: bool = True) -> None:
        self.set_width(self.settings.close_value_shot_cup, speed=speed, wait=wait)

    def is_moving(self) -> Optional[bool]:
        mc = self._mc()
        try:
            state = mc.is_gripper_moving()
        except AttributeError:
            return None
        if state is None:
            return None
        return bool(state == 1)

    def wait_until_done(self, timeout_s: float = 2.0) -> None:
        moving = self.is_moving()
        if moving is None:
            # Firmware doesn't expose the polling endpoint; fall back to a fixed wait.
            time.sleep(float(self.settings.blocking_wait_s))
            return
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            moving = self.is_moving()
            if moving is False:
                return
            time.sleep(0.05)

    def get_value(self) -> Optional[int]:
        mc = self._mc()
        try:
            v = mc.get_gripper_value()
        except AttributeError:
            return None
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
