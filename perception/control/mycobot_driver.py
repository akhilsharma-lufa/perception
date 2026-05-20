from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence


class RobotMotionError(RuntimeError):
    """Raised when a motion command fails or times out."""


def _import_mycobot_class():
    """Import a MyCobot 280 driver class from pymycobot, tolerating layout
    differences across pymycobot versions.

    Different releases expose the 280 class in different places:
        - pymycobot.mycobot280.MyCobot280  (newer)
        - pymycobot.MyCobot280             (some versions)
        - pymycobot.MyCobot                (generic, works for 280)
    """
    attempts = [
        ("pymycobot.mycobot280", "MyCobot280"),
        ("pymycobot", "MyCobot280"),
        ("pymycobot", "MyCobot"),
    ]
    import importlib

    errors: list[str] = []
    for module_name, class_name in attempts:
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"  - import {module_name}: {exc}")
            continue
        cls = getattr(mod, class_name, None)
        if cls is not None:
            return cls
        errors.append(f"  - {module_name}.{class_name}: attribute missing")
    raise RuntimeError(
        "Could not locate a MyCobot driver class in pymycobot. Attempts:\n"
        + "\n".join(errors)
        + "\n"
        "If pymycobot is installed, your version may expose a different class. "
        "Run: python3 -c \"import pymycobot, pkgutil; print(pymycobot.__version__); "
        "print([m.name for m in pkgutil.iter_modules(pymycobot.__path__)])\""
    )


@dataclass
class MyCobotDriverSettings:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1_000_000
    default_speed: int = 30
    coord_mode: int = 1  # 0=angular interpolation, 1=linear interpolation
    home_angles_deg: Sequence[float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    wait_poll_interval_s: float = 0.08
    default_wait_timeout_s: float = 25.0
    position_tolerance_mm: float = 5.0
    # If pymycobot.is_moving() returns ambiguous values (None/-1) for this many
    # consecutive polls AND the position has been stable, assume motion is done.
    motion_stable_polls: int = 4
    motion_stable_tolerance_mm: float = 1.5


class MyCobotDriver:
    """Thin wrapper around pymycobot.MyCobot280.

    pymycobot uses these conventions:
    - send_coords expects [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    - get_coords returns the same; None if the robot has no current pose
    - speed is 1..100, integer
    - mode: 0 angular interpolation, 1 linear (cartesian) interpolation

    We expose meters/radians-friendly entry points internally and convert at
    the pymycobot boundary so the rest of the codebase stays SI.
    """

    def __init__(self, settings: Optional[MyCobotDriverSettings] = None):
        self.settings = settings or MyCobotDriverSettings()
        self._mc = None

    @property
    def is_connected(self) -> bool:
        return self._mc is not None

    def connect(self) -> None:
        if self._mc is not None:
            return
        cls = _import_mycobot_class()
        self._mc = cls(self.settings.port, self.settings.baudrate)
        # The mycobot needs a moment after opening the serial port before it accepts commands.
        time.sleep(0.2)

    def disconnect(self) -> None:
        if self._mc is None:
            return
        try:
            self._mc.close()
        except AttributeError:
            pass  # older pymycobot versions don't expose close()
        self._mc = None

    # ---- Power / state -------------------------------------------------

    def power_on(self) -> None:
        self._require_connected().power_on()
        time.sleep(0.5)

    def power_off(self) -> None:
        self._require_connected().power_off()

    def is_power_on(self) -> bool:
        result = self._require_connected().is_power_on()
        return bool(result == 1)

    def release_all_servos(self) -> None:
        self._require_connected().release_all_servos()

    # ---- Pose I/O ------------------------------------------------------

    def get_coords_mm_deg(self, retries: int = 5) -> list[float]:
        """Returns [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        mc = self._require_connected()
        for _ in range(max(1, retries)):
            coords = mc.get_coords()
            if coords:
                return list(coords)
            time.sleep(0.05)
        raise RobotMotionError("get_coords returned empty after retries")

    def get_angles_deg(self, retries: int = 5) -> list[float]:
        mc = self._require_connected()
        for _ in range(max(1, retries)):
            angles = mc.get_angles()
            if angles:
                return list(angles)
            time.sleep(0.05)
        raise RobotMotionError("get_angles returned empty after retries")

    def send_coords_mm_deg(
        self,
        coords: Sequence[float],
        speed: Optional[int] = None,
        mode: Optional[int] = None,
    ) -> None:
        """coords = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        mc = self._require_connected()
        if len(coords) != 6:
            raise ValueError(f"coords must have 6 elements, got {len(coords)}")
        s = int(speed if speed is not None else self.settings.default_speed)
        m = int(mode if mode is not None else self.settings.coord_mode)
        s = max(1, min(100, s))
        mc.send_coords(list(coords), s, m)

    def send_angles_deg(
        self,
        angles: Sequence[float],
        speed: Optional[int] = None,
    ) -> None:
        mc = self._require_connected()
        if len(angles) != 6:
            raise ValueError(f"angles must have 6 elements, got {len(angles)}")
        s = int(speed if speed is not None else self.settings.default_speed)
        s = max(1, min(100, s))
        mc.send_angles(list(angles), s)

    # ---- Motion completion helpers ------------------------------------

    def wait_until_done(
        self,
        timeout_s: Optional[float] = None,
        strict: bool = True,
    ) -> None:
        """Block until the arm reports motion complete.

        Primary signal: pymycobot.is_moving() returning 0. That endpoint is
        unreliable on some firmware (returns None / -1 / stuck at 1), so we
        also accept "the reported pose has not changed for several polls" as
        evidence the arm has settled. If neither signal fires within the
        timeout, raise RobotMotionError (strict=True) or return quietly
        (strict=False) so callers can decide.
        """
        mc = self._require_connected()
        timeout = float(
            timeout_s if timeout_s is not None else self.settings.default_wait_timeout_s
        )
        deadline = time.monotonic() + max(0.1, timeout)
        # Give the arm a moment to actually start moving before we poll.
        time.sleep(0.15)

        prev_coords: Optional[list[float]] = None
        stable_polls = 0
        tol = float(self.settings.motion_stable_tolerance_mm)
        need_stable = int(self.settings.motion_stable_polls)

        while time.monotonic() < deadline:
            moving: Optional[int]
            try:
                moving = mc.is_moving()
            except Exception:
                moving = None

            if moving == 0:
                return

            # Fallback: if is_moving is unreliable, fall back to detecting that
            # the pose has stopped changing.
            if moving not in (0, 1):
                try:
                    coords = mc.get_coords()
                except Exception:
                    coords = None
                if coords and prev_coords is not None and len(coords) >= 3:
                    dx = coords[0] - prev_coords[0]
                    dy = coords[1] - prev_coords[1]
                    dz = coords[2] - prev_coords[2]
                    if (dx * dx + dy * dy + dz * dz) ** 0.5 <= tol:
                        stable_polls += 1
                        if stable_polls >= need_stable:
                            return
                    else:
                        stable_polls = 0
                if coords:
                    prev_coords = list(coords)

            time.sleep(self.settings.wait_poll_interval_s)

        if strict:
            raise RobotMotionError(f"Motion did not finish within {timeout:.1f}s")

    def is_in_position_mm_deg(
        self,
        coords: Sequence[float],
        tolerance_mm: Optional[float] = None,
    ) -> bool:
        tol = float(
            tolerance_mm if tolerance_mm is not None else self.settings.position_tolerance_mm
        )
        try:
            current = self.get_coords_mm_deg(retries=2)
        except RobotMotionError:
            return False
        dx = current[0] - coords[0]
        dy = current[1] - coords[1]
        dz = current[2] - coords[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5 <= tol

    # ---- Convenience --------------------------------------------------

    def home(self, speed: Optional[int] = None, wait: bool = True) -> None:
        self.send_angles_deg(self.settings.home_angles_deg, speed=speed)
        if wait:
            self.wait_until_done()

    # ---- Internal -----------------------------------------------------

    def _require_connected(self):
        if self._mc is None:
            raise RuntimeError("MyCobotDriver.connect() not called")
        return self._mc

    # Allow `with MyCobotDriver(...) as drv:` usage.
    def __enter__(self) -> "MyCobotDriver":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.is_connected:
                self.release_all_servos()
        except Exception:
            pass
        self.disconnect()
