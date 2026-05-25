"""Re-export of the interactive touch-calibration CLI.

Lets the teammate run ``python -m perception.cup_locator.calibrate`` without
having to learn that the implementation lives under ``perception.demos``. All
flags (``--port``, ``--profile``, ``--squares-x``, ``--square-mm``, ...) are
identical to ``perception.demos.touch_calibrate``.
"""
from perception.demos.touch_calibrate import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
