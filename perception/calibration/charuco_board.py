"""ChArUco board detection and pose solving.

Provides a single high-level entry point `detect_board_pose(rgb, K, cfg)` that
returns a `CharucoDetection` (the board pose in the camera frame, the detected
inner-chessboard corner pixels, and the reprojection error).

The detector is tolerant of both opencv API generations:
- Newer (>= 4.7) `cv2.aruco.ArucoDetector` / `cv2.aruco.CharucoDetector`.
- Older (<= 4.6) `cv2.aruco.detectMarkers` / `cv2.aruco.interpolateCornersCharuco`.
We solve the pose ourselves via `cv2.solvePnP` so the API split for
`estimatePoseCharucoBoard` (which has moved a few times) doesn't matter.

opencv-contrib-python is required because the aruco module is in the contrib
package. The base opencv-python wheel does not ship `cv2.aruco` at all.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CharucoBoardConfig:
    """Geometry of the physical ChArUco target.

    Letter paper (8.5x11 in = 21.6x27.94 cm) fits ~10x13 squares at 2 cm; we
    default to 7x10 leaving generous margins so the entire board (including
    the outer black border ArUco needs) sits on the page.

    `legacy_pattern`: OpenCV 4.7 changed the ChArUco marker placement
    convention. Boards generated with OpenCV <= 4.6 (or popular tools like
    chev.me/arucogen, calib.io) use the LEGACY layout, where the top-left
    square holds an ArUco marker. OpenCV 4.7+ defaults to a NEW layout where
    the top-left square is BLACK. If your printed board has a marker in the
    top-left square, set `legacy_pattern=True`; if it has a black square at
    top-left, set False. Default True because pre-printed boards usually do.
    """
    squares_x: int = 7
    squares_y: int = 10
    square_length_m: float = 0.020
    marker_length_m: float = 0.015
    dictionary_name: str = "DICT_4X4_50"
    legacy_pattern: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CharucoBoardConfig":
        return cls(**d)


@dataclass
class CharucoDetection:
    t_camera_board: np.ndarray  # (4, 4) pose of the board in camera frame
    corners_image: np.ndarray   # (N, 2) sub-pixel chessboard-corner pixel coords
    corner_ids: np.ndarray      # (N,) ids into the board's chessboard corner list
    reprojection_error_px: float
    n_corners: int


def _resolve_dictionary(cv2_module, name: str):
    """Look up an aruco predefined dictionary constant by string name and
    return the dictionary object (API differs between opencv 4.6 and 4.7+)."""
    dict_const = getattr(cv2_module.aruco, name, None)
    if dict_const is None:
        raise ValueError(
            f"opencv.aruco does not expose dictionary '{name}'. "
            f"Available examples: DICT_4X4_50, DICT_4X4_100, DICT_5X5_50."
        )
    if hasattr(cv2_module.aruco, "getPredefinedDictionary"):
        return cv2_module.aruco.getPredefinedDictionary(int(dict_const))
    # Old API
    return cv2_module.aruco.Dictionary_get(int(dict_const))


def _build_board(cv2_module, cfg: CharucoBoardConfig):
    dictionary = _resolve_dictionary(cv2_module, cfg.dictionary_name)
    board = None
    # opencv 4.7+ constructor expects (size_tuple, square, marker, dict)
    if hasattr(cv2_module.aruco, "CharucoBoard"):
        try:
            board = cv2_module.aruco.CharucoBoard(
                (int(cfg.squares_x), int(cfg.squares_y)),
                float(cfg.square_length_m),
                float(cfg.marker_length_m),
                dictionary,
            )
        except TypeError:
            board = None
    if board is None and hasattr(cv2_module.aruco, "CharucoBoard_create"):
        board = cv2_module.aruco.CharucoBoard_create(
            int(cfg.squares_x),
            int(cfg.squares_y),
            float(cfg.square_length_m),
            float(cfg.marker_length_m),
            dictionary,
        )
    if board is None:
        raise RuntimeError(
            "opencv.aruco missing CharucoBoard constructor. Install opencv-contrib-python."
        )
    # opencv 4.7+: switch the marker placement to the legacy (<= 4.6) convention
    # if the user's printed board has a marker in the top-left square.
    if bool(cfg.legacy_pattern) and hasattr(board, "setLegacyPattern"):
        try:
            board.setLegacyPattern(True)
        except Exception:
            pass
    return dictionary, board


def _detect_markers(cv2_module, gray: np.ndarray, dictionary):
    if hasattr(cv2_module.aruco, "ArucoDetector"):
        params = cv2_module.aruco.DetectorParameters()
        det = cv2_module.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = det.detectMarkers(gray)
        return corners, ids
    # old API
    params = cv2_module.aruco.DetectorParameters_create()
    corners, ids, _ = cv2_module.aruco.detectMarkers(
        gray, dictionary, parameters=params
    )
    return corners, ids


def _interpolate_charuco_corners(
    cv2_module,
    gray: np.ndarray,
    marker_corners,
    marker_ids,
    board,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if hasattr(cv2_module.aruco, "CharucoDetector"):
        det = cv2_module.aruco.CharucoDetector(board)
        corners, ids, _marker_c, _marker_i = det.detectBoard(gray)
        return corners, ids
    # old API
    retval, charuco_corners, charuco_ids = cv2_module.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if retval is None or int(retval) < 1:
        return None, None
    return charuco_corners, charuco_ids


def _board_chessboard_corners(board) -> np.ndarray:
    """Get the inner-chessboard corner 3D positions in board frame."""
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    # very old fallback
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float64).reshape(-1, 3)
    raise RuntimeError("CharucoBoard exposes neither getChessboardCorners() nor chessboardCorners")


def detect_board_pose(
    rgb: np.ndarray,
    intrinsic_3x3: np.ndarray,
    cfg: CharucoBoardConfig,
    dist_coeffs: Optional[np.ndarray] = None,
    min_corners: int = 6,
) -> Optional[CharucoDetection]:
    """Detect the ChArUco board in `rgb` and return its pose in the camera frame.

    Returns None if the board is not found or fewer than `min_corners` inner
    chessboard corners were interpolated. The board pose is solved via
    `cv2.solvePnP` on the (board-frame 3D corner, image-pixel 2D corner)
    correspondences, which is consistent across opencv API versions.
    """
    import cv2  # deferred import keeps this module loadable without cv2

    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(rgb, dtype=np.uint8)

    dictionary, board = _build_board(cv2, cfg)
    marker_corners, marker_ids = _detect_markers(cv2, gray, dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        return None

    charuco_corners, charuco_ids = _interpolate_charuco_corners(
        cv2, gray, marker_corners, marker_ids, board
    )
    if (
        charuco_corners is None
        or charuco_ids is None
        or len(charuco_ids) < int(min_corners)
    ):
        return None

    img_pts = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int64).reshape(-1)

    board_corners_3d = _board_chessboard_corners(board)
    if int(ids.max()) >= board_corners_3d.shape[0]:
        return None
    obj_pts = board_corners_3d[ids]

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)
    k = np.asarray(intrinsic_3x3, dtype=np.float64).reshape(3, 3)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float64),
        img_pts.astype(np.float64),
        k,
        dist_coeffs.astype(np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    # Mean reprojection error (px)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, dist_coeffs)
    proj = proj.reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1))))

    rot, _ = cv2.Rodrigues(rvec)
    t_camera_board = np.eye(4, dtype=np.float64)
    t_camera_board[:3, :3] = rot
    t_camera_board[:3, 3] = tvec.reshape(3)

    return CharucoDetection(
        t_camera_board=t_camera_board,
        corners_image=img_pts,
        corner_ids=ids,
        reprojection_error_px=err,
        n_corners=int(img_pts.shape[0]),
    )


def board_extents_m(cfg: CharucoBoardConfig) -> Tuple[float, float]:
    """Return (width_x_m, height_y_m) of the printed board (outer dimensions)."""
    return (
        float(cfg.squares_x) * float(cfg.square_length_m),
        float(cfg.squares_y) * float(cfg.square_length_m),
    )
