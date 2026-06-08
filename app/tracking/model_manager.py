"""MediaPipe HandLandmarker model file management.

Downloads the hand_landmarker.task model from Google Storage on first use
if not already bundled with the project.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Look for model in three locations (priority order):
# 1. app/tracking/hand_landmarker.task (bundled with package)
# 2. hand_tracking/hand_landmarker.task (legacy prototype location)
# 3. Download to app/tracking/hand_landmarker.task

_PACKAGE_DIR = Path(__file__).parent
_LEGACY_DIR = _PACKAGE_DIR.parent.parent / "hand_tracking"

_SEARCH_PATHS = [
    _PACKAGE_DIR / "hand_landmarker.task",
    _LEGACY_DIR / "hand_landmarker.task",
]


def ensure_model() -> Path:
    """Return the path to the HandLandmarker model, downloading if needed.

    Returns
    -------
    Path
        Path to the hand_landmarker.task model file.

    Raises
    ------
    RuntimeError
        If download fails and no local copy exists.
    """
    for path in _SEARCH_PATHS:
        if path.exists():
            return path

    # Download to package directory
    target = _SEARCH_PATHS[0]
    print("Hand tracking model not found. Downloading from Google Storage...")
    try:
        urllib.request.urlretrieve(MODEL_URL, target)
        print(f"Model downloaded to {target}")
        return target
    except Exception as exc:
        raise RuntimeError(
            f"Could not download hand landmarker model: {exc}\n"
            f"Download manually from:\n  {MODEL_URL}\n"
            f"Place at:\n  {target}"
        ) from exc
