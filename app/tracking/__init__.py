"""Hand tracking integration — gesture-based 3D model control via MediaPipe."""

from app.tracking.gesture_mapper import GestureMapper
from app.tracking.model_manager import ensure_model

__all__ = ["GestureMapper", "ensure_model"]
