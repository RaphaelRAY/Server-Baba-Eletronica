class InferenceEngine:
    """Stub inference engine."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        # Real implementation would load a ML model

    def infer(self, frame):
        """Return empty detections list."""
        return []
