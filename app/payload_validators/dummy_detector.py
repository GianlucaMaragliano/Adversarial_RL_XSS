
from app.payload_validators.interfaces.abstract_xss_detector import AbstractXssDetector


class DummyDetector(AbstractXssDetector):
    def detect_valid_payload(self, payload: str) -> bool:
        return True
    