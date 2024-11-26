from abc import abstractmethod

from app.payload_validators.interfaces.payload_validator_interface import PayloadValidatorInterface

#The detector does not perform any sanitization
class AbstractXssDetector(PayloadValidatorInterface):
    def sanitize_payload(self, payload: str) -> str:
        return payload
    
    @abstractmethod
    def detect_valid_payload(self, payload: str) -> bool:
        return super().detect_valid_payload(payload)