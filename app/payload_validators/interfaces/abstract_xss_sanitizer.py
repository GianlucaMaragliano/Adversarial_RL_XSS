from abc import abstractmethod

from app.payload_validators.interfaces.payload_validator_interface import PayloadValidatorInterface


#The sanitizer does not perform any detection
class AbstractXssSanitizer(PayloadValidatorInterface):

    def detect_valid_payload(self, payload: str) -> bool:
        return True
    
    @abstractmethod
    def sanitize_payload(self, payload: str) -> str:
        return super().sanitize_payload(payload)