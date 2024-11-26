from app.payload_validators.dummy_detector import DummyDetector

from app.payload_validators.interfaces.payload_validator_interface import PayloadValidatorInterface


class ValidatorSelector:
    def __init__(self):
        self.validators = {"dummy_detector":DummyDetector}

    def select(self, payload: str) -> PayloadValidatorInterface:
        if payload in self.validators.keys():
            return self.validators[payload]()
        raise ValueError(f"No validator found for {payload}")