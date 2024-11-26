from abc import ABC, abstractmethod
from typing import Tuple

class PayloadValidatorInterface(ABC):
    '''
        This interface exposes the methods to validate the payloads
    '''

    @abstractmethod
    def detect_valid_payload(self, payload:str)-> bool:
        """
        This methods, given a specified payload, analyzes it and detects if it safe to be executed

        Parameters
        ----------
        payload : str
            Analyzed paylod

        Returns
        -------
        bool
            True if the payload is valid, False otherwise

        Raises
        ------
        NotImplementedError
            The method is not implemented
        """
        raise NotImplementedError
    
    @abstractmethod
    def sanitize_payload(self,payload:str)->str:
        """ 
        This methods, given a specified payload, analyzes it and sanitizes it

        Parameters
        ----------
        payload : str
            Payload to be sanitized

        Returns
        -------
        str
            Sanitized payload

        Raises
        ------
        NotImplementedError
            The method is not implemented
        """
        raise NotImplementedError
    
    def validate(self,payload:str) -> Tuple[bool, str]:
        sanitized = self.sanitize_payload(payload)
        return self.detect_valid_payload(sanitized), sanitized



