class XSSException(Exception):
    def __init__(self):
        self.message = "The payload contains an XSS attack"
        super().__init__(self.message )