class ExceptionWithMessage(Exception):
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)    


class ModelNotAvailableError(ExceptionWithMessage):
    pass


class OperationalError(ExceptionWithMessage):
    pass
