class ExceptionWithMessage(Exception):
    """Base exception that includes a message to be returned by the API."""

    def __init__(self, message: str):
        """
        Args:
            message: A user-friendly message explaining the error.
        """
        self.message = message
        super().__init__(message)


class ModelNotAvailableError(ExceptionWithMessage):
    """A requested model is unavailable."""
    pass


class OperationalError(ExceptionWithMessage):
    """An operational error has occurred."""
    pass
