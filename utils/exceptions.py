# utils/exceptions.py

"""
Custom exception definitions for the trading bot project.

Centralizes custom error types to improve error handling and clarity
throughout the application.
"""

class TemporalSafetyError(Exception):
    """
    Custom exception raised when temporal safety validation fails during feature engineering.
    This indicates potential lookahead bias in the generated features.
    """
    def __init__(self, message: str, features: list = None):
        """
        Initializes the TemporalSafetyError.

        Args:
            message (str): A descriptive error message.
            features (list, optional): A list of feature names that failed validation.
                                       Defaults to None.
        """
        super().__init__(message)
        self.features = features if features is not None else []
        self.message = message # Store message explicitly

class ExchangeConnectionError(Exception):
    """
    Exception raised for errors related to exchange connectivity or authentication.
    This includes issues during client initialization, API key problems,
    or general network connection failures with the exchange.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        """
        Initializes the ExchangeConnectionError.

        Args:
            message (str): A descriptive error message.
            original_exception (Exception, optional): The underlying exception that caused this error.
                                                      Defaults to None.
        """
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception

class OrderExecutionError(Exception):
    """
    Exception raised for errors during order placement, modification, or cancellation.
    This includes issues like insufficient funds, invalid order parameters,
    or exchange-side rejections of order requests.
    """
    def __init__(self, message: str, order_details: dict = None, original_exception: Exception = None):
        """
        Initializes the OrderExecutionError.

        Args:
            message (str): A descriptive error message.
            order_details (dict, optional): A dictionary containing details about the order
                                            that failed (e.g., symbol, side, quantity, price, error code).
                                            Defaults to None.
            original_exception (Exception, optional): The underlying exception that caused this error.
                                                      Defaults to None.
        """
        super().__init__(message)
        self.message = message
        self.order_details = order_details if order_details is not None else {}
        self.original_exception = original_exception

class ModelAnalysisError(Exception):
    """Custom exception for errors during model analysis."""
    pass

# You can define other custom exceptions here as your project grows,
# e.g., ConfigurationError, DataProcessingError.
