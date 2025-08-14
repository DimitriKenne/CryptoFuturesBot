# config/exchange_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, Literal
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """
    Defines configuration parameters for exchange interactions.
    """
    exchange: Literal['binance'] = 'binance' # Supported: 'binance'
    testnet: bool = True # Use exchange testnet
    api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    default_type: Literal['future', 'spot'] = 'future'
    options: Dict[str, Any] = field(default_factory=lambda: {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 10000 # Max is 60000 for Binance
    })
    rateLimit: int = 100 # Max requests per second
    enableRateLimit: bool = True
    verbose: bool = False # Set to True for verbose API output (debugging)
    timeout: int = 30000 # API call timeout in milliseconds
    tld: Literal['com', 'us'] = field(default_factory=lambda: os.getenv('BINANCE_TLD', 'com')) # 'com' for international, 'us' for Binance.us

    # NEW: Price and Quantity Precision and Minimums (Crucial for Order Management)
    price_precision: int = 4 # Number of decimal places for prices (e.g., 4 for ADAUSDT)
    quantity_precision: int = 2 # Number of decimal places for quantities (e.g., 2 for ADAUSDT)
    min_quantity: float = 1.0 # Minimum order quantity for the symbol (e.g., 1.0 ADA)
    min_notional: float = 5.0 # Minimum order notional value in USDT (e.g., 5.0 USDT)


    def __post_init__(self):
        if self.exchange not in ['binance']:
            raise ValueError("Unsupported exchange. Currently only 'binance' is supported.")
        if not isinstance(self.testnet, bool):
            raise TypeError("testnet must be a boolean.")
        if not isinstance(self.api_key, str) or not isinstance(self.api_secret, str):
            raise TypeError("api_key and api_secret must be strings.")
        if self.default_type not in ['future', 'spot']:
            raise ValueError("default_type must be 'future' or 'spot'.")
        if not isinstance(self.options, dict):
            raise TypeError("options must be a dictionary.")
        if not isinstance(self.rateLimit, int) or self.rateLimit <= 0:
            raise ValueError("rateLimit must be a positive integer.")
        if not isinstance(self.enableRateLimit, bool):
            raise TypeError("enableRateLimit must be a boolean.")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValueError("timeout must be a positive integer.")
        if self.tld not in ['com', 'us']:
            raise ValueError("tld must be 'com' or 'us'.")
        
        # Validation for API keys if exchange is Binance
        if self.exchange == 'binance':
            if not self.api_key or not self.api_secret:
                logger.warning("Binance API Key or Secret is missing but required for the selected exchange type. Trading operations may fail.")
        
        # New validations for precision and minimums
        if not isinstance(self.price_precision, int) or self.price_precision < 0:
            raise ValueError("price_precision must be a non-negative integer.")
        if not isinstance(self.quantity_precision, int) or self.quantity_precision < 0:
            raise ValueError("quantity_precision must be a non-negative integer.")
        if not isinstance(self.min_quantity, (int, float)) or self.min_quantity < 0:
            raise ValueError("min_quantity must be a non-negative number.")
        if not isinstance(self.min_notional, (int, float)) or self.min_notional < 0:
            raise ValueError("min_notional must be a non-negative number.")


# Default configuration instance
DEFAULT_EXCHANGE_CONFIG = ExchangeConfig()
