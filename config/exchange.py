# config/exchange.py

from dataclasses import dataclass, field
from typing import Dict, Any, Literal
import os

@dataclass
class ExchangeConfig:
    """
    Exchange connection and order settings.
    """
    exchange: Literal['binance'] = 'binance'
    testnet: bool = True
    api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    default_type: Literal['future', 'spot'] = 'future'
    options: Dict[str, Any] = field(default_factory=lambda: {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 10000
    })
    rateLimit: int = 100
    enableRateLimit: bool = True
    verbose: bool = False
    timeout: int = 30000
    tld: Literal['com', 'us'] = field(default_factory=lambda: os.getenv('BINANCE_TLD', 'com'))

    price_precision: int = 2 # Price precision for orders to retrieve from your exchange depending on the market
    quantity_precision: int = 3 # Quantity precision for orders to retrieve from your exchange depending on the market
    min_quantity: float = 0.001 # Minimum quantity for orders to retrieve from your exchange depending on the market
    min_notional: float = 5.0 # Minimum notional for orders to retrieve from your exchange depending on the market

# Default config instance
DEFAULT_EXCHANGE_CONFIG = ExchangeConfig()
