from dataclasses import dataclass, field
from typing import Dict, Any, Literal
import os

SYMBOL_PARAMS: Dict[str, Dict[str, Any]] = {
    "BTCUSDT": {
        "min_notional": 5.0,
        "min_quantity": 0.001,
        "price_precision": 2,
        "quantity_precision": 3
    },
    "ADAUSDT": {
        "min_notional": 1.0,
        "min_quantity": 1.0,
        "price_precision": 4,
        "quantity_precision": 1
    },
    "XRPUSDT": {
    "min_notional": 5.0,
    "min_quantity": 1.0,
    "price_precision": 4,
    "quantity_precision": 0
}
    # Add more pairs as needed
}

@dataclass
class ExchangeConfig:
    """
    Exchange connection and order settings. 
    Store all static configuration related to connecting to an exchange (e.g. Binance).
    """
    exchange: Literal['binance'] = 'binance'
    testnet: bool = False
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

    def get_symbol_params(self, symbol: str) -> Dict[str, Any]:
        """Retrieve symbol-specific parameters or raise error if missing."""
        params = SYMBOL_PARAMS.get(symbol)
        if params is None:
            params = SYMBOL_PARAMS["ADAUSDT"]
            print(f"Symbol parameters not found for {symbol}. Using ADAUSDT defaults.")
        return params

DEFAULT_EXCHANGE_CONFIG = ExchangeConfig()