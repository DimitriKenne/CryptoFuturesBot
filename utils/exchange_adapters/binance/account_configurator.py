# utils/exchange_adapters/binance/binance_account_configurator.py

"""
Handles account-level configurations for a specific symbol on Binance,
such as setting leverage and margin mode.
"""

import logging
from typing import Any

from binance import AsyncClient
from binance.exceptions import BinanceAPIException

# Add project root to Python path for imports, adjusting for new depth
import sys
from pathlib import Path
# Current file is in utils/exchange_adapters/binance, need to go up 3 levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from exceptions
from utils.exceptions import ExchangeConnectionError
# Import decorator from utils.exchange_adapters.binance.decorators as it's specifically for Binance API errors
from utils.exchange_adapters.binance.decorators import async_retry_api_call

logger = logging.getLogger(__name__)

# Specific Binance API error code for "no change needed" on margin type
MARGIN_TYPE_NO_CHANGE_CODE = -4046

class BinanceAccountConfigurator:
    """
    Handles account-level configurations for a specific symbol on Binance,
    such as setting leverage and margin mode.
    """
    def __init__(self, client: AsyncClient, symbol: str, leverage: int, logger: logging.Logger):
        """
        Initializes the BinanceAccountConfigurator.

        Args:
            client (AsyncClient): An initialized Binance AsyncClient instance.
            symbol (str): The trading pair symbol (e.g., "BTCUSDT").
            leverage (int): The leverage to use for trading.
            logger (logging.Logger): Logger instance for logging messages.
        """
        self._client = client
        self.symbol = symbol.upper()
        self.leverage = leverage
        self.logger = logger

    def set_client(self, client: AsyncClient):
        """
        Allows setting the client after initialization or updating it.
        This is useful if the client is created asynchronously.
        """
        self._client = client

    @async_retry_api_call()
    async def set_leverage(self):
        """
        Sets the leverage for the primary symbol on Binance Futures.
        """
        if not self._client:
            raise ExchangeConnectionError("Binance client not initialized for setting leverage.")
        try:
            leverage_int = int(self.leverage)
            self.logger.info(f"Setting leverage to {leverage_int}x for {self.symbol}...")
            response = await self._client.futures_change_leverage(symbol=self.symbol, leverage=leverage_int)
            self.logger.info(f"Leverage set successfully for {self.symbol}. Response: {response}")
        except BinanceAPIException as e:
            if e.code == MARGIN_TYPE_NO_CHANGE_CODE:
                 self.logger.info(f"Leverage for {self.symbol} is already {self.leverage}x (API code {e.code}). No change needed.")
                 return
            self.logger.error(f"Failed to set leverage for {self.symbol}: {e.code} - {e.message}", exc_info=False)
            raise ExchangeConnectionError(f"Failed to set leverage for {self.symbol}: {e.message}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error setting leverage for {self.symbol}: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Unexpected error setting leverage for {self.symbol}: {e}") from e

    @async_retry_api_call(max_retries=3, initial_delay=2)
    async def set_margin_mode(self, margin_mode: str = "ISOLATED"):
        """
        Sets the margin mode (e.g., ISOLATED or CROSSED) for the trading pair.
        """
        if not self._client:
            raise ExchangeConnectionError("Binance client not initialized for setting margin mode.")
        self.logger.info(f"Attempting to set margin mode to {margin_mode} for {self.symbol}...")
        try:
            resp = await self._client.futures_change_margin_type(symbol=self.symbol, marginType=margin_mode)
            self.logger.info(f"Margin mode set to {margin_mode} for {self.symbol}. Response: {resp}")
        except BinanceAPIException as e:
            if e.code == MARGIN_TYPE_NO_CHANGE_CODE:
                self.logger.info(f"Margin mode for {self.symbol} is already {margin_mode}. No change needed (API code {e.code}).")
                return
            self.logger.error(f"Binance API Exception setting margin mode ({self.symbol}): {e.code} - {e.message}", exc_info=False)
            raise ExchangeConnectionError(f"Failed to set margin mode for {self.symbol}: {e.message}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error setting margin mode for {self.symbol}: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Failed to set margin mode for {self.symbol}: {e}") from e

