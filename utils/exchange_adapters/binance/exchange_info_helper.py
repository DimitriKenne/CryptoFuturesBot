# utils/exchange_adapters/binance/binance_exchange_info_helper.py

"""
Provides helper functionalities for fetching, caching, and querying Binance
exchange information, including symbol filters for precision and minimums.
"""

import logging
import math
from typing import Dict, Any, Optional, Union

import pandas as pd

from binance import AsyncClient

# Add project root to Python path for imports, adjusting for new depth
import sys
from pathlib import Path
# Current file is in utils/exchange_adapters/binance, need to go up 3 levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from config and exceptions
from config.params import FLOAT_EPSILON
from utils.exceptions import ExchangeConnectionError
# Import decorator from utils.exchange_adapters.binance.decorators as it's specifically for Binance API errors
from utils.exchange_adapters.binance.decorators import async_retry_api_call

logger = logging.getLogger(__name__)

class BinanceExchangeInfoHelper:
    """
    Manages fetching and caching Binance exchange information, including symbol filters.
    Provides methods to query precision, minimums, and adjust values based on these filters.
    """
    def __init__(self, client: AsyncClient, exchange_config: Any, logger: logging.Logger):
        """
        Initializes the BinanceExchangeInfoHelper.

        Args:
            client (AsyncClient): An initialized Binance AsyncClient instance.
            exchange_config (Any): The exchange-specific configuration object
                                   (e.g., AppConfig.exchange).
            logger (logging.Logger): Logger instance for logging messages.
        """
        self._client = client
        self.exchange_config = exchange_config
        self.logger = logger
        self.symbol_info_cache: Dict[str, Any] = {}
        self._exchange_info_fetched = False

        # Fallback values from AppConfig's ExchangeConfig for cases where API info isn't available
        self._default_quantity_precision = self.exchange_config.quantity_precision
        self._default_price_precision = self.exchange_config.price_precision
        self._default_min_quantity = self.exchange_config.min_quantity
        self._default_min_notional = self.exchange_config.min_notional

    def set_client(self, client: AsyncClient):
        """
        Allows setting the client after initialization or updating it.
        This is useful if the client is created asynchronously.
        """
        self._client = client

    @async_retry_api_call(max_retries=2, initial_delay=0.5)
    async def fetch_and_cache_info(self):
        """
        Fetches and caches exchange information (symbol details, filters) from Binance.
        This should be called once during the adapter's async setup.
        """
        if self._exchange_info_fetched:
            self.logger.debug("Exchange info already fetched.")
            return
        if not self._client:
            raise ExchangeConnectionError("Binance client not initialized for fetching exchange info.")
        try:
            self.logger.info("Fetching Binance exchange information...")
            info = await self._client.futures_exchange_info()
            if info and 'symbols' in info:
                self.symbol_info_cache = {s['symbol']: s for s in info['symbols']}
                self._exchange_info_fetched = True
                self.logger.info(f"Cached exchange info for {len(self.symbol_info_cache)} symbols.")
            else:
                self.logger.warning("Received unexpected format for exchange info from Binance.")
                self.symbol_info_cache = {}
                self._exchange_info_fetched = False
        except Exception as e:
            self.logger.error(f"Failed to fetch or process Binance exchange info: {e}", exc_info=True)
            self.symbol_info_cache = {}
            self._exchange_info_fetched = False
            raise ExchangeConnectionError(f"Failed to fetch exchange info: {e}") from e

    def _get_cached_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached information for a specific symbol.
        """
        symbol = symbol.upper()
        if not self._exchange_info_fetched:
            self.logger.warning("Attempted to get symbol info before exchange info was fetched successfully. Returning None.")
            return None
        info = self.symbol_info_cache.get(symbol)
        if not info:
            self.logger.warning(f"Symbol '{symbol}' not found in cached exchange info. This might indicate an invalid symbol or stale cache.")
        return info

    def _get_filter_value(self, symbol: str, filter_type: str, key: str) -> Optional[Union[float, str]]:
        """
        Helper method to get a specific value from a symbol's filter list.
        """
        info = self._get_cached_symbol_info(symbol)
        if info and 'filters' in info:
            for f in info['filters']:
                if f.get('filterType') == filter_type:
                    value = f.get(key)
                    if value is not None:
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Filter value '{value}' for key '{key}' in filter '{filter_type}' for {symbol} could not be converted to float. Returning as string.")
                            return str(value)
                    else:
                        self.logger.warning(f"Key '{key}' found but has no value in filter '{filter_type}' for {symbol}.")
                        return None
            self.logger.warning(f"Filter type '{filter_type}' not found for symbol '{symbol}'.")
            return None
        self.logger.warning(f"No exchange info or filters available for symbol '{symbol}'.")
        return None

    def get_quantity_precision(self, symbol: str) -> int:
        """
        Determines the quantity precision (number of decimal places) for a symbol
        based on the exchange's LOT_SIZE filter stepSize.
        """
        step_size = self._get_filter_value(symbol, 'LOT_SIZE', 'stepSize')
        if isinstance(step_size, float) and step_size > 0:
            try:
                # Calculate precision based on step size (e.g., 0.001 -> 3, 1.0 -> 0)
                if 0 < step_size < 1:
                    return int(round(-math.log10(step_size) + FLOAT_EPSILON))
                elif step_size >= 1:
                    return 0 # Whole numbers have 0 precision
            except Exception as e:
                self.logger.error(f"Error calculating quantity precision for {symbol} with stepSize {step_size}: {e}", exc_info=True)
        self.logger.warning(f"Could not determine quantity precision for {symbol} from exchange info. Using configured default: {self._default_quantity_precision}.")
        return self._default_quantity_precision

    def get_price_precision(self, symbol: str) -> int:
        """
        Determines the price precision (number of decimal places) for a symbol
        based on the exchange's PRICE_FILTER tickSize.
        """
        tick_size = self._get_filter_value(symbol, 'PRICE_FILTER', 'tickSize')
        if isinstance(tick_size, float) and tick_size > 0:
            try:
                # Calculate precision based on tick size (e.g., 0.001 -> 3, 1.0 -> 0)
                if 0 < tick_size < 1:
                    return int(round(-math.log10(tick_size) + FLOAT_EPSILON))
                elif tick_size >= 1:
                    return 0 # Whole numbers have 0 precision
            except Exception as e:
                self.logger.error(f"Error calculating price precision for {symbol} with tickSize {tick_size}: {e}", exc_info=True)
        self.logger.warning(f"Could not determine price precision for {symbol} from exchange info. Using configured default: {self._default_price_precision}.")
        return self._default_price_precision

    def get_min_quantity(self, symbol: str) -> float:
        """
        Retrieves the minimum order quantity allowed for a symbol.
        """
        min_qty = self._get_filter_value(symbol, 'LOT_SIZE', 'minQty')
        if isinstance(min_qty, float) and min_qty >= 0:
            return min_qty
        self.logger.warning(f"Could not determine minimum quantity for {symbol} from exchange info. Using configured default: {self._default_min_quantity}.")
        return self._default_min_quantity

    def get_min_notional(self, symbol: str) -> float:
        """
        Retrieves the minimum notional value (price * quantity) allowed for an order.
        """
        min_notional = self._get_filter_value(symbol, 'MIN_NOTIONAL', 'minNotional')
        if isinstance(min_notional, float) and min_notional >= 0:
            return min_notional
        self.logger.warning(f"Could not determine minimum notional for {symbol} from exchange info. Using configured default: {self._default_min_notional}.")
        return self._default_min_notional

    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Adjusts the given quantity to the nearest valid step size multiple,
        rounding down to ensure validity.
        """
        if quantity is None or not pd.notna(quantity) or quantity <= FLOAT_EPSILON:
            return 0.0

        step_size = self._get_filter_value(symbol, 'LOT_SIZE', 'stepSize')
        precision = self.get_quantity_precision(symbol)

        if isinstance(step_size, float) and step_size > 0 and precision is not None:
            try:
                # Floor division to ensure we don't exceed allowed quantity
                adjusted_qty = math.floor(quantity / step_size) * step_size
                # Format to precision to avoid floating point inaccuracies
                final_qty = float(f"{adjusted_qty:.{precision}f}")
                return final_qty if final_qty > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting quantity {quantity} for {symbol} with step {step_size}: {e}", exc_info=True)
        else:
            self.logger.warning(f"Using fallback quantity adjustment for {symbol}. Precision: {self._default_quantity_precision}.")
            try:
                # Fallback: simple rounding to default precision
                final_qty = float(f"{quantity:.{self._default_quantity_precision}f}")
                return final_qty if final_qty > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting quantity {quantity} with default precision for {symbol}: {e}", exc_info=True)
        return 0.0


    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Adjusts the given price to the nearest valid tick size multiple.
        """
        if price is None or not pd.notna(price) or price <= FLOAT_EPSILON:
            return 0.0

        tick_size = self._get_filter_value(symbol, 'PRICE_FILTER', 'tickSize')
        precision = self.get_price_precision(symbol)

        if isinstance(tick_size, float) and tick_size > 0 and precision is not None:
            try:
                # Round to the nearest tick size
                adjusted_price = round(price / tick_size) * tick_size
                # Format to precision to avoid floating point inaccuracies
                final_price = float(f"{adjusted_price:.{precision}f}")
                return final_price if final_price > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting price {price} for {symbol} with tick {tick_size}: {e}", exc_info=True)
        else:
            self.logger.warning(f"Using fallback price adjustment for {symbol}. Precision: {self._default_price_precision}.")
            try:
                # Fallback: simple rounding to default precision
                final_price = float(f"{price:.{self._default_price_precision}f}")
                return final_price if final_price > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting price {price} with default precision for {symbol}: {e}", exc_info=True)
        return 0.0

