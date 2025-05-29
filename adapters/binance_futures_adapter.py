# adapters/binance_futures_adapter.py

"""
Concrete implementation of the ExchangeInterface for Binance Futures.

Handles API interactions, data fetching, order management, and provides methods
to query exchange information like precision and minimums. Includes asynchronous
operations and retry logic for API calls.
"""

import asyncio
import logging
import json # <--- ADDED THIS IMPORT for json.dumps
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timezone
import math # For floor in quantity adjustment
import numpy as np # For np.nan

# Import Binance client and exceptions
from binance import AsyncClient # Use AsyncClient for async operations
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Add project root to Python path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom exceptions and base interface
from utils.exchange_interface import ExchangeInterface
from utils.exceptions import ExchangeConnectionError, OrderExecutionError # <--- ENSURED THESE ARE IMPORTED

# Define FLOAT_EPSILON from params.py for consistency
try:
    from config.params import FLOAT_EPSILON
except ImportError:
    FLOAT_EPSILON = 1e-9 # Fallback if params.py not available or FLOAT_EPSILON missing
    logging.warning("FLOAT_EPSILON not imported from config.params. Using default 1e-9.")

# --- Constants for Binance API Error Codes ---
ORDER_NOT_FOUND_CODE = -2011
INSUFFICIENT_FUNDS_CODES = [-2019, -4003, -4007, -4014] # Common margin/fund errors
RATE_LIMIT_CODES = [-1003, -1015, -1120] # Common rate limit errors
INVALID_FILTER_CODES = [-1013, -2010] # Price/quantity filter errors
REDUCE_ONLY_REJECTED_CODE = -2022 # ReduceOnly order rejected (e.g., no position)
INVALID_API_KEY_CODE = -2008 # Invalid API Key or IP Access

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Async Retry Decorator ---
def async_retry_api_call(max_retries: int = 3, initial_delay: float = 1.0, max_delay: float = 10.0):
    """
    Decorator for retrying asynchronous Binance API calls with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay between retries in seconds.
        max_delay (float): Maximum delay between retries in seconds.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries + 1): # +1 to include the initial try
                try:
                    # Add a small, fixed pre-call delay to generally respect rate limits
                    await asyncio.sleep(0.05)
                    return await func(*args, **kwargs) # Await the decorated async function
                except (BinanceAPIException, BinanceRequestException) as e:
                    last_exception = e
                    # Non-retryable errors (order logic, invalid input, auth)
                    if e.code in [ORDER_NOT_FOUND_CODE, REDUCE_ONLY_REJECTED_CODE, INVALID_API_KEY_CODE] or \
                       e.code in INSUFFICIENT_FUNDS_CODES or \
                       e.code in INVALID_FILTER_CODES or \
                       (e.status_code >= 400 and e.status_code < 500 and e.status_code not in [429]): # Client-side errors (4xx) are usually not retryable, except 429 (rate limit)
                        logger.error(f"Non-retryable Binance API error on {func.__name__} (Code: {e.code}, Status: {e.status_code}): {e}", exc_info=False) # Log less verbosely
                        raise # Re-raise immediately
                    # Retryable errors (rate limits, server issues)
                    elif e.code in RATE_LIMIT_CODES or e.status_code in [429, 500, 502, 503, 504]:
                        if attempt < max_retries:
                            logger.warning(f"Retryable API error on {func.__name__} (Code: {e.code}, Status: {e.status_code}). Retry {attempt+1}/{max_retries}. Sleeping {delay:.2f}s.")
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, max_delay) # Exponential backoff with max delay
                        else:
                            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__} after retryable error (Code: {e.code}, Status: {e.status_code}).")
                            raise ConnectionError(f"Max retries exceeded for {func.__name__} after API error {e.code}") from e
                    else:
                        # Unexpected API error codes - treat as non-retryable for safety
                        logger.error(f"Unexpected non-retryable Binance API error on {func.__name__} (Code: {e.code}, Status: {e.status_code}): {e}", exc_info=True)
                        raise # Re-raise unexpected API errors
                except asyncio.TimeoutError as e: # Catch potential timeout errors from underlying library
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Request timeout on {func.__name__}. Retry {attempt+1}/{max_retries}. Sleeping {delay:.2f}s.")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__} after timeout.")
                        raise ConnectionError(f"Max retries exceeded for {func.__name__} after timeout") from e
                except ConnectionError as e: # Catch potential network errors
                     last_exception = e
                     if attempt < max_retries:
                          logger.warning(f"Network connection error on {func.__name__}: {e}. Retry {attempt+1}/{max_retries}. Sleeping {delay:.2f}s.")
                          await asyncio.sleep(delay)
                          delay = min(delay * 2, max_delay)
                     else:
                          logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__} after connection error.")
                          raise ConnectionError(f"Max retries exceeded for {func.__name__} after connection error") from e
                except Exception as e:
                    # Catch any other unexpected errors
                    logger.error(f"Unexpected error during API call {func.__name__}: {e}", exc_info=True)
                    raise # Re-raise immediately, as the cause is unknown
            # This point should only be reached if all retries failed for a retryable error
            # Raise the last exception encountered
            if last_exception:
                raise last_exception
            else:
                # Should not happen, but raise a generic error if it does
                raise ConnectionError(f"API call {func.__name__} failed after {max_retries} retries without a specific exception.")

        return wrapper
    return decorator


class BinanceFuturesAdapter(ExchangeInterface):
    """
    Adapter for interacting with the Binance Futures API.

    Implements the ExchangeInterface methods for fetching data, managing orders,
    and retrieving account/symbol information. Uses asynchronous operations and
    includes retry logic for robustness.
    """

    def __init__(self, api_key: str, api_secret: str, symbol: str, leverage: int, logger: logging.Logger, config: Dict[str, Any]):
        """
        Initializes the BinanceFuturesAdapter.

        Args:
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
            symbol (str): The primary trading pair symbol for this adapter instance (e.g., 'BTCUSDT').
            leverage (int): The leverage to set for the symbol.
            logger (logging.Logger): Logger instance for logging messages.
            config (Dict[str, Any]): Dictionary containing adapter configurations:
                - testnet (bool): True for testnet, False for production (default: False).
                - tld (str): Top-level domain ('com', 'us', etc.) (default: 'com').
                - timeout (int): API request timeout in seconds (default: 30).
                - quantity_precision (int): Default fallback quantity precision.
                - price_precision (int): Default fallback price precision.
                - min_quantity (float): Default fallback minimum order quantity.
                - min_notional (float): Default fallback minimum order notional value.
        """
        super().__init__(symbol, leverage, logger, config)
        self.api_key = api_key
        self.api_secret = api_secret
        self.client: Optional[AsyncClient] = None
        self.exchange_info: Dict[str, Any] = {} # To store symbol filters (precision, min_qty, etc.)
        self.symbol_info_cache: Dict[str, Any] = {} # Specific info for self.symbol
        self._exchange_info_fetched = False # Flag to track if info has been fetched

        # Store fallback parameters from config, providing reasonable defaults
        # These will be updated by _fetch_exchange_info if successful
        self._default_quantity_precision = int(config.get('quantity_precision', 8))
        self._default_price_precision = int(config.get('price_precision', 8))
        self._default_min_quantity = float(config.get('min_quantity', 0.00001))
        self._default_min_notional = float(config.get('min_notional', 5.0))

        self.logger.info(f"BinanceFuturesAdapter initialized for {self.symbol} (Testnet: {self.testnet})")
        self.logger.info(f"Adapter defaults: QtyPrec={self._default_quantity_precision}, PricePrec={self._default_price_precision}, MinQty={self._default_min_quantity}, MinNotional={self._default_min_notional}")


    async def async_setup(self):
        """
        Establishes an asynchronous connection to Binance and sets up account parameters.
        This method is called after adapter instantiation to perform async initialization.
        """
        self.logger.info("Connecting to Binance Futures API and performing async setup...")
        try:
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                tld=self.config.get('tld', 'com'),
                testnet=self.testnet
            )
            self.logger.info("Binance AsyncClient created.")

            # Fetch exchange info to get symbol details (precision, min_qty, etc.)
            await self._fetch_exchange_info()

            # Set leverage
            await self._set_leverage()

            # Set margin mode (ISOLATED/CROSSED)
            await self._set_margin_mode()

            self.logger.info("Binance Futures API async setup complete.")
        except BinanceAPIException as e:
            self.logger.critical(f"Binance API Exception during async setup: {e.code} - {e.message}", exc_info=True)
            raise ExchangeConnectionError(f"Binance API setup failed: {e.message}") from e
        except BinanceRequestException as e:
            self.logger.critical(f"Binance Request Exception during async setup: {e.status_code} - {e.message}", exc_info=True)
            raise ExchangeConnectionError(f"Binance request failed during setup: {e.message}") from e
        except Exception as e:
            self.logger.critical(f"Unexpected error during Binance async setup: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Unexpected error during Binance setup: {e}") from e

    @async_retry_api_call()
    async def _set_leverage(self):
        """Sets the leverage for the primary symbol of this adapter instance."""
        if not self.client:
            raise ExchangeConnectionError("Binance client not initialized for setting leverage.")
        try:
            leverage_int = int(self.leverage)
            self.logger.info(f"Setting leverage to {leverage_int}x for {self.symbol}...")
            response = await self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage_int)
            self.logger.info(f"Leverage set successfully for {self.symbol}. Response: {response}")
        except BinanceAPIException as e:
            if e.code == -4046: # Already at target leverage
                self.logger.info(f"Leverage for {self.symbol} is already {self.leverage}x.")
            else:
                self.logger.error(f"Failed to set leverage for {self.symbol}: {e.code} - {e.message}", exc_info=True)
                raise ExchangeConnectionError(f"Failed to set leverage: {e.message}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error setting leverage: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Unexpected error setting leverage: {e}") from e

    @async_retry_api_call()
    async def _set_margin_mode(self):
        """Sets the margin mode (ISOLATED or CROSSED) for the trading symbol."""
        if not self.client:
            raise ExchangeConnectionError("Binance client not initialized for setting margin mode.")
        margin_mode = self.config.get('default_margin_mode', 'ISOLATED').upper()
        try:
            resp = await self.client.futures_change_margin_type(symbol=self.symbol, marginType=margin_mode)
            self.logger.info(f"Margin mode set to {margin_mode} for {self.symbol}. Response: {resp}")
        except BinanceAPIException as e:
            if e.code == -4059: # Margin type already set
                self.logger.info(f"Margin type for {self.symbol} is already {margin_mode}.")
            else:
                self.logger.error(f"Failed to set margin mode for {self.symbol}: {e.code} - {e.message}", exc_info=True)
                raise ExchangeConnectionError(f"Failed to set margin mode: {e.message}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error setting margin mode: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Unexpected error setting margin mode: {e}") from e


    @async_retry_api_call(max_retries=2, initial_delay=0.5) # Fewer retries for info fetch
    async def _fetch_exchange_info(self):
        """Fetches and caches exchange information (symbol details, filters)."""
        if self._exchange_info_fetched:
            self.logger.debug("Exchange info already fetched.")
            return
        if not self.client:
            raise ExchangeConnectionError("Binance client not initialized for fetching exchange info.")
        try:
            self.logger.info("Fetching exchange information...")
            info = await self.client.futures_exchange_info() # Await the async client call
            if info and 'symbols' in info:
                self.symbol_info_cache = {s['symbol']: s for s in info['symbols']}
                self._exchange_info_fetched = True
                self.logger.info(f"Cached exchange info for {len(self.symbol_info_cache)} symbols.")

                # Update adapter's precision/min values for its primary symbol
                if self.symbol in self.symbol_info_cache:
                    s_info = self.symbol_info_cache[self.symbol]
                    for f in s_info['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            self.price_precision = int(math.log10(1/float(f['tickSize'])))
                            self.logger.debug(f"Updated price_precision from exchange: {self.price_precision}")
                        elif f['filterType'] == 'LOT_SIZE':
                            self.quantity_precision = int(math.log10(1/float(f['stepSize'])))
                            self.min_quantity = float(f['minQty'])
                            self.logger.debug(f"Updated quantity_precision from exchange: {self.quantity_precision}, min_quantity: {self.min_quantity}")
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            self.min_notional = float(f['notional'])
                            self.logger.debug(f"Updated min_notional from exchange: {self.min_notional}")
            else:
                self.logger.warning("Received unexpected format for exchange info.")
                self.symbol_info_cache = {} # Clear cache on bad response
                self._exchange_info_fetched = False
        except Exception as e:
            self.logger.error(f"Failed to fetch or process exchange info: {e}", exc_info=True)
            self.symbol_info_cache = {} # Clear cache on error
            self._exchange_info_fetched = False
            raise # Re-raise to allow retry decorator/caller to handle


    def _get_cached_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached information for a specific symbol."""
        symbol = symbol.upper()
        if not self._exchange_info_fetched:
            self.logger.warning("Attempted to get symbol info before exchange info was fetched successfully.")
            return None
        info = self.symbol_info_cache.get(symbol)
        if not info:
            self.logger.warning(f"Symbol '{symbol}' not found in cached exchange info.")
        return info


    def _get_filter_value(self, symbol: str, filter_type: str, key: str) -> Optional[Union[float, str]]:
        """Helper to get a specific value (float or string) from a symbol's filter list."""
        info = self._get_cached_symbol_info(symbol)
        if info and 'filters' in info:
            for f in info['filters']:
                if f.get('filterType') == filter_type:
                    value = f.get(key)
                    if value is not None:
                        # Attempt to convert to float if it looks numeric, otherwise return as string
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return str(value) # Return as string if not convertible to float
                    else:
                        # Key exists in filter but has no value (should be rare)
                        self.logger.warning(f"Key '{key}' found but has no value in filter '{filter_type}' for {symbol}.")
                        return None
            # Filter type not found for the symbol
            return None
        # Symbol info not found
        return None

    # --- Precision and Filter Methods (Synchronous - Use Cached Data) ---

    def get_quantity_precision(self, symbol: str) -> int:
        """Gets quantity precision (decimal places) from LOT_SIZE filter stepSize."""
        step_size = self._get_filter_value(symbol, 'LOT_SIZE', 'stepSize')
        if isinstance(step_size, float) and step_size > 0:
            try:
                # Calculate precision from step size (e.g., 0.001 -> 3)
                if 0 < step_size < 1:
                    # Use round with epsilon for robustness
                    return int(round(-math.log10(step_size) + FLOAT_EPSILON))
                elif step_size >= 1:
                    return 0 # No decimal places if step size is 1 or more
            except Exception as e:
                self.logger.error(f"Error calculating qty precision for {symbol} (step={step_size}): {e}")
        self.logger.warning(f"Could not determine quantity precision for {symbol}. Using default: {self._default_quantity_precision}.")
        return self._default_quantity_precision

    def get_price_precision(self, symbol: str) -> int:
        """Gets price precision (decimal places) from PRICE_FILTER tickSize."""
        tick_size = self._get_filter_value(symbol, 'PRICE_FILTER', 'tickSize')
        if isinstance(tick_size, float) and tick_size > 0:
            try:
                if 0 < tick_size < 1:
                    # Use round with epsilon for robustness
                    return int(round(-math.log10(tick_size) + FLOAT_EPSILON))
                elif tick_size >= 1:
                    return 0
            except Exception as e:
                self.logger.error(f"Error calculating price precision for {symbol} (tick={tick_size}): {e}")
        self.logger.warning(f"Could not determine price precision for {symbol}. Using default: {self._default_price_precision}.")
        return self._default_price_precision

    def get_min_quantity(self, symbol: str) -> float:
        """Gets minimum order quantity from LOT_SIZE filter minQty."""
        min_qty = self._get_filter_value(symbol, 'LOT_SIZE', 'minQty')
        if isinstance(min_qty, float) and min_qty >= 0:
            return min_qty
        self.logger.warning(f"Could not determine min quantity for {symbol}. Using default: {self._default_min_quantity}.")
        return self._default_min_quantity

    def get_min_notional(self, symbol: str) -> float:
        """Gets minimum notional value from MIN_NOTIONAL filter minNotional."""
        min_notional = self._get_filter_value(symbol, 'MIN_NOTIONAL', 'minNotional')
        if isinstance(min_notional, float) and min_notional >= 0:
            return min_notional
        self.logger.warning(f"Could not determine min notional for {symbol}. Using default: {self._default_min_notional}.")
        return self._default_min_notional

    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Adjusts the quantity DOWN to the nearest valid step size multiple based on LOT_SIZE filter.

        Args:
            symbol (str): The symbol to adjust for.
            quantity (float): The raw quantity.

        Returns:
            float: The adjusted quantity, or 0.0 if adjustment fails or results in zero/negative.
        """
        if quantity is None or quantity <= FLOAT_EPSILON: return 0.0
        step_size = self._get_filter_value(symbol, 'LOT_SIZE', 'stepSize')
        precision = self.get_quantity_precision(symbol) # Use the determined precision

        if isinstance(step_size, float) and step_size > 0 and precision is not None:
            try:
                # Floor the quantity based on step size
                adjusted_qty = math.floor(quantity / step_size) * step_size
                # Format to the correct precision to avoid floating point issues
                formatted_qty = f"{adjusted_qty:.{precision}f}"
                final_qty = float(formatted_qty)
                # Return 0.0 if the adjusted quantity is effectively zero
                return final_qty if final_qty > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting quantity {quantity} for {symbol} (step={step_size}): {e}")
        else:
            # Fallback: Use default precision if step_size invalid or precision unknown
            self.logger.warning(f"Using default quantity precision {self._default_quantity_precision} for adjustment of {symbol}.")
            try:
                # Format to default precision first
                formatted_qty = f"{quantity:.{self._default_quantity_precision}f}"
                # Floor based on the implied step size from default precision
                implied_step = 1 / (10**self._default_quantity_precision)
                adjusted_qty = math.floor(float(formatted_qty) / implied_step) * implied_step
                # Re-format after flooring
                final_formatted_qty = f"{adjusted_qty:.{self._default_quantity_precision}f}"
                final_qty = float(final_formatted_qty)
                return final_qty if final_qty > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting quantity {quantity} with default precision for {symbol}: {e}")

        return 0.0 # Return 0.0 if adjustment fails

    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Adjusts the price to the nearest valid tick size multiple based on PRICE_FILTER.

        Args:
            symbol (str): The symbol to adjust for.
            price (float): The raw price.

        Returns:
            float: The adjusted price, or 0.0 if adjustment fails or results in zero/negative.
        """
        if price is None or price <= FLOAT_EPSILON: return 0.0
        tick_size = self._get_filter_value(symbol, 'PRICE_FILTER', 'tickSize')
        precision = self.get_price_precision(symbol)

        if isinstance(tick_size, float) and tick_size > 0 and precision is not None:
            try:
                # Round price to the nearest tick size multiple
                adjusted_price = round(price / tick_size) * tick_size
                # Format to the correct precision
                formatted_price = f"{adjusted_price:.{precision}f}"
                final_price = float(formatted_price)
                # Return 0.0 if the adjusted price is effectively zero
                return final_price if final_price > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting price {price} for {symbol} (tick={tick_size}): {e}")
        else:
            # Fallback: Use default precision if tick_size invalid or precision unknown
            self.logger.warning(f"Using default price precision {self._default_price_precision} for adjustment of {symbol}.")
            try:
                formatted_price = f"{price:.{self._default_price_precision}f}"
                final_price = float(formatted_price)
                return final_price if final_price > FLOAT_EPSILON else 0.0
            except Exception as e:
                self.logger.error(f"Error adjusting price {price} with default precision for {symbol}: {e}")

        return 0.0 # Return 0.0 if adjustment fails

    # --- Asynchronous Data Fetching Methods ---

    @async_retry_api_call()
    async def get_historical_candles(self, symbol: str, interval: str, start_time: Optional[pd.Timestamp] = None, end_time: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical klines (candlesticks) for the given symbol and interval,
        handling pagination automatically.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1m', '5m', '1h').
            start_time (Optional[pd.Timestamp]): Start time (UTC, inclusive). If None, fetches from the earliest available.
            end_time (Optional[pd.Timestamp]): End time (UTC, inclusive). If None, fetches up to the present.

        Returns:
            Optional[pd.DataFrame]: DataFrame with OHLCV data indexed by open time (UTC),
                                    or an empty DataFrame if no data or an error occurs after retries.
        """
        symbol = symbol.upper()
        all_klines = []
        limit = 1500 # Max limit for futures historical klines per request

        # Convert Timestamps to UTC milliseconds integer
        start_ts = int(start_time.tz_convert('UTC').timestamp() * 1000) if start_time and start_time.tzinfo else (int(start_time.timestamp() * 1000) if start_time else None)
        end_ts = int(end_time.tz_convert('UTC').timestamp() * 1000) if end_time and end_time.tzinfo else (int(end_time.timestamp() * 1000) if end_time else None)
        current_start_ts = start_ts

        self.logger.info(f"Fetching historical klines for {symbol} ({interval}) from {start_time} to {end_time}...")

        while True:
            try:
                # Fetch klines for the current time range
                # Pass timestamps as milliseconds
                klines = await self.client.futures_klines( # Use await self.client.futures_klines
                    symbol=symbol,
                    interval=interval,
                    startTime=current_start_ts, # Use startTime
                    endTime=end_ts,             # Use endTime
                    limit=limit
                )

                if not klines:
                    self.logger.debug(f"No more klines received from timestamp {current_start_ts}.")
                    break # Exit loop if no more data

                all_klines.extend(klines)
                last_open_time_ms = klines[-1][0]

                # Prepare for the next iteration: start after the last received kline
                current_start_ts = last_open_time_ms + 1 # Start from the next millisecond

                # Check exit conditions
                if end_ts is not None and last_open_time_ms >= end_ts:
                    self.logger.debug("Reached or exceeded end timestamp.")
                    break
                if len(klines) < limit:
                    self.logger.debug("Received fewer klines than limit, assuming end of data for the range.")
                    break

                self.logger.debug(f"Fetched {len(klines)} klines, total {len(all_klines)}. Next start: {current_start_ts}")
                await asyncio.sleep(0.1) # Small delay between paginated requests

            except (BinanceAPIException, BinanceRequestException) as e:
                 self.logger.error(f"API error during historical fetch pagination for {symbol}: {e}", exc_info=False)
                 raise # Re-raise for retry decorator
            except Exception as e:
                 self.logger.error(f"Unexpected error during historical fetch pagination for {symbol}: {e}", exc_info=True)
                 return pd.DataFrame() # Return empty on unexpected error

        if not all_klines:
            self.logger.warning(f"No historical data found for {symbol} {interval} in the specified range.")
            return pd.DataFrame()

        # --- Process fetched klines into DataFrame ---
        try:
            df = pd.DataFrame(all_klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            # Convert open_time to DatetimeIndex (UTC)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df.set_index('open_time', inplace=True)
            # Convert OHLCV columns to numeric, coercing errors
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop rows with NaNs in essential columns after conversion
            df.dropna(subset=numeric_cols, inplace=True)
            # Ensure chronological order
            df.sort_index(inplace=True)

            # Filter final DataFrame based on original start/end times if provided
            # This ensures exact boundaries even if pagination slightly overshoots
            if start_time:
                start_time_utc = start_time.tz_convert('UTC') if start_time.tzinfo else start_time.tz_localize('UTC')
                df = df[df.index >= start_time_utc]
            if end_time:
                end_time_utc = end_time.tz_convert('UTC') if end_time.tzinfo else end_time.tz_localize('UTC')
                df = df[df.index <= end_time_utc]

            self.logger.info(f"Finished fetching historical data for {symbol}. Total records: {len(df)}")
            return df[numeric_cols] # Return only OHLCV columns

        except Exception as e:
            self.logger.error(f"Error processing fetched historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame() # Return empty DataFrame on processing error


    @async_retry_api_call()
    async def fetch_recent_candles(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetches the most recent N candles."""
        if not self.client:
            self.logger.error("Binance client not initialized.")
            return None
        symbol = symbol.upper()
        try:
            klines = await self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                self.logger.warning(f"No recent candles received for {symbol} {interval}.")
                return pd.DataFrame()

            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
            df = pd.DataFrame(klines, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)
            df.sort_index(inplace=True)
            return df[numeric_cols]
        except Exception as e:
            self.logger.error(f"Error fetching recent candles for {symbol} ({interval}): {e}", exc_info=True)
            raise # Let retry decorator handle


    @async_retry_api_call()
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Gets the latest mark price (preferred) or last price for a symbol."""
        if not self.client:
            self.logger.error("Binance client not initialized.")
            return None
        symbol = symbol.upper()
        price = None
        try:
            # Prefer mark price for futures
            ticker = await self.client.futures_mark_price(symbol=symbol)
            if ticker and 'markPrice' in ticker:
                try: price = float(ticker['markPrice'])
                except (ValueError, TypeError): logger.warning(f"Could not convert mark price to float for {symbol}: {ticker.get('markPrice')}")
            # Fallback to last price ticker if mark price fails or is invalid
            if price is None or price <= FLOAT_EPSILON:
                 logger.debug(f"Mark price for {symbol} invalid or missing, falling back to last price.")
                 # Use await self.client.futures_symbol_ticker for async call
                 ticker = await self.client.futures_symbol_ticker(symbol=symbol)
                 if ticker and 'price' in ticker:
                      try: price = float(ticker['price'])
                      except (ValueError, TypeError): logger.warning(f"Could not convert last price to float for {symbol}: {ticker.get('price')}")

            return price if price and price > FLOAT_EPSILON else None
        except Exception as e:
            self.logger.error(f"Error fetching latest price for {symbol}: {e}", exc_info=True)
            raise


    @async_retry_api_call()
    async def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieves all currently open positions for a specific symbol.

        Args:
            symbol (str): Trading pair symbol.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an open position.
                                  Expected keys: 'symbol', 'direction' ('long'/'short'), 'quantity',
                                  'entryPrice', 'unrealizedPnl', 'leverage', 'entryMargin', 'liquidationPrice',
                                  'entryTime' (timestamp in ms).
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            return []
        try:
            account_info = await self.client.futures_account() # Use await
            positions = []
            for pos_info in account_info['positions']:
                if pos_info['symbol'] == symbol and float(pos_info['positionAmt']) != 0:
                    position_amount = float(pos_info['positionAmt'])
                    direction = 'long' if position_amount > 0 else 'short'
                    entry_price = float(pos_info['entryPrice'])
                    unrealized_pnl = float(pos_info['unRealizedProfit'])
                    liquidation_price = float(pos_info['liquidationPrice']) if float(pos_info['liquidationPrice']) > 0 else np.nan # Use np.nan for consistency
                    leverage = int(pos_info['leverage']) if pos_info.get('leverage') else self.leverage # Fallback to configured leverage

                    # Binance API doesn't directly provide 'entryTime' for positions.
                    # This would typically be tracked internally by the bot or inferred from first trade.
                    entry_time = None # Placeholder

                    positions.append({
                        'symbol': pos_info['symbol'],
                        'direction': direction,
                        'quantity': abs(position_amount),
                        'entryPrice': entry_price,
                        'unrealizedPnl': unrealized_pnl,
                        'leverage': leverage,
                        'entryMargin': float(pos_info['isolatedMargin']) if pos_info['isolatedMargin'] else None,
                        'liquidationPrice': liquidation_price,
                        'entryTime': entry_time # This needs to be managed by the bot's state
                    })
            self.logger.debug(f"Fetched {len(positions)} open positions for {symbol}.")
            return positions
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception getting open positions ({symbol}): {e.code} - {e.message}", exc_info=True)
            return []
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception getting open positions ({symbol}): {e.status_code} - {e.message}", exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting open positions ({symbol}): {e}", exc_info=True)
            return []

    @async_retry_api_call()
    async def get_position_liquidation_price(self, symbol: str) -> Optional[float]:
        """Gets the liquidation price (float or NaN) for the open position on a specific symbol."""
        symbol = symbol.upper()
        try:
            positions = await self.get_open_positions(symbol=symbol) # Already handles parsing
            if not positions:
                self.logger.debug(f"No open position found for {symbol} to get liquidation price.")
                return None
            # Return the liquidation price (which could be NaN if unavailable/invalid)
            liq_price = positions[0].get('liquidationPrice')
            return liq_price if pd.notna(liq_price) else None
        except Exception as e:
            self.logger.error(f"Error fetching liquidation price for {symbol}: {e}", exc_info=True)
            return None # Return None on error

    @async_retry_api_call()
    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieves all currently open orders for a specific symbol.

        Args:
            symbol (str): Trading pair symbol.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an open order.
                                  Expected keys: 'orderId', 'symbol', 'side', 'type', 'price',
                                  'origQty', 'executedQty', 'status'.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            return []
        try:
            orders = await self.client.futures_get_open_orders(symbol=symbol)
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'orderId': str(order['orderId']), # Ensure orderId is string
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'price': float(order['price']),
                    'origQty': float(order['origQty']),
                    'executedQty': float(order['executedQty']),
                    'status': order['status'],
                    'updateTime': pd.to_datetime(order['updateTime'], unit='ms', utc=True) # Add update time for context
                })
            self.logger.debug(f"Fetched {len(formatted_orders)} open orders for {symbol}.")
            return formatted_orders
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception getting open orders ({symbol}): {e.code} - {e.message}", exc_info=True)
            return []
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception getting open orders ({symbol}): {e.status_code} - {e.message}", exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting open orders ({symbol}): {e}", exc_info=True)
            return []

    @async_retry_api_call()
    async def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Places a market order on Binance Futures.

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Quantity to trade.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details (e.g., 'orderId', 'status', 'executedQty', 'avgPrice').
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")

        adjusted_qty = self.adjust_quantity_precision(symbol, quantity)
        if adjusted_qty <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted quantity {adjusted_qty} is too small or invalid for market order.")
            return {"status": "REJECTED", "message": "Invalid quantity"}

        self.logger.info(f"Placing MARKET {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} (ReduceOnly: {reduce_only}).")
        try:
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=adjusted_qty,
                reduceOnly='true' if reduce_only else 'false'
            )
            self.logger.info(f"Market order placed: {order}")
            # Extract relevant info from the response
            executed_qty = 0.0
            avg_price = 0.0
            cum_quote = 0.0
            if order.get('fills'):
                for fill in order['fills']:
                    executed_qty += float(fill.get('qty', 0))
                    cum_quote += float(fill.get('quoteQty', 0))
                if executed_qty > FLOAT_EPSILON:
                    avg_price = cum_quote / executed_qty

            return {
                'orderId': str(order.get('orderId')),
                'symbol': order.get('symbol'),
                'status': order.get('status'),
                'executedQty': executed_qty,
                'avgPrice': avg_price,
                'cumQuote': cum_quote, # Useful for fee calculation
                'time': pd.to_datetime(order.get('updateTime'), unit='ms', utc=True)
            }
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception placing market order ({symbol}, {side}, {quantity}): {e.code} - {e.message}", exc_info=False) # Reduced verbosity for common errors
            raise OrderExecutionError(f"Market order failed: {e.message} (Code: {e.code})", order_details={"code": e.code, "message": e.message}) from e
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception placing market order ({symbol}, {side}, {quantity}): {e.status_code} - {e.message}", exc_info=True)
            raise OrderExecutionError(f"Market order failed: {e.message} (Status: {e.status_code})") from e
        except Exception as e:
            self.logger.error(f"Unexpected error placing market order ({symbol}, {side}, {quantity}): {e}", exc_info=True)
            raise OrderExecutionError(f"Unexpected error placing market order: {e}") from e

    @async_retry_api_call()
    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Places a limit order on Binance Futures.

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            price (float): Price at which to place the order.
            quantity (float): Quantity to trade.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")

        adjusted_qty = self.adjust_quantity_precision(symbol, quantity)
        adjusted_price = self.adjust_price_precision(symbol, price)

        if adjusted_qty <= FLOAT_EPSILON or adjusted_price <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted qty ({adjusted_qty}) or price ({adjusted_price}) is too small or invalid for limit order.")
            return {"status": "REJECTED", "message": "Invalid quantity or price"}

        self.logger.info(f"Placing LIMIT {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} @ {adjusted_price:.{self.price_precision}f} (ReduceOnly: {reduce_only}).")
        try:
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC', # Good Till Cancel
                quantity=adjusted_qty,
                price=adjusted_price,
                reduceOnly='true' if reduce_only else 'false'
            )
            self.logger.info(f"Limit order placed: {order}")
            return {
                'orderId': str(order.get('orderId')),
                'symbol': order.get('symbol'),
                'status': order.get('status'),
                'origQty': float(order.get('origQty', 0)),
                'price': float(order.get('price', 0)),
                'time': pd.to_datetime(order.get('updateTime'), unit='ms', utc=True)
            }
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception placing limit order ({symbol}, {side}, {quantity}, {price}): {e.code} - {e.message}", exc_info=False)
            raise OrderExecutionError(f"Limit order failed: {e.message} (Code: {e.code})") from e
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception placing limit order ({symbol}, {side}, {quantity}, {price}): {e.status_code} - {e.message}", exc_info=True)
            raise OrderExecutionError(f"Limit order failed: {e.message} (Status: {e.status_code})") from e
        except Exception as e:
            self.logger.error(f"Unexpected error placing limit order ({symbol}, {side}, {quantity}, {price}): {e}", exc_info=True)
            raise OrderExecutionError(f"Unexpected error placing limit order: {e}") from e

    @async_retry_api_call()
    async def place_stop_market_order(self, symbol: str, side: str, quantity: float, stop_price: float, reduce_only: bool = True) -> Dict[str, Any]:
        """
        Places a STOP_MARKET order on Binance Futures.

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Quantity to trade.
            stop_price (float): The price at which the market order will be triggered.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")

        adjusted_qty = self.adjust_quantity_precision(symbol, quantity)
        adjusted_stop_price = self.adjust_price_precision(symbol, stop_price)

        if adjusted_qty <= FLOAT_EPSILON or adjusted_stop_price <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted qty ({adjusted_qty}) or stop_price ({adjusted_stop_price}) is too small or invalid for stop market order.")
            return {"status": "REJECTED", "message": "Invalid quantity or stop price"}

        self.logger.info(f"Placing STOP_MARKET {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} @ StopPrice {adjusted_stop_price:.{self.price_precision}f} (ReduceOnly: {reduce_only}).")
        try:
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=adjusted_qty,
                stopPrice=adjusted_stop_price,
                closePosition='true' if reduce_only else 'false', # Use closePosition for reduceOnly on STOP_MARKET
                # NewOrderRespType='FULL' # Request full response for more details
            )
            self.logger.info(f"Stop Market order placed: {order}")
            return {
                'orderId': str(order.get('orderId')),
                'symbol': order.get('symbol'),
                'status': order.get('status'),
                'origQty': float(order.get('origQty', 0)),
                'price': float(order.get('price', 0)), # This will be 0 for market orders
                'stopPrice': float(order.get('stopPrice', 0)),
                'time': pd.to_datetime(order.get('updateTime'), unit='ms', utc=True)
            }
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception placing stop market order ({symbol}, {side}, {quantity}, {stop_price}): {e.code} - {e.message}", exc_info=False)
            raise OrderExecutionError(f"Stop Market order failed: {e.message} (Code: {e.code})") from e
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception placing stop market order ({symbol}, {side}, {quantity}, {stop_price}): {e.status_code} - {e.message}", exc_info=True)
            raise OrderExecutionError(f"Stop Market order failed: {e.message} (Status: {e.status_code})") from e
        except Exception as e:
            self.logger.error(f"Unexpected error placing stop market order ({symbol}, {side}, {quantity}, {stop_price}): {e}", exc_info=True)
            raise OrderExecutionError(f"Unexpected error placing stop market order: {e}") from e

    @async_retry_api_call()
    async def place_take_profit_market_order(self, symbol: str, side: str, quantity: float, stop_price: float, reduce_only: bool = True) -> Dict[str, Any]:
        """
        Places a TAKE_PROFIT_MARKET order on Binance Futures.

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Quantity to trade.
            stop_price (float): The price at which the market order will be triggered.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")

        adjusted_qty = self.adjust_quantity_precision(symbol, quantity)
        adjusted_stop_price = self.adjust_price_precision(symbol, stop_price)

        if adjusted_qty <= FLOAT_EPSILON or adjusted_stop_price <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted qty ({adjusted_qty}) or stop_price ({adjusted_stop_price}) is too small or invalid for take profit market order.")
            return {"status": "REJECTED", "message": "Invalid quantity or stop price"}

        self.logger.info(f"Placing TAKE_PROFIT_MARKET {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} @ StopPrice {adjusted_stop_price:.{self.price_precision}f} (ReduceOnly: {reduce_only}).")
        try:
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                quantity=adjusted_qty,
                stopPrice=adjusted_stop_price,
                closePosition='true' if reduce_only else 'false', # Use closePosition for reduceOnly on TAKE_PROFIT_MARKET
                # NewOrderRespType='FULL' # Request full response for more details
            )
            self.logger.info(f"Take Profit Market order placed: {order}")
            return {
                'orderId': str(order.get('orderId')),
                'symbol': order.get('symbol'),
                'status': order.get('status'),
                'origQty': float(order.get('origQty', 0)),
                'price': float(order.get('price', 0)), # This will be 0 for market orders
                'stopPrice': float(order.get('stopPrice', 0)),
                'time': pd.to_datetime(order.get('updateTime'), unit='ms', utc=True)
            }
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception placing take profit market order ({symbol}, {side}, {quantity}, {stop_price}): {e.code} - {e.message}", exc_info=False)
            raise OrderExecutionError(f"Take Profit Market order failed: {e.message} (Code: {e.code})") from e
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception placing take profit market order ({symbol}, {side}, {quantity}, {stop_price}): {e.status_code} - {e.message}", exc_info=True)
            raise OrderExecutionError(f"Take Profit Market order failed: {e.message} (Status: {e.status_code})") from e
        except Exception as e:
            self.logger.error(f"Unexpected error placing take profit market order ({symbol}, {side}, {quantity}, {stop_price}): {e}", exc_info=True)
            raise OrderExecutionError(f"Unexpected error placing take profit market order: {e}") from e

    @async_retry_api_call(max_retries=2, initial_delay=0.2) # Fewer retries for cancellation
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancels a specific open order on Binance Futures.

        Args:
            symbol (str): Trading pair symbol.
            order_id (str): The ID of the order to cancel.

        Returns:
            Dict[str, Any]: Dictionary containing cancellation details.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")
        self.logger.info(f"Cancelling order {order_id} for {symbol}.")
        try:
            # Binance API expects orderId as int for some calls, but str for others.
            # It's safer to pass as str if the API expects it or handles conversion.
            # The interface defines order_id as str, so we pass it as str.
            result = await self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Order {order_id} cancelled: {result}")
            return result
        except BinanceAPIException as e:
            if e.code == ORDER_NOT_FOUND_CODE:
                self.logger.warning(f"Order {order_id} for {symbol} not found during cancellation (already filled/cancelled?).")
                return {"orderId": order_id, "status": "ALREADY_DONE", "message": e.message}
            self.logger.error(f"Binance API Exception cancelling order ({symbol}, {order_id}): {e.code} - {e.message}", exc_info=False)
            raise OrderExecutionError(f"Order cancellation failed: {e.message} (Code: {e.code})") from e
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception cancelling order ({symbol}, {order_id}): {e.status_code} - {e.message}", exc_info=True)
            raise OrderExecutionError(f"Order cancellation failed: {e.message} (Status: {e.status_code})") from e
        except Exception as e:
            self.logger.error(f"Unexpected error cancelling order ({symbol}, {order_id}): {e}", exc_info=True)
            raise OrderExecutionError(f"Unexpected error cancelling order: {e}") from e

    @async_retry_api_call()
    async def cancel_multiple_orders(self, symbol: str, order_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Cancels multiple open orders on Binance Futures.
        Attempts bulk cancellation first, falls back to individual if bulk fails.

        Args:
            symbol (str): Trading pair symbol.
            order_ids (List[str]): A list of order IDs to cancel.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each with cancellation details for an order.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")
        if not order_ids:
            return []

        results = []
        self.logger.info(f"Attempting to cancel {len(order_ids)} orders for {symbol}: {order_ids}")

        # Attempt Bulk Cancellation
        try:
            # Binance API expects orderIdList as a JSON string of a list of strings
            order_id_list_json = json.dumps(order_ids) # Convert list of strings to JSON string
            bulk_results = await self.client.futures_cancel_orders(symbol=symbol, orderIdList=order_id_list_json)
            self.logger.info(f"Bulk cancellation request sent for {symbol}. Results: {bulk_results}")

            processed_results = []
            for res in bulk_results:
                if isinstance(res, dict) and 'code' in res: # It's an error object
                    processed_results.append({'orderId': res.get('origClientOrderId') or res.get('orderId', 'UNKNOWN'), 'status': 'FAILED', 'message': res.get('msg'), 'code': res.get('code')})
                elif isinstance(res, dict): # Assume it's a success object (contains order details)
                    processed_results.append(res) # Keep the success dict as is
                else: # Unexpected format
                     processed_results.append({'orderId': 'UNKNOWN', 'status': 'UNKNOWN_FORMAT', 'message': str(res)})
            return processed_results

        except (BinanceAPIException, BinanceRequestException) as e_bulk:
            self.logger.warning(f"Bulk cancellation failed for {symbol} ({e_bulk}). Falling back to individual cancellation.")
            # Fallback to individual calls
            for order_id in order_ids:
                try:
                    individual_result = await self.cancel_order(symbol, order_id)
                    if individual_result:
                        results.append(individual_result)
                    else:
                        results.append({'orderId': order_id, 'status': 'FAILED', 'message': 'Individual cancellation returned None.'})
                    await asyncio.sleep(0.1) # Small delay between individual cancels
                except Exception as e_single:
                    self.logger.warning(f"Individual cancel failed for order {order_id} after retries: {e_single}")
                    results.append({'orderId': order_id, 'status': 'FAILED', 'message': str(e_single)})
            return results
        except Exception as e:
            self.logger.error(f"Unexpected error during bulk cancellation attempt for {symbol}: {e}", exc_info=True)
            return [{'orderId': oid, 'status': 'UNEXPECTED_ERROR', 'message': str(e)} for oid in order_ids]

    @async_retry_api_call()
    async def cancel_all_orders(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Cancels ALL open orders for a specific symbol."""
        if not self.client:
            self.logger.error("Binance client not initialized.")
            raise OrderExecutionError("Binance client not initialized.")
        symbol = symbol.upper()
        self.logger.info(f"Attempting to cancel ALL open orders for {symbol}...")
        try:
            result = await self.client.futures_cancel_all_open_orders(symbol=symbol) # Use await
            self.logger.info(f"Cancel ALL orders request successful for {symbol}. Result: {result}")
            return [result] if result else None
        except Exception as e:
            self.logger.error(f"Error cancelling ALL orders for {symbol}: {e}", exc_info=True)
            raise

    @async_retry_api_call()
    async def get_order_info(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves information about a specific order on Binance Futures.

        Args:
            symbol (str): Trading pair symbol.
            order_id (str): The ID of the order to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing order details, or None if not found.
                                     Expected keys: 'orderId', 'symbol', 'status', 'executedQty', 'avgPrice', 'cumQuote'.
        """
        if not self.client:
            self.logger.error("Binance client not initialized.")
            return None
        try:
            # Binance API expects orderId as int for some calls, but str for others.
            # It's safer to pass as str if the API expects it or handles conversion.
            order = await self.client.futures_get_order(symbol=symbol, orderId=order_id)
            if order:
                executed_qty = float(order.get('executedQty', 0))
                cum_quote = float(order.get('cumQuote', 0))
                avg_price = cum_quote / executed_qty if executed_qty > FLOAT_EPSILON else 0.0

                return {
                    'orderId': str(order.get('orderId')),
                    'symbol': order.get('symbol'),
                    'status': order.get('status'),
                    'executedQty': executed_qty,
                    'avgPrice': avg_price,
                    'cumQuote': cum_quote,
                    'origQty': float(order.get('origQty', 0)),
                    'price': float(order.get('price', 0)),
                    'type': order.get('type'),
                    'side': order.get('side'),
                    'time': pd.to_datetime(order.get('updateTime'), unit='ms', utc=True)
                }
            return None
        except BinanceAPIException as e:
            if e.code == ORDER_NOT_FOUND_CODE:
                self.logger.warning(f"Order {order_id} for {symbol} not found: {e.message}")
                return None
            self.logger.error(f"Binance API Exception getting order info ({symbol}, {order_id}): {e.code} - {e.message}", exc_info=False)
            return None
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception getting order info ({symbol}, {order_id}): {e.status_code} - {e.message}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting order info ({symbol}, {order_id}): {e}", exc_info=True)
            return None

    async def close_connection(self):
        """
        Closes the Binance AsyncClient connection.
        """
        if self.client:
            self.logger.info("Closing Binance AsyncClient connection.")
            await self.client.close_connection()
            self.client = None
        # BinanceSocketManager is not used in this adapter, so no need to close it.
