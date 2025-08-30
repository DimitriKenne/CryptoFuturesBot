# utils/exchange_adapters/binance/binance_futures_adapter.py

"""
Refactored concrete implementation of the ExchangeInterface for Binance Futures.

This module now acts as an orchestrator, delegating specific responsibilities to
dedicated helper classes located in the 'binance' sub-directory:
- BinanceAPIClientManager: Handles the AsyncClient connection lifecycle and core API calls.
- BinanceExchangeInfoHelper: Manages exchange-specific information like precision and filters.
- BinanceAccountConfigurator: Handles account-level settings such as leverage and margin mode.

This separation of concerns makes the adapter leaner, more maintainable, and easier to test.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timezone
import numpy as np

# Import Binance client and exceptions (still needed for exception handling)
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Add project root to Python path for imports
import sys
from pathlib import Path
# Current file is in utils/exchange_adapters/binance, need to go up 4 levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom exceptions and base interface
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.exceptions import ExchangeConnectionError, OrderExecutionError, ConfigurationError

# Import AppConfig for type hinting and configuration access
from config.params import AppConfig, FLOAT_EPSILON

# Import the new helper utilities from their specific subfolder
from utils.exchange_adapters.binance.client_manager import BinanceAPIClientManager
from utils.exchange_adapters.binance.exchange_info_helper import BinanceExchangeInfoHelper
from utils.exchange_adapters.binance.account_configurator import BinanceAccountConfigurator
from utils.exchange_adapters.binance.decorators import async_retry_api_call, ORDER_NOT_FOUND_CODE # Import decorator and constants from new path

# --- Logger Setup ---
logger = logging.getLogger(__name__)


class BinanceFuturesAdapter(ExchangeInterface):
    """
    Adapter for interacting with the Binance Futures API.
    This class now acts as an orchestrator, delegating to specialized helper utilities
    for client management, exchange info, and account configuration.
    """

    def __init__(self, app_config: AppConfig, symbol: str, logger: logging.Logger):
        """
        Initializes the BinanceFuturesAdapter by setting up its internal helper instances.

        Args:
            app_config (AppConfig): The comprehensive application configuration object.
            symbol (str): The primary trading pair symbol for this adapter instance (e.g., 'BTCUSDT').
            logger (logging.Logger): Logger instance for logging messages.
        """
        # Ensure ExchangeConfig is available within app_config
        if not hasattr(app_config, 'exchange') or not app_config.exchange:
            raise ConfigurationError("Exchange configuration (app_config.exchange) is missing.")
        if not hasattr(app_config, 'trading') or not app_config.trading or not app_config.trading.risk:
            raise ConfigurationError("Trading risk configuration (app_config.trading.risk) is missing.")


        # Pass relevant config to superclass (ExchangeInterface) for common attributes
        super().__init__(symbol, app_config.trading.risk.leverage, logger, app_config.exchange)

        self.app_config = app_config # Store full AppConfig for access to nested configs
        self.symbol = symbol.upper() # Ensure symbol is uppercase

        # Initialize helper instances WITHOUT passing the client immediately.
        # The client will be set later in async_setup() after connect() is called.
        self.client_manager = BinanceAPIClientManager(
            api_key=self.app_config.exchange.api_key,
            api_secret=self.app_config.exchange.api_secret,
            exchange_config=self.app_config.exchange,
            logger=self.logger
        )
        # Pass None for client initially, then set it properly in async_setup
        self.exchange_info_helper = BinanceExchangeInfoHelper(
            client=None,
            symbol=self.symbol,
            exchange_config=self.app_config.exchange,
            logger=self.logger
        )
        self.account_configurator = BinanceAccountConfigurator(
            client=None,
            symbol=self.symbol,
            leverage=self.leverage,
            logger=self.logger
        )

        self.logger.info(f"BinanceFuturesAdapter initialized for {self.symbol} (Testnet: {self.app_config.exchange.testnet})")
        # These will be updated from exchange_info_helper during async_setup
        self.price_precision = self.app_config.exchange.get_symbol_params(self.symbol).get("price_precision", 2)
        self.quantity_precision = self.app_config.exchange.get_symbol_params(self.symbol).get("quantity_precision", 3)
        self.min_quantity = self.app_config.exchange.get_symbol_params(self.symbol).get("min_quantity", 0.001)
        self.min_notional = self.app_config.exchange.get_symbol_params(self.symbol).get("min_notional", 5.0)


    async def async_setup(self):
        """
        Performs asynchronous setup by connecting to Binance API,
        fetching exchange info, and configuring account settings.
        This method must be called after adapter instantiation.
        """
        self.logger.info(f"Connecting to Binance Futures API for {self.symbol} and performing async setup...")
        try:
            # 1. Connect the client
            await self.client_manager.connect()

            # 2. Set the *actual* initialized client in helpers using the new set_client method
            client = self.client_manager.get_client
            self.exchange_info_helper.set_client(client)
            self.account_configurator.set_client(client)

            # 3. Fetch and cache exchange information
            await self.exchange_info_helper.fetch_and_cache_info()

            # 4. Update the adapter's precision/min values from the helper's cached info
            self.price_precision = self.exchange_info_helper.get_price_precision(self.symbol)
            self.quantity_precision = self.exchange_info_helper.get_quantity_precision(self.symbol)
            self.min_quantity = self.exchange_info_helper.get_min_quantity(self.symbol)
            self.min_notional = self.exchange_info_helper.get_min_notional(self.symbol)
            self.logger.info(f"Adapter settings updated from exchange info: Price Prec={self.price_precision}, Qty Prec={self.quantity_precision}, Min Qty={self.min_quantity}, Min Notional={self.min_notional}")

            # 5. Set leverage and margin mode
            await self.account_configurator.set_leverage()
            # await self.account_configurator.set_margin_mode()

            self.logger.info("Binance Futures API async setup complete.")
        except Exception as e:
            self.logger.critical(f"Binance Futures Adapter async setup failed for {self.symbol}: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Binance Futures Adapter setup failed: {e}") from e


    def set_client(self, client: Any):
        """
        Sets the underlying exchange client. This is a pass-through method
        for the ExchangeInterface abstract method, used here to ensure
        BinanceFuturesAdapter itself conforms to the interface, even though
        its internal helpers manage the client directly.
        """
        # In this adapter's design, the actual client is managed by client_manager
        # and passed to helpers. This method is primarily for interface compliance.
        self.client_manager.set_client(client)
        self.exchange_info_helper.set_client(client)
        self.account_configurator.set_client(client)


    # --- Asynchronous Data Fetching Methods ---

    @async_retry_api_call()
    async def get_historical_candles(self, symbol: str, interval: str, start_time: Optional[pd.Timestamp] = None, end_time: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical klines (candlesticks) for the given symbol and interval,
        handling pagination automatically.
        """
        client = self.client_manager.get_client # Access client via manager
        symbol = symbol.upper()
        all_klines = []
        limit = 1500 # Max limit for futures historical klines per request

        # Convert timestamps to milliseconds, handling timezone if present
        start_ts = int(start_time.tz_convert('UTC').timestamp() * 1000) if start_time and start_time.tzinfo else (int(start_time.timestamp() * 1000) if start_time else None)
        end_ts = int(end_time.tz_convert('UTC').timestamp() * 1000) if end_time and end_time.tzinfo else (int(end_time.timestamp() * 1000) if end_time else None)
        current_start_ts = start_ts

        self.logger.info(f"Fetching historical klines for {symbol} ({interval}) from {start_time} to {end_time}...")

        while True:
            try:
                klines = await client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_start_ts,
                    endTime=end_ts,
                    limit=limit
                )

                if not klines:
                    self.logger.debug(f"No more klines received from timestamp {current_start_ts}.")
                    break

                all_klines.extend(klines)
                last_open_time_ms = klines[-1][0]
                current_start_ts = last_open_time_ms + 1

                # Stop if we've reached or exceeded the requested end time
                if end_ts is not None and last_open_time_ms >= end_ts:
                    self.logger.debug("Reached or exceeded end timestamp.")
                    break
                # Stop if the number of klines received is less than the limit (implies end of data)
                if len(klines) < limit:
                    self.logger.debug("Received fewer klines than limit, assuming end of data for the range.")
                    break

                self.logger.debug(f"Fetched {len(klines)} klines, total {len(all_klines)}. Next start: {datetime.fromtimestamp(current_start_ts / 1000, tz=timezone.utc)}")
                await asyncio.sleep(0.1) # Small delay to avoid hitting rate limits on continuous fetching

            except (BinanceAPIException, BinanceRequestException) as e:
                 self.logger.error(f"API error during historical fetch pagination for {symbol}: {e.code} - {e.message}", exc_info=False)
                 raise ExchangeConnectionError(f"Historical data fetch failed due to API error: {e.message}") from e
            except Exception as e:
                 self.logger.error(f"Unexpected error during historical fetch pagination for {symbol}: {e}", exc_info=True)
                 raise ExchangeConnectionError(f"Historical data fetch failed: {e}") from e

        if not all_klines:
            self.logger.warning(f"No historical data found for {symbol} {interval} in the specified range.")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(all_klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df.set_index('open_time', inplace=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)
            df.sort_index(inplace=True)

            # Apply exact time filters after initial fetch and processing
            if start_time:
                start_time_utc = start_time.tz_convert('UTC') if start_time.tzinfo else start_time.tz_localize('UTC')
                df = df[df.index >= start_time_utc]
            if end_time:
                end_time_utc = end_time.tz_convert('UTC') if end_time.tzinfo else end_time.tz_localize('UTC')
                df = df[df.index <= end_time_utc]

            self.logger.info(f"Finished fetching historical data for {symbol}. Total records: {len(df)}")
            return df[numeric_cols]

        except Exception as e:
            self.logger.error(f"Error processing fetched historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()


    @async_retry_api_call()
    async def fetch_recent_candles(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Fetches the most recent N candles for a given symbol and interval.
        """
        client = self.client_manager.get_client # Access client via manager
        symbol = symbol.upper()
        try:
            klines = await client.futures_klines(symbol=symbol, interval=interval, limit=limit)
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
            raise ExchangeConnectionError(f"Failed to fetch recent candles: {e}") from e


    @async_retry_api_call()
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Gets the latest mark price (preferred) or last price for a symbol.
        """
        client = self.client_manager.get_client # Access client via manager
        symbol = symbol.upper()
        price = None
        try:
            # Try to get mark price first
            ticker = await client.futures_mark_price(symbol=symbol)
            if ticker and 'markPrice' in ticker:
                try:
                    price = float(ticker['markPrice'])
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert mark price to float for {symbol}: {ticker.get('markPrice')}")
            
            # If mark price is invalid or missing, fall back to last price
            if price is None or price <= FLOAT_EPSILON:
                 self.logger.debug(f"Mark price for {symbol} invalid or missing, falling back to last price.")
                 ticker = await client.futures_symbol_ticker(symbol=symbol)
                 if ticker and 'price' in ticker:
                      try:
                          price = float(ticker['price'])
                      except (ValueError, TypeError):
                          self.logger.warning(f"Could not convert last price to float for {symbol}: {ticker.get('price')}")
            
            return price if price and price > FLOAT_EPSILON else None
        except Exception as e:
            self.logger.error(f"Error fetching latest price for {symbol}: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Failed to get latest price: {e}") from e


    # --- Position Management Methods ---

    @async_retry_api_call()
    async def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieves all currently open positions for a specific symbol.
        """
        client = self.client_manager.get_client # Access client via manager
        try:
            account_info = await client.futures_account()
            positions = []
            for pos_info in account_info['positions']:
                if pos_info['symbol'] == symbol and float(pos_info['positionAmt']) != 0:
                    position_amount = float(pos_info['positionAmt'])
                    direction = 'long' if position_amount > 0 else 'short'
                    entry_price = float(pos_info['entryPrice'])
                    unrealized_pnl = float(pos_info.get('unRealizedProfit', 0.0))
                    liquidation_price_raw = float(pos_info.get('liquidationPrice', 0.0))
                    liquidation_price = liquidation_price_raw if liquidation_price_raw > 0 else np.nan
                    leverage = int(pos_info['leverage']) if pos_info.get('leverage') else self.leverage
                    entry_time = None # Binance API does not directly provide entry time for positions here
                    margin_val = pos_info.get('isolatedMargin')
                    try:
                        entry_margin = float(margin_val) if margin_val not in (None, '', 0) else None
                    except (ValueError, TypeError):
                        entry_margin = None

                    positions.append({
                        'symbol': pos_info['symbol'],
                        'direction': direction,
                        'quantity': abs(position_amount),
                        'entryPrice': entry_price,
                        'unrealizedPnl': unrealized_pnl,
                        'leverage': leverage,
                        'entryMargin': entry_margin,
                        'liquidationPrice': liquidation_price,
                        'entryTime': entry_time # This will be None, to be potentially enriched elsewhere
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
        """
        Gets the liquidation price for the open position on a specific symbol.
        """
        symbol = symbol.upper()
        try:
            positions = await self.get_open_positions(symbol=symbol)
            if not positions:
                self.logger.debug(f"No open position found for {symbol} to get liquidation price.")
                return None
            liq_price = positions[0].get('liquidationPrice')
            return liq_price if pd.notna(liq_price) else None
        except Exception as e:
            self.logger.error(f"Error fetching liquidation price for {symbol}: {e}", exc_info=True)
            raise ExchangeConnectionError(f"Failed to get liquidation price: {e}") from e


    @async_retry_api_call()
    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieves all currently open orders for a specific symbol.
        """
        client = self.client_manager.get_client # Access client via manager
        try:
            orders = await client.futures_get_open_orders(symbol=symbol)
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'orderId': str(order['orderId']),
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'price': float(order['price']),
                    'origQty': float(order['origQty']),
                    'executedQty': float(order['executedQty']),
                    'status': order['status'],
                    'updateTime': pd.to_datetime(order['updateTime'], unit='ms', utc=True)
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

    # --- Order Placement Methods ---

    @async_retry_api_call()
    async def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Places a market order on Binance Futures.
        Delegates quantity precision adjustment to the exchange info helper.
        """
        client = self.client_manager.get_client # Access client via manager
        adjusted_qty = self.exchange_info_helper.adjust_quantity_precision(symbol, quantity)
        if adjusted_qty <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted quantity {adjusted_qty} is too small or invalid for market order.")
            raise OrderExecutionError("Invalid quantity for market order after precision adjustment.")

        self.logger.info(f"Placing MARKET {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} (ReduceOnly: {reduce_only}).")
        try:
            order = await client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=adjusted_qty,
                reduceOnly='true' if reduce_only else 'false'
            )
            self.logger.info(f"Market order placed: {order}")
            
            # Parse fills for executed quantity and average price
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
                'cumQuote': cum_quote,
                'time': pd.to_datetime(order.get('updateTime'), unit='ms', utc=True)
            }
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception placing market order ({symbol}, {side}, {quantity}): {e.code} - {e.message}", exc_info=False)
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
        Delegates quantity and price precision adjustments to the exchange info helper.
        """
        client = self.client_manager.get_client # Access client via manager
        adjusted_qty = self.exchange_info_helper.adjust_quantity_precision(symbol, quantity)
        adjusted_price = self.exchange_info_helper.adjust_price_precision(symbol, price)

        if adjusted_qty <= FLOAT_EPSILON or adjusted_price <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted qty ({adjusted_qty}) or price ({adjusted_price}) is too small or invalid for limit order.")
            raise OrderExecutionError("Invalid quantity or price for limit order after precision adjustment.")

        self.logger.info(f"Placing LIMIT {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} @ {adjusted_price:.{self.price_precision}f} (ReduceOnly: {reduce_only}).")
        try:
            order = await client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC', # Good 'Till Cancelled
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
        Places a STOP_MARKET order on Binance Futures (often used for Stop Loss).
        Delegates quantity and price precision adjustments to the exchange info helper.
        """
        client = self.client_manager.get_client # Access client via manager
        adjusted_qty = self.exchange_info_helper.adjust_quantity_precision(symbol, quantity)
        adjusted_stop_price = self.exchange_info_helper.adjust_price_precision(symbol, stop_price)

        if adjusted_qty <= FLOAT_EPSILON or adjusted_stop_price <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted qty ({adjusted_qty}) or stop_price ({adjusted_stop_price}) is too small or invalid for stop market order.")
            raise OrderExecutionError("Invalid quantity or stop price for stop market order after precision adjustment.")

        self.logger.info(f"Placing STOP_MARKET {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} @ StopPrice {adjusted_stop_price:.{self.price_precision}f} (ReduceOnly: {reduce_only}).")
        try:
            order = await client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=adjusted_qty,
                stopPrice=adjusted_stop_price,
                closePosition='true' if reduce_only else 'false', # For stop-loss, closePosition is typically 'true'
            )
            self.logger.info(f"Stop Market order placed: {order}")
            return {
                'orderId': str(order.get('orderId')),
                'symbol': order.get('symbol'),
                'status': order.get('status'),
                'origQty': float(order.get('origQty', 0)),
                'price': float(order.get('price', 0)), # Price might be 0 for market orders
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
        Places a TAKE_PROFIT_MARKET order on Binance Futures (often used for Take Profit).
        Delegates quantity and price precision adjustments to the exchange info helper.
        """
        client = self.client_manager.get_client # Access client via manager
        adjusted_qty = self.exchange_info_helper.adjust_quantity_precision(symbol, quantity)
        adjusted_stop_price = self.exchange_info_helper.adjust_price_precision(symbol, stop_price)

        if adjusted_qty <= FLOAT_EPSILON or adjusted_stop_price <= FLOAT_EPSILON:
            self.logger.warning(f"Adjusted qty ({adjusted_qty}) or stop_price ({adjusted_stop_price}) is too small or invalid for take profit market order.")
            raise OrderExecutionError("Invalid quantity or stop price for take profit market order after precision adjustment.")

        self.logger.info(f"Placing TAKE_PROFIT_MARKET {side} order for {adjusted_qty:.{self.quantity_precision}f} {symbol} @ StopPrice {adjusted_stop_price:.{self.price_precision}f} (ReduceOnly: {reduce_only}).")
        try:
            order = await client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                quantity=adjusted_qty,
                stopPrice=adjusted_stop_price,
                closePosition='true' if reduce_only else 'false', # For take-profit, closePosition is typically 'true'
            )
            self.logger.info(f"Take Profit Market order placed: {order}")
            return {
                'orderId': str(order.get('orderId')),
                'symbol': order.get('symbol'),
                'status': order.get('status'),
                'origQty': float(order.get('origQty', 0)),
                'price': float(order.get('price', 0)), # Price might be 0 for market orders
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

    # --- Order Cancellation Methods ---

    @async_retry_api_call(max_retries=2, initial_delay=0.2)
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancels a specific open order on Binance Futures.
        """
        client = self.client_manager.get_client # Access client via manager
        self.logger.info(f"Cancelling order {order_id} for {symbol}.")
        try:
            result = await client.futures_cancel_order(symbol=symbol, orderId=order_id)
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
        """
        client = self.client_manager.get_client # Access client via manager
        if not order_ids:
            return []

        results = []
        self.logger.info(f"Attempting to cancel {len(order_ids)} orders for {symbol}: {order_ids}")

        try:
            order_id_list_json = json.dumps(order_ids) # Binance API expects orderIdList as JSON string
            bulk_results = await client.futures_cancel_orders(symbol=symbol, orderIdList=order_id_list_json)
            self.logger.info(f"Bulk cancellation request sent for {symbol}. Results: {bulk_results}")

            # Process bulk results (Binance returns a list of dicts with status/code)
            processed_results = []
            for res in bulk_results:
                if isinstance(res, dict) and 'code' in res and 'msg' in res: # Error format
                    processed_results.append({'orderId': res.get('origClientOrderId') or res.get('orderId', 'UNKNOWN'), 'status': 'FAILED', 'message': res.get('msg'), 'code': res.get('code')})
                elif isinstance(res, dict): # Success format
                    processed_results.append(res)
                else: # Unexpected format
                     processed_results.append({'orderId': 'UNKNOWN', 'status': 'UNKNOWN_FORMAT', 'message': str(res)})
            return processed_results

        except (BinanceAPIException, BinanceRequestException) as e_bulk:
            self.logger.warning(f"Bulk cancellation failed for {symbol} ({e_bulk.code} - {e_bulk.message}). Falling back to individual cancellation.")
            for order_id in order_ids:
                try:
                    individual_result = await self.cancel_order(symbol, order_id)
                    if individual_result:
                        results.append(individual_result)
                    else:
                        results.append({'orderId': order_id, 'status': 'FAILED', 'message': 'Individual cancellation returned None.'})
                    await asyncio.sleep(0.1) # Small delay between individual cancellations
                except Exception as e_single:
                    self.logger.warning(f"Individual cancel failed for order {order_id} after retries: {e_single}")
                    results.append({'orderId': order_id, 'status': 'FAILED', 'message': str(e_single)})
            return results
        except Exception as e:
            self.logger.error(f"Unexpected error during bulk cancellation attempt for {symbol}: {e}", exc_info=True)
            # Return failed status for all requested order_ids if unexpected error
            return [{'orderId': oid, 'status': 'UNEXPECTED_ERROR', 'message': str(e)} for oid in order_ids]

    @async_retry_api_call()
    async def cancel_all_orders(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Cancels ALL open orders for a specific symbol.
        """
        client = self.client_manager.get_client # Access client via manager
        symbol = symbol.upper()
        self.logger.info(f"Attempting to cancel ALL open orders for {symbol}...")
        try:
            result = await client.futures_cancel_all_open_orders(symbol=symbol)
            self.logger.info(f"Cancel ALL orders request successful for {symbol}. Result: {result}")
            # Binance's futures_cancel_all_open_orders returns a list of dicts for cancelled orders
            return result if result else []
        except Exception as e:
            self.logger.error(f"Error cancelling ALL orders for {symbol}: {e}", exc_info=True)
            raise OrderExecutionError(f"Failed to cancel all orders: {e}") from e

    @async_retry_api_call()
    async def get_order_info(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves information about a specific order on Binance Futures.
        """
        client = self.client_manager.get_client # Access client via manager
        try:
            order = await client.futures_get_order(symbol=symbol, orderId=order_id)
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
            raise ExchangeConnectionError(f"Failed to get order info: {e.message}") from e
        except BinanceRequestException as e:
            self.logger.error(f"Binance Request Exception getting order info ({symbol}, {order_id}): {e.status_code} - {e.message}", exc_info=True)
            raise ExchangeConnectionError(f"Failed to get order info: {e.message}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error getting order info ({symbol}, {order_id}): {e}", exc_info=True)
            raise ExchangeConnectionError(f"Failed to get order info: {e}") from e

    @async_retry_api_call()
    async def close_position(self, symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Closes the open position for the given symbol by placing a market order in the opposite direction.

        Args:
            symbol (str): Trading pair symbol.
            position (Dict[str, Any]): The position dict, must contain 'direction' and 'quantity'.

        Returns:
            Dict[str, Any]: Resulting order details.
        """
        symbol = symbol.upper()
        direction = position.get('direction')
        quantity = position.get('quantity')
        if not direction or not quantity:
            raise OrderExecutionError("Position must contain 'direction' and 'quantity' to close.")

        # Determine opposite side
        side = 'SELL' if direction == 'long' else 'BUY'
        # Place market order with reduceOnly=True to close the position
        self.logger.info(f"Closing position for {symbol}: {direction}, Qty={quantity} with side {side}")
        close_order = await self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=True
        )
        self.logger.info(f"Position closed for {symbol}: {close_order}")
        return close_order

    # --- Utility Methods (delegated to exchange info helper) ---

    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Adjusts a quantity to the exchange's required precision, delegating to the helper.
        """
        return self.exchange_info_helper.adjust_quantity_precision(symbol, quantity)

    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Adjusts a price to the exchange's required precision, delegating to the helper.
        """
        return self.exchange_info_helper.adjust_price_precision(symbol, price)

    def get_min_quantity(self, symbol: str) -> float:
        """
        Returns the minimum quantity allowed for a trade on the exchange for a given symbol,
        delegating to the helper.
        """
        return self.exchange_info_helper.get_min_quantity(symbol)

    def get_min_notional(self, symbol: str) -> float:
        """
        Returns the minimum notional value allowed for a trade on the exchange for a given symbol,
        delegating to the helper.
        """
        return self.exchange_info_helper.get_min_notional(symbol)


    # --- Connection Closure ---

    async def close_connection(self):
        """
        Closes the Binance AsyncClient connection by delegating to the client manager.
        """
        await self.client_manager.close_connection()

