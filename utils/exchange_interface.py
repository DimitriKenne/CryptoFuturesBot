# utils/exchange_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import logging # Import logging here

class ExchangeInterface(ABC):
    """
    Abstract Base Class (ABC) for exchange adapters.
    Defines the common interface that all exchange adapters must implement.
    This ensures consistency and allows the trading bot to work with different
    exchanges by simply swapping out the adapter.
    """

    def __init__(self, symbol: str, leverage: int, logger: Any, config: Dict[str, Any]):
        """
        Initializes the exchange interface with common parameters.

        Args:
            symbol (str): The trading pair symbol (e.g., "BTCUSDT").
            leverage (int): The leverage to use for trading.
            logger (Any): A logger instance for logging messages.
            config (Dict[str, Any]): Exchange-specific configuration parameters.
        """
        self.symbol = symbol
        self.leverage = leverage
        self.logger = logger
        self.config = config
        self.testnet = config.get('testnet', True) # Default to testnet for safety
        # These will be updated by the concrete adapter from exchange info
        self.price_precision = config.get('price_precision')
        self.quantity_precision = config.get('quantity_precision')
        self.min_quantity = config.get('min_quantity')
        self.min_notional = config.get('min_notional')

    @abstractmethod
    async def async_setup(self):
        """
        Performs asynchronous setup for the exchange connection, e.g., setting leverage.
        This method should be called once after the bot starts.
        """
        pass

    @abstractmethod
    async def fetch_recent_candles(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Fetches recent OHLCV (Open, High, Low, Close, Volume) candles.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Candle interval (e.g., "1m", "5m", "1h").
            limit (int): Number of recent candles to fetch.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame with OHLCV data,
                                    indexed by datetime, or None if data cannot be fetched.
                                    Columns must be 'open', 'high', 'low', 'close', 'volume'.
        """
        pass

    @abstractmethod
    async def get_historical_candles(self, symbol: str, interval: str, start_time: Optional[pd.Timestamp] = None, end_time: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical kline/candlestick data within a specified time range, handling pagination.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The time interval for candles (e.g., '1m', '5m').
            start_time (Optional[pd.Timestamp]): The start time for fetching data (inclusive). Defaults to None.
            end_time (Optional[pd.Timestamp]): The end time for fetching data (inclusive). Defaults to None.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the historical data, or None on failure or if no data.
                                    Indexed by open time (UTC), with 'open', 'high', 'low', 'close', 'volume' columns.
        """
        pass

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the current market price for a given symbol.

        Args:
            symbol (str): Trading pair symbol.

        Returns:
            Optional[float]: The current price, or None if not available.
        """
        pass

    @abstractmethod
    async def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieves all currently open positions for the account or a specific symbol.

        Args:
            symbol (str): Trading pair symbol.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an open position.
                                  Expected keys: 'symbol', 'direction' ('long'/'short'), 'quantity',
                                  'entryPrice', 'unrealizedPnl', 'leverage', 'entryMargin', 'liquidationPrice',
                                  'entryTime' (timestamp).
        """
        pass

    @abstractmethod
    async def get_position_liquidation_price(self, symbol: str) -> Optional[float]:
        """
        Gets the liquidation price for the open position on a specific symbol.

        Args:
            symbol (str): Trading pair symbol.

        Returns:
            Optional[float]: The liquidation price as a float, or None if no open position
                             or liquidation price is not available/relevant.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Places a market order.

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Quantity to trade.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details (e.g., 'orderId', 'status', 'executedQty', 'avgPrice').
        """
        pass

    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Places a limit order.

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            price (float): Price at which to place the order.
            quantity (float): Quantity to trade.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details.
        """
        pass

    @abstractmethod
    async def place_stop_market_order(self, symbol: str, side: str, quantity: float, stop_price: float, reduce_only: bool = True) -> Dict[str, Any]:
        """
        Places a STOP_MARKET order (often used for Stop Loss).

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Quantity to trade.
            stop_price (float): The price at which the market order will be triggered.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details.
        """
        pass

    @abstractmethod
    async def place_take_profit_market_order(self, symbol: str, side: str, quantity: float, stop_price: float, reduce_only: bool = True) -> Dict[str, Any]:
        """
        Places a TAKE_PROFIT_MARKET order (often used for Take Profit).

        Args:
            symbol (str): Trading pair symbol.
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Quantity to trade.
            stop_price (float): The price at which the market order will be triggered.
            reduce_only (bool): If True, order will only reduce an existing position.

        Returns:
            Dict[str, Any]: Dictionary containing order details.
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancels a specific open order.

        Args:
            symbol (str): Trading pair symbol.
            order_id (str): The ID of the order to cancel.

        Returns:
            Dict[str, Any]: Dictionary containing cancellation details.
        """
        pass

    @abstractmethod
    async def cancel_multiple_orders(self, symbol: str, order_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Cancels multiple open orders.

        Args:
            symbol (str): Trading pair symbol.
            order_ids (List[str]): A list of order IDs to cancel.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each with cancellation details for an order.
        """
        pass

    @abstractmethod
    async def get_order_info(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves information about a specific order.

        Args:
            symbol (str): Trading pair symbol.
            order_id (str): The ID of the order to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing order details, or None if not found.
                                     Expected keys: 'orderId', 'symbol', 'status', 'executedQty', 'avgPrice', 'cumQuote'.
        """
        pass

    @abstractmethod
    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Adjusts a quantity to the exchange's required precision.

        Args:
            symbol (str): Trading pair symbol.
            quantity (float): The raw quantity.

        Returns:
            float: The quantity adjusted to the correct precision.
        """
        pass

    @abstractmethod
    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Adjusts a price to the exchange's required precision.

        Args:
            symbol (str): Trading pair symbol.
            price (float): The raw price.

        Returns:
            float: The price adjusted to the correct precision.
        """
        pass

    @abstractmethod
    def get_min_quantity(self, symbol: str) -> float:
        """
        Returns the minimum quantity allowed for a trade on the exchange for a given symbol.
        """
        pass

    @abstractmethod
    def get_min_notional(self, symbol: str) -> float:
        """
        Returns the minimum notional value allowed for a trade on the exchange for a given symbol.
        """
        pass

    @abstractmethod
    async def close_connection(self):
        """
        Closes any open connections to the exchange (e.g., websockets).
        """
        pass
