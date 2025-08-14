# utils/strategy_execution/exit_conditions.py

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any

# Import FLOAT_EPSILON from params.py
from config.params import FLOAT_EPSILON

# Import external modules for liquidation check and order precision
from utils.trade_core.liquidation import LiquidationEstimator
from utils.trade_core.order_precision import OrderPrecisionHandler

logger = logging.getLogger(__name__)

class ExitConditions:
    """
    Evaluates various conditions to determine if an open trade should be exited.
    This includes Stop Loss, Take Profit, Max Holding Period, Liquidation,
    and neutral signal conditions. This class *checks* conditions based on pre-defined
    SL/TP levels and market data, it does not calculate new SL/TP levels.
    """

    def __init__(self, strategy_config: Any, backtest_config: Any, exchange_config: Any): # ADDED exchange_config
        """
        Initializes ExitConditions with strategy-specific, backtest-specific,
        and exchange-specific parameters.

        Args:
            strategy_config (Any): The instance of StrategyConfig.
            backtest_config (Any): The instance of BacktestConfig.
            exchange_config (Any): The instance of ExchangeConfig. # NEW arg
        """
        self.strategy_config = strategy_config
        self.backtest_config = backtest_config # For liquidation_fee_rate primarily
        self.exchange_config = exchange_config # Store exchange_config

        # Initialize helper classes
        self.liquidation_estimator = LiquidationEstimator()
        # Initialize OrderPrecisionHandler with parameters from exchange_config
        self.order_precision_handler = OrderPrecisionHandler(
            price_precision=self.exchange_config.price_precision,
            quantity_precision=self.exchange_config.quantity_precision,
            min_quantity=self.exchange_config.min_quantity,
            min_notional=self.exchange_config.min_notional
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("ExitConditions initialized.")

    def check_conditions(self,
                         current_bar: pd.Series,
                         position_direction: int,
                         current_trade_sl_price: float,
                         current_trade_tp_price: float,
                         trade_open_bar_index: int,
                         current_bar_index: int,
                         trade_max_holding_bars: Optional[int], # This is the parameter
                         liquidation_price: Optional[float],
                         price_precision: int # Pass price_precision for rounding and logging
                         ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Checks if any exit condition is met for an open position.

        Args:
            current_bar (pd.Series): The current OHLCV bar with features and signal.
            position_direction (int): Current position state (1: long, -1: short).
            current_trade_sl_price (float): Stop loss price for the current trade.
            current_trade_tp_price (float): Take profit price for the current trade.
            trade_open_bar_index (int): iloc index of the bar where trade was entered.
            current_bar_index (int): iloc index of the current bar being processed.
            trade_max_holding_bars (Optional[int]): Max holding period for the current trade.
            liquidation_price (Optional[float]): The estimated liquidation price from the trade manager.
            price_precision (int): Decimal places for price rounding and logging.

        Returns:
            Tuple[bool, Optional[str], Optional[float]]:
                - True if an exit condition is met, False otherwise.
                - Reason for exit ('stop_loss', 'take_profit', 'max_holding', 'liquidation', 'signal_neutral', 'invalid_ohlc', None).
                - Price at which to exit (adjusted for slippage), or None.
        """
        if position_direction == 0:
            return False, None, None # No position to exit

        current_open = current_bar['open']
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        current_signal = current_bar['signal']

        # --- 0. Check for Invalid OHLCV Data (e.g., if price becomes NaN) ---
        if pd.isna(current_close) or current_close <= 0:
            self.logger.error("Current OHLCV close price is NaN or non-positive. Exiting position due to invalid data.")
            exit_price = current_close # Pass the problematic price; TradeManager will handle
            return True, 'invalid_ohlc', exit_price

        # --- 1. Stop Loss Check ---
        if pd.notna(current_trade_sl_price):
            if position_direction == 1: # Long position
                if current_low <= current_trade_sl_price:
                    # Fill at SL price, capped by current bar's low or open
                    exit_price = max(current_trade_sl_price, current_open) if current_open < current_trade_sl_price else current_trade_sl_price
                    exit_price *= (1 - self.strategy_config.slippage_tolerance_pct)
                    exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
                    self.logger.info(f"STOP LOSS HIT (Long). Current Low: {current_low:.{price_precision}f}, SL: {current_trade_sl_price:.{price_precision}f}. Exiting at {exit_price:.{price_precision}f}")
                    return True, 'stop_loss', exit_price
            elif position_direction == -1: # Short position
                if current_high >= current_trade_sl_price:
                    # Fill at SL price, capped by current bar's high or open
                    exit_price = min(current_trade_sl_price, current_open) if current_open > current_trade_sl_price else current_trade_sl_price
                    exit_price *= (1 + self.strategy_config.slippage_tolerance_pct)
                    exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
                    self.logger.info(f"STOP LOSS HIT (Short). Current High: {current_high:.{price_precision}f}, SL: {current_trade_sl_price:.{price_precision}f}. Exiting at {exit_price:.{price_precision}f}")
                    return True, 'stop_loss', exit_price
        
        # --- 2. Take Profit Check ---
        if pd.notna(current_trade_tp_price):
            if position_direction == 1: # Long position
                if current_high >= current_trade_tp_price:
                    # Fill at TP price, capped by current bar's high or open
                    exit_price = min(current_trade_tp_price, current_open) if current_open > current_trade_tp_price else current_trade_tp_price
                    exit_price *= (1 - self.strategy_config.slippage_tolerance_pct) # Slight adverse slippage
                    exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
                    self.logger.info(f"TAKE PROFIT HIT (Long). Current High: {current_high:.{price_precision}f}, TP: {current_trade_tp_price:.{price_precision}f}. Exiting at {exit_price:.{price_precision}f}")
                    return True, 'take_profit', exit_price
            elif position_direction == -1: # Short position
                if current_low <= current_trade_tp_price:
                    # Fill at TP price, capped by current bar's low or open
                    exit_price = max(current_trade_tp_price, current_open) if current_open < current_trade_tp_price else current_trade_tp_price
                    exit_price *= (1 + self.strategy_config.slippage_tolerance_pct) # Slight adverse slippage
                    exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
                    self.logger.info(f"TAKE PROFIT HIT (Short). Current Low: {current_low:.{price_precision}f}, TP: {current_trade_tp_price:.{price_precision}f}. Exiting at {exit_price:.{price_precision}f}")
                    return True, 'take_profit', exit_price
        
        # --- 3. Max Holding Period Check ---
        # Using the passed parameters directly
        if trade_max_holding_bars is not None and trade_open_bar_index != -1:
            bars_held = current_bar_index - trade_open_bar_index
            if bars_held >= trade_max_holding_bars:
                # Exit at current close, with slippage
                exit_price = current_close * (1 - self.strategy_config.slippage_tolerance_pct if position_direction == 1 else 1 + self.strategy_config.slippage_tolerance_pct)
                exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
                self.logger.info(f"MAX HOLDING PERIOD HIT. Bars Held: {bars_held}. Exiting at {exit_price:.{price_precision}f}.")
                return True, 'max_holding', exit_price

        # --- 4. Liquidation Check ---
        if pd.notna(liquidation_price) and liquidation_price > 0:
            if (position_direction == 1 and current_low <= liquidation_price) or \
               (position_direction == -1 and current_high >= liquidation_price):
                # Assume liquidation occurs at the estimated liquidation price, with additional slippage
                exit_price = liquidation_price * (1 + self.strategy_config.slippage_tolerance_pct if position_direction == 1 else 1 - self.strategy_config.slippage_tolerance_pct)
                exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
                self.logger.critical(f"LIQUIDATION HIT! Est. Liq Price: {liquidation_price:.{price_precision}f}. Exiting at {exit_price:.{price_precision}f}.")
                return True, 'liquidation', exit_price

        # --- 5. Neutral Signal Exit (if enabled) ---
        if self.strategy_config.exit_on_neutral_signal and current_signal == 0:
            exit_price = current_close * (1 - self.strategy_config.slippage_tolerance_pct if position_direction == 1 else 1 + self.strategy_config.slippage_tolerance_pct)
            exit_price = self.order_precision_handler.round_price(exit_price) # No need to pass precision here, it's internal
            self.logger.info(f"NEUTRAL SIGNAL ({current_signal}) received. Exiting position at {exit_price:.{price_precision}f}.")
            return True, 'signal_neutral', exit_price
            
        return False, None, None
