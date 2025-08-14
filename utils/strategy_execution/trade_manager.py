# utils/strategy_execution/trade_manager.py

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

# Import necessary utilities from trade_core
from utils.trade_core.financial_math import FinancialMath
from utils.trade_core.position_sizing import PositionSizer
from utils.trade_core.liquidation import LiquidationEstimator
from utils.trade_core.order_precision import OrderPrecisionHandler # Updated import for class name

# Import FLOAT_EPSILON and app_config from params.py
from config.params import FLOAT_EPSILON, app_config 

logger = logging.getLogger(__name__)

class TradeManager:
    """
    Manages the lifecycle of a single trade, from calculating entry/exit
    parameters to tracking its state. It orchestrates calls to lower-level
    trade_core utilities.
    """
    def __init__(self,
                 strategy_config: Any, # Instance of StrategyConfig
                 exchange_config: Any, # Instance of ExchangeConfig
                 backtest_config: Any, # Instance of BacktestConfig
                 symbol: str, # NEW: Add symbol
                 interval: str, # NEW: Add interval
                 model_type: str # NEW: Add model_type
                 ):
        """
        Initializes the TradeManager with configuration objects and trade metadata.

        Args:
            strategy_config (Any): Instance of StrategyConfig.
            exchange_config (Any): Instance of ExchangeConfig.
            backtest_config (Any): Instance of BacktestConfig.
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The OHLCV interval (e.g., '1h', '5m').
            model_type (str): The type of model used (e.g., 'xgboost').
        """
        self.strategy_config = strategy_config
        self.exchange_config = exchange_config
        self.backtest_config = backtest_config
        self.symbol = symbol # Store symbol
        self.interval = interval # Store interval
        self.model_type = model_type # Store model_type

        # Initialize internal state for the current trade
        self._initialize_trade_state()

        # Initialize helper classes (these don't take config params in their __init__ anymore)
        self.financial_math = FinancialMath()
        self.position_sizer = PositionSizer()
        self.liquidation_estimator = LiquidationEstimator()
        # OrderPrecisionHandler needs precision values, which should come from exchange_config
        self.order_precision_handler = OrderPrecisionHandler(
            price_precision=self.exchange_config.price_precision,
            quantity_precision=self.exchange_config.quantity_precision,
            min_quantity=self.exchange_config.min_quantity,
            min_notional=self.exchange_config.min_notional
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("TradeManager initialized with core trade utilities.")


    def _initialize_trade_state(self):
        """Resets the internal state variables for an open trade."""
        self.position_direction = 0 # 1: long, -1: short, 0: flat
        self.entry_price = np.nan
        self.entry_time = pd.NaT
        self.position_value_entry_usd = 0.0
        self.position_asset_qty = 0.0
        self.liquidation_price = np.nan
        self.current_trade_sl_price = np.nan
        self.current_trade_tp_price = np.nan
        self.trade_open_bar_index = -1
        self.trade_max_holding_bars = None # Can be dynamically set per trade
        self.current_capital = np.nan # Initial capital for the specific trade context


    def open_trade(self,
                   signal: int,
                   current_capital: float,
                   current_bar_data: pd.Series,
                   bar_index: int,
                   ) -> Optional[Dict[str, Any]]:
        """
        Attempts to open a trade based on the given signal and current market conditions.

        Args:
            signal (int): The trade signal (1 for long, -1 for short).
            current_capital (float): The current available capital in the account.
            current_bar_data (pd.Series): The current OHLCV bar data including features.
            bar_index (int): The current bar's index in the overall data sequence.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of trade details if trade is opened, else None.
        """
        if self.position_direction != 0:
            self.logger.warning("Attempted to open a trade while a position is already open. Skipping.")
            return None
        if signal == 0:
            self.logger.debug("Attempted to open trade with neutral signal. Skipping.")
            return None

        # Price and quantity precision are now passed during init to OrderPrecisionHandler
        # So we use self.order_precision_handler's stored precision
        price_precision = self.order_precision_handler.price_precision
        quantity_precision = self.order_precision_handler.quantity_precision
        min_qty = self.order_precision_handler.min_quantity
        min_notional = self.order_precision_handler.min_notional
        
        # Max quantity could be dynamic or from exchange_config
        # For simplicity, assuming no max quantity limit for now or deriving from exchange_config
        # max_qty = self.exchange_config.max_quantity # If it exists

        # Assume entry at current bar's close for simplicity in backtest for now
        # Or current_bar_data['open'] if next bar entry is preferred
        entry_price = current_bar_data['close'] 

        # Apply slippage to entry price
        if signal == 1: # Long entry
            entry_price *= (1 + self.strategy_config.slippage_tolerance_pct)
        elif signal == -1: # Short entry
            entry_price *= (1 - self.strategy_config.slippage_tolerance_pct)
        
        entry_price = self.order_precision_handler.round_price(entry_price) # No need to pass precision here

        if pd.isna(entry_price) or entry_price <= 0:
            self.logger.error(f"Invalid entry price {entry_price} for signal {signal}. Cannot open position.")
            return None

        # Calculate SL/TP prices and estimated liquidation price
        sl_price, tp_price, estimated_liq_price = self._calculate_sl_tp_and_liq_prices(
            current_price=entry_price,
            direction=signal,
            latest_features=current_bar_data # Pass the Series directly
        )

        if pd.isna(sl_price) or sl_price <= 0:
            self.logger.error(f"Calculated Stop Loss price is invalid ({sl_price}). Cannot open position.")
            return None
        # TP price can be NaN if not used, log warning only
        if pd.isna(tp_price) or tp_price <= 0:
            self.logger.warning(f"Calculated Take Profit price is invalid ({tp_price}). Proceeding without TP.")
            tp_price = np.nan # Ensure it's NaN if problematic

        # Calculate position size using PositionSizer
        asset_qty = self.position_sizer.calculate_quantity(
            current_capital=current_capital,
            current_price=entry_price,
            stop_loss_price=sl_price,
            risk_per_trade_fraction=self.strategy_config.risk_per_trade_pct,
            leverage=self.strategy_config.leverage
        )

        # Validate quantity and notional using OrderPrecisionHandler
        if not self.order_precision_handler.validate_quantity_and_notional(
            quantity=asset_qty,
            price=entry_price,
            # Min_quantity and min_notional are now handled by the handler itself via its init
            # No need to pass them explicitly here
        ):
            self.logger.error("Calculated quantity or notional value is invalid after min checks. Cannot open position.")
            return None
        
        # Round quantity to exchange's precision (step size) AFTER validation
        asset_qty = self.order_precision_handler._round_quantity(asset_qty) # Use internal _round_quantity

        # Re-validate quantity after rounding, in case rounding made it too small
        if not self.order_precision_handler.validate_quantity_and_notional(
            quantity=asset_qty,
            price=entry_price,
        ):
            self.logger.error("Rounded quantity is invalid after min checks. Cannot open position.")
            return None


        # Determine max holding bars
        self.trade_max_holding_bars = self._determine_max_holding_bars(
            current_bar_data=current_bar_data,
            volatility_regime_max_holding_bars=self.strategy_config.volatility_regime_params.max_holding_bars if self.strategy_config.volatility_regime_filter_enabled else None,
            volatility_regime_col_name=app_config.features.volatility_regime_col_name if self.strategy_config.volatility_regime_filter_enabled else None # Corrected source for volatility_regime_col_name
        )

        # Set internal state
        self.position_direction = signal
        self.entry_price = entry_price
        self.entry_time = current_bar_data.name # Index of the current bar (timestamp)
        self.position_asset_qty = asset_qty
        self.position_value_entry_usd = self.position_asset_qty * self.entry_price
        self.current_trade_sl_price = sl_price
        self.current_trade_tp_price = tp_price
        self.liquidation_price = estimated_liq_price
        self.trade_open_bar_index = bar_index
        self.current_capital = current_capital # Store capital at trade entry

        # Simulate initial funding and fees (entry fee)
        # We need the full PnL function from FinancialMath to get the entry_fee specifically
        # For entry, exit_price is effectively entry_price for fee calc.
        # CORRECTED: Unpack the tuple return from calculate_pnl_and_fees
        gross_pnl_dummy, entry_fee, exit_fee_dummy, total_fees_dummy = self.financial_math.calculate_pnl_and_fees(
            direction=signal,
            entry_price=self.entry_price,
            exit_price=self.entry_price, # For fee calc, use entry price
            quantity=self.position_asset_qty,
            trading_fee_rate=self.strategy_config.trading_fee_rate,
            exit_reason=None # Not an exit yet
        )
        
        trade_details = {
            'direction': 'LONG' if signal == 1 else 'SHORT',
            'entry_price': self.entry_price,
            'quantity': self.position_asset_qty,
            'entry_time': self.entry_time.isoformat(), # Store as ISO format string
            'sl_price': self.current_trade_sl_price,
            'tp_price': self.current_trade_tp_price,
            'liquidation_price': self.liquidation_price,
            'entry_fee': entry_fee, # Now correctly unpacked
            'position_value_entry_usd': self.position_value_entry_usd,
            'trade_open_bar_index': self.trade_open_bar_index,
            'trade_max_holding_bars': self.trade_max_holding_bars,
            'symbol': self.symbol, # Use self.symbol
            'interval': self.interval, # Use self.interval
            'model_type': self.model_type, # Use self.model_type
            'status': 'OPEN' # Indicate trade is open
        }
        self.logger.info(f"OPEN {trade_details['direction']} {self.position_asset_qty:.{quantity_precision}f} @ {self.entry_price:.{price_precision}f} (Fee: {entry_fee:.4f} USD).")
        self.logger.info(f"SL: {self.current_trade_sl_price:.{price_precision}f}, TP: {self.current_trade_tp_price:.{price_precision}f}, Max Holding: {self.trade_max_holding_bars} bars, Liq Price: {self.liquidation_price:.{price_precision}f}.")
        
        return trade_details

    def close_trade(self,
                    open_trade: Dict[str, Any], # Pass the open trade dict directly
                    exit_price: float,
                    exit_time: datetime,
                    exit_reason: str,
                    current_capital: float, # Pass current capital for PnL context
                    price_precision: int, # Pass for rounding and logging
                    quantity_precision: int # Pass for logging
                    ) -> Optional[Dict[str, Any]]:
        """
        Closes the current open trade, calculates PnL and fees, and resets trade state.

        Args:
            open_trade (Dict[str, Any]): The open trade dictionary passed from outside.
            exit_price (float): The price at which the trade is closed.
            exit_time (datetime): The timestamp of the exit.
            exit_reason (str): The reason for closing the trade.
            current_capital (float): The current total capital before this trade closure.
            price_precision (int): Decimal places for price rounding and logging.
            quantity_precision (int): Decimal places for quantity logging.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of completed trade record, else None if no position open.
        """
        if not open_trade or open_trade.get('status') == 'CLOSED':
            self.logger.warning("Attempted to close an invalid or already closed trade.")
            return None
        
        # Extract necessary details from the passed open_trade dictionary
        trade_direction = open_trade['direction'] # 'LONG' or 'SHORT'
        # Convert string direction back to int for financial_math
        direction_int = 1 if trade_direction == 'LONG' else -1
        entry_price = open_trade['entry_price']
        quantity = open_trade['quantity']
        
        exit_price_adjusted = exit_price
        # Apply slippage to exit price, unless it's a liquidation (slippage already implicitly in liquidation price)
        if exit_reason != 'liquidation':
            exit_price_adjusted *= (1 - self.strategy_config.slippage_tolerance_pct if direction_int == 1 else 1 + self.strategy_config.slippage_tolerance_pct)
        exit_price_adjusted = self.order_precision_handler.round_price(exit_price_adjusted) # No need to pass precision

        if pd.isna(exit_price_adjusted) or exit_price_adjusted <= 0:
            self.logger.error(f"Invalid exit price {exit_price_adjusted} for reason {exit_reason}. Cannot close position.")
            return None


        # Determine fee rate to use
        fee_rate_to_apply = self.backtest_config.liquidation_fee_rate if exit_reason == 'liquidation' else self.strategy_config.trading_fee_rate

        # Calculate PnL and fees using FinancialMath
        # CORRECTED: Unpack the tuple return from calculate_pnl_and_fees
        gross_pnl, entry_fee, exit_fee, total_fees = self.financial_math.calculate_pnl_and_fees(
            direction=direction_int,
            entry_price=entry_price,
            exit_price=exit_price_adjusted,
            quantity=quantity,
            trading_fee_rate=fee_rate_to_apply,
            liquidation_fee_rate=self.backtest_config.liquidation_fee_rate, # Pass liquidation_fee_rate
            exit_reason=exit_reason
        )
        
        # Populate the trade record with calculated PnL and fees
        open_trade.update({
            'exit_time': exit_time.isoformat(),
            'exit_price': exit_price_adjusted,
            'gross_pnl': gross_pnl, # Now correctly unpacked
            'net_pnl': gross_pnl - total_fees, # Net PnL is gross PnL minus total fees
            'entry_fee': entry_fee, # Now correctly unpacked
            'exit_fee': exit_fee, # Now correctly unpacked
            'total_fees': total_fees, # Now correctly unpacked
            'exit_reason': exit_reason,
            'status': 'CLOSED',
        })

        # Calculate bars_held
        open_trade_entry_time = pd.to_datetime(open_trade['entry_time'])
        interval_seconds = self._get_interval_seconds(self.interval) # Use self.interval
        if interval_seconds > FLOAT_EPSILON:
            open_trade['bars_held'] = (exit_time - open_trade_entry_time).total_seconds() / interval_seconds
        else:
            open_trade['bars_held'] = np.nan
            self.logger.warning(f"Could not calculate bars_held: Interval '{self.interval}' resulted in zero or invalid seconds.")


        self.logger.info(f"CLOSED {open_trade['direction']} trade. PnL: {open_trade['net_pnl']:.4f} (Gross: {open_trade['gross_pnl']:.4f}, Fees: {open_trade['total_fees']:.4f}). Reason: {exit_reason}.")
        
        self._initialize_trade_state() # Reset internal state for the TradeManager instance

        return open_trade

    def get_current_trade_state(self) -> Dict[str, Any]:
        """Returns the current state of the open trade, if any."""
        return {
            'position_direction': self.position_direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if pd.notna(self.entry_time) else None, # Store as ISO string
            'position_asset_qty': self.position_asset_qty,
            'position_value_entry_usd': self.position_value_entry_usd,
            'liquidation_price': self.liquidation_price,
            'current_trade_sl_price': self.current_trade_sl_price,
            'current_trade_tp_price': self.current_trade_tp_price,
            'trade_open_bar_index': self.trade_open_bar_index,
            'trade_max_holding_bars': self.trade_max_holding_bars,
            'current_capital': self.current_capital, # Include current capital at time of state retrieval
        }

    def load_trade_state(self, state: Dict[str, Any]):
        """Loads a trade state into the TradeManager, typically from a saved bot state."""
        self.position_direction = state.get('position_direction', 0)
        self.entry_price = state.get('entry_price', np.nan)
        self.entry_time = pd.to_datetime(state['entry_time']) if state.get('entry_time') else pd.NaT
        self.position_asset_qty = state.get('position_asset_qty', 0.0)
        self.position_value_entry_usd = state.get('position_value_entry_usd', 0.0)
        self.liquidation_price = state.get('liquidation_price', np.nan)
        self.current_trade_sl_price = state.get('current_trade_sl_price', np.nan)
        self.current_trade_tp_price = state.get('current_trade_tp_price', np.nan)
        self.trade_open_bar_index = state.get('trade_open_bar_index', -1)
        self.trade_max_holding_bars = state.get('trade_max_holding_bars')
        self.current_capital = state.get('current_capital', np.nan) # Load current capital
        self.logger.info("Trade state loaded into TradeManager.")

    def update_current_bar_info(self, current_bar_data: pd.Series):
        """
        Updates the TradeManager with the latest bar data, primarily for
        dynamic calculations like ATR-based SL/TP or volatility regime.
        """
        # This method can be expanded if TradeManager needs more dynamic access to current bar features
        # For now, it's implicitly used when _calculate_sl_tp_and_liq_prices is called during open_trade
        pass # No action needed here currently, as `open_trade` already receives `current_bar_data`

    def calculate_unrealized_pnl(self, current_price: float, open_trade: Dict[str, Any]) -> float:
        """
        Calculates the unrealized Profit and Loss for an open trade.
        Does NOT include fees.

        Args:
            current_price (float): The current market price.
            open_trade (Dict[str, Any]): The dictionary representing the open trade.

        Returns:
            float: The unrealized PnL.
        """
        if pd.isna(current_price) or current_price <= 0:
            self.logger.warning("Cannot calculate unrealized PnL: current_price is invalid.")
            return 0.0
        if not open_trade or open_trade.get('status') == 'CLOSED':
            self.logger.warning("Cannot calculate unrealized PnL: No valid open trade.")
            return 0.0
        
        direction_int = 1 if open_trade['direction'] == 'LONG' else -1
        entry_price = open_trade['entry_price']
        quantity = open_trade['quantity']

        # Simplified PnL calculation without fees
        return (current_price - entry_price) * quantity * direction_int

    def _calculate_sl_tp_and_liq_prices(self,
                                        current_price: float,
                                        direction: int,
                                        latest_features: pd.Series
                                        ) -> Tuple[float, float, Optional[float]]:
        """
        Calculates Stop Loss (SL) and Take Profit (TP) prices, and estimated
        liquidation price. Adjusts SL if too close to liquidation.

        Args:
            current_price (float): The current market price (entry price for a new trade).
            direction (int): 1 for long, -1 for short.
            latest_features (pd.Series): Series containing the latest bar's features.

        Returns:
            Tuple[float, float, Optional[float]]: (stop_loss_price, take_profit_price, liquidation_price).
        """
        if pd.isna(current_price) or current_price <= 0:
            self.logger.error("Cannot calculate SL/TP: current_price is invalid.")
            return np.nan, np.nan, np.nan

        sl_pct = self.strategy_config.sltp_params.fixed_stop_loss_pct / 100.0
        tp_pct = self.strategy_config.sltp_params.fixed_take_profit_pct / 100.0
        
        # Volatility adjustment
        if self.strategy_config.sltp_params.enabled:
            atr_col = f'atr_{self.strategy_config.sltp_params.volatility_window_bars}' # Derived ATR column name
            # Check if ATR column exists in the Series and is not NaN
            if atr_col in latest_features.index and pd.notna(latest_features[atr_col]):
                current_atr = latest_features[atr_col]
                if current_atr > FLOAT_EPSILON: # Avoid division by zero
                    # Ensure alpha values are applied correctly to yield a fraction
                    sl_pct = (self.strategy_config.sltp_params.alpha_stop_loss * current_atr) / current_price
                    tp_pct = (self.strategy_config.sltp_params.alpha_take_profit * current_atr) / current_price
                    self.logger.debug(f"Volatility-adjusted SL/TP: ATR={current_atr:.4f}, SL_pct={sl_pct*100:.2f}%, TP_pct={tp_pct*100:.2f}%")
                else:
                    self.logger.warning(f"ATR is zero or near zero ({current_atr}). Reverting to fixed SL/TP.")
            else:
                self.logger.warning(f"Volatility adjustment enabled but ATR column '{atr_col}' missing or NaN in features. Using fixed SL/TP.")


        # Apply a minimum percentage for SL/TP to prevent excessively tight stops
        min_pct = self.strategy_config.min_liq_distance_pct / 100.0 # Changed to min_sl_tp_pct
        sl_pct = max(sl_pct, min_pct)
        tp_pct = max(tp_pct, min_pct)


        if direction == 1: # Long position
            stop_loss_price = current_price * (1 - sl_pct)
            take_profit_price = current_price * (1 + tp_pct)
        elif direction == -1: # Short position
            stop_loss_price = current_price * (1 + sl_pct)
            take_profit_price = current_price * (1 - tp_pct)
        else:
            self.logger.error(f"Invalid direction '{direction}' for SL/TP calculation.")
            return np.nan, np.nan, np.nan
        
        # Round prices using the OrderPrecisionHandler
        stop_loss_price = self.order_precision_handler.round_price(stop_loss_price)
        take_profit_price = self.order_precision_handler.round_price(take_profit_price)

        # Ensure SL and TP are positive and distinct from entry price
        if stop_loss_price <= 0 or take_profit_price <= 0:
            self.logger.error(f"Calculated SL ({stop_loss_price}) or TP ({take_profit_price}) is non-positive. Returning NaN.")
            return np.nan, np.nan, np.nan
        if abs(stop_loss_price - current_price) < FLOAT_EPSILON or abs(take_profit_price - current_price) < FLOAT_EPSILON:
             self.logger.warning("Calculated SL or TP is too close to entry price. May indicate an issue with calculations or inputs.")

        # Ensure SL is below TP for long, or above TP for short
        if (direction == 1 and stop_loss_price >= take_profit_price) or \
           (direction == -1 and stop_loss_price <= take_profit_price):
            self.logger.error(f"Invalid SL/TP relationship for direction {direction}: SL={stop_loss_price:.{self.order_precision_handler.price_precision}f}, TP={take_profit_price:.{self.order_precision_handler.price_precision}f}. Returning NaN.")
            return np.nan, np.nan, np.nan

        # Calculate liquidation price
        liquidation_price = self.liquidation_estimator.estimate_liquidation_price(
            entry_price=current_price,
            leverage=self.strategy_config.leverage,
            direction=direction,
            maintenance_margin_rate=self.backtest_config.maintenance_margin_rate # From backtest config
        )

        # Check if SL is dangerously close to liquidation. If so, adjust SL.
        if pd.notna(liquidation_price) and liquidation_price > 0 and \
           not self.liquidation_estimator.is_sl_safe_from_liquidation(
               sl_price=stop_loss_price,
               liquidation_price=liquidation_price,
               direction=direction,
               min_distance_pct=self.strategy_config.min_liq_distance_pct
           ):
            self.logger.warning(f"Calculated SL ({stop_loss_price:.{self.order_precision_handler.price_precision}f}) is too close to estimated liquidation price ({liquidation_price:.{self.order_precision_handler.price_precision}f}). Adjusting SL to be safer.")
            # For simplicity, if unsafe, adjust SL to a minimal safe distance
            if direction == 1: # Long
                stop_loss_price = liquidation_price * (1 + self.strategy_config.min_liq_distance_pct * 1.5) # Move slightly away
            else: # Short
                stop_loss_price = liquidation_price * (1 - self.strategy_config.min_liq_distance_pct * 1.5) # Move slightly away
            stop_loss_price = self.order_precision_handler.round_price(stop_loss_price) # Re-round adjusted SL
            self.logger.info(f"Adjusted SL to {stop_loss_price:.{self.order_precision_handler.price_precision}f} to avoid liquidation.")

        return stop_loss_price, take_profit_price, liquidation_price

    def _determine_max_holding_bars(self,
                                    current_bar_data: pd.Series,
                                    volatility_regime_max_holding_bars: Optional[Dict[int, int]] = None,
                                    volatility_regime_col_name: Optional[str] = None
                                    ) -> Optional[int]:
        """
        Determines the maximum holding period for the current trade,
        prioritizing volatility regime-based limits if enabled and available.

        Args:
            current_bar_data (pd.Series): The current OHLCV bar data including features.
            volatility_regime_max_holding_bars (Optional[Dict]): Mapping of regime to max holding bars.
            volatility_regime_col_name (Optional[str]): Name of the volatility regime column.

        Returns:
            Optional[int]: The maximum number of bars to hold the trade, or None if no limit.
        """
        estimated_max_holding = None

        if volatility_regime_max_holding_bars and volatility_regime_col_name:
            current_regime = current_bar_data.get(volatility_regime_col_name, pd.NA)
            if pd.notna(current_regime):
                try:
                    current_regime_int = int(current_regime)
                    if current_regime_int in volatility_regime_max_holding_bars:
                        estimated_max_holding = volatility_regime_max_holding_bars[current_regime_int]
                        self.logger.debug(f"Setting max holding bars based on volatility regime {current_regime_int}: {estimated_max_holding} bars.")
                except (ValueError, TypeError):
                    self.logger.warning(f"Volatility regime value '{current_regime}' is not an integer. Cannot apply regime-based max holding.")
            else:
                self.logger.warning(f"Volatility regime column '{volatility_regime_col_name}' is missing or NaN. Cannot apply regime-based max holding.")

        # Fallback to general strategy max_holding_period_bars_default if regime-based is None or not set
        if estimated_max_holding is None:
            if self.strategy_config.max_holding_period_bars_default is not None:
                estimated_max_holding = self.strategy_config.max_holding_period_bars_default
                self.logger.debug(f"Using default max holding bars from strategy config: {estimated_max_holding} bars.")
            else:
                self.logger.debug("No specific max holding period configured. Max holding will be effectively unlimited.")

        return estimated_max_holding

    def _get_interval_seconds(self, interval_str: str) -> float:
        """Helper to convert interval string to seconds for approximate bar duration."""
        try:
            if not isinstance(interval_str, str) or len(interval_str) < 2:
                self.logger.warning(f"Invalid interval string format: '{interval_str}'. Returning 60 seconds as default.")
                return 60.0

            unit = interval_str[-1].lower()
            value = int(interval_str[:-1])

            if unit == 'm': return value * 60.0
            if unit == 'h': return value * 3600.0
            if unit == 'd': return value * 86400.0
            if unit == 'w': return value * 604800.0
            if unit == 'M': return value * 2592000.0 # Approximate month as 30 days
            self.logger.warning(f"Unsupported interval unit '{unit}' for time calculation. Returning 60 seconds as default.")
            return 60.0
        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to parse interval string '{interval_str}': {e}. Returning 60 seconds as default.", exc_info=True)
            return 60.0
