# utils/strategy_execution/trade_execution_engine.py

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Import the main AppConfig from your centralized configuration
from config.params import AppConfig, FLOAT_EPSILON
# Import the TradeCalculationHelpers
from utils.strategy_execution.trade_calculation_helpers import TradeCalculationHelpers

logger = logging.getLogger(__name__)

class TradeExecutionEngine:
    """
    Central engine for handling all trade-related calculations and strategy logic.
    It encapsulates trade entry, exit, position sizing, SL/TP, PnL, and filtering.
    Designed for reuse in both backtesting and live trading environments,
    without direct interaction with exchange APIs or data fetching.
    """

    def __init__(self, app_config: AppConfig):
        """
        Initializes the TradeExecutionEngine by extracting all necessary configuration
        parameters from the provided AppConfig object.

        Args:
            app_config (AppConfig): The global application configuration object,
                                    containing all sub-configurations (trading, exchange, features).
                                    It is assumed that this app_config has already been
                                    validated by config/validator.py externally.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TradeExecutionEngine...")

        # --- Configuration Parsing: Extracting relevant sub-configs ---
        self.risk_config = app_config.trading.risk
        self.trade_execution_config = app_config.trading.trade_execution
        self.entry_filter_config = app_config.trading.entry_filter
        self.volatility_regime_config = app_config.trading.volatility_regime
        self.sltp_config = app_config.trading.sltp
        self.backtest_config = app_config.trading.backtest

        self.exchange_config = app_config.exchange
        self.feature_config = app_config.features # For volatility regime column name reference

        # --- Pre-calculated Rates for Efficiency ---
        self.trading_fee_rate = self.trade_execution_config.trading_fee_pct / 100.0
        self.slippage_tolerance_rate = self.trade_execution_config.slippage_tolerance_pct / 100.0
        self.maintenance_margin_rate = self.backtest_config.maintenance_margin_pct / 100.0
        self.liquidation_fee_rate = self.backtest_config.liquidation_fee_pct / 100.0

        # --- Initialize TradeCalculationHelpers ---
        # This instance provides access to all helper functions
        self.trade_calculation_helpers = TradeCalculationHelpers(app_config=app_config)

        # Cache column names for quick access, derived from app_config via TradeCalculationHelpers
        self.volatility_regime_col_name = self.trade_calculation_helpers.volatility_regime_col_name
        self.atr_vol_adj_col_name = self.trade_calculation_helpers.atr_vol_adj_col_name


        self.logger.info("TradeExecutionEngine initialized with configurations and helpers.")

    # ====================================================================
    # --- Public API for Trade Management (Called by Backtester/TradingBot) ---
    # ====================================================================

    def calculate_entry_details(
        self,
        signal: int,
        current_capital: float,
        current_price: float, # The price at which the entry would be considered (e.g., next bar's open)
        current_bar_features: pd.Series, # Features for the bar that generated the signal (e.g., bar 'i')
        model_probabilities: Optional[pd.Series] = None, # Model probabilities for the signal bar's prediction
        current_bar_index: Optional[int] = None # The iloc index of the current bar
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates all necessary details for a potential trade entry.
        This includes applying entry filters, determining position size,
        calculating Stop Loss (SL) and Take Profit (TP) prices, and
        estimating the liquidation price.

        Returns None if the trade does not pass filters or cannot be afforded.
        """
        self.logger.debug(f"Attempting to calculate entry for signal {signal} at {current_price:.{self.exchange_config.price_precision}f} on bar index {current_bar_index}")

        # 1. Input Validation: Basic sanity checks
        if signal == 0:
            self.logger.debug(f"Signal is 0 (neutral). No entry calculation needed.")
            return None
        if pd.isna(current_capital) or current_capital <= FLOAT_EPSILON:
            self.logger.warning(f"Invalid current capital ({current_capital}). Cannot calculate entry.")
            return None
        if pd.isna(current_price) or current_price <= FLOAT_EPSILON:
            self.logger.warning(f"Invalid current price ({current_price}). Cannot calculate entry.")
            return None
        if current_bar_features.empty:
            self.logger.warning("Current bar features are empty. Cannot calculate entry.")
            return None

        # Determine trade side string
        side = 'buy' if signal == 1 else 'sell'
        direction_str = 'long' if signal == 1 else 'short'


        # 2. Apply Entry Filters 🛡️
        filtered_signal = self.trade_calculation_helpers.apply_entry_filters(
            signal=signal,
            latest_features=current_bar_features,
            latest_probabilities=model_probabilities
        )

        if filtered_signal == 0:
            self.logger.info(f"Signal {signal} for {current_bar_features.name} was filtered out. No entry.")
            return None

        # 3. Adjust Entry Price for Slippage
        # For 'buy' (long), we assume slight upward slippage, so entry price increases.
        # For 'sell' (short), we assume slight downward slippage, so entry price decreases.
        slippage_adjusted_price = current_price * (1 + self.slippage_tolerance_rate * filtered_signal)
        adjusted_entry_price = self.trade_calculation_helpers._round_price(slippage_adjusted_price)

        if pd.isna(adjusted_entry_price) or adjusted_entry_price <= FLOAT_EPSILON:
            self.logger.error(f"Adjusted entry price invalid ({adjusted_entry_price}). Cannot proceed with entry.")
            return None

        # 4. Calculate Stop Loss (SL) and Take Profit (TP) Prices 🎯
        latest_atr = current_bar_features.get(self.atr_vol_adj_col_name) # Get ATR from features
        stop_loss_price, take_profit_price = self.trade_calculation_helpers.calculate_sl_tp_prices(
            side=side,
            current_price=adjusted_entry_price, # Use adjusted price as basis for SL/TP
            latest_atr=latest_atr
        )

        if pd.isna(stop_loss_price) or stop_loss_price <= FLOAT_EPSILON:
            self.logger.warning(f"Stop loss price could not be calculated or is invalid ({stop_loss_price}). Blocking entry.")
            return None

        # 5. Estimate Liquidation Price 📉
        liquidation_price = self.trade_calculation_helpers.estimate_liquidation_price(
            side=side,
            entry_price=adjusted_entry_price
        )

        if pd.isna(liquidation_price) or liquidation_price <= FLOAT_EPSILON:
            self.logger.warning(f"Liquidation price could not be estimated or is invalid ({liquidation_price}). Blocking entry.")
            return None

        # 6. Check SL Safety from Liquidation ✅
        is_sl_safe = self.trade_calculation_helpers.is_sl_safe_from_liquidation(
            side=side,
            stop_loss_price=stop_loss_price,
            liquidation_price=liquidation_price
        )
        if not is_sl_safe:
            self.logger.warning(f"Stop loss ({stop_loss_price:.{self.exchange_config.price_precision}f}) is too close to liquidation price ({liquidation_price:.{self.exchange_config.price_precision}f}). Blocking entry.")
            return None

        # 7. Determine Position Size 📏
        adjusted_quantity, notional_value = self.trade_calculation_helpers.calculate_position_size(
            current_equity=current_capital,
            current_price=adjusted_entry_price,
            stop_loss_price=stop_loss_price,
            trade_direction=filtered_signal
        )

        if adjusted_quantity is None or adjusted_quantity <= FLOAT_EPSILON:
            self.logger.warning(f"Position size could not be determined or is zero ({adjusted_quantity}). Blocking entry.")
            return None

        # 8. Calculate Initial Margin and Entry Fee 💲
        # Initial margin is (notional value / leverage)
        initial_margin = notional_value / self.risk_config.leverage

        # Entry fee is (notional value * trading fee rate)
        entry_fee = notional_value * self.trading_fee_rate

        # 9. Determine Max Holding Bars ⏱️
        # Get the current volatility regime from the features (it will be an integer: 0, 1, or 2)
        current_regime = current_bar_features.get(self.volatility_regime_col_name, 0)
        # Ensure it's an integer for dictionary lookup
        if pd.isna(current_regime):
            self.logger.warning(f"Volatility regime for current bar is NaN. Defaulting max_holding_bars to 0 (no time limit).")
            max_holding_bars = 0 # Or a fallback value if regime is missing
        else:
            try:
                current_regime_int = int(current_regime)
                # Look up max holding bars for this regime from config
                max_holding_bars = self.volatility_regime_config.max_holding_bars.get(current_regime_int, 0)
                if max_holding_bars is None: # Handle if a regime exists but has no configured max_holding_bars
                    max_holding_bars = 0
                    self.logger.warning(f"Max holding bars not configured for regime {current_regime_int}. Defaulting to 0.")
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid volatility regime value '{current_regime}'. Defaulting max_holding_bars to 0.")
                max_holding_bars = 0


        # 10. Construct and Return Entry Details Dictionary 📦
        entry_details = {
            'direction_int': filtered_signal,
            'direction_str': direction_str,
            'entry_price': adjusted_entry_price,
            'quantity': adjusted_quantity,
            'notional_value': notional_value,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'liquidation_price': liquidation_price,
            'initial_margin': initial_margin,
            'entry_fee': entry_fee,
            'max_holding_bars': max_holding_bars,
            'entry_time': current_bar_features.name, # Use timestamp from bar index
            'entry_bar_index': current_bar_index, # The index of the bar that triggered entry
            'model_probabilities': model_probabilities.to_dict() if model_probabilities is not None else {},
            'entry_reason': 'ML_signal_entry'
        }

        self.logger.info(f"Calculated entry details for {direction_str} trade at {adjusted_entry_price:.{self.exchange_config.price_precision}f} with quantity {adjusted_quantity:.{self.exchange_config.quantity_precision}f}.")
        return entry_details

    def check_exit_conditions(
        self,
        open_trade: Dict[str, Any],
        current_bar_data: pd.Series, # OHLC + signal + probabilities for the current bar being processed
        current_bar_index: int # The iloc index of the current bar
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Checks if any 'hard' or strategic exit condition (Stop Loss, Take Profit, Liquidation,
        Max Holding Period, or a Filtered Reversal Signal) is met for an open trade
        within the price range or context of the current bar.

        Returns:
            Tuple[bool, Optional[str], Optional[float]]:
                - True if an exit condition is met, False otherwise.
                - The reason for exit ('stop_loss', 'take_profit', 'liquidation', 'max_holding',
                  'reversal_signal', 'invalid_ohlc').
                - The determined exit price.
        """
        self.logger.debug(f"Checking exit conditions for trade {open_trade.get('direction_str')} (Entry: {open_trade.get('entry_price'):.{self.exchange_config.price_precision}f}, SL: {open_trade.get('stop_loss_price'):.{self.exchange_config.price_precision}f}, TP: {open_trade.get('take_profit_price'):.{self.exchange_config.price_precision}f}) at bar index {current_bar_index}")

        # Ensure essential data points are present
        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in current_bar_data.index or pd.isna(current_bar_data[col]):
                self.logger.error(f"Current bar data missing or invalid OHLC value for '{col}'. Exiting trade.")
                return True, 'invalid_ohlc', np.nan # Exit immediately on invalid data

        # Extract relevant info for brevity
        trade_direction_int = open_trade.get('direction_int')
        sl_price = open_trade.get('stop_loss_price')
        tp_price = open_trade.get('take_profit_price')
        liq_price = open_trade.get('liquidation_price')
        
        current_open = current_bar_data['open']
        current_high = current_bar_data['high']
        current_low = current_bar_data['low']
        current_close = current_bar_data['close']
        
        # Determine effective exit price based on bar's prices and hit level
        exit_price_candidate = np.nan # Initialize as NaN


        # --- 1. Liquidation Check (Highest Priority) ---
        if pd.notna(liq_price) and liq_price > FLOAT_EPSILON:
            if trade_direction_int == 1: # Long position
                if current_low <= liq_price + FLOAT_EPSILON:
                    self.logger.warning(f"Long position liquidated at {liq_price:.{self.exchange_config.price_precision}f} (current_low: {current_low:.{self.exchange_config.price_precision}f}).")
                    exit_price_candidate = liq_price # Exit at liquidation price
                    return True, 'liquidation', self.trade_calculation_helpers._round_price(exit_price_candidate)
            elif trade_direction_int == -1: # Short position
                if current_high >= liq_price - FLOAT_EPSILON:
                    self.logger.warning(f"Short position liquidated at {liq_price:.{self.exchange_config.price_precision}f} (current_high: {current_high:.{self.exchange_config.price_precision}f}).")
                    exit_price_candidate = liq_price # Exit at liquidation price
                    return True, 'liquidation', self.trade_calculation_helpers._round_price(exit_price_candidate)


        # --- 2. Stop Loss (SL) Hit Check ---
        # Only check SL if it was set and is valid
        if pd.notna(sl_price) and sl_price > FLOAT_EPSILON:
            if trade_direction_int == 1: # Long position
                if current_low <= sl_price + FLOAT_EPSILON:
                    # If SL hit, the exit price is the SL price itself
                    self.logger.info(f"Long position Stop Loss hit at {sl_price:.{self.exchange_config.price_precision}f} (current_low: {current_low:.{self.exchange_config.price_precision}f}).")
                    exit_price_candidate = sl_price
                    return True, 'stop_loss', self.trade_calculation_helpers._round_price(exit_price_candidate)
            elif trade_direction_int == -1: # Short position
                if current_high >= sl_price - FLOAT_EPSILON:
                    # If SL hit, the exit price is the SL price itself
                    self.logger.info(f"Short position Stop Loss hit at {sl_price:.{self.exchange_config.price_precision}f} (current_high: {current_high:.{self.exchange_config.price_precision}f}).")
                    exit_price_candidate = sl_price
                    return True, 'stop_loss', self.trade_calculation_helpers._round_price(exit_price_candidate)


        # --- 3. Take Profit (TP) Hit Check ---
        # Only check TP if it was set and is valid
        if pd.notna(tp_price) and tp_price > FLOAT_EPSILON:
            if trade_direction_int == 1: # Long position
                if current_high >= tp_price - FLOAT_EPSILON:
                    # If TP hit, the exit price is the TP price itself
                    self.logger.info(f"Long position Take Profit hit at {tp_price:.{self.exchange_config.price_precision}f} (current_high: {current_high:.{self.exchange_config.price_precision}f}).")
                    exit_price_candidate = tp_price
                    return True, 'take_profit', self.trade_calculation_helpers._round_price(exit_price_candidate)
            elif trade_direction_int == -1: # Short position
                if current_low <= tp_price + FLOAT_EPSILON:
                    # If TP hit, the exit price is the TP price itself
                    self.logger.info(f"Short position Take Profit hit at {tp_price:.{self.exchange_config.price_precision}f} (current_low: {current_low:.{self.exchange_config.price_precision}f}).")
                    exit_price_candidate = tp_price
                    return True, 'take_profit', self.trade_calculation_helpers._round_price(exit_price_candidate)


        # --- 4. Max Holding Period Reached ---
        max_holding_bars = open_trade.get('max_holding_bars', 0)
        entry_bar_index = open_trade.get('entry_bar_index')

        if max_holding_bars > 0 and entry_bar_index is not None:
            if current_bar_index - entry_bar_index >= max_holding_bars:
                self.logger.info(f"Max holding period of {max_holding_bars} bars reached. Exiting trade.")
                # Exit at current bar's close price for time-based exit
                exit_price_candidate = current_close
                return True, 'max_holding', self.trade_calculation_helpers._round_price(exit_price_candidate)


        # --- 5. Filtered Reversal Signal ---
        # This signal should already be filtered by the MarketDataHandler using apply_entry_filters
        current_bar_signal = current_bar_data.get('signal')

        if current_bar_signal is not None and current_bar_signal != 0: # Ensure there's an actual signal
            if (trade_direction_int == 1 and current_bar_signal == -1) or \
               (trade_direction_int == -1 and current_bar_signal == 1):
                self.logger.info(f"Reversal signal ({current_bar_signal}) detected for open {open_trade.get('direction_str')} position. Exiting trade.")
                # Exit at current bar's close price for reversal signal
                exit_price_candidate = current_close
                return True, 'reversal_signal', self.trade_calculation_helpers._round_price(exit_price_candidate)


        # --- No Exit Condition Met ---
        self.logger.debug(f"No exit conditions met for trade {open_trade.get('direction_str')} at bar index {current_bar_index}.")
        return False, None, None

    def calculate_exit_details(
        self,
        open_trade: Dict[str, Any],
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        current_bar_index: Optional[int] = None # Added for holding_bars calculation in backtest
    ) -> Dict[str, Any]:
        """
        Calculates the financial outcome of a trade closure (Gross PnL, Net PnL, fees)
        and returns a comprehensive completed trade record.
        This method finalizes the trade details for historical record-keeping.

        Args:
            open_trade (Dict[str, Any]): The details of the currently open trade.
            exit_price (float): The price at which the trade was exited.
            exit_time (datetime): The timestamp of the exit.
            exit_reason (str): The reason for the exit ('stop_loss', 'take_profit', 'liquidation',
                               'max_holding', 'reversal_signal', 'invalid_ohlc').
            current_bar_index (Optional[int]): The index of the bar at which the trade exited.
                                                Required for calculating holding duration in bars.

        Returns:
            Dict[str, Any]: A comprehensive dictionary representing the completed trade record.
        """
        self.logger.debug(f"Calculating exit details for trade (entry {open_trade.get('entry_price')}) at exit price {exit_price} due to {exit_reason}")

        # 1. Input Validation: Basic sanity checks
        if not open_trade or pd.isna(exit_price) or exit_price <= FLOAT_EPSILON or not isinstance(exit_time, datetime):
            self.logger.error("Invalid input for calculate_exit_details. Cannot calculate exit details.")
            return {} # Return empty dict or raise error as appropriate for your error handling

        # Determine actual exit price with conditional slippage
        actual_exit_price = exit_price
        if exit_reason != 'liquidation':
            # Apply slippage to exit price (opposite direction of entry)
            # If trade was long (1), selling to close is -1, so price decreases.
            # If trade was short (-1), buying to close is 1, so price increases.
            direction_int = open_trade.get('direction_int', 0)
            slippage_multiplier = -1 if direction_int == 1 else (1 if direction_int == -1 else 0)

            actual_exit_price = exit_price * (1 + self.slippage_tolerance_rate * slippage_multiplier)
            actual_exit_price = self.trade_calculation_helpers._round_price(actual_exit_price)
            if pd.isna(actual_exit_price) or actual_exit_price <= FLOAT_EPSILON:
                self.logger.error(f"Actual exit price invalid ({actual_exit_price}) after slippage adjustment. Using original exit price.")
                actual_exit_price = exit_price # Fallback to original if slippage calculation fails

        # Calculate notional value at exit for fee calculations
        quantity = open_trade.get('quantity', 0.0)
        notional_value_at_exit = actual_exit_price * quantity


        # 3. Calculate All PnL and Fees 💲
        gross_pnl, exit_fee, liquidation_fee, net_pnl = \
            self.trade_calculation_helpers.calculate_pnl_and_fees(
                trade_direction_int=open_trade.get('direction_int', 0),
                entry_price=open_trade.get('entry_price', 0.0),
                actual_exit_price=actual_exit_price,
                quantity=quantity,
                entry_fee=open_trade.get('entry_fee', 0.0),
                trading_fee_rate=self.trading_fee_rate,
                liquidation_fee_rate=self.liquidation_fee_rate,
                notional_value_at_exit=notional_value_at_exit,
                exit_reason=exit_reason
            )

        # 4. Calculate Holding Duration
        holding_bars = None
        if current_bar_index is not None and open_trade.get('entry_bar_index') is not None:
            holding_bars = current_bar_index - open_trade['entry_bar_index']
            if holding_bars < 0: # Defensive check
                self.logger.warning(f"Calculated holding_bars is negative ({holding_bars}). Setting to 0.")
                holding_bars = 0

        holding_duration_seconds = 0.0
        if isinstance(open_trade.get('entry_time'), datetime) and isinstance(exit_time, datetime):
            holding_duration_seconds = (exit_time - open_trade['entry_time']).total_seconds()
            if holding_duration_seconds < 0: # Defensive check
                self.logger.warning(f"Calculated holding_duration_seconds is negative ({holding_duration_seconds}). Setting to 0.")
                holding_duration_seconds = 0.0


        # 5. Construct Completed Trade Record 📦
        completed_trade = open_trade.copy() # Start with all original entry details
        completed_trade.update({
            'exit_price': actual_exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'gross_pnl': gross_pnl,
            'exit_fee': exit_fee,
            'liquidation_fee': liquidation_fee, # Explicitly include liquidation fee
            'total_fees': open_trade.get('entry_fee', 0.0) + exit_fee + liquidation_fee, # Sum all fees
            'net_pnl': net_pnl,
            'holding_bars': holding_bars,
            'holding_duration_seconds': holding_duration_seconds,
            'is_closed': True, # Mark the trade as closed
            'notional_value_at_exit': notional_value_at_exit # Add notional value at exit for reference
        })

        self.logger.info(f"Trade closed due to '{exit_reason}' at {actual_exit_price:.{self.exchange_config.price_precision}f}. Net PnL: {net_pnl:.2f}.")
        return completed_trade

