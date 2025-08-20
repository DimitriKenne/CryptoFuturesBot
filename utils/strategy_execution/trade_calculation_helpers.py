# utils/strategy_execution/trade_calculation_helpers.py

import pandas as pd
import numpy as np
import math
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

# Import configuration from params.py
try:
    from config.params import AppConfig, FLOAT_EPSILON
except ImportError as e:
    logging.critical(f"Failed to import necessary configuration modules: {e}. Ensure config/params.py exists and is correctly structured.", exc_info=True)
    raise # Re-raise to prevent unconfigured helpers from being used

logger = logging.getLogger(__name__)

class TradeCalculationHelpers:
    """
    Provides helper methods for trade-related calculations such as position sizing,
    stop-loss/take-profit price determination, liquidation price estimation,
    and applying various entry filters.
    """

    def __init__(self, app_config: AppConfig):
        """
        Initializes the TradeCalculationHelpers with the application configuration.

        Args:
            app_config (AppConfig): The comprehensive application configuration object.
                                    It is assumed that this app_config has already been
                                    validated by config/validator.py externally.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = app_config # Store the full AppConfig

        # Initialize precision and minimums from exchange config for direct access
        # These are frequently used in rounding and validation methods.
        self.price_precision = self.config.exchange.price_precision
        self.quantity_precision = self.config.exchange.quantity_precision
        self.min_quantity = self.config.exchange.min_quantity
        self.min_notional = self.config.exchange.min_notional

        # Derived column names for features from FeatureConfig and TradingConfig
        # These depend on specific config values and are best calculated once here.
        self.volatility_regime_col_name = self.config.features.volatility_regime_col_name
        self.atr_vol_adj_col_name = f'atr_{self.config.trading.sltp.volatility_window_bars}'
        self.ema_filter_col_name = f'ema_{self.config.trading.entry_filter.trend_filter_ema_period}'


    def _round_price(self, price: Optional[float]) -> Optional[float]:
        """
        Rounds a price to the configured precision (decimal places).

        Args:
            price (Optional[float]): The price to round.

        Returns:
            Optional[float]: The rounded price, or NaN if input is NaN/invalid.
        """
        if pd.isna(price): return np.nan
        if not isinstance(self.price_precision, int) or self.price_precision < 0:
             self.logger.warning(f"Invalid price_precision: {self.price_precision}. Cannot round price.")
             return price

        try:
            rounded_price = round(price, self.price_precision)
            self.logger.debug(f"Rounded price: {rounded_price}")
            return rounded_price
        except (TypeError, ValueError) as e:
             self.logger.warning(f"Could not round price {price} to precision {self.price_precision}: {e}")
             return np.nan

    def _round_quantity(self, quantity: Optional[float]) -> Optional[float]:
        """
        Rounds a quantity DOWN to the configured precision (step size).
        This is crucial for ensuring trade quantities adhere to exchange requirements.

        Args:
            quantity (Optional[float]): The quantity to round.

        Returns:
            Optional[float]: The rounded-down quantity, or NaN if input is NaN/invalid.
        """
        if pd.isna(quantity): return np.nan
        if not isinstance(self.quantity_precision, int) or self.quantity_precision < 0:
             self.logger.warning(f"Invalid quantity_precision: {self.quantity_precision}. Cannot round quantity.")
             return quantity

        try:
            factor = 10 ** self.quantity_precision
            # Floor division equivalent for floating point precision
            rounded_quantity = math.floor(quantity * factor) / factor
            self.logger.debug(f"Rounded quantity: {rounded_quantity}")
            return rounded_quantity
        except (TypeError, ValueError) as e:
             self.logger.warning(f"Could not round quantity {quantity} to precision {self.quantity_precision}: {e}")
             return np.nan

    def _validate_quantity_and_notional(self, quantity: Optional[float], price: Optional[float]) -> bool:
        """
        Validates if the given quantity meets minimum requirements and if the
        notional value (quantity * price) meets minimum notional requirements.

        Args:
            quantity (Optional[float]): The quantity to validate.
            price (Optional[float]): The price associated with the quantity.

        Returns:
            bool: True if both quantity and notional meet minimums, False otherwise.
        """
        if pd.isna(quantity) or quantity <= 0:
            self.logger.warning(f"Invalid quantity ({quantity}). Must be positive.")
            return False
        if pd.isna(price) or price <= 0:
            self.logger.warning(f"Invalid price ({price}). Must be positive.")
            return False

        if quantity < self.min_quantity - FLOAT_EPSILON: # Allow for tiny float differences
            self.logger.warning(f"Quantity ({quantity:.8f}) is below minimum allowed ({self.min_quantity:.8f}).")
            return False

        notional_value = quantity * price
        if notional_value < self.min_notional - FLOAT_EPSILON: # Allow for tiny float differences
            self.logger.warning(f"Notional value ({notional_value:.2f}) is below minimum allowed ({self.min_notional:.2f}).")
            return False

        self.logger.debug(f"Quantity {quantity:.8f} and Notional {notional_value:.2f} are valid.")
        return True


    def calculate_sl_tp_prices(self, side: str, current_price: float, latest_atr: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        self.logger.info(
            f"🎯 SL/TP Calculation | Side: {side.upper()} | Price: {current_price:.4f} | ATR: {latest_atr:.4f}"
        )

        # Access SLTP config via self.config
        fixed_sl_pct_fraction = self.config.trading.sltp.fixed_stop_loss_pct / 100.0
        fixed_tp_pct_fraction = self.config.trading.sltp.fixed_take_profit_pct / 100.0

        sl_pct_fraction = fixed_sl_pct_fraction
        tp_pct_fraction = fixed_tp_pct_fraction

        if self.config.trading.sltp.enabled:
            if pd.isna(latest_atr) or latest_atr <= FLOAT_EPSILON:
                self.logger.warning(f"Volatility adjustment enabled but ATR is invalid ({latest_atr}). Using fixed SL/TP percentages.")
            else:
                try:
                    # Convert ATR to a percentage of the current price
                    vol_pct_of_price = latest_atr / current_price
                    dynamic_tp_frac = self.config.trading.sltp.alpha_take_profit * vol_pct_of_price
                    dynamic_sl_frac = self.config.trading.sltp.alpha_stop_loss * vol_pct_of_price
                    # Use the larger of fixed or dynamic percentage for TP/SL levels
                    tp_pct_fraction = max(fixed_tp_pct_fraction, dynamic_tp_frac)
                    sl_pct_fraction = max(fixed_sl_pct_fraction, dynamic_sl_frac)
                    self.logger.debug(f"Dynamic SL/TP based on ATR {latest_atr:.8f}: SL%={dynamic_sl_frac*100:.4f}, TP%={dynamic_tp_frac*100:.4f}")
                except Exception as e:
                    self.logger.error(f"Error calculating dynamic SL/TP: {e}. Using fixed percentages.", exc_info=True)
                    # Fallback to fixed on error (values already set from config)
        else:
             self.logger.debug(f"Using fixed barrier percentages: TP%={tp_pct_fraction*100:.2f}, SL%={sl_pct_fraction*100:.2f}")

        # Ensure minimum SL/TP distance is met
        min_sl_tp_pct_rate = self.config.trading.sltp.min_sl_tp_pct / 100.0
        sl_pct_fraction = max(sl_pct_fraction, min_sl_tp_pct_rate)
        tp_pct_fraction = max(tp_pct_fraction, min_sl_tp_pct_rate)


        # Final check on calculated fractions
        if sl_pct_fraction <= FLOAT_EPSILON:
             self.logger.error(f"Calculated SL% ({sl_pct_fraction*100:.4f}) is zero or negative. Cannot set SL.")
             stop_loss_price = np.nan # Indicate failure by returning NaN
        else:
             # Calculate SL price based on direction:
             # Long position (buy): SL below entry (1 - SL_fraction)
             # Short position (sell): SL above entry (1 + SL_fraction)
             direction_multiplier = 1 if side == 'buy' else -1
             stop_loss_price = current_price * (1 - direction_multiplier * sl_pct_fraction)
             stop_loss_price = self._round_price(stop_loss_price)
             if pd.isna(stop_loss_price) or stop_loss_price <= FLOAT_EPSILON:
                  self.logger.error(f"SL price invalid ({stop_loss_price}) after rounding. Cannot set SL.")
                  stop_loss_price = np.nan

        if tp_pct_fraction <= FLOAT_EPSILON:
             self.logger.warning(f"Calculated TP% ({tp_pct_fraction*100:.4f}) is zero or negative. Setting TP to NaN.")
             take_profit_price = np.nan
        else:
             # Calculate TP price based on direction:
             # Long position (buy): TP above entry (1 + TP_fraction)
             # Short position (sell): TP below entry (1 - TP_fraction)
             direction_multiplier = 1 if side == 'buy' else -1
             take_profit_price = current_price * (1 + direction_multiplier * tp_pct_fraction)
             take_profit_price = self._round_price(take_profit_price)
             if pd.isna(take_profit_price) or take_profit_price <= FLOAT_EPSILON:
                  self.logger.warning(f"TP price invalid ({take_profit_price}) after rounding. Setting TP to NaN.")
                  take_profit_price = np.nan

        self.logger.info(
            f"🎯 SL/TP Result | SL: {stop_loss_price:.4f} | TP: {take_profit_price:.4f}"
        )
        return stop_loss_price, take_profit_price

    def calculate_position_size(self,
                                current_equity: float,
                                current_price: float,
                                stop_loss_price: float,
                                trade_direction: int
                               ) -> Tuple[Optional[float], Optional[float]]:
        self.logger.info(
            f"📏 Position Size | Equity: {current_equity:.2f} | Price: {current_price:.4f} | SL: {stop_loss_price:.4f} | Dir: {trade_direction}"
        )

        # Access risk and trade execution config via self.config
        risk_per_trade_fraction = self.config.trading.risk.risk_per_trade_pct / 100.0
        leverage = self.config.trading.risk.leverage
        trading_fee_rate = self.config.trading.trade_execution.trading_fee_pct / 100.0


        if pd.isna(current_price) or current_price <= FLOAT_EPSILON or \
           pd.isna(stop_loss_price) or stop_loss_price <= FLOAT_EPSILON or \
           current_equity <= FLOAT_EPSILON or leverage <= FLOAT_EPSILON or \
           risk_per_trade_fraction <= FLOAT_EPSILON:
            self.logger.warning("Invalid input for position size calculation. Returning zero size.")
            return None, None

        # --- 1. Calculate Risk-Based Size ---
        capital_to_risk = current_equity * risk_per_trade_fraction
        stop_loss_distance = abs(current_price - stop_loss_price)

        if stop_loss_distance <= FLOAT_EPSILON: # Very small SL distance, potentially invalid or dangerous
            self.logger.warning(f"Stop loss distance is zero or too small ({stop_loss_distance:.{self.price_precision+2}f}). Cannot calculate risk-based size meaningfully.")
            return None, None

        # For futures, quantity = Capital at Risk / SL Distance (in price units)
        risk_based_quantity = capital_to_risk / stop_loss_distance
        self.logger.debug(f"Risk Calc: Equity={current_equity:.2f}, RiskAmt={capital_to_risk:.2f}, SLDist={stop_loss_distance:.{self.price_precision}f}, RiskBasedQty={risk_based_quantity:.{self.quantity_precision+4}f}")

        # --- 2. Calculate Max Size Allowed by Margin ---
        # Max Position Value = Balance / (Initial Margin Rate + Entry Fee Rate)
        # Note: 'current_equity' is used here to represent the capital available in simulation.
        initial_margin_rate = 1.0 / leverage
        effective_cost_rate = initial_margin_rate + trading_fee_rate

        if effective_cost_rate <= FLOAT_EPSILON:
             self.logger.warning(f"Effective cost rate ({effective_cost_rate:.6f}) non-positive. Cannot calculate max margin size.")
             max_allowed_quantity = float('inf') # Effectively no margin limit if calculation fails
        else:
             max_position_value_usd = current_equity / effective_cost_rate
             max_allowed_quantity = max_position_value_usd / current_price if current_price > FLOAT_EPSILON else 0.0
             self.logger.debug(f"Margin Calc: Balance={current_equity:.2f}, MaxValue={max_position_value_usd:.2f}, MaxAllowedQty={max_allowed_quantity:.{self.quantity_precision+4}f}")

        # --- 3. Determine Final Quantity ---
        # Use the minimum of risk-based and margin-based quantities
        final_raw_quantity = min(risk_based_quantity, max_allowed_quantity)

        # Apply exchange quantity precision rules (rounding down)
        adjusted_quantity = self._round_quantity(final_raw_quantity)

        if adjusted_quantity is None or adjusted_quantity <= FLOAT_EPSILON:
             self.logger.warning(f"Final adjusted quantity ({adjusted_quantity}) invalid or zero. Setting size to 0.")
             return None, None

        # --- 4. Check Exchange Minimums ---
        notional_value = adjusted_quantity * current_price

        if not self._validate_quantity_and_notional(adjusted_quantity, current_price): # Re-use internal validation
             self.logger.warning(f"Calculated quantity {adjusted_quantity:.8f} or notional value {notional_value:.2f} are below exchange minimums. Setting size to 0.")
             return None, None

        self.logger.info(
            f"📏 Position Size Result | Qty: {adjusted_quantity:.4f} | Notional: {notional_value:.2f}"
        )
        return adjusted_quantity, notional_value

    def estimate_liquidation_price(self, side: str, entry_price: float) -> Optional[float]:
        self.logger.info(
            f"⚠️ Liquidation Price | Side: {side.upper()} | Entry: {entry_price:.4f}"
        )

        # Access risk and backtest config via self.config
        leverage = self.config.trading.risk.leverage
        maint_margin_rate = self.config.trading.backtest.maintenance_margin_pct / 100.0

        if pd.isna(entry_price) or entry_price <= FLOAT_EPSILON:
             self.logger.error("Invalid entry price for liquidation estimation.")
             return None

        try:
            initial_margin_rate = 1.0 / leverage

            if initial_margin_rate <= maint_margin_rate + FLOAT_EPSILON:
                self.logger.warning(f"Initial margin rate ({initial_margin_rate:.4f}) <= maintenance margin rate ({maint_margin_rate:.4f}). Liquidation likely immediate.")
                return entry_price * (1 - (1 if side == 'buy' else -1) * FLOAT_EPSILON * 100)

            direction_multiplier = 1 if side == 'buy' else -1
            liq_price = entry_price * (1 - direction_multiplier * (initial_margin_rate - maint_margin_rate))

            liq_price = self._round_price(liq_price)

            self.logger.info(
                f"⚠️ Liquidation Price Result: {liq_price:.4f}"
            )
            return liq_price if pd.notna(liq_price) and liq_price > FLOAT_EPSILON else None

        except Exception as e:
            self.logger.error(f"Error estimating liquidation price: {e}", exc_info=True)
            return None

    def is_sl_safe_from_liquidation(self, side: str, stop_loss_price: float, liquidation_price: float) -> bool:
        self.logger.info(
            f"🔒 SL Safety | SL: {stop_loss_price:.4f} | Liq: {liquidation_price:.4f} | Side: {side}"
        )

        if pd.isna(stop_loss_price) or pd.isna(liquidation_price):
             self.logger.warning("Cannot check SL safety: Invalid SL or Liquidation price (NaN).")
             return False

        # Access trade execution config via self.config
        min_liq_distance_fraction = self.config.trading.trade_execution.min_liq_distance_pct / 100.0
        safety_buffer = liquidation_price * min_liq_distance_fraction

        if side == 'buy': # Long position: SL must be above (Liquidation Price + Buffer)
            safe_sl_level = liquidation_price + safety_buffer
            is_safe = stop_loss_price > safe_sl_level - FLOAT_EPSILON # Use tolerance for comparison
            if not is_safe:
                self.logger.warning(f"Long SL check: SL ({stop_loss_price:.8f}) NOT above Liq ({liquidation_price:.8f}) + Buffer ({safety_buffer:.8f}) = {safe_sl_level:.8f}")
        else: # Short position: SL must be below (Liquidation Price - Buffer)
            safe_sl_level = liquidation_price - safety_buffer
            is_safe = stop_loss_price < safe_sl_level + FLOAT_EPSILON # Use tolerance for comparison
            if not is_safe:
                self.logger.warning(f"Short SL check: SL ({stop_loss_price:.8f}) NOT below Liq ({liquidation_price:.8f}) - Buffer ({safety_buffer:.8f}) = {safe_sl_level:.8f}")
        self.logger.info(
            f"🔒 SL Safety Result: {is_safe}"
        )
        return is_safe

    def apply_entry_filters(self,
                            signal: int,
                            latest_features: pd.Series, # Changed to pd.Series for consistency
                            latest_probabilities: Optional[pd.Series] # Changed to pd.Series for consistency
                           ) -> int:
        """
        Applies configured entry filters (e.g., volatility regime, confidence threshold,
        and trend) to a raw model signal.

        Args:
            signal (int): The raw signal from the model (1 for long, -1 for short).
                          Expected to be non-zero for filtering.
            latest_features (pd.Series): Series containing engineered features
                                            for the most recent bar. Must include 'close',
                                            and relevant EMA/ATR/regime columns if filters
                                            are enabled.
            latest_probabilities (Optional[pd.Series]): Series containing model
                                                            probability scores for the
                                                            most recent bar (indexed by signal: -1, 0, 1).
                                                            Required if confidence filter is enabled.

        Returns:
            int: The filtered signal: 1 for a valid long entry, -1 for a valid short entry,
                 or 0 if the signal is filtered out by any active condition.
        """
        if signal == 0: return 0 # Defensive: should not be called with neutral signal
        if latest_features.empty: # Check for empty series
             self.logger.warning("Input data for entry filters is empty. Blocking entry.")
             return 0

        current_bar = latest_features # latest_features is already a Series for the current bar
        current_timestamp = latest_features.name # Get timestamp from Series name
        current_close = current_bar['close']

        # Access entry filter, volatility regime, and feature config via self.config
        # --- Allowed Trade Directions Filter ---
        if (signal == 1 and not self.config.trading.entry_filter.allow_long_trades) or \
           (signal == -1 and not self.config.trading.entry_filter.allow_short_trades):
            self.logger.debug(f"Candle {current_timestamp}: Signal {signal} blocked by allowed trade direction filter.")
            return 0


        # --- Volatility Regime Filter ---
        if self.config.trading.entry_filter.volatility_regime_filter_enabled:
             try:
                  current_regime = current_bar.get(self.volatility_regime_col_name, pd.NA)

                  if pd.isna(current_regime):
                       self.logger.warning(f"Candle {current_timestamp}: Volatility regime value is NaN. Volatility regime filter blocks entry.")
                       return 0

                  # Ensure current_regime is an int before checking in dict
                  current_regime_int = int(current_regime)

                  if current_regime_int not in self.config.trading.volatility_regime.allow_trading:
                       self.logger.warning(f"Candle {current_timestamp}: Volatility regime {current_regime_int} not configured in allow_trading. Blocking entry.")
                       return 0

                  if not self.config.trading.volatility_regime.allow_trading.get(current_regime_int, False):
                       self.logger.debug(f"Candle {current_timestamp}: Trading is not allowed in volatility regime {current_regime_int}. Volatility regime filter blocks signal {signal}.")
                       return 0
                  self.logger.debug(f"Candle {current_timestamp}: Trading is allowed in volatility regime {current_regime_int}. Passed volatility regime filter.")

             except KeyError:
                  self.logger.error(f"Volatility regime filter enabled, but column '{self.volatility_regime_col_name}' not found in latest features. Blocking trade.")
                  return 0
             except ValueError:
                  self.logger.error(f"Invalid integer conversion for volatility regime '{current_regime}'. Blocking entry.")
                  return 0
             except Exception as e:
                  self.logger.error(f"Error during volatility regime filter check at candle {current_timestamp}: {e}. Blocking entry.", exc_info=True)
                  return 0


        # --- Confidence Threshold Filter ---
        if self.config.trading.entry_filter.confidence_filter_enabled:
             confidence_threshold_pct = (self.config.trading.entry_filter.confidence_threshold_long_pct) if signal == 1 else (self.config.trading.entry_filter.confidence_threshold_short_pct)
             confidence_threshold = confidence_threshold_pct / 100.0 # Convert percentage to fraction

             if confidence_threshold > FLOAT_EPSILON:
                  if latest_probabilities is None or latest_probabilities.empty:
                       self.logger.warning("Confidence filter enabled but probabilities are missing or empty. Blocking entry.")
                       return 0

                  try:
                       # Assuming latest_probabilities is a Series indexed by signal (-1, 0, 1)
                       confidence = latest_probabilities.get(signal, 0.0)
                       if pd.isna(confidence):
                            self.logger.debug(f"Candle {current_timestamp}: Confidence score is NaN for signal {signal}. Confidence filter blocks entry.")
                            return 0
                       if confidence < confidence_threshold:
                            self.logger.debug(f"Candle {current_timestamp}: Confidence {confidence:.2f} for signal {signal} is below threshold {confidence_threshold:.2f}. Confidence filter blocks entry.")
                            return 0
                       self.logger.debug(f"Candle {current_timestamp}: Confidence {confidence:.2f} for signal {signal} meets threshold.")
                  except Exception as e:
                       self.logger.error(f"Error during confidence filter check at candle {current_timestamp}: {e}. Blocking entry.", exc_info=True)
                       return 0


        # --- EMA Trend Filter ---
        if self.config.trading.entry_filter.trend_filter_enabled:
            try:
                if self.ema_filter_col_name not in current_bar.index: # Check if column exists in Series index
                     self.logger.warning(f"Candle {current_timestamp}: Trend filter enabled, but EMA column '{self.ema_filter_col_name}' not found in features. Blocking trade.")
                     return 0

                latest_ema = current_bar.get(self.ema_filter_col_name)

                if pd.isna(current_close) or pd.isna(latest_ema):
                     self.logger.warning(f"Candle {current_timestamp}: Trend filter cannot be applied due to NaN close or EMA value. Blocking trade.")
                     return 0

                if (signal == 1 and current_close <= latest_ema + FLOAT_EPSILON) or \
                   (signal == -1 and current_close >= latest_ema - FLOAT_EPSILON):
                    self.logger.info(f"Candle {current_timestamp}: Trend filter blocked signal {signal} (Close={current_close:.4f}, EMA={latest_ema:.4f}).")
                    return 0
                else:
                    self.logger.debug(f"Candle {current_timestamp}: Trend filter passed signal {signal} (Close={current_close:.4f}, EMA={latest_ema:.4f}).")

            except Exception as e:
                 self.logger.error(f"Error applying trend filter at candle {current_timestamp}: {e}. Blocking trade.", exc_info=True)
                 return 0

        self.logger.info(
            f"Applying entry filters: Signal={signal}, Timestamp={latest_features.name}, Probabilities={latest_probabilities.to_dict() if latest_probabilities is not None else '{}'}"
        )
        self.logger.debug(f"Candle {current_timestamp}: Signal {signal} passed all active filters.")
        return signal # Signal passes all active filters

    def calculate_pnl_and_fees(
        self,
        trade_direction_int: int,
        entry_price: float,
        actual_exit_price: float,
        quantity: float,
        entry_fee: float,
        trading_fee_rate: float,
        liquidation_fee_rate: float,
        notional_value_at_exit: float,
        exit_reason: str,
    ) -> Tuple[float, float, float, float]:
        self.logger.info(
            f"💸 PnL/Fees | Dir: {trade_direction_int} | Entry: {entry_price:.4f} | Exit: {actual_exit_price:.4f} | Qty: {quantity} | EntryFee: {entry_fee:.2f} | Reason: {exit_reason}"
        )

        if pd.isna(entry_price) or pd.isna(actual_exit_price) or pd.isna(quantity) or quantity <= FLOAT_EPSILON:
            self.logger.error("Invalid input for PnL and fee calculation.")
            return 0.0, 0.0, 0.0, 0.0

        # 1. Calculate Gross PnL
        if trade_direction_int == 1: # Long position
            gross_pnl = (actual_exit_price - entry_price) * quantity
        elif trade_direction_int == -1: # Short position
            gross_pnl = (entry_price - actual_exit_price) * quantity
        else:
            self.logger.error(f"Invalid trade_direction_int: {trade_direction_int}. Gross PnL set to 0.")
            gross_pnl = 0.0

        # 2. Calculate Exit Fee
        exit_fee = notional_value_at_exit * trading_fee_rate
        if pd.isna(exit_fee): # Defensive check for NaN from invalid multiplication
            exit_fee = 0.0
            self.logger.warning(f"Calculated exit_fee is NaN. Setting to 0. Notional: {notional_value_at_exit}, Rate: {trading_fee_rate}")

        # 3. Calculate Liquidation Fee (if applicable)
        liquidation_fee = 0.0
        if exit_reason == 'liquidation':
            liquidation_fee = notional_value_at_exit * liquidation_fee_rate
            if pd.isna(liquidation_fee):
                liquidation_fee = 0.0
                self.logger.warning(f"Calculated liquidation_fee is NaN. Setting to 0. Notional: {notional_value_at_exit}, Rate: {liquidation_fee_rate}")

        # 4. Calculate Total Fees
        total_fees = entry_fee + exit_fee + liquidation_fee

        # 5. Calculate Net PnL
        net_pnl = gross_pnl - total_fees

        self.logger.info(
            f"💸 PnL/Fees Result | GrossPnL: {gross_pnl:.2f} | ExitFee: {exit_fee:.2f} | LiqFee: {liquidation_fee:.2f} | NetPnL: {net_pnl:.2f}"
        )

        return gross_pnl, exit_fee, liquidation_fee, net_pnl

