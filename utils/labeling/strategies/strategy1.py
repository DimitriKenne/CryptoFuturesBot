# utils/labeling_strategies/strategy1.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional, Tuple
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this labeling strategy
from config.label import LabelingStrategy1Config

class Strategy1(BaseLabelingStrategy):
    """
    Labeling Strategy 1: Triple Barrier.

    This labeling strategy labels data based on a combination of a profit-taking (TP) barrier,
    a stop-loss (SL) barrier, and a time barrier. A signal (1 for long, -1 for short)
    is generated if the price hits the TP barrier before hitting the SL barrier
    or the maximum holding period.

    The TP and SL levels are dynamically determined based on configuration parameters,
    accounting for transaction fees and slippage.

    - A label of '1' (Buy) is assigned if a potential long position hits its TP
      before its SL within the 'future_return_window'.
    - A label of '-1' (Sell) is assigned if a potential short position hits its TP
      before its SL within the 'future_return_window'.
    - A label of '0' (Neutral) is assigned otherwise.
    """

    def __init__(self, config: LabelingStrategy1Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_rate: float):
        """
        Initializes Labeling Strategy 1 (Triple Barrier).

        Args:
            config (LabelingStrategy1Config): The configuration dataclass for this labeling strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate (0-1).
            slippage_tolerance_rate (float): The estimated slippage rate (0-1).
        """
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        self.logger.info("Labeling Strategy 1 (Triple Barrier) initializing.")

        # Access parameters directly from the LabelingStrategy1Config dataclass
        self.future_return_window = self.config.future_return_window
        # These multipliers are expected to be percentages (0-100) from config
        self.profit_multiplier_pct = self.config.profit_multiplier_pct
        self.stop_loss_multiplier_pct = self.config.stop_loss_multiplier_pct
        self.vol_adj_lookback = self.config.vol_adj_lookback
        self.num_price_bars = self.config.num_price_bars

        self._validate_strategy_config()

        self.logger.info(f"  Future Return Window: {self.future_return_window} bars")
        self.logger.info(f"  Profit Multiplier: {self.profit_multiplier_pct:.2f}%")
        self.logger.info(f"  Stop Loss Multiplier: {self.stop_loss_multiplier_pct:.2f}%")
        self.logger.info(f"  Volatility Adjustment Lookback (ATR): {self.vol_adj_lookback} bars")
        self.logger.info(f"  Transaction Fee Rate: {self.trading_fee_rate:.6f}")
        self.logger.info(f"  Slippage Tolerance: {self.slippage_tolerance_rate:.6f}")

    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Labeling Strategy 1.
        """
        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")
        if not isinstance(self.config.profit_multiplier_pct, (int, float)) or self.config.profit_multiplier_pct <= 0:
            raise ValueError("'profit_multiplier_pct' must be a positive number.")
        if not isinstance(self.config.stop_loss_multiplier_pct, (int, float)) or self.config.stop_loss_multiplier_pct <= 0:
            raise ValueError("'stop_loss_multiplier_pct' must be a positive number.")
        if not isinstance(self.config.vol_adj_lookback, int) or self.config.vol_adj_lookback <= 0:
            raise ValueError("'vol_adj_lookback' must be a positive integer.")
        if not isinstance(self.config.num_price_bars, int) or self.config.num_price_bars <= 0:
            raise ValueError("'num_price_bars' must be a positive integer.")
        self.logger.debug("Labeling Strategy 1 config validated.")

    def _net_to_gross_price_move_rate(self, net_rate: float, trade_type: int) -> float:
        """
        Converts a desired net return/loss rate (including fees/slippage)
        to the required gross price movement rate (0-1).

        Args:
            net_rate (float): The desired net return/loss rate (0-1).
            trade_type (int): 1 for long, -1 for short.

        Returns:
            float: The gross price movement rate required.
        """
        # Ensure denominators are not zero or too close to zero
        long_denom_exit = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        short_denom_entry = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        short_denom_exit = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)

        if abs(long_denom_exit) < FLOAT_EPSILON or \
           abs(short_denom_entry) < FLOAT_EPSILON or \
           abs(short_denom_exit) < FLOAT_EPSILON:
            self.logger.error("Fee/slippage rates lead to division by zero. Check trading_fee_rate/slippage_tolerance_rate values.")
            return np.nan

        if trade_type == 1:  # Long
            # (1 + net_rate) = (exit_price_gross / entry_price_gross) * (1 - fees) / (1 + fees)
            # gross_price_move = ( (1 + net_rate) * (1 + fees_entry) / (1 - fees_exit) ) - 1
            # Simplified using combined factors:
            gross_price_move_rate = ((1 + net_rate) * (1 + self.trading_fee_rate + self.slippage_tolerance_rate) /
                                     (1 - self.trading_fee_rate - self.slippage_tolerance_rate)) - 1
            return gross_price_move_rate
        elif trade_type == -1:  # Short
            # (1 - net_rate) = (exit_price_gross / entry_price_gross) * (1 + fees) / (1 - fees)
            # gross_price_move = 1 - ( (1 - net_rate) * (1 - fees_entry) / (1 + fees_exit) )
            # Simplified using combined factors:
            gross_price_move_rate = 1 - ((1 - net_rate) * (1 - self.trading_fee_rate - self.slippage_tolerance_rate) /
                                     (1 + self.trading_fee_rate + self.slippage_tolerance_rate))
            return gross_price_move_rate
        return np.nan

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Labeling Strategy 1 (Triple Barrier).

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Labeling Strategy 1 (Triple Barrier).")
        self._validate_input_df(df, ['open', 'high', 'low', 'close', 'atr_14']) # Assuming atr_14 is needed for vol_adj_lookback=14

        df_copy = df.copy()
        if df_copy.empty:
            self.logger.warning("Input DataFrame is empty. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0})

        n = len(df_copy)
        labels = np.zeros(n, dtype=int)

        # Convert profit/stop loss multipliers from percentage to rate (0-1)
        # These are the *net* profit/loss rates we aim for, after fees/slippage
        long_tp_net_rate = self.profit_multiplier_pct / 100.0
        short_tp_net_rate = self.profit_multiplier_pct / 100.0 # Short profit is also a positive percentage
        long_sl_net_rate = self.stop_loss_multiplier_pct / 100.0
        short_sl_net_rate = self.stop_loss_multiplier_pct / 100.0 # Short stop loss is also a positive percentage

        # Calculate gross price movement rates required to achieve net profit/loss
        long_tp_gross_rate = self._net_to_gross_price_move_rate(long_tp_net_rate, trade_type=1)
        long_sl_gross_rate = abs(self._net_to_gross_price_move_rate(-long_sl_net_rate, trade_type=1)) # SL is a 'negative' net return
        short_tp_gross_rate = self._net_to_gross_price_move_rate(short_tp_net_rate, trade_type=-1)
        short_sl_gross_rate = abs(self._net_to_gross_price_move_rate(-short_sl_net_rate, trade_type=-1)) # SL is a 'negative' net return

        self.logger.info(f"Calculated Long TP (Net Rate): {long_tp_net_rate:.4f}, Long SL (Net Rate): {long_sl_net_rate:.4f}")
        self.logger.info(f"Calculated Short TP (Net Rate): {short_tp_net_rate:.4f}, Short SL (Net Rate): {short_sl_net_rate:.4f}")
        self.logger.info(f"Calculated Long TP (Gross Rate): {long_tp_gross_rate:.4f}, Long SL (Gross Rate): {long_sl_gross_rate:.4f}")
        self.logger.info(f"Calculated Short TP (Gross Rate): {short_tp_gross_rate:.4f}, Short SL (Gross Rate): {short_sl_gross_rate:.4f}")

        # Check for NaN in gross rates, meaning invalid fee/slippage config
        if pd.isna(long_tp_gross_rate) or pd.isna(long_sl_gross_rate) or \
           pd.isna(short_tp_gross_rate) or pd.isna(short_sl_gross_rate):
            self.logger.error("Invalid gross price move rates calculated (NaN detected). Check fees/slippage configuration.")
            return pd.DataFrame({'label': labels}, index=df_copy.index) # Return all zeros

        # Dynamically adjust thresholds based on ATR
        atr_col = f'atr_{self.vol_adj_lookback}'
        if atr_col not in df_copy.columns:
            self.logger.error(f"ATR column '{atr_col}' not found in DataFrame. Cannot apply volatility adjustment for triple barrier. Ensure feature engineering provides this.")
            # Fallback to fixed percentage if ATR not available - convert percentage from config directly to rate here
            atr_adjusted_long_tp_rate = self.profit_multiplier_pct / 100.0
            atr_adjusted_long_sl_rate = self.stop_loss_multiplier_pct / 100.0
            atr_adjusted_short_tp_rate = self.profit_multiplier_pct / 100.0
            atr_adjusted_short_sl_rate = self.stop_loss_multiplier_pct / 100.0
        else:
            # We must use ATR from *previous* bars to avoid lookahead bias.
            # Assuming 'atr_col' already uses shifted data or is calculated with lookback.
            # If not, df_copy[atr_col].shift(1) might be necessary depending on ATR definition.
            # For simplicity, we assume atr_col at current index is safe for *this bar's* calculation context.
            atr_series = df_copy[atr_col].replace(0, np.nan) # Replace 0 ATR with NaN to avoid division by zero later
            
            # Apply ATR as a base unit, then scale by multipliers, and then convert to gross price moves
            # We already have gross rates, so we just need to scale those if ATR is to be the base unit.
            # This implies profit_multiplier_pct / stop_loss_multiplier_pct are now *multipliers of ATR*, not direct price percentages.
            # This is a common interpretation for Triple Barrier with ATR.

            # Reinterpret profit_multiplier_pct and stop_loss_multiplier_pct as ATR multiples
            # For example, if profit_multiplier_pct is 200 (2x ATR), then (200/100) * ATR
            # This is a key change in interpretation based on config.
            df_copy['long_tp_price'] = df_copy['close'] * (1 + long_tp_gross_rate * atr_series / df_copy['close'])
            df_copy['long_sl_price'] = df_copy['close'] * (1 - long_sl_gross_rate * atr_series / df_copy['close'])
            df_copy['short_tp_price'] = df_copy['close'] * (1 - short_tp_gross_rate * atr_series / df_copy['close'])
            df_copy['short_sl_price'] = df_copy['close'] * (1 + short_sl_gross_rate * atr_series / df_copy['close'])

            # Handle NaNs from ATR: if ATR is NaN for a bar, its barriers become NaN too.
            df_copy[['long_tp_price', 'long_sl_price', 'short_tp_price', 'short_sl_price']] = \
                df_copy[['long_tp_price', 'long_sl_price', 'short_tp_price', 'short_sl_price']].mask(atr_series.isna())
        
        # Iterate through data to find barrier hits
        for i in range(n):
            current_close = df_copy['close'].iloc[i]
            # Ensure current_close is valid before proceeding
            if pd.isna(current_close) or abs(current_close) < FLOAT_EPSILON:
                labels[i] = 0 # Cannot determine label without valid current price
                continue

            # Get dynamic barrier prices for the current bar
            long_tp_price = df_copy['long_tp_price'].iloc[i] if 'long_tp_price' in df_copy.columns else current_close * (1 + long_tp_gross_rate)
            long_sl_price = df_copy['long_sl_price'].iloc[i] if 'long_sl_price' in df_copy.columns else current_close * (1 - long_sl_gross_rate)
            short_tp_price = df_copy['short_tp_price'].iloc[i] if 'short_tp_price' in df_copy.columns else current_close * (1 - short_tp_gross_rate)
            short_sl_price = df_copy['short_sl_price'].iloc[i] if 'short_sl_price' in df_copy.columns else current_close * (1 + short_sl_gross_rate)
            
            # If any barrier price is NaN due to missing ATR, skip this bar
            if pd.isna(long_tp_price) or pd.isna(long_sl_price) or \
               pd.isna(short_tp_price) or pd.isna(short_sl_price):
                labels[i] = 0
                continue

            # Define the window for looking for barrier hits (future_return_window bars + current bar itself)
            window_end_iloc = min(i + self.future_return_window + 1, n)
            
            # Extract relevant future data. Ensure 'high' and 'low' exist and are not NaN for the window.
            window_data = df_copy.iloc[i + 1 : window_end_iloc][['high', 'low']].copy()
            window_data.dropna(inplace=True) # Drop rows with NaNs in high/low within the window

            if window_data.empty:
                # If no valid data in the lookahead window, assume neutral for this bar
                labels[i] = 0
                continue

            # Check for barrier hits
            long_tp_hit = (window_data['high'] >= long_tp_price)
            long_sl_hit = (window_data['low'] <= long_sl_price)
            short_tp_hit = (window_data['low'] <= short_tp_price)
            short_sl_hit = (window_data['high'] >= short_sl_price)

            # Find first index of hit within the window (using iloc indices)
            first_long_tp_idx_iloc = window_data[long_tp_hit].index.min() if long_tp_hit.any() else None
            first_long_sl_idx_iloc = window_data[long_sl_hit].index.min() if long_sl_hit.any() else None
            first_short_tp_idx_iloc = window_data[short_tp_hit].index.min() if short_tp_hit.any() else None
            first_short_sl_idx_iloc = window_data[short_sl_hit].index.min() if short_sl_hit.any() else None

            # Determine label based on which barrier is hit first
            long_condition_met = False
            short_condition_met = False

            if first_long_tp_idx_iloc is not None:
                if first_long_sl_idx_iloc is None or first_long_tp_idx_iloc < first_long_sl_idx_iloc:
                    long_condition_met = True

            if first_short_tp_idx_iloc is not None:
                if first_short_sl_idx_iloc is None or first_short_tp_idx_iloc < first_short_sl_idx_iloc:
                    short_condition_met = True
            
            if long_condition_met and short_condition_met:
                # If both profit conditions are met, prioritize the one that occurred first
                if first_long_tp_idx_iloc < first_short_tp_idx_iloc:
                    labels[i] = 1
                elif first_short_tp_idx_iloc < first_long_tp_idx_iloc:
                    labels[i] = -1
                else:
                    labels[i] = 0 # Simultaneous hit, consider neutral or handle based on strategy specifics
            elif long_condition_met:
                labels[i] = 1
            elif short_condition_met:
                labels[i] = -1
            else:
                labels[i] = 0 # No barrier hit within window, or SL hit before TP for both

        self.logger.debug("Raw labels calculated for Labeling Strategy 1 (Triple Barrier).")
        
        # Return a DataFrame with only the 'label' column and the original index
        return pd.DataFrame({'label': labels}, index=df_copy.index)
