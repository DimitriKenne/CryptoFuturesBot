# utils/labeling_strategies/triple_barrier.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON # Corrected import of FLOAT_EPSILON

class TripleBarrierStrategy(BaseLabelingStrategy):
    """
    Labels data based on a two-step process:
    1. Check if the future net return over 'max_holding_bars' meets a specified TP quantile.
    2. If it does, then check if the calculated SL barrier was NOT hit within 'max_holding_bars'.
       If SL was hit, the signal is neutral (0). Otherwise, it's an active signal (1 or -1).
    The time barrier is implicitly handled by the 'max_holding_bars' window.

    - 1: Long future net return meets quantile AND SL was NOT hit.
    - -1: Short future net return meets quantile AND SL was NOT hit.
    - 0: Future net return does not meet quantile, OR SL was hit.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the TripleBarrierStrategy.

        Config parameters:
        - 'max_holding_bars': Maximum trade duration (int).
        - 'long_tp_quantile_pct': Percentile for positive future net return to set TP for long (float).
        - 'short_tp_quantile_pct': Percentile for negative future net return to set TP for short (float).
        - 'rr_ratio': Desired Risk-Reward ratio (float).
        - 'fee_range': Transaction fee for range calculation (float, e.g., 0.0005).
        - 'slippage_range': Estimated slippage for range calculation (float, e.0.0001).
        """
        super().__init__(config, logger)
        self.logger.info("TripleBarrierStrategy initializing with revised logic: TP quantile + SL disqualifier.")

        self.max_holding_bars = self.config.get('max_holding_bars')
        self.long_tp_quantile_pct = self.config.get('long_tp_quantile_pct')
        self.short_tp_quantile_pct = self.config.get('short_tp_quantile_pct')
        self.rr_ratio = self.config.get('rr_ratio')
        self.fee_range = self.config.get('fee_range')
        self.slippage_range = self.config.get('slippage_range')

        # Initialize attributes to store calculated NET TP/SL percentages (used internally for barrier setting)
        self.long_tp_net_pct: float = 0.0
        self.long_sl_net_pct: float = 0.0
        self.short_tp_net_pct: float = 0.0
        self.short_sl_net_pct: float = 0.0

        # Initialize attributes to store calculated GROSS TP/SL percentages (for external use like backtester)
        self.long_tp_gross_pct: float = 0.0
        self.long_sl_gross_pct: float = 0.0 # Will be positive magnitude for SL
        self.short_tp_gross_pct: float = 0.0
        self.short_sl_gross_pct: float = 0.0 # Will be positive magnitude for SL


        # Now, validate the assigned attributes
        self._validate_strategy_config()

        self.logger.info(f"  Max Holding Period: {self.max_holding_bars} bars")
        self.logger.info(f"  Long TP Quantile Percentile: {self.long_tp_quantile_pct}%")
        self.logger.info(f"  Short TP Quantile Percentile: {self.short_tp_quantile_pct}%")
        self.logger.info(f"  Desired Risk-Reward Ratio: {self.rr_ratio}")
        self.logger.info(f"  Transaction Fee: {self.fee_range}")
        self.logger.info(f"  Slippage: {self.slippage_range}")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Triple Barrier strategy.
        """
        if self.max_holding_bars is None or not isinstance(self.max_holding_bars, int) or self.max_holding_bars <= 0:
            raise ValueError("'max_holding_bars' must be a positive integer.")

        if self.long_tp_quantile_pct is None or not isinstance(self.long_tp_quantile_pct, (int, float)) or not (0 < self.long_tp_quantile_pct < 100):
            raise ValueError("'long_tp_quantile_pct' must be a number between 0 and 100 (exclusive).")

        if self.short_tp_quantile_pct is None or not isinstance(self.short_tp_quantile_pct, (int, float)) or not (0 < self.short_tp_quantile_pct < 100):
            raise ValueError("'short_tp_quantile_pct' must be a number between 0 and 100 (exclusive).")

        if self.rr_ratio is None or not isinstance(self.rr_ratio, (int, float)) or self.rr_ratio <= 0:
            raise ValueError("'rr_ratio' must be a positive number.")

        if self.fee_range is None or not isinstance(self.fee_range, (int, float)) or self.fee_range < 0:
            raise ValueError("'fee_range' must be a non-negative number.")

        if self.slippage_range is None or not isinstance(self.slippage_range, (int, float)) or self.slippage_range < 0:
            raise ValueError("'slippage_range' must be a non-negative number.")

        self.logger.debug("TripleBarrierStrategy config validated.")

    def _calculate_net_return_scalar(self, entry_price: float, exit_price: float, trade_type: int) -> float:
        """
        Calculates net return for scalar values, accounting for fees and slippage.
        Returns percentage.

        Args:
            entry_price (float): The price at which the trade is considered entered.
            exit_price (float): The price at which the trade is considered exited.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            float: Net return percentage.
        """
        if abs(entry_price) < FLOAT_EPSILON:
            return np.nan # Avoid division by zero

        # Factors for fees and slippage
        # For entry: price * (1 + fee + slippage)
        # For exit: price * (1 - fee - slippage)
        entry_cost_factor = (1 + self.fee_range + self.slippage_range)
        exit_revenue_factor = (1 - self.fee_range - self.slippage_range)

        if trade_type == 1: # Long trade: Buy at entry, Sell at exit
            cost_to_enter = entry_price * entry_cost_factor
            revenue_from_exit = exit_price * exit_revenue_factor
            net_return = ((revenue_from_exit - cost_to_enter) / cost_to_enter) * 100.0
        elif trade_type == -1: # Short trade: Sell at entry, Buy at exit
            revenue_from_enter = entry_price * exit_revenue_factor # Sell at entry, so revenue is reduced by costs
            cost_to_exit = exit_price * entry_cost_factor # Buy back at exit, so cost is increased by costs
            net_return = ((revenue_from_enter - cost_to_exit) / revenue_from_enter) * 100.0
        else:
            net_return = np.nan # Should not happen for labels 1 or -1

        return net_return

    def _net_to_gross_price_move_pct(self, net_pct: float, trade_type: int) -> float:
        """
        Converts a desired net return/loss percentage (including fees/slippage)
        to the required gross price movement percentage.

        Args:
            net_pct (float): The desired net return/loss percentage.
                             For profit: positive. For loss: negative.
                             e.g., 5 for 5% net profit, -2 for 2% net loss.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            float: The gross price movement percentage required.
                   For long TP: positive (price increase).
                   For long SL: negative (price decrease).
                   For short TP: negative (price decrease).
                   For short SL: positive (price increase).
        """
        # Ensure denominators are not zero or near zero
        if abs(1 - self.fee_range - self.slippage_range) < FLOAT_EPSILON or \
           abs(1 + self.fee_range + self.slippage_range) < FLOAT_EPSILON:
            self.logger.error("Fee/slippage factors lead to division by zero. Check fee_range/slippage_range values.")
            return np.nan

        if trade_type == 1: # Long trade (Buy then Sell)
            gross_price_move_pct = (((1 + net_pct / 100) * (1 + self.fee_range + self.slippage_range)) / \
                                    (1 - self.fee_range - self.slippage_range) - 1) * 100
            return gross_price_move_pct

        elif trade_type == -1: # Short trade (Sell then Buy)
            gross_price_move_pct = (1 - ((1 - net_pct / 100) * (1 - self.fee_range - self.slippage_range)) / \
                                    (1 + self.fee_range + self.slippage_range)) * 100
            return gross_price_move_pct
        
        return np.nan # Should not happen


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Triple Barrier strategy with data-driven TP/SL.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Triple Barrier strategy with data-driven barriers (revised logic).")
        self._validate_input_df(df, ['open', 'high', 'low', 'close'])

        df_copy = df.copy() # Work on a copy

        # Handle empty DataFrame early
        if df_copy.empty:
            self.logger.warning("Input DataFrame is empty. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0})

        n = len(df_copy) # Define n here

        # --- Step 1: Calculate Future Net Returns at Max Holding for TP Quantile Estimation ---
        # This needs to be done over the entire dataset to get a representative quantile
        future_close_series = df_copy['close'].shift(-self.max_holding_bars)

        # Define factors once here
        entry_cost_long_factor = (1 + self.fee_range + self.slippage_range)
        exit_revenue_long_factor = (1 - self.fee_range - self.slippage_range)
        entry_revenue_short_factor = (1 - self.fee_range - self.slippage_range)
        exit_cost_short_factor = (1 + self.fee_range + self.slippage_range)

        safe_current_close = df_copy['close'].replace(0, np.nan)

        long_returns_at_max_holding = (future_close_series * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) / (safe_current_close * entry_cost_long_factor) * 100.0
        short_returns_at_max_holding = (safe_current_close * entry_revenue_short_factor - future_close_series * exit_cost_short_factor) / (safe_current_close * entry_revenue_short_factor) * 100.0

        # --- Step 2: Determine TP and SL Percentage Thresholds (NET & GROSS) from Quantiles and RR ---
        long_positive_returns = long_returns_at_max_holding[long_returns_at_max_holding > FLOAT_EPSILON].dropna()
        short_negative_returns = short_returns_at_max_holding[short_returns_at_max_holding < -FLOAT_EPSILON].dropna()

        # Calculate NET TP/SL percentages using distinct quantiles
        long_tp_net_pct = 0.0
        if not long_positive_returns.empty:
            long_tp_net_pct = np.percentile(long_positive_returns, self.long_tp_quantile_pct)
            long_tp_net_pct = max(long_tp_net_pct, FLOAT_EPSILON) # Ensure TP is positive
        else:
            self.logger.warning("No positive long returns at max holding found for TP quantile calculation. Long signals will be 0.")

        short_tp_net_pct = 0.0
        if not short_negative_returns.empty:
            # For short, we want the magnitude of the negative return, so we take the absolute value of a low percentile
            short_tp_net_pct = np.abs(np.percentile(short_negative_returns, 100 - self.short_tp_quantile_pct)) # Use short_tp_quantile_pct here
            short_tp_net_pct = max(short_tp_net_pct, FLOAT_EPSILON) # Ensure TP is positive (magnitude)
        else:
            self.logger.warning("No negative short returns at max holding found for TP quantile calculation. Short signals will be 0.")

        # Derive NET SL percentages based on desired RR ratio
        long_sl_net_pct = long_tp_net_pct / self.rr_ratio
        short_sl_net_pct = short_tp_net_pct / self.rr_ratio

        # Store these calculated NET values as instance attributes
        self.long_tp_net_pct = long_tp_net_pct
        self.long_sl_net_pct = long_sl_net_pct
        self.short_tp_net_pct = short_tp_net_pct
        self.short_sl_net_pct = short_sl_net_pct

        # Calculate and store GROSS TP/SL percentages for external use
        self.long_tp_gross_pct = self._net_to_gross_price_move_pct(self.long_tp_net_pct, trade_type=1)
        self.long_sl_gross_pct = abs(self._net_to_gross_price_move_pct(-self.long_sl_net_pct, trade_type=1)) # Pass negative for loss, then take abs for magnitude
        self.short_tp_gross_pct = self._net_to_gross_price_move_pct(self.short_tp_net_pct, trade_type=-1)
        self.short_sl_gross_pct = abs(self._net_to_gross_price_move_pct(-self.short_sl_net_pct, trade_type=-1)) # Pass negative for loss, then take abs for magnitude

        self.logger.info(f"Calculated Long TP (Net): {self.long_tp_net_pct:.4f}%, Long SL (Net): {self.long_sl_net_pct:.4f}%")
        self.logger.info(f"Calculated Short TP (Net): {self.short_tp_net_pct:.4f}%, Short SL (Net): {self.short_sl_net_pct:.4f}%")
        self.logger.info(f"Calculated Long TP (Gross Price Move): {self.long_tp_gross_pct:.4f}%, Long SL (Gross Price Move): {self.long_sl_gross_pct:.4f}%")
        self.logger.info(f"Calculated Short TP (Gross Price Move): {self.short_tp_gross_pct:.4f}%, Short SL (Gross Price Move): {self.short_sl_gross_pct:.4f}%")


        labels = np.zeros(n, dtype=int)

        # --- Step 3: Iterate and Apply Revised Triple Barrier Logic ---
        for i in range(n):
            current_close = df_copy['close'].iloc[i]
            if pd.isna(current_close) or abs(current_close) < FLOAT_EPSILON:
                continue

            # Check if this bar is a potential long signal based on future return at max holding
            # Ensure long_returns_at_max_holding is valid at this index
            is_potential_long_signal = (
                i < len(long_returns_at_max_holding) and
                pd.notna(long_returns_at_max_holding.iloc[i]) and
                long_returns_at_max_holding.iloc[i] >= self.long_tp_net_pct
            )

            # Check if this bar is a potential short signal based on future return at max holding
            # Ensure short_returns_at_max_holding is valid at this index
            is_potential_short_signal = (
                i < len(short_returns_at_max_holding) and
                pd.notna(short_returns_at_max_holding.iloc[i]) and
                short_returns_at_max_holding.iloc[i] <= -self.short_tp_net_pct # Note the negative for short
            )

            # Define the lookahead window for this bar (for SL check)
            window_end_iloc = min(i + self.max_holding_bars + 1, n)
            window_data = df_copy.iloc[i + 1 : window_end_iloc].copy()

            if window_data.empty:
                continue # No future data, label remains 0

            # --- Long Signal Logic ---
            if is_potential_long_signal and self.long_sl_net_pct > FLOAT_EPSILON:
                # Calculate the actual SL price for this specific bar, accounting for fees/slippage
                long_sl_price = current_close * (1 - self.long_sl_net_pct / 100) / (1 + self.fee_range + self.slippage_range)
                
                # Check if SL was hit within the window
                long_sl_hit_idx = np.where(window_data['low'] <= long_sl_price)[0]

                if len(long_sl_hit_idx) == 0: # SL was NOT hit within max_holding_bars
                    labels[i] = 1 # This is a valid long signal
                # else: label remains 0 (SL was hit or no valid thresholds)

            # --- Short Signal Logic ---
            if is_potential_short_signal and self.short_sl_net_pct > FLOAT_EPSILON:
                # Calculate the actual SL price for this specific bar, accounting for fees/slippage
                short_sl_price = current_close * (1 + self.short_sl_net_pct / 100) / (1 - self.fee_range - self.slippage_range)
                
                # Check if SL was hit within the window
                short_sl_hit_idx = np.where(window_data['high'] >= short_sl_price)[0]

                if len(short_sl_hit_idx) == 0: # SL was NOT hit within max_holding_bars
                    # Only assign -1 if it wasn't already labeled as 1 (should be mutually exclusive with proper thresholds)
                    if labels[i] == 0:
                        labels[i] = -1
                # else: label remains 0 (SL was hit or no valid thresholds)

        return pd.DataFrame({'label': labels}, index=df_copy.index)
