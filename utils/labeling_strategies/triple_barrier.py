# utils/labeling_strategies/triple_barrier.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON
from ta.volatility import AverageTrueRange # For fallback ATR calculation

class TripleBarrierStrategy(BaseLabelingStrategy):
    """
    Labels data based on the first barrier hit among three:
    Take Profit (TP), Stop Loss (SL), or Time (max_holding_bars).
    Barriers can be fixed percentage or volatility-adjusted (ATR-based).

    - 1: Long TP hit first (and short TP not hit earlier)
    - -1: Short TP hit first (and long TP not hit earlier)
    - 0: Timeout, SL hit, or ambiguous outcome
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the TripleBarrierStrategy.

        Config parameters:
        - 'max_holding_bars': Maximum trade duration (int).
        - 'fixed_take_profit_pct': Fixed TP percentage (float).
        - 'fixed_stop_loss_pct': Fixed SL percentage (float).
        - 'use_volatility_adjustment': Enable ATR-based barriers (bool).
        - 'vol_adj_lookback': ATR lookback period (int) if volatility adjustment is used.
        - 'alpha_take_profit': ATR multiplier for TP (float) if volatility adjustment is used.
        - 'alpha_stop_loss': ATR multiplier for SL (float) if volatility adjustment is used.
        """
        super().__init__(config, logger)
        self.logger.info("TripleBarrierStrategy initializing...")

        # Assign attributes directly from config using .get() with defaults where appropriate
        # This ensures the attributes always exist on the instance, even if config keys are missing
        self.max_holding_bars = self.config.get('max_holding_bars')
        self.fixed_take_profit_pct = self.config.get('fixed_take_profit_pct')
        self.fixed_stop_loss_pct = self.config.get('fixed_stop_loss_pct')
        self.use_volatility_adjustment = self.config.get('use_volatility_adjustment', False) # Default to False

        if self.use_volatility_adjustment:
            self.vol_adj_lookback = self.config.get('vol_adj_lookback')
            self.alpha_take_profit = self.config.get('alpha_take_profit')
            self.alpha_stop_loss = self.config.get('alpha_stop_loss')
            self.atr_column_name = f'atr_{self.vol_adj_lookback}' if self.vol_adj_lookback is not None else None
        else:
            # Explicitly set these to None if volatility adjustment is not used
            self.vol_adj_lookback = None
            self.alpha_take_profit = None
            self.alpha_stop_loss = None
            self.atr_column_name = None

        # Now, validate the assigned attributes
        self._validate_strategy_config()

        self.logger.info(f"  Max Holding Period: {self.max_holding_bars} bars")
        if self.use_volatility_adjustment:
            self.logger.info(f"  Volatility Adjustment Enabled: ATR Lookback={self.vol_adj_lookback}, TP Alpha={self.alpha_take_profit}, SL Alpha={self.alpha_stop_loss}")
        else:
            self.logger.info(f"  Volatility Adjustment Disabled: Fixed TP={self.fixed_take_profit_pct}%, Fixed SL={self.fixed_stop_loss_pct}%")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the Triple Barrier strategy.
        Now validates the *instance attributes* which should already be set by __init__.
        """
        # Check for None values and types for required attributes
        if self.max_holding_bars is None:
            raise KeyError("'max_holding_bars' is missing or None in config.")
        if not isinstance(self.max_holding_bars, int) or self.max_holding_bars <= 0:
            raise ValueError("'max_holding_bars' must be a positive integer.")

        if self.fixed_take_profit_pct is None:
            raise KeyError("'fixed_take_profit_pct' is missing or None in config.")
        if not isinstance(self.fixed_take_profit_pct, (int, float)) or self.fixed_take_profit_pct <= 0:
            raise ValueError("'fixed_take_profit_pct' must be a positive number.")

        if self.fixed_stop_loss_pct is None:
            raise KeyError("'fixed_stop_loss_pct' is missing or None in config.")
        if not isinstance(self.fixed_stop_loss_pct, (int, float)) or self.fixed_stop_loss_pct <= 0:
            raise ValueError("'fixed_stop_loss_pct' must be a positive number.")

        if self.use_volatility_adjustment is None: # Should not be None due to .get(..., False)
            raise KeyError("'use_volatility_adjustment' is missing or None in config.")
        if not isinstance(self.use_volatility_adjustment, bool):
            raise ValueError("'use_volatility_adjustment' must be a boolean.")

        if self.use_volatility_adjustment:
            if self.vol_adj_lookback is None:
                raise KeyError("'vol_adj_lookback' is missing or None when volatility adjustment is enabled.")
            if not isinstance(self.vol_adj_lookback, int) or self.vol_adj_lookback <= 0:
                raise ValueError("'vol_adj_lookback' must be a positive integer when volatility adjustment is enabled.")

            if self.alpha_take_profit is None:
                raise KeyError("'alpha_take_profit' is missing or None when volatility adjustment is enabled.")
            if not isinstance(self.alpha_take_profit, (int, float)) or self.alpha_take_profit <= 0:
                raise ValueError("'alpha_take_profit' must be a positive number when volatility adjustment is enabled.")

            if self.alpha_stop_loss is None:
                raise KeyError("'alpha_stop_loss' is missing or None when volatility adjustment is enabled.")
            if not isinstance(self.alpha_stop_loss, (int, float)) or self.alpha_stop_loss <= 0:
                raise ValueError("'alpha_stop_loss' must be a positive number when volatility adjustment is enabled.")

        self.logger.debug("TripleBarrierStrategy config validated.")


    def _calculate_atr_fallback(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculates ATR as a fallback if the required ATR column is not present.
        Assumes df has 'high', 'low', 'close' columns.
        """
        self.logger.warning(f"Calculating ATR (period={period}) fallback within TripleBarrierStrategy. "
                            "It's recommended to pre-calculate ATR during feature engineering.")
        if len(df) < period:
            self.logger.warning(f"DataFrame too short ({len(df)} bars) for ATR calculation with period {period}. Returning NaNs.")
            return pd.Series(np.nan, index=df.index)

        return AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period,
            fillna=False # Do not fill NaNs, let them propagate
        ).average_true_range()


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the Triple Barrier strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and potentially 'atr_{period}' features,
                               indexed by time. Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Triple Barrier strategy.")
        # Validate input DataFrame for essential columns
        self._validate_input_df(df, ['open', 'high', 'low', 'close'])

        df_copy = df.copy() # Work on a copy

        # Ensure ATR is available if volatility adjustment is enabled
        if self.use_volatility_adjustment:
            if self.atr_column_name not in df_copy.columns:
                self.logger.warning(f"ATR column '{self.atr_column_name}' not found in input DataFrame. "
                                    f"Attempting to calculate fallback ATR. Ensure feature engineering includes ATR.")
                df_copy[self.atr_column_name] = self._calculate_atr_fallback(df_copy, self.vol_adj_lookback)
                # If fallback ATR calculation results in all NaNs, we might need to disable vol adjustment or exit
                if df_copy[self.atr_column_name].isnull().all():
                    self.logger.error(f"Fallback ATR calculation for period {self.vol_adj_lookback} resulted in all NaNs. Cannot use volatility adjustment.")
                    # Fallback to fixed percentage if ATR is not available
                    self.use_volatility_adjustment = False
                    self.logger.info("Falling back to fixed percentage barriers due to unavailable ATR.")
            else:
                # Ensure ATR column is numeric and handle NaNs
                df_copy[self.atr_column_name] = pd.to_numeric(df_copy[self.atr_column_name], errors='coerce')
                # Fill NaNs in ATR with a small value or mean/median if appropriate, or drop rows.
                # For labeling, it's safer to drop rows if ATR is critical.
                initial_rows = len(df_copy)
                df_copy.dropna(subset=[self.atr_column_name], inplace=True)
                if len(df_copy) < initial_rows:
                    self.logger.warning(f"Dropped {initial_rows - len(df_copy)} rows due to NaNs in ATR column '{self.atr_column_name}'.")
                if df_copy.empty:
                    self.logger.error("DataFrame became empty after dropping ATR NaNs. Cannot generate labels.")
                    # CRITICAL FIX: Return a DataFrame with the original index and all 0 labels
                    return pd.DataFrame(index=df.index, data={'label': 0})


        # Initialize labels
        df_copy['label'] = 0 # Default to neutral

        # Iterate through each bar to set up barriers and check for hits
        # This is typically a loop-based operation for triple barrier
        # due to the need to look forward and check multiple conditions.
        n = len(df_copy)
        labels = np.zeros(n, dtype=int) # Use numpy array for efficiency

        for i in range(n):
            current_close = df_copy['close'].iloc[i]
            if pd.isna(current_close): # Skip if current close is NaN
                continue

            # Define barriers
            if self.use_volatility_adjustment:
                current_atr = df_copy[self.atr_column_name].iloc[i]
                if pd.isna(current_atr) or current_atr <= FLOAT_EPSILON:
                    # If ATR is invalid, fall back to fixed percentage for this specific bar
                    long_tp_price = current_close * (1 + self.fixed_take_profit_pct / 100)
                    long_sl_price = current_close * (1 - self.fixed_stop_loss_pct / 100)
                    short_tp_price = current_close * (1 - self.fixed_take_profit_pct / 100)
                    short_sl_price = current_close * (1 + self.fixed_stop_loss_pct / 100)
                else:
                    long_tp_price = current_close + (self.alpha_take_profit * current_atr)
                    long_sl_price = current_close - (self.alpha_stop_loss * current_atr)
                    short_tp_price = current_close - (self.alpha_take_profit * current_atr)
                    short_sl_price = current_close + (self.alpha_stop_loss * current_atr)
            else:
                long_tp_price = current_close * (1 + self.fixed_take_profit_pct / 100)
                long_sl_price = current_close * (1 - self.fixed_stop_loss_pct / 100)
                short_tp_price = current_close * (1 - self.fixed_take_profit_pct / 100)
                short_sl_price = current_close * (1 + self.fixed_stop_loss_pct / 100)

            # Define the lookahead window for this bar
            window_end_iloc = min(i + self.max_holding_bars + 1, n) # +1 because slicing is exclusive of end

            # Get the price data for the lookahead window
            # Exclude the current bar (i) from the barrier check, start from i+1
            window_data = df_copy.iloc[i + 1 : window_end_iloc]

            if window_data.empty:
                # No future data within max_holding_bars, label remains 0
                continue

            # Check for barrier hits within the window
            long_tp_hit_idx = np.where(window_data['high'] >= long_tp_price)[0]
            long_sl_hit_idx = np.where(window_data['low'] <= long_sl_price)[0]
            short_tp_hit_idx = np.where(window_data['low'] <= short_tp_price)[0]
            short_sl_hit_idx = np.where(window_data['high'] >= short_sl_price)[0]

            # Get the first hit index for each barrier type
            first_long_tp_hit = window_data.index[long_tp_hit_idx[0]] if len(long_tp_hit_idx) > 0 else None
            first_long_sl_hit = window_data.index[long_sl_hit_idx[0]] if len(long_sl_hit_idx) > 0 else None
            first_short_tp_hit = window_data.index[short_tp_hit_idx[0]] if len(short_tp_hit_idx) > 0 else None
            first_short_sl_hit = window_data.index[short_sl_hit_idx[0]] if len(short_sl_hit_idx) > 0 else None

            # Collect all hit timestamps (if any) and their corresponding labels
            hits = []
            if first_long_tp_hit: hits.append((first_long_tp_hit, 1))
            if first_long_sl_hit: hits.append((first_long_sl_hit, 0)) # Long SL means neutral/exit
            if first_short_tp_hit: hits.append((first_short_tp_hit, -1))
            if first_short_sl_hit: hits.append((first_short_sl_hit, 0)) # Short SL means neutral/exit

            if hits:
                # Sort hits by timestamp to find the first one
                hits.sort(key=lambda x: x[0])
                first_hit_time, first_hit_label = hits[0]

                # Determine the final label based on the first hit
                if first_hit_label == 1: # Long TP hit first
                    labels[i] = 1
                elif first_hit_label == -1: # Short TP hit first
                    labels[i] = -1
                else: # SL hit first (either long or short)
                    labels[i] = 0
            else:
                # No barriers hit within max_holding_bars, so it's a timeout
                labels[i] = 0 # Neutral due to timeout

        # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
        # and only the 'label' column. This prevents index corruption.
        return pd.DataFrame({'label': labels}, index=df_copy.index)
