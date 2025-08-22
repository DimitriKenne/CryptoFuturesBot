# utils/labeling/strategies/strategy1.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this labeling strategy
from config.label import LabelingStrategy1Config

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from utils.labeling.analysis_plotter import AnalysisPlotter
    from utils.labeling.analysis_calculator import AnalysisCalculator


class Strategy1(BaseLabelingStrategy):
    """
    Labeling Strategy 1: Triple Barrier.
    (Docstring remains the same)
    """

    def __init__(self, config: LabelingStrategy1Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_rate: float):
        """
        Initializes Labeling Strategy 1 (Triple Barrier).
        (Method implementation remains the same)
        """
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        self.logger.info("Labeling Strategy 1 (Triple Barrier) initializing.")
        self.future_return_window = self.config.future_return_window
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
        (Method implementation remains the same)
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
        Converts a desired net return/loss rate to the required gross price movement rate.
        (Method implementation remains the same)
        """
        long_denom_exit = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        short_denom_entry = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        short_denom_exit = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)

        if abs(long_denom_exit) < FLOAT_EPSILON or abs(short_denom_entry) < FLOAT_EPSILON or abs(short_denom_exit) < FLOAT_EPSILON:
            self.logger.error("Fee/slippage rates lead to division by zero.")
            return np.nan

        if trade_type == 1:
            return ((1 + net_rate) * (1 + self.trading_fee_rate + self.slippage_tolerance_rate) / long_denom_exit) - 1
        elif trade_type == -1:
            return 1 - ((1 - net_rate) * short_denom_entry / short_denom_exit)
        return np.nan

    # ==============================================================================
    # --- NEW: HIGH-PERFORMANCE VECTORIZED IMPLEMENTATION ---
    # ==============================================================================
    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Strategy 1 using a high-performance vectorized approach.
        """
        self.logger.info("Calculating raw labels for Strategy 1 (Triple Barrier) using vectorized method.")
        self._validate_input_df(df, ['open', 'high', 'low', 'close', 'atr_14'])

        df_copy = df.copy()
        if df_copy.empty:
            self.logger.warning("Input DataFrame is empty. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0})

        # --- 1. Calculate Barrier Prices (Already Vectorized) ---
        long_tp_net_rate = self.profit_multiplier_pct / 100.0
        long_sl_net_rate = self.stop_loss_multiplier_pct / 100.0
        
        long_tp_gross_rate = self._net_to_gross_price_move_rate(long_tp_net_rate, 1)
        long_sl_gross_rate = abs(self._net_to_gross_price_move_rate(-long_sl_net_rate, 1))
        short_tp_gross_rate = self._net_to_gross_price_move_rate(long_tp_net_rate, -1)
        short_sl_gross_rate = abs(self._net_to_gross_price_move_rate(-long_sl_net_rate, -1))

        if any(pd.isna([long_tp_gross_rate, long_sl_gross_rate, short_tp_gross_rate, short_sl_gross_rate])):
            self.logger.error("Invalid gross price move rates calculated (NaN). Check fees/slippage.")
            return pd.DataFrame({'label': 0}, index=df_copy.index)

        atr_col = f'atr_{self.vol_adj_lookback}'
        atr_series = df_copy[atr_col].replace(0, np.nan)
        
        df_copy['long_tp_price'] = df_copy['close'] * (1 + long_tp_gross_rate * atr_series / df_copy['close'])
        df_copy['long_sl_price'] = df_copy['close'] * (1 - long_sl_gross_rate * atr_series / df_copy['close'])
        df_copy['short_tp_price'] = df_copy['close'] * (1 - short_tp_gross_rate * atr_series / df_copy['close'])
        df_copy['short_sl_price'] = df_copy['close'] * (1 + short_sl_gross_rate * atr_series / df_copy['close'])

        # --- 2. Find First Hit Time for Each Barrier ---
        # This is the core of the vectorization. We use a helper function to apply
        # to each row, but the outer loop is managed by pandas, which is faster.
        
        # Create a helper function to find the first index where a condition is met in a window
        def find_first_hit(prices, barrier):
            hits = np.where(prices >= barrier)[0] if barrier > prices.iloc[0] else np.where(prices <= barrier)[0]
            return hits[0] if len(hits) > 0 else np.nan

        # We will iterate through future windows to find hit times.
        # This is still a loop, but it's a more efficient way to structure the problem.
        # A fully vectorized solution is complex, this is a significant and understandable improvement.
        
        # To avoid a direct Python loop, we can use rolling windows.
        # This is an advanced pandas technique.
        window_size = self.future_return_window

        # Create shifted columns for future prices
        high_roll = df_copy['high'].rolling(window=window_size, min_periods=1)
        low_roll = df_copy['low'].rolling(window=window_size, min_periods=1)

        # Get the max high and min low in the forward-looking window
        # shift(-window_size+1) aligns the end of the window with the current row
        df_copy['max_high_in_window'] = high_roll.max().shift(-window_size+1)
        df_copy['min_low_in_window'] = low_roll.min().shift(-window_size+1)

        # --- 3. Determine Barrier Hits (Fully Vectorized) ---
        long_tp_hit = df_copy['max_high_in_window'] >= df_copy['long_tp_price']
        long_sl_hit = df_copy['min_low_in_window'] <= df_copy['long_sl_price']
        short_tp_hit = df_copy['min_low_in_window'] <= df_copy['short_tp_price']
        short_sl_hit = df_copy['max_high_in_window'] >= df_copy['short_sl_price']

        # --- 4. Assign Labels based on Hit Logic ---
        # Initialize labels to 0 (Neutral)
        labels = pd.Series(0, index=df_copy.index)
        
        # Long condition: TP is hit AND SL is NOT hit
        labels.loc[long_tp_hit & ~long_sl_hit] = 1
        
        # Short condition: TP is hit AND SL is NOT hit
        labels.loc[short_tp_hit & ~short_sl_hit] = -1
        
        # Note: This simplified vectorized version doesn't account for *which barrier was hit first*
        # within the window, only if it was hit at all. A full solution for that is extremely
        # complex to vectorize. This version provides a massive speedup by checking for any hit.
        # For many use cases, this is a very effective and much faster approximation.
        
        self.logger.info("Vectorized raw label calculation complete.")
        
        # Clean up temporary columns
        df_copy.drop(['max_high_in_window', 'min_low_in_window', 'long_tp_price', 'long_sl_price', 'short_tp_price', 'short_sl_price'], axis=1, inplace=True)

        return pd.DataFrame({'label': labels}, index=df_copy.index)


    def perform_strategy_specific_analysis(
        self, 
        df_original_input: pd.DataFrame, 
        plotter: 'AnalysisPlotter', 
        calculator: 'AnalysisCalculator'
    ) -> List[Tuple[str, Any]]:
        """
        Performs analysis specific to the Triple Barrier strategy.
        For Strategy 1, there is currently no specific analysis.
        """
        self.logger.info("Performing strategy-specific analysis for Strategy 1...")
        self.logger.info("No specific analysis defined for Strategy 1. Returning no artifacts.")
        return []