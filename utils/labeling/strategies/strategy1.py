# utils/labeling_strategies/strategy1.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional, Tuple # Use Any for config type hint initially
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this strategy
from config.label_config_schema import Strategy1Config

class Strategy1(BaseLabelingStrategy):
    """
    Strategy 1: Triple Barrier.

    This strategy labels data based on a combination of a profit-taking (TP) barrier,
    a stop-loss (SL) barrier, and a time barrier. A signal (1 for long, -1 for short)
    is generated if the price hits the TP barrier before hitting the SL barrier
    or the maximum holding period.

    The TP and SL levels are dynamically determined based on historical future net
    returns and a specified risk-reward (RR) ratio, accounting for transaction fees and slippage.

    - A label of '1' (Buy) is assigned if a potential long position hits its TP
      before its SL within the 'future_return_window' (formerly max_holding_bars) window.
    - A label of '-1' (Sell) is assigned if a potential short position hits its TP
      before its SL within the 'future_return_window' (formerly max_holding_bars) window.
    - A label of '0' (Neutral) is assigned otherwise (e.g., if SL is hit first,
      or if neither TP nor SL is hit within the time barrier).
    """

    def __init__(self, config: Strategy1Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_pct: float):
        """
        Initializes Strategy 1 (Triple Barrier Strategy).

        Args:
            config (Strategy1Config): The configuration dataclass for this strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate.
            slippage_tolerance_pct (float): The estimated slippage rate.
        """
        # Pass config to the superclass, which now also accepts fee/slippage
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_pct)
        self.logger.info("Strategy 1 (Triple Barrier) initializing with revised logic: TP quantile + SL disqualifier.")

        # Access parameters directly from the Strategy1Config dataclass
        self.future_return_window = self.config.future_return_window # This is the time barrier
        self.profit_multiplier = self.config.profit_multiplier
        self.stop_loss_multiplier = self.config.stop_loss_multiplier
        self.vol_adj_lookback = self.config.vol_adj_lookback
        self.num_price_bars = self.config.num_price_bars # Not directly used in label calculation logic currently

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

        self.logger.info(f"  Future Return Window (Time Barrier): {self.future_return_window} bars")
        self.logger.info(f"  Profit Multiplier: {self.profit_multiplier}")
        self.logger.info(f"  Stop Loss Multiplier: {self.stop_loss_multiplier}")
        self.logger.info(f"  Volatility Adjustment Lookback (ATR): {self.vol_adj_lookback}")
        self.logger.info(f"  Trading Fee Rate: {self.trading_fee_rate:.4f}")
        self.logger.info(f"  Slippage Tolerance: {self.slippage_tolerance_pct:.6f}")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Strategy 1,
        now accessing directly from self.config (Strategy1Config).
        """
        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")

        if not isinstance(self.config.profit_multiplier, (int, float)) or self.config.profit_multiplier <= 0:
            raise ValueError("'profit_multiplier' must be a positive number.")

        if not isinstance(self.config.stop_loss_multiplier, (int, float)) or self.config.stop_loss_multiplier <= 0:
            raise ValueError("'stop_loss_multiplier' must be a positive number.")
        
        if not isinstance(self.config.vol_adj_lookback, int) or self.config.vol_adj_lookback <= 0:
            raise ValueError("'vol_adj_lookback' must be a positive integer.")
        
        if not isinstance(self.config.num_price_bars, int) or self.config.num_price_bars <= 0:
            raise ValueError("'num_price_bars' must be a positive integer.")

        self.logger.debug("Strategy 1 config validated.")

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

        # Factors for fees and slippage, using self.trading_fee_rate and self.slippage_tolerance_pct
        entry_cost_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        exit_revenue_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)

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
        if abs(1 - self.trading_fee_rate - self.slippage_tolerance_pct) < FLOAT_EPSILON or \
           abs(1 + self.trading_fee_rate + self.slippage_tolerance_pct) < FLOAT_EPSILON:
            self.logger.error("Fee/slippage factors lead to division by zero. Check trading_fee_rate/slippage_tolerance_pct values.")
            return np.nan

        if trade_type == 1: # Long trade (Buy then Sell)
            gross_price_move_pct = (((1 + net_pct / 100) * (1 + self.trading_fee_rate + self.slippage_tolerance_pct)) / \
                                    (1 - self.trading_fee_rate - self.slippage_tolerance_pct) - 1) * 100
            return gross_price_move_pct

        elif trade_type == -1: # Short trade (Sell then Buy)
            gross_price_move_pct = (1 - ((1 - net_pct / 100) * (1 - self.trading_fee_rate - self.slippage_tolerance_pct)) / \
                                    (1 + self.trading_fee_rate + self.slippage_tolerance_pct)) * 100
            return gross_price_move_pct
        
        return np.nan # Should not happen


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Strategy 1 (Triple Barrier).

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for Strategy 1 (Triple Barrier) with data-driven barriers (revised logic).")
        self._validate_input_df(df, ['open', 'high', 'low', 'close'])

        df_copy = df.copy() # Work on a copy

        # Handle empty DataFrame early
        if df_copy.empty:
            self.logger.warning("Input DataFrame is empty. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0})

        n = len(df_copy) # Define n here

        # --- Step 1: Calculate Future Net Returns at Max Holding for TP Quantile Estimation ---
        future_close_series = df_copy['close'].shift(-self.future_return_window)

        # Define factors once here, using attributes from base class
        entry_cost_long_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        exit_revenue_long_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)
        entry_revenue_short_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)
        exit_cost_short_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)

        safe_current_close = df_copy['close'].replace(0, np.nan)

        long_returns_at_max_holding = (future_close_series * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) / (safe_current_close * entry_cost_long_factor) * 100.0
        short_returns_at_max_holding = (safe_current_close * entry_revenue_short_factor - future_close_series * exit_cost_short_factor) / (safe_current_close * entry_revenue_short_factor) * 100.0

        # --- Step 2: Determine TP and SL Percentage Thresholds (NET & GROSS) from Quantiles and RR ---
        # Strategy1Config uses profit_multiplier and stop_loss_multiplier directly as target distances,
        # not quantiles. Re-interpreting the old 'long_tp_quantile_pct' and 'short_tp_quantile_pct'
        # as a direct target return (if they were intended as such) or removing them.
        # Based on Strategy1Config: profit_multiplier & stop_loss_multiplier are fixed ratios to ATR or similar.
        # The previous code for Strategy1 used 'long_tp_quantile_pct' and 'short_tp_quantile_pct'
        # which isn't in Strategy1Config.

        # The current Strategy1Config is: profit_multiplier, stop_loss_multiplier, future_return_window, vol_adj_lookback, num_price_bars.
        # This implies fixed barriers or ATR-adjusted fixed barriers, not quantile-based TP.
        # Let's align the logic with the Strategy1Config parameters.
        # The original code's "TP quantile estimation" seems to be a mismatch with the new Strategy1Config.
        # Assuming profit_multiplier and stop_loss_multiplier define the fixed net % targets.

        self.long_tp_net_pct = self.profit_multiplier
        self.short_tp_net_pct = self.profit_multiplier # Assuming symmetric profit for short
        self.long_sl_net_pct = self.stop_loss_multiplier
        self.short_sl_net_pct = self.stop_loss_multiplier # Assuming symmetric stop-loss for short

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

            # Define the lookahead window for this bar (for barrier check)
            window_end_iloc = min(i + self.future_return_window + 1, n)
            window_data = df_copy.iloc[i + 1 : window_end_iloc].copy() # Slice from next bar

            if window_data.empty:
                continue # No future data in window, label remains 0

            # Calculate TP/SL prices for the current bar
            long_tp_price = current_close * (1 + self.long_tp_gross_pct / 100)
            long_sl_price = current_close * (1 - self.long_sl_gross_pct / 100) # SL is a decrease for long

            short_tp_price = current_close * (1 - self.short_tp_gross_pct / 100) # TP is a decrease for short
            short_sl_price = current_close * (1 + self.short_sl_gross_pct / 100) # SL is an increase for short

            # Check for barrier hits within the window_data
            long_tp_hit = (window_data['high'] >= long_tp_price).any()
            long_sl_hit = (window_data['low'] <= long_sl_price).any()

            short_tp_hit = (window_data['low'] <= short_tp_price).any()
            short_sl_hit = (window_data['high'] >= short_sl_price).any()

            # Determine which barrier was hit first (if any)
            # Find first index where TP/SL was hit
            first_long_tp_idx = (window_data['high'] >= long_tp_price).idxmax() if long_tp_hit else None
            first_long_sl_idx = (window_data['low'] <= long_sl_price).idxmax() if long_sl_hit else None

            first_short_tp_idx = (window_data['low'] <= short_tp_price).idxmax() if short_tp_hit else None
            first_short_sl_idx = (window_data['high'] >= short_sl_price).idxmax() if short_sl_hit else None

            # Logic for Long Label (1)
            if long_tp_hit and (not long_sl_hit or first_long_tp_idx < first_long_sl_idx):
                labels[i] = 1

            # Logic for Short Label (-1)
            elif short_tp_hit and (not short_sl_hit or first_short_tp_idx < first_short_sl_idx):
                labels[i] = -1
            
            # If neither TP nor SL is hit (time barrier is hit implicitly), or SL is hit first, label remains 0.

        return pd.DataFrame({'label': labels}, index=df_copy.index)
