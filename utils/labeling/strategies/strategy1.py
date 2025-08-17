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

    def __init__(self, config: LabelingStrategy1Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_pct: float):
        """
        Initializes Labeling Strategy 1 (Triple Barrier).

        Args:
            config (LabelingStrategy1Config): The configuration dataclass for this labeling strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate.
            slippage_tolerance_pct (float): The estimated slippage rate.
        """
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_pct)
        self.logger.info("Labeling Strategy 1 (Triple Barrier) initializing.")

        # Access parameters directly from the LabelingStrategy1Config dataclass
        self.future_return_window = self.config.future_return_window
        self.profit_multiplier = self.config.profit_multiplier
        self.stop_loss_multiplier = self.config.stop_loss_multiplier
        self.vol_adj_lookback = self.config.vol_adj_lookback
        self.num_price_bars = self.config.num_price_bars

        self._validate_strategy_config()

        self.logger.info(f"  Future Return Window: {self.future_return_window} bars")
        self.logger.info(f"  Profit Multiplier: {self.profit_multiplier}")
        self.logger.info(f"  Stop Loss Multiplier: {self.stop_loss_multiplier}")
        self.logger.info(f"  Volatility Adjustment Lookback (ATR): {self.vol_adj_lookback}")
        self.logger.info(f"  Trading Fee Rate: {self.trading_fee_rate:.4f}")
        self.logger.info(f"  Slippage Tolerance: {self.slippage_tolerance_pct:.6f}")

    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Labeling Strategy 1.
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
        self.logger.debug("Labeling Strategy 1 config validated.")

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
            return np.nan

        entry_cost_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        exit_revenue_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)

        if trade_type == 1:  # Long
            cost_to_enter = entry_price * entry_cost_factor
            revenue_from_exit = exit_price * exit_revenue_factor
            net_return = ((revenue_from_exit - cost_to_enter) / cost_to_enter) * 100.0
        elif trade_type == -1:  # Short
            revenue_from_enter = entry_price * exit_revenue_factor
            cost_to_exit = exit_price * entry_cost_factor
            net_return = ((revenue_from_enter - cost_to_exit) / revenue_from_enter) * 100.0
        else:
            net_return = np.nan

        return net_return

    def _net_to_gross_price_move_pct(self, net_pct: float, trade_type: int) -> float:
        """
        Converts a desired net return/loss percentage (including fees/slippage)
        to the required gross price movement percentage.

        Args:
            net_pct (float): The desired net return/loss percentage.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            float: The gross price movement percentage required.
        """
        if abs(1 - self.trading_fee_rate - self.slippage_tolerance_pct) < FLOAT_EPSILON or \
           abs(1 + self.trading_fee_rate + self.slippage_tolerance_pct) < FLOAT_EPSILON:
            self.logger.error("Fee/slippage factors lead to division by zero. Check trading_fee_rate/slippage_tolerance_pct values.")
            return np.nan

        if trade_type == 1:  # Long
            gross_price_move_pct = (((1 + net_pct / 100) * (1 + self.trading_fee_rate + self.slippage_tolerance_pct)) /
                                    (1 - self.trading_fee_rate - self.slippage_tolerance_pct) - 1) * 100
            return gross_price_move_pct
        elif trade_type == -1:  # Short
            gross_price_move_pct = (1 - ((1 - net_pct / 100) * (1 - self.trading_fee_rate - self.slippage_tolerance_pct)) /
                                    (1 + self.trading_fee_rate + self.slippage_tolerance_pct)) * 100
            return gross_price_move_pct
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
        self._validate_input_df(df, ['open', 'high', 'low', 'close'])

        df_copy = df.copy()
        if df_copy.empty:
            self.logger.warning("Input DataFrame is empty. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0})

        n = len(df_copy)

        # Set TP/SL net percentages from config
        self.long_tp_net_pct = self.profit_multiplier
        self.short_tp_net_pct = self.profit_multiplier
        self.long_sl_net_pct = self.stop_loss_multiplier
        self.short_sl_net_pct = self.stop_loss_multiplier

        # Calculate gross price move percentages
        self.long_tp_gross_pct = self._net_to_gross_price_move_pct(self.long_tp_net_pct, trade_type=1)
        self.long_sl_gross_pct = abs(self._net_to_gross_price_move_pct(-self.long_sl_net_pct, trade_type=1))
        self.short_tp_gross_pct = self._net_to_gross_price_move_pct(self.short_tp_net_pct, trade_type=-1)
        self.short_sl_gross_pct = abs(self._net_to_gross_price_move_pct(-self.short_sl_net_pct, trade_type=-1))

        self.logger.info(f"Calculated Long TP (Net): {self.long_tp_net_pct:.4f}%, Long SL (Net): {self.long_sl_net_pct:.4f}%")
        self.logger.info(f"Calculated Short TP (Net): {self.short_tp_net_pct:.4f}%, Short SL (Net): {self.short_sl_net_pct:.4f}%")
        self.logger.info(f"Calculated Long TP (Gross): {self.long_tp_gross_pct:.4f}%, Long SL (Gross): {self.long_sl_gross_pct:.4f}%")
        self.logger.info(f"Calculated Short TP (Gross): {self.short_tp_gross_pct:.4f}%, Short SL (Gross): {self.short_sl_gross_pct:.4f}%")

        labels = np.zeros(n, dtype=int)

        for i in range(n):
            current_close = df_copy['close'].iloc[i]
            if pd.isna(current_close) or abs(current_close) < FLOAT_EPSILON:
                continue

            window_end_iloc = min(i + self.future_return_window + 1, n)
            window_data = df_copy.iloc[i + 1 : window_end_iloc].copy()

            if window_data.empty:
                continue

            long_tp_price = current_close * (1 + self.long_tp_gross_pct / 100)
            long_sl_price = current_close * (1 - self.long_sl_gross_pct / 100)
            short_tp_price = current_close * (1 - self.short_tp_gross_pct / 100)
            short_sl_price = current_close * (1 + self.short_sl_gross_pct / 100)

            long_tp_hit = (window_data['high'] >= long_tp_price).any()
            long_sl_hit = (window_data['low'] <= long_sl_price).any()
            short_tp_hit = (window_data['low'] <= short_tp_price).any()
            short_sl_hit = (window_data['high'] >= short_sl_price).any()

            first_long_tp_idx = (window_data['high'] >= long_tp_price).idxmax() if long_tp_hit else None
            first_long_sl_idx = (window_data['low'] <= long_sl_price).idxmax() if long_sl_hit else None
            first_short_tp_idx = (window_data['low'] <= short_tp_price).idxmax() if short_tp_hit else None
            first_short_sl_idx = (window_data['high'] >= short_sl_price).idxmax() if short_sl_hit else None

            if long_tp_hit and (not long_sl_hit or first_long_tp_idx < first_long_sl_idx):
                labels[i] = 1
            elif short_tp_hit and (not short_sl_hit or first_short_tp_idx < first_short_sl_idx):
                labels[i] = -1

        return pd.DataFrame({'label': labels}, index=df_copy.index)
