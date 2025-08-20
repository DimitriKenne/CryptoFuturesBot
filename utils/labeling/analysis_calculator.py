# utils/labeling/analysis_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple

# Import global constants like FLOAT_EPSILON from your config
from config.params import FLOAT_EPSILON

logger = logging.getLogger(__name__)

class AnalysisCalculator:
    """
    Utility methods for financial calculations in label analysis,
    including net return, Maximum Favorable Return (MFR), and Maximum Adverse Loss (MAL),
    accounting for trading fees and slippage.
    """

    def __init__(self, trading_fee_rate: float, slippage_tolerance_rate: float):
        """
        Initializes the calculator with global transaction cost parameters.

        Args:
            trading_fee_rate (float): Transaction fee rate (0-1, e.g., 0.0005).
            slippage_tolerance_rate (float): Estimated slippage rate (0-1, e.g., 0.0001).
        """
        self.trading_fee_rate = trading_fee_rate
        self.slippage_tolerance_rate = slippage_tolerance_rate
        logger.debug(f"AnalysisCalculator initialized with fee={self.trading_fee_rate:.6f}, slippage={self.slippage_tolerance_rate:.6f}.")

    def calculate_net_return_scalar(self, entry_price: float, exit_price: float, trade_type: int) -> float:
        """
        Calculates net return for a single trade (percentage), accounting for fees/slippage.

        Args:
            entry_price (float): Price at entry.
            exit_price (float): Price at exit.
            trade_type (int): 1 for long (buy-sell), -1 for short (sell-buy).

        Returns:
            float: Net return percentage. NaN if entry_price is zero/invalid.
        """
        if abs(entry_price) < FLOAT_EPSILON:
            logger.warning("Zero/near-zero entry price. Returning NaN.")
            return np.nan

        # Effective cost/revenue factors considering fees and slippage
        effective_cost_factor_long_entry = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        effective_revenue_factor_long_exit = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)

        effective_revenue_factor_short_entry = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        effective_cost_factor_short_exit = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)

        if trade_type == 1: # Long trade
            cost_to_enter = entry_price * effective_cost_factor_long_entry
            revenue_from_exit = exit_price * effective_revenue_factor_long_exit
            net_return = ((revenue_from_exit - cost_to_enter) / cost_to_enter) * 100.0
        elif trade_type == -1: # Short trade
            revenue_from_enter = entry_price * effective_revenue_factor_short_entry
            cost_to_exit = exit_price * effective_cost_factor_short_exit
            net_return = ((revenue_from_enter - cost_to_exit) / revenue_from_enter) * 100.0
        else:
            logger.warning(f"Invalid trade_type '{trade_type}'. Expected 1 or -1. Returning NaN.")
            net_return = np.nan

        return net_return

    def calculate_max_return_loss_within_window(
        self,
        df_segment: pd.DataFrame,
        entry_price: float,
        trade_type: int
    ) -> Tuple[float, float]:
        """
        Calculates Maximum Favorable Return (MFR) and Maximum Adverse Loss (MAL)
        for a price segment, from an entry price, accounting for fees/slippage.

        MFR: highest net profit during segment.
        MAL: largest net loss (positive magnitude) during segment.

        Args:
            df_segment (pd.DataFrame): OHLC data slice for trade duration (must have 'high', 'low').
            entry_price (float): Trade entry price.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            Tuple[float, float]: (Max Favorable Return percentage, Max Adverse Loss percentage).
                                 (np.nan, np.nan) if segment is empty or invalid.
        """
        if df_segment.empty:
            logger.debug("Empty DataFrame segment for MFR/MAL. Returning NaN, NaN.")
            return np.nan, np.nan

        df_segment_clean = df_segment.copy()
        df_segment_clean['high'] = pd.to_numeric(df_segment_clean['high'], errors='coerce')
        df_segment_clean['low'] = pd.to_numeric(df_segment_clean['low'], errors='coerce')
        df_segment_clean.dropna(subset=['high', 'low'], inplace=True)

        if df_segment_clean.empty:
            logger.warning("Segment empty after dropping NaNs in 'high'/'low'. Cannot calculate MFR/MAL.")
            return np.nan, np.nan

        current_max_favorable = -np.inf
        current_max_adverse = -np.inf

        if trade_type == 1: # Long trade
            for _, row in df_segment_clean.iterrows():
                favorable_return = self.calculate_net_return_scalar(entry_price, row['high'], trade_type=1)
                if pd.notna(favorable_return):
                    current_max_favorable = max(current_max_favorable, favorable_return)

                adverse_return = self.calculate_net_return_scalar(entry_price, row['low'], trade_type=1)
                if pd.notna(adverse_return):
                    current_max_adverse = max(current_max_adverse, abs(adverse_return))

        elif trade_type == -1: # Short trade
            for _, row in df_segment_clean.iterrows():
                favorable_return = self.calculate_net_return_scalar(entry_price, row['low'], trade_type=-1)
                if pd.notna(favorable_return):
                    current_max_favorable = max(current_max_favorable, favorable_return)

                adverse_return = self.calculate_net_return_scalar(entry_price, row['high'], trade_type=-1)
                if pd.notna(adverse_return):
                    current_max_adverse = max(current_max_adverse, abs(adverse_return))

        max_favorable_return_pct = current_max_favorable if current_max_favorable != -np.inf else np.nan
        max_adverse_loss_pct = current_max_adverse if current_max_adverse != -np.inf else np.nan

        return max_favorable_return_pct, max_adverse_loss_pct

