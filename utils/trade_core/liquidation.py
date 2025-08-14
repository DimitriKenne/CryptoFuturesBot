# utils/trade_core/liquidation.py

import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

class LiquidationEstimator:
    """
    Estimates liquidation prices and provides utilities to assess the safety
    of stop-loss levels relative to the estimated liquidation price.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("LiquidationEstimator initialized.")

    def estimate_liquidation_price(self,
                                   entry_price: float,
                                   leverage: int,
                                   direction: int,
                                   maintenance_margin_rate: float
                                   ) -> float:
        """
        Estimates the liquidation price for a futures position.
        This is a simplified estimation and real exchange formulas might differ.

        Args:
            entry_price (float): The price at which the position was opened.
            leverage (int): The leverage applied to the position.
            direction (int): 1 for long, -1 for short.
            maintenance_margin_rate (float): The exchange's maintenance margin rate (e.g., 0.005).

        Returns:
            float: The estimated liquidation price. Returns NaN if inputs are invalid.
        """
        if pd.isna(entry_price) or entry_price <= 0 or leverage <= 0 or maintenance_margin_rate < 0:
            self.logger.error("Invalid input for liquidation price estimation.")
            return np.nan

        try:
            if direction == 1: # Long position
                liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
            elif direction == -1: # Short position
                liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate)
            else:
                self.logger.error(f"Invalid direction '{direction}' for liquidation price estimation.")
                return np.nan
            
            if liquidation_price <= 0:
                self.logger.warning(f"Calculated liquidation price is non-positive ({liquidation_price}). Returning NaN.")
                return np.nan

            return liquidation_price
        except Exception as e:
            self.logger.error(f"Error estimating liquidation price: {e}", exc_info=True)
            return np.nan

    def is_sl_safe_from_liquidation(self,
                                     sl_price: float,
                                     liquidation_price: float,
                                     direction: int,
                                     min_distance_pct: float
                                    ) -> bool:
        """
        Checks if the Stop Loss price is safely above/below the liquidation price
        by a minimum percentage distance.

        Args:
            sl_price (float): The calculated Stop Loss price.
            liquidation_price (float): The estimated liquidation price.
            direction (int): 1 for long, -1 for short.
            min_distance_pct (float): Minimum percentage distance (e.g., 0.01 for 1%) to maintain
                                      between SL and liquidation price.

        Returns:
            bool: True if SL is safe, False otherwise.
        """
        if pd.isna(sl_price) or pd.isna(liquidation_price) or sl_price <= 0 or liquidation_price <= 0:
            self.logger.warning("Cannot check SL safety: SL or liquidation price is invalid.")
            return False
        
        # Calculate the absolute difference and the required minimum distance
        abs_diff = abs(sl_price - liquidation_price)
        required_abs_distance = liquidation_price * min_distance_pct

        if direction == 1: # Long position
            # SL must be above liquidation price, and by at least min_distance_pct
            # sl_price > liquidation_price + required_abs_distance
            return sl_price > liquidation_price and abs_diff >= required_abs_distance
        elif direction == -1: # Short position
            # SL must be below liquidation price, and by at least min_distance_pct
            # sl_price < liquidation_price - required_abs_distance
            return sl_price < liquidation_price and abs_diff >= required_abs_distance
        else:
            self.logger.error(f"Invalid direction '{direction}' for SL safety check.")
            return False

