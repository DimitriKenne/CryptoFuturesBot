# utils/trade_core/position_sizing.py

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

FLOAT_EPSILON = 1e-9

class PositionSizer:
    """
    Calculates the appropriate position size based on risk management parameters.
    This class is designed to be reusable for both live trading and backtesting.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("PositionSizer initialized.")

    def calculate_quantity(self,
                           current_capital: float,
                           current_price: float,
                           stop_loss_price: float,
                           risk_per_trade_fraction: float,
                           leverage: int
                          ) -> Optional[float]:
        """
        Calculates the position size in base currency (e.g., BTC for BTCUSDT)
        based on risk management parameters.

        Args:
            current_capital (float): The current available capital in the account.
            current_price (float): The current market price.
            stop_loss_price (float): The calculated stop loss price for the potential trade.
            risk_per_trade_fraction (float): The fraction of capital to risk per trade (e.g., 0.01 for 1%).
            leverage (int): The leverage to apply to the trade.

        Returns:
            Optional[float]: The calculated quantity in base asset, or None if calculation fails.
        """
        if pd.isna(current_capital) or current_capital <= 0:
            self.logger.error("Invalid current_capital for position sizing.")
            return None
        if pd.isna(current_price) or current_price <= 0 or pd.isna(stop_loss_price) or stop_loss_price <= 0:
            self.logger.error("Invalid current_price or stop_loss_price for position sizing.")
            return None
        if not (0 < risk_per_trade_fraction <= 1):
            self.logger.error(f"Invalid risk_per_trade_fraction: {risk_per_trade_fraction}. Must be > 0 and <= 1.")
            return None
        if leverage <= 0:
            self.logger.error(f"Invalid leverage: {leverage}. Must be positive.")
            return None

        # Calculate the price difference between entry and stop loss
        price_diff = abs(current_price - stop_loss_price)
        if price_diff < FLOAT_EPSILON:
            self.logger.warning("Price difference (entry - SL) is zero or too small. Cannot calculate position size, risk is undefined.")
            return None

        # Calculate the maximum loss allowed per trade in USD
        max_loss_usd = current_capital * risk_per_trade_fraction

        # Calculate the quantity based on max_loss_usd and price_diff
        # (max_loss_usd / price_diff) gives the number of base asset units
        # that would result in max_loss_usd if price moves by price_diff.
        calculated_quantity = (max_loss_usd / price_diff) * leverage
        
        if calculated_quantity <= 0 or pd.isna(calculated_quantity):
            self.logger.warning(f"Calculated quantity is non-positive or NaN ({calculated_quantity:.8f}).")
            return None

        self.logger.debug(f"Calculated raw position size: {calculated_quantity:.8f}")
        return calculated_quantity

