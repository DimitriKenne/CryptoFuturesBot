# utils/trade_core/financial_math.py

import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

FLOAT_EPSILON = 1e-9

class FinancialMath:
    """
    Provides core financial calculation utilities for trading, including PnL, fees,
    and margin calculations. Designed to be agnostic to live or backtest execution.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("FinancialMath initialized.")

    def calculate_pnl_and_fees(self,
                                direction: int,
                                entry_price: float,
                                exit_price: float,
                                quantity: float,
                                trading_fee_rate: float,
                                liquidation_fee_rate: float = 0.0, # Used only if exit_reason is liquidation
                                exit_reason: str = None
                                ) -> Tuple[float, float, float, float]:
        """
        Calculates gross PnL, entry fee, exit fee, and total fees for a trade.

        Args:
            direction (int): 1 for long, -1 for short.
            entry_price (float): The price at which the position was entered.
            exit_price (float): The price at which the position was exited.
            quantity (float): The quantity of the base asset traded.
            trading_fee_rate (float): The exchange's trading fee rate (e.g., 0.0004 for 0.04%).
            liquidation_fee_rate (float): Specific fee rate for liquidation, if applicable.
            exit_reason (str): The reason for exit ('liquidation' or other).

        Returns:
            Tuple[float, float, float, float]:
                - gross_pnl (float): PnL before fees.
                - entry_fee (float): Fee incurred at entry.
                - exit_fee (float): Fee incurred at exit.
                - total_fees (float): Sum of entry and exit fees.
        """
        if pd.isna(entry_price) or pd.isna(exit_price) or pd.isna(quantity) or \
           entry_price <= 0 or exit_price <= 0 or quantity <= 0:
            self.logger.error("Invalid input for PnL calculation: prices or quantity are non-positive or NaN.")
            return 0.0, 0.0, 0.0, 0.0

        position_value_at_entry = quantity * entry_price
        position_value_at_exit = quantity * exit_price

        if direction == 1: # Long position
            gross_pnl = position_value_at_exit - position_value_at_entry
        elif direction == -1: # Short position
            gross_pnl = position_value_at_entry - position_value_at_exit
        else:
            self.logger.error(f"Invalid direction for PnL calculation: {direction}")
            return 0.0, 0.0, 0.0, 0.0

        entry_fee = position_value_at_entry * trading_fee_rate
        
        # Apply liquidation fee if the exit reason is liquidation
        if exit_reason == 'liquidation':
            exit_fee = position_value_at_exit * liquidation_fee_rate
            self.logger.debug(f"Applied liquidation fee: {exit_fee:.4f}")
        else:
            exit_fee = position_value_at_exit * trading_fee_rate
            self.logger.debug(f"Applied standard exit fee: {exit_fee:.4f}")

        total_fees = entry_fee + exit_fee

        return gross_pnl, entry_fee, exit_fee, total_fees

    def calculate_initial_margin(self, notional_value: float, leverage: int) -> float:
        """
        Calculates the initial margin required for a position.

        Args:
            notional_value (float): The total value of the position (quantity * entry_price).
            leverage (int): The leverage used for the trade.

        Returns:
            float: The initial margin required.
        """
        if pd.isna(notional_value) or notional_value <= 0 or leverage <= 0:
            self.logger.error("Invalid input for initial margin calculation: notional_value or leverage invalid.")
            return 0.0
        return notional_value / leverage

    def calculate_maintenance_margin_level(self, entry_price: float, maintenance_margin_rate: float, leverage: int, direction: int) -> float:
        """
        Calculates the price at which the maintenance margin level is reached,
        useful for understanding proximity to liquidation.

        Args:
            entry_price (float): The entry price of the position.
            maintenance_margin_rate (float): The exchange's maintenance margin rate (e.g., 0.005).
            leverage (int): The leverage used.
            direction (int): 1 for long, -1 for short.

        Returns:
            float: The price level at which maintenance margin is met.
        """
        if pd.isna(entry_price) or entry_price <= 0 or maintenance_margin_rate < 0 or leverage <= 0:
            self.logger.error("Invalid input for maintenance margin level calculation.")
            return np.nan

        # Formula for maintenance margin level price (approximate for futures)
        # This can vary slightly by exchange and instrument type.
        if direction == 1: # Long
            return entry_price * (1 - maintenance_margin_rate / leverage)
        elif direction == -1: # Short
            return entry_price * (1 + maintenance_margin_rate / leverage)
        else:
            self.logger.error(f"Invalid direction '{direction}' for maintenance margin level calculation.")
            return np.nan

