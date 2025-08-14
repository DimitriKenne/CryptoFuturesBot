# utils/trade_core/order_precision.py

import logging
import numpy as np
import math
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class OrderPrecisionHandler:
    """
    Handles exchange-specific price and quantity precision rounding,
    and checks for minimum/maximum order quantities and notional values.
    """

    def __init__(self, price_precision: int, quantity_precision: int,
                 min_quantity: float, min_notional: float):
        """
        Initializes the OrderPrecisionHandler.

        Args:
            price_precision (int): Number of decimal places for price.
            quantity_precision (int): Number of decimal places for quantity.
            min_quantity (float): Minimum order quantity allowed by the exchange.
            min_notional (float): Minimum notional value allowed by the exchange (quantity * price).
        """
        if not isinstance(price_precision, int) or price_precision < 0:
            raise ValueError(f"price_precision must be a non-negative integer, got {price_precision}.")
        if not isinstance(quantity_precision, int) or quantity_precision < 0:
            raise ValueError(f"quantity_precision must be a non-negative integer, got {quantity_precision}.")
        if not isinstance(min_quantity, (int, float)) or min_quantity < 0:
            raise ValueError(f"min_quantity must be a non-negative number, got {min_quantity}.")
        if not isinstance(min_notional, (int, float)) or min_notional < 0:
            raise ValueError(f"min_notional must be a non-negative number, got {min_notional}.")

        self.price_precision = price_precision
        self.quantity_precision = quantity_precision
        self.min_quantity = min_quantity
        self.min_notional = min_notional
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("OrderPrecisionHandler initialized.")

    def round_price(self, price: Optional[float]) -> Optional[float]:
        """Rounds a price to the configured precision."""
        if pd.isna(price): return np.nan
        # Ensure price_precision is a non-negative integer
        if not isinstance(self.price_precision, int) or self.price_precision < 0:
             self.logger.warning(f"Invalid price_precision: {self.price_precision}. Cannot round price.")
             return price # Return original price if precision is invalid

        try:
            return round(price, self.price_precision)
        except (TypeError, ValueError) as e:
             self.logger.warning(f"Could not round price {price} to precision {self.price_precision}: {e}")
             return np.nan # Return NaN if rounding fails

    def _round_quantity(self, quantity: Optional[float]) -> Optional[float]:
        """Rounds a quantity DOWN to the configured precision (step size)."""
        if pd.isna(quantity): return np.nan
        # Ensure quantity_precision is a non-negative integer
        if not isinstance(self.quantity_precision, int) or self.quantity_precision < 0:
             self.logger.warning(f"Invalid quantity_precision: {self.quantity_precision}. Cannot round quantity.")
             return quantity # Return original quantity if precision is invalid

        try:
            factor = 10 ** self.quantity_precision
            # Floor division equivalent for floating point precision
            # Multiply by factor, floor, then divide by factor
            return math.floor(quantity * factor) / factor
        except (TypeError, ValueError) as e:
             self.logger.warning(f"Could not round quantity {quantity} to precision {self.quantity_precision}: {e}")
             return np.nan # Return NaN if rounding fails

    def validate_quantity_and_notional(self, quantity: Optional[float], price: Optional[float]) -> bool:
        """
        Validates if the given quantity meets minimum requirements and if the
        notional value (quantity * price) meets minimum notional requirements.

        Args:
            quantity (Optional[float]): The quantity of the asset.
            price (Optional[float]): The price of the asset.

        Returns:
            bool: True if both quantity and notional meet minimums, False otherwise.
        """
        if pd.isna(quantity) or quantity <= 0:
            self.logger.warning(f"Invalid quantity ({quantity}). Must be positive.")
            return False
        if pd.isna(price) or price <= 0:
            self.logger.warning(f"Invalid price ({price}). Must be positive.")
            return False

        if quantity < self.min_quantity:
            self.logger.warning(f"Quantity ({quantity:.8f}) is below minimum allowed ({self.min_quantity:.8f}).")
            return False

        notional_value = quantity * price
        if notional_value < self.min_notional:
            self.logger.warning(f"Notional value ({notional_value:.2f}) is below minimum allowed ({self.min_notional:.2f}).")
            return False

        self.logger.debug(f"Quantity {quantity:.8f} and Notional {notional_value:.2f} are valid.")
        return True
