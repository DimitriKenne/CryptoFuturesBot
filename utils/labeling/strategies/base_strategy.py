# utils/labeling/strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Import global constants and configuration ---
from config.params import app_config, FLOAT_EPSILON
from config.label import LabelConfig

# Set up a module-level logger for all labeling strategies
logger = logging.getLogger(__name__)

class BaseLabelingStrategy(ABC):
    """
    Abstract Base Class (ABC) for all labeling strategies.
    Defines the common interface that all concrete labeling strategies must implement.
    This ensures consistency and allows the LabelGenerator to work with different
    labeling strategies interchangeably.
    """

    def __init__(self, config: Any, logger: logging.Logger, trading_fee_rate: Optional[float] = None, slippage_tolerance_pct: Optional[float] = None):
        """
        Initializes the base labeling strategy with common parameters.

        Args:
            config (Any): The specific configuration object (dataclass instance) for the labeling strategy.
                          This will be like LabelingStrategy1Config, LabelingStrategy2Config etc.
            logger (logging.Logger): A logger instance for logging messages specific
                                     to this labeling strategy.
            trading_fee_rate (float): The transaction fee rate to apply to calculations.
                                      If None, uses app_config.labeling.trading_fee_rate.
            slippage_tolerance_pct (float): The estimated slippage rate to apply to calculations.
                                            If None, uses app_config.labeling.slippage_tolerance_pct.
        """
        self.config = config
        self.logger = logger
        # Use config values if provided, else fallback to global config
        self.trading_fee_rate = trading_fee_rate if trading_fee_rate is not None else app_config.labeling.trading_fee_rate
        self.slippage_tolerance_pct = slippage_tolerance_pct if slippage_tolerance_pct is not None else app_config.labeling.slippage_tolerance_pct
        
        self.logger.debug(f"BaseLabelingStrategy initialized for {self.__class__.__name__}.")
        self.logger.debug(f"  Trading Fee Rate: {self.trading_fee_rate}")
        self.logger.debug(f"  Slippage Tolerance: {self.slippage_tolerance_pct}")

    @abstractmethod
    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw trading labels (1 for Buy, -1 for Sell, 0 for Neutral)
        based on the specific logic of the labeling strategy.

        This method should NOT apply label propagation; that is handled by LabelGenerator.
        The DataFrame returned should contain at least a 'label' column and
        maintain the original DatetimeIndex.

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data and any necessary
                               features that the labeling strategy relies upon.
                               Must have a DatetimeIndex.

        Returns:
            pd.DataFrame: A DataFrame with the raw 'label' column and the same index
                          as the input df.
        Raises:
            ValueError: If input DataFrame is missing required columns or features for this labeling strategy.
        """
        pass

    def _validate_input_df(self, df: pd.DataFrame, required_cols: Optional[List[str]] = None):
        """
        Helper method for labeling strategies to validate their input DataFrame.
        Checks for DatetimeIndex and presence of required columns.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.error("Input DataFrame must have a DatetimeIndex.")
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Input DataFrame is missing required columns for this labeling strategy: {missing_cols}")
                raise ValueError(f"Input DataFrame is missing required columns for this labeling strategy: {missing_cols}")

        # Check for NaNs in critical OHLCV columns (already done by LabelGenerator, but good for robustness)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in df.columns and df[col].isnull().any():
                self.logger.warning(f"Input DataFrame contains NaN values in critical OHLCV column '{col}'. This should ideally be handled before labeling strategy calculation.")
                # Depending on how strict we want to be, we could raise an error here.
                # For now, we assume LabelGenerator has handled it.
