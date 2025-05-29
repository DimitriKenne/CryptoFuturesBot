# utils/labeling_strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Set up a module-level logger for all labeling strategies
# This logger will inherit the configuration from utils/logger_config.py
logger = logging.getLogger(__name__)

# Define FLOAT_EPSILON here as it's a common constant for numerical stability
FLOAT_EPSILON = 1e-9

class BaseLabelingStrategy(ABC):
    """
    Abstract Base Class (ABC) for all trading labeling strategies.
    Defines the common interface that all concrete labeling strategies must implement.
    This ensures consistency and allows the LabelGenerator to work with different
    strategies interchangeably.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the base labeling strategy with common parameters.

        Args:
            config (Dict[str, Any]): The configuration dictionary for the strategy.
                                     This will typically be a subset of LABELING_CONFIG
                                     from config.params.
            logger (logging.Logger): A logger instance for logging messages specific
                                     to this strategy.
        """
        # Removed super().__init__(config, logger) as object.__init__ takes no args
        self.config = config
        self.logger = logger
        self.logger.debug(f"BaseLabelingStrategy initialized with config: {self.config}")

        self._validate_common_config()

    def _validate_common_config(self):
        """
        Validates common configuration parameters that all strategies might expect.
        Concrete strategies should override or extend this for their specific parameters.
        """
        # No common parameters directly validated here as strategies only produce raw labels.
        # Specific strategies will validate their own parameters.
        pass

    @abstractmethod
    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to calculate raw trading labels (1, -1, or 0) for the input DataFrame.
        This method should NOT apply any label propagation or smoothing.
        It should focus solely on the core logic of the specific labeling strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and required features,
                               indexed by time. Must include 'open', 'high', 'low', 'close'.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: A DataFrame with at least the original index and a 'label' column
                          containing the raw labels (1, -1, 0). Other columns may be preserved
                          or dropped as needed by the strategy.
                          IMPORTANT: The returned DataFrame MUST have a DatetimeIndex
                                     that matches the input df's index.
        Raises:
            ValueError: If input DataFrame is missing required columns or features for this strategy.
        """
        pass

    def _validate_input_df(self, df: pd.DataFrame, required_cols: Optional[List[str]] = None):
        """
        Helper method for strategies to validate their input DataFrame.
        Checks for DatetimeIndex and presence of required columns.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Input DataFrame is missing required columns for this strategy: {missing_cols}")

        # Check for NaNs in critical OHLCV columns (already done by LabelGenerator, but good for robustness)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in df.columns and df[col].isnull().any():
                self.logger.warning(f"Input DataFrame contains NaN values in critical OHLCV column '{col}'. This should ideally be handled before strategy calculation.")
                # Depending on how strict we want to be, we could raise an error here.
                # For now, we assume LabelGenerator has handled it.
                pass
