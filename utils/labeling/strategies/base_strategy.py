# utils/labeling/strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Any, List, Tuple, Optional
from pathlib import Path

from config.params import FLOAT_EPSILON

logger = logging.getLogger(__name__)

class BaseLabelingStrategy(ABC):
    """
    Abstract Base Class (ABC) for all labeling strategies.
    Defines the common interface that all concrete labeling strategies must implement.
    """

    def __init__(self, config: Any, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_rate: float):
        """
        Initializes the base labeling strategy with common parameters.

        Args:
            config (Any): The specific configuration object for the labeling strategy.
            logger (logging.Logger): A logger instance for logging messages.
            trading_fee_rate (float): The transaction fee rate (0-1).
            slippage_tolerance_rate (float): The estimated slippage rate (0-1).
        """
        self.config = config
        self.logger = logger
        self.trading_fee_rate = trading_fee_rate
        self.slippage_tolerance_rate = slippage_tolerance_rate
        self.logger.debug(f"BaseLabelingStrategy initialized for {self.__class__.__name__}.")

    @abstractmethod
    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw trading labels (1 for Buy, -1 for Sell, 0 for Neutral).

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data and required features.

        Returns:
            pd.DataFrame: A DataFrame with the raw 'label' column.
        """
        pass

    @abstractmethod
    def perform_strategy_specific_analysis(
        self,
        df_original_input: pd.DataFrame,
        plotter: Any,
        calculator: Any
    ) -> List[Tuple[str, Any]]:
        """
        Performs analysis specific to this strategy and returns artifacts to be saved.

        Args:
            df_original_input (pd.DataFrame): The input DataFrame for analysis context.
            plotter (Any): An instance of AnalysisPlotter for creating plots.
            calculator (Any): An instance of AnalysisCalculator for calculations.

        Returns:
            List[Tuple[str, Any]]: A list of tuples, where each tuple contains:
                                   (analysis_type_suffix, artifact_to_save).
                                   The artifact can be a matplotlib Figure or a pandas DataFrame.
                                   Example: [('my_plot', fig_object), ('my_table', df_object)]
        """
        pass

    def _validate_input_df(self, df: pd.DataFrame, required_cols: Optional[List[str]] = None):
        """
        Helper method for labeling strategies to validate their input DataFrame.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Input DataFrame is missing required columns for this strategy: {missing_cols}")