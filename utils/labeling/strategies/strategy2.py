# utils/labeling/strategies/strategy2.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Tuple
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON
from config.label import LabelingStrategy2Config

class Strategy2(BaseLabelingStrategy):
    """
    Labeling Strategy 2: Net Forward Return Quantile.
    """

    def __init__(self, config: LabelingStrategy2Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_rate: float):
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        self.logger.info("Labeling Strategy 2 (Net Forward Return Quantile) initializing...")
        self._validate_strategy_config()

        self.future_return_window = self.config.future_return_window
        self.quantile_threshold_long_pct = self.config.quantile_threshold_long_pct
        self.quantile_threshold_short_pct = self.config.quantile_threshold_short_pct
        self._intermediate_analysis_df: pd.DataFrame = pd.DataFrame()

    def _validate_strategy_config(self):
        """Validates configuration parameters specific to Labeling Strategy 2."""
        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")
        if not (0 <= self.config.quantile_threshold_long_pct < 100):
            raise ValueError("'quantile_threshold_long_pct' must be between 0 and 100.")
        if not (0 <= self.config.quantile_threshold_short_pct < 100):
            raise ValueError("'quantile_threshold_short_pct' must be between 0 and 100.")
        self.logger.debug("Labeling Strategy 2 config validated.")

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates raw labels for Strategy 2."""
        self.logger.debug("Calculating raw labels for Strategy 2.")
        self._validate_input_df(df, ['close'])
        df_copy = df.copy()

        df_copy['Future_Close'] = df_copy['close'].shift(-self.future_return_window)
        
        entry_cost_long = df_copy['close'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        exit_revenue_long = df_copy['Future_Close'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        df_copy['Net_Return_Long'] = (exit_revenue_long - entry_cost_long) / entry_cost_long

        entry_revenue_short = df_copy['close'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        exit_cost_short = df_copy['Future_Close'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        df_copy['Net_Return_Short'] = (entry_revenue_short - exit_cost_short) / entry_revenue_short

        self._intermediate_analysis_df = df_copy.copy()

        df_copy.dropna(subset=['Future_Close', 'Net_Return_Long', 'Net_Return_Short'], inplace=True)
        
        positive_returns = df_copy['Net_Return_Long'][df_copy['Net_Return_Long'] > 0].dropna()
        profitable_short_returns = df_copy['Net_Return_Short'][df_copy['Net_Return_Short'] > 0].dropna()
        
        buy_threshold = np.percentile(positive_returns, self.quantile_threshold_long_pct) if not positive_returns.empty else 0.0
        sell_threshold = np.percentile(profitable_short_returns, self.quantile_threshold_short_pct) if not profitable_short_returns.empty else 0.0

        df_copy['label'] = 0
        df_copy.loc[df_copy['Net_Return_Long'] >= buy_threshold, 'label'] = 1
        df_copy.loc[df_copy['Net_Return_Short'] >= sell_threshold, 'label'] = -1
        
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)

    def perform_strategy_specific_analysis(
        self,
        df_original_input: pd.DataFrame,
        plotter: Any,
        calculator: Any
    ) -> List[Tuple[str, Any]]:
        """Performs analysis for Strategy 2 and returns the plot figure."""
        self.logger.info("Performing strategy-specific analysis for Strategy 2...")
        if self._intermediate_analysis_df is None or self._intermediate_analysis_df.empty:
            self.logger.warning("No intermediate analysis data found for Strategy 2. Skipping analysis.")
            return []

        df_net_returns = self._intermediate_analysis_df[['Net_Return_Long', 'Net_Return_Short']].dropna().copy()
        df_net_returns['Net_Return_Long'] *= 100.0
        df_net_returns['Net_Return_Short'] *= 100.0

        if df_net_returns.empty:
            self.logger.warning("No valid net returns for plotting in Strategy 2.")
            return []

        fig = plotter.plot_net_return_distributions(df_net_returns)
        
        return [("net_return_distributions", fig)]