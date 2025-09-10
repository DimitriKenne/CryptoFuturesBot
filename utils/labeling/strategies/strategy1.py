import pandas as pd
import numpy as np
import logging
from typing import Any, List, Tuple, TYPE_CHECKING
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON
from config.label import LabelingStrategy1Config

if TYPE_CHECKING:
    from utils.labeling.analysis_plotter import AnalysisPlotter
    from utils.labeling.analysis_calculator import AnalysisCalculator

class Strategy1(BaseLabelingStrategy):
    """
    Vectorized triple-barrier labeling for ternary classification.
    For each bar:
      - If long TP is hit before long SL, label is 1.
      - If short TP is hit before short SL, label is -1.
      - Otherwise, label is 0.
    """

    def __init__(self, config: LabelingStrategy1Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_rate: float):
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        self.logger.info("Labeling Strategy 1 (Vectorized triple-barrier) initializing.")
        self.take_profit_pct = self.config.take_profit_pct
        self.stop_loss_pct = self.config.stop_loss_pct
        self.lookahead_bars = self.config.lookahead_bars

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating raw labels for Strategy 1 (vectorized triple-barrier).")
        self._validate_input_df(df, ['high', 'low', 'close'])

        n = len(df)
        tp_pct = self.take_profit_pct / 100.0
        sl_pct = self.stop_loss_pct / 100.0
        lookahead = self.lookahead_bars

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Precompute barriers for all bars
        tp_long = closes * (1 + tp_pct)
        sl_long = closes * (1 - sl_pct)
        tp_short = closes * (1 - tp_pct)
        sl_short = closes * (1 + sl_pct)

        labels = np.zeros(n, dtype=int)

        # For each bar, vectorized lookahead window comparisons
        for i in range(n):
            end = min(i + 1 + lookahead, n)
            hi_window = highs[i+1:end]
            lo_window = lows[i+1:end]

            # --- Long trade window ---
            long_tp_hits = np.where(hi_window >= tp_long[i])[0]
            long_sl_hits = np.where(lo_window <= sl_long[i])[0]
            long_label = 0
            if long_tp_hits.size and (not long_sl_hits.size or long_tp_hits[0] < long_sl_hits[0]):
                long_label = 1
            elif long_sl_hits.size and (not long_tp_hits.size or long_sl_hits[0] < long_tp_hits[0]):
                long_label = 0

            # --- Short trade window ---
            short_tp_hits = np.where(lo_window <= tp_short[i])[0]
            short_sl_hits = np.where(hi_window >= sl_short[i])[0]
            short_label = 0
            if short_tp_hits.size and (not short_sl_hits.size or short_tp_hits[0] < short_sl_hits[0]):
                short_label = -1
            elif short_sl_hits.size and (not short_tp_hits.size or short_sl_hits[0] < short_tp_hits[0]):
                short_label = 0

            # Assign label: long priority, else short, else neutral
            if long_label == 1:
                labels[i] = 1
            elif short_label == -1:
                labels[i] = -1
            else:
                labels[i] = 0

        return pd.DataFrame({'label': labels}, index=df.index)

    def perform_strategy_specific_analysis(
        self, 
        df_original_input: pd.DataFrame, 
        plotter: 'AnalysisPlotter', 
        calculator: 'AnalysisCalculator'
    ) -> List[Tuple[str, Any]]:
        self.logger.info("No specific analysis defined for Strategy 1.")
        return []