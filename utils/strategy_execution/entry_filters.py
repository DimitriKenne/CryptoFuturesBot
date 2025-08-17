# utils/strategy_execution/entry_filters.py

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EntryFilters:
    """
    Applies entry filters to determine if a trade should be opened.
    Each utility unpacks its config as needed.
    """

    def __init__(self, app_config):
        self.strategy_config = app_config.trading
        self.entry_filter_config = app_config.trading.entry_filter
        self.features_config = app_config.features

    def apply_filters(self,
                      current_signal: int,
                      current_bar: pd.Series,
                      current_long_proba: float,
                      current_short_proba: float,
                      price_precision: int = None
                      ) -> bool:
        # Filter: Only allow non-neutral signals
        if current_signal == 0:
            return False

        # Filter: Only allow long/short trades if enabled in config
        if current_signal == 1 and not getattr(self.entry_filter_config, "allow_long_trades", True):
            return False
        if current_signal == -1 and not getattr(self.entry_filter_config, "allow_short_trades", True):
            return False

        # Filter: Probability thresholds
        if current_signal == 1 and hasattr(self.entry_filter_config, 'min_long_proba'):
            if current_long_proba < self.entry_filter_config.min_long_proba:
                return False
        if current_signal == -1 and hasattr(self.entry_filter_config, 'min_short_proba'):
            if current_short_proba < self.entry_filter_config.min_short_proba:
                return False

        # Filter: Confidence thresholds (percentage)
        if current_signal == 1 and hasattr(self.entry_filter_config, 'confidence_threshold_long_pct'):
            if current_long_proba * 100 < self.entry_filter_config.confidence_threshold_long_pct:
                return False
        if current_signal == -1 and hasattr(self.entry_filter_config, 'confidence_threshold_short_pct'):
            if current_short_proba * 100 < self.entry_filter_config.confidence_threshold_short_pct:
                return False

        # Filter: Trend filter
        if getattr(self.entry_filter_config, 'trend_filter_enabled', False):
            ema_col = f"ema_{self.entry_filter_config.trend_filter_ema_period}"
            if ema_col in current_bar and not pd.isna(current_bar[ema_col]):
                if current_signal == 1 and current_bar['close'] < current_bar[ema_col]:
                    return False
                if current_signal == -1 and current_bar['close'] > current_bar[ema_col]:
                    return False

        # Filter: Volatility regime filter
        if getattr(self.entry_filter_config, 'volatility_regime_filter_enabled', False):
            regime_col = getattr(self.features_config, 'volatility_regime_col_name', 'volatility_regime')
            if regime_col in current_bar and not pd.isna(current_bar[regime_col]):
                allowed_regimes = getattr(self.entry_filter_config, 'allowed_volatility_regimes', None)
                if allowed_regimes is not None and int(current_bar[regime_col]) not in allowed_regimes:
                    return False

        return True
