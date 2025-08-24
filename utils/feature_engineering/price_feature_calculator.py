# utils/feature_engineering/price_feature_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Optional

# --- Import configuration ---
try:
    from config.feature import FeatureConfig
    from config.params import FLOAT_EPSILON
except ImportError as e:
    logging.critical(f"Failed to import necessary configuration modules: {e}")
    raise

logger = logging.getLogger(__name__)

class PriceFeatureCalculator:
    """
    Calculates basic price transformations and ratios from OHLCV data.
    These are generally direct manipulations of price components (open, high, low, close).
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("PriceFeatureCalculator initialized.")

    def add_price_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds basic price transformations (e.g., log returns, typical price) to the DataFrame.
        These are calculated based on past data to ensure temporal safety.

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data.

        Returns:
            pd.DataFrame: DataFrame with added price transformation features.
        """
        df_transformed = pd.DataFrame(index=df.index)
        
        # Shift close prices for log returns to prevent lookahead
        # For current bar `t`, we use `close[t-1] / close[t-2]`
        df_transformed['log_returns'] = np.log(df['close'].shift(1) / df['close'].shift(2))
        
        # Typical price of the *previous* bar
        df_transformed['typical_price'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3

        # Add other essential price differences based on *shifted* data
        df_transformed['mid_price'] = (df['high'].shift(1) + df['low'].shift(1)) / 2
        df_transformed['body_range'] = df['high'].shift(1) - df['low'].shift(1)
        df_transformed['open_close_diff'] = df['close'].shift(1) - df['open'].shift(1)
        df_transformed['high_low_diff'] = df['high'].shift(1) - df['low'].shift(1)

        self.logger.debug("Price transformations added.")
        return df_transformed
