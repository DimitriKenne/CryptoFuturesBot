# utils/labeling_strategies/ema_return_percentile.py

import pandas as pd
import numpy as np
import logging # Added logging import
from typing import Dict, Any, List, Optional
from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

class EmaReturnPercentileStrategy(BaseLabelingStrategy):
    """
    Labels data based on future returns relative to a backward EMA,
    with thresholds derived from percentiles of Open-Close changes.

    - 1: Future Return >= alpha AND Future Return <= beta_threshold (Buy)
    - -1: Future Return <= -alpha AND Future Return >= -beta_threshold (Sell)
    - 0: Otherwise (Neutral)

    Parameters:
    - 'f_window': Forward lookahead window for future close (int).
    - 'b_window': Backward EMA window for reference price (int).
    - 'fee': Transaction fee (float, e.g., 0.0005).
    - 'beta_increment': Increment factor for beta per forward window (float, e.g., 0.1 for 10%).
    - 'lower_percentile': Percentile for alpha (float, e.g., 85 for 85th percentile).
    - 'upper_percentile': Percentile for beta (float, e.g., 99.7 for 99.7th percentile).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the EmaReturnPercentileStrategy.
        """
        super().__init__(config, logger)
        self.logger.info("EmaReturnPercentileStrategy initializing...")
        self._validate_strategy_config()

        self.f_window = self.config['f_window']
        self.b_window = self.config['b_window']
        self.fee = self.config['fee']
        self.beta_increment = self.config['beta_increment']
        self.lower_percentile = self.config['lower_percentile']
        self.upper_percentile = self.config['upper_percentile']

        self.logger.info(f"  Forward Window (f_window): {self.f_window} bars")
        self.logger.info(f"  Backward EMA Window (b_window): {self.b_window} bars")
        self.logger.info(f"  Transaction Fee: {self.fee}")
        self.logger.info(f"  Beta Increment: {self.beta_increment}")
        self.logger.info(f"  Lower Percentile (alpha): {self.lower_percentile}")
        self.logger.info(f"  Upper Percentile (beta): {self.upper_percentile}")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to the EMA Return Percentile strategy.
        """
        required_keys = ['f_window', 'b_window', 'fee', 'beta_increment', 'lower_percentile', 'upper_percentile']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for EmaReturnPercentileStrategy: '{key}'")

        if not isinstance(self.config['f_window'], int) or self.config['f_window'] <= 0:
            raise ValueError("'f_window' must be a positive integer.")
        if not isinstance(self.config['b_window'], int) or self.config['b_window'] <= 0:
            raise ValueError("'b_window' must be a positive integer.")
        if not isinstance(self.config['fee'], (int, float)) or self.config['fee'] < 0:
            raise ValueError("'fee' must be a non-negative number.")
        if not isinstance(self.config['beta_increment'], (int, float)) or self.config['beta_increment'] < 0:
            raise ValueError("'beta_increment' must be a non-negative number.")
        if not isinstance(self.config['lower_percentile'], (int, float)) or not (0 < self.config['lower_percentile'] < 100):
            raise ValueError("'lower_percentile' must be a number between 0 and 100 (exclusive).")
        if not isinstance(self.config['upper_percentile'], (int, float)) or not (0 < self.config['upper_percentile'] < 100):
            raise ValueError("'upper_percentile' must be a number between 0 and 100 (exclusive).")
        if self.config['lower_percentile'] >= self.config['upper_percentile']:
            raise ValueError("'lower_percentile' must be less than 'upper_percentile'.")

        self.logger.debug("EmaReturnPercentileStrategy config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for the EMA Return Percentile strategy.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data, indexed by time.
                               Assumed to be cleaned (no NaNs in OHLCV) by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
        """
        self.logger.debug("Calculating raw labels for EMA Return Percentile strategy.")
        self._validate_input_df(df, ['open', 'close'])

        df_copy = df.copy() # Work on a copy

        # Compute Backward EMA as reference
        # The EWM calculation should be done on the 'close' price
        # Ensure enough data for EMA calculation
        if len(df_copy) < self.b_window:
            self.logger.warning(f"DataFrame too short ({len(df_copy)} bars) for Backward EMA calculation with window {self.b_window}. Returning all 0 labels.")
            # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
            # and only the 'label' column. This prevents index corruption.
            return pd.DataFrame(index=df_copy.index, data={'label': 0})

        df_copy['Backward_EMA'] = df_copy['close'].ewm(span=self.b_window, adjust=False).mean()

        # Compute future close and future return using Backward_EMA as the reference
        # Shift future_close by f_window
        df_copy['Future_Close'] = df_copy['close'].shift(-self.f_window)

        # Calculate Return: ((1 - fee) * Future_Close - (1 + fee) * Backward_EMA) / ((1 + fee) * Backward_EMA)
        # Ensure Backward_EMA is not zero for division
        safe_backward_ema = df_copy['Backward_EMA'].replace(0, np.nan)
        df_copy['Return'] = ((1 - self.fee) * df_copy['Future_Close'] - (1 + self.fee) * df_copy['Backward_EMA']) / ((1 + self.fee) * safe_backward_ema)

        # Drop NaN values (caused by shifting and initial EMA calculation)
        initial_rows = len(df_copy)
        df_copy.dropna(subset=['Future_Close', 'Return', 'Backward_EMA'], inplace=True)
        if len(df_copy) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_copy)} rows with NaNs after calculating future returns and EMA.")
        if df_copy.empty:
            self.logger.error("DataFrame is empty after dropping NaNs. Cannot generate labels.")
            # CRITICAL FIX: Return a DataFrame with the original index and all 0 labels
            return pd.DataFrame(index=df_copy.index, data={'label': 0})


        # Compute Open-Close percentage change for threshold analysis
        # Ensure open price is not zero for division
        safe_open = df_copy['open'].replace(0, np.nan)
        df_copy['Open_Close_Change'] = (df_copy['close'] - df_copy['open']) / safe_open


        # Compute α (lower percentile) and β (upper percentile)
        # Ensure there are enough non-NaN values for percentile calculation
        oc_change_dropna = df_copy['Open_Close_Change'].dropna()
        if oc_change_dropna.empty:
            self.logger.warning("Open-Close Change column is empty after dropping NaNs. Cannot calculate percentiles. All labels will be 0.")
            # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
            # and only the 'label' column. This prevents index corruption.
            return pd.DataFrame(index=df_copy.index, data={'label': 0})

        # Calculate alpha and beta using the non-NaN values
        alpha = np.percentile(oc_change_dropna, self.lower_percentile)
        beta = np.percentile(oc_change_dropna, self.upper_percentile)

        # Increment β based on forward_window (linear increase)
        beta_threshold = beta * (1 + self.f_window * self.beta_increment)

        # Ensure alpha is positive for comparison, and beta_threshold is greater than alpha
        alpha = abs(alpha) # Alpha should represent a magnitude
        if beta_threshold < alpha: # Ensure beta_threshold is at least alpha
            beta_threshold = alpha + FLOAT_EPSILON
            self.logger.warning(f"Adjusted beta_threshold to {beta_threshold} as it was less than alpha {alpha}.")

        self.logger.debug(f"Calculated alpha: {alpha}, beta: {beta}, beta_threshold: {beta_threshold}")

        # Assign Labels
        df_copy['label'] = 0 # Default to neutral

        # Buy (1): Return is within [alpha, beta_threshold]
        df_copy.loc[(df_copy['Return'] >= alpha) & (df_copy['Return'] <= beta_threshold), 'label'] = 1

        # Sell (-1): Return is within [-beta_threshold, -alpha]
        df_copy.loc[(df_copy['Return'] <= -alpha) & (df_copy['Return'] >= -beta_threshold), 'label'] = -1

        # Drop temporary columns
        df_copy.drop(columns=['Future_Close', 'Return', 'Backward_EMA', 'Open_Close_Change'], inplace=True)

        self.logger.debug("Raw labels calculated for EMA Return Percentile strategy.")
        
        # CRITICAL FIX: Ensure the returned DataFrame has the original DatetimeIndex
        # and only the 'label' column. This prevents index corruption.
        return pd.DataFrame({'label': df_copy['label']}, index=df_copy.index)
