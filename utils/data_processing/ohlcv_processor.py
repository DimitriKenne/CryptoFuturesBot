# utils/data_processing/ohlcv_processor.py

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, List

# --- Technical Analysis Library Import ---
# Attempt to import 'ta' library for indicators. Handle missing library gracefully.
try:
    from ta.volatility import AverageTrueRange
    from ta.trend import EMAIndicator
    TA_AVAILABLE = True
except ImportError:
    logging.warning(
        "Technical Analysis library 'ta' not found. Install using 'pip install ta'. "
        "Fallback ATR/EMA calculation will be attempted if features are missing."
    )
    AverageTrueRange = None
    EMAIndicator = None
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)

class OHLCVProcessor:
    """
    Handles common OHLCV (Open, High, Low, Close, Volume) data processing tasks,
    including validation, numeric type conversion, calculation of missing indicators (as fallback),
    aligning model signals and probabilities, and handling NaN values.

    This class centralizes data preparation logic used by both backtesting and
    live trading components.
    """

    def __init__(self):
        """
        Initializes the OHLCVProcessor.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("OHLCVProcessor initialized.")

    def prepare_data(self,
                     data: pd.DataFrame,
                     model_predict: Optional[pd.Series] = None,
                     model_proba: Optional[pd.DataFrame] = None,
                     volatility_adjustment_enabled: bool = False,
                     trend_filter_enabled: bool = False,
                     volatility_regime_filter_enabled: bool = False,
                     atr_col_name: str = 'atr_14',
                     ema_col_name: str = 'ema_200',
                     volatility_regime_col_name: str = 'volatility_regime'
                     ) -> pd.DataFrame:
        """
        Prepares OHLCV data by validating columns, converting types,
        calculating missing indicators (as a fallback), aligning signals and probabilities,
        and cleaning NaNs.

        Args:
            data (pd.DataFrame): Input DataFrame with OHLCV data.
            model_predict (Optional[pd.Series]): Series with model predictions (-1, 0, 1).
            model_proba (Optional[pd.DataFrame]): DataFrame with model probability scores.
            volatility_adjustment_enabled (bool): Whether volatility adjustment is active.
            trend_filter_enabled (bool): Whether trend filtering is active.
            volatility_regime_filter_enabled (bool): Whether volatility regime filtering is active.
            atr_col_name (str): Expected ATR column name if volatility adjustment is enabled.
            ema_col_name (str): Expected EMA column name if trend filter is enabled.
            volatility_regime_col_name (str): Expected volatility regime column name.

        Returns:
            pd.DataFrame: Cleaned and prepared DataFrame.

        Raises:
            ValueError: If critical OHLCV columns are missing or DataFrame becomes empty.
            ImportError: If 'ta' library is required but not available.
        """
        self.logger.debug("Starting data preparation...")

        if data.empty:
            self.logger.warning("Input data for preparation is empty.")
            return pd.DataFrame()

        processed_data = data.copy()

        # --- 1. Validate and Coerce OHLCV Columns to numeric ---
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in processed_data.columns for col in required_ohlcv):
            missing = [col for col in required_ohlcv if col not in processed_data.columns]
            raise ValueError(f"Input data missing required OHLCV columns: {missing}")

        for col in required_ohlcv:
            # Coerce to numeric, errors='coerce' will turn non-numeric into NaN
            if not pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                self.logger.debug(f"Coerced column '{col}' to numeric type.")

        if processed_data[required_ohlcv].isnull().any().any():
            self.logger.warning("NaN values detected in OHLCV data after numeric conversion. These might be dropped later.")

        # --- 2. Calculate Missing Indicators (ATR, EMA) as Fallback ---
        # Note: Primary feature generation should occur in FeatureEngineer.
        # This is a fallback if the required indicator columns are not present.
        if volatility_adjustment_enabled and atr_col_name not in processed_data.columns:
            self._calculate_missing_indicator(processed_data, 'ATR', atr_col_name)

        if trend_filter_enabled and ema_col_name not in processed_data.columns:
            self._calculate_missing_indicator(processed_data, 'EMA', ema_col_name)

        # Ensure calculated indicators are numeric and ffill NaNs
        for col_name in [atr_col_name, ema_col_name]:
            if col_name in processed_data.columns:
                if not pd.api.types.is_numeric_dtype(processed_data[col_name]):
                    processed_data[col_name] = pd.to_numeric(processed_data[col_name], errors='coerce')
                    self.logger.debug(f"Coerced calculated indicator '{col_name}' to numeric.")
                if processed_data[col_name].isnull().any():
                    self.logger.warning(f"NaN values found in calculated indicator '{col_name}'. Forward filling NaNs.")
                    processed_data[col_name].ffill(inplace=True)

        # --- 3. Handle Volatility Regime Column ---
        if volatility_regime_filter_enabled:
            if volatility_regime_col_name not in processed_data.columns:
                self.logger.error(f"Volatility regime filter enabled, but column '{volatility_regime_col_name}' is missing from data. Setting to pd.NA.")
                processed_data[volatility_regime_col_name] = pd.NA
            else:
                # Ensure it's a nullable integer type
                # Use pd.Int8Dtype() for nullable integer, which can store pd.NA
                if not isinstance(processed_data[volatility_regime_col_name].dtype, pd.Int8Dtype):
                    self.logger.warning(f"Volatility regime column '{volatility_regime_col_name}' has incorrect dtype {processed_data[volatility_regime_col_name].dtype}. Attempting cast to Int8Dtype.")
                    try:
                        processed_data[volatility_regime_col_name] = processed_data[volatility_regime_col_name].astype(pd.Int8Dtype())
                    except Exception as e:
                        self.logger.error(f"Failed to cast '{volatility_regime_col_name}' to Int8Dtype: {e}. NaNs in this column may cause errors later.", exc_info=True)
                # Fill NaNs in volatility_regime with a default or most frequent if appropriate, for now use -1
                if processed_data[volatility_regime_col_name].isnull().any():
                    self.logger.warning(f"NaNs found in '{volatility_regime_col_name}'. Filling with pd.NA for consistent processing.")
                    processed_data[volatility_regime_col_name].fillna(pd.NA, inplace=True) # Fill with nullable NA

        # --- 4. Align Model Predictions and Probabilities ---
        if model_predict is not None:
            if not model_predict.index.equals(processed_data.index):
                self.logger.warning("Model prediction index does not match data index. Reindexing predictions.")
                processed_data['signal'] = model_predict.reindex(processed_data.index).fillna(0).astype(int)
            else:
                processed_data['signal'] = model_predict.fillna(0).astype(int)
            self.logger.debug("Model predictions aligned with data index.")
        else:
            self.logger.warning("No model predictions provided. 'signal' column will not be available.")
            processed_data['signal'] = 0 # Default to neutral if no predictions

        if model_proba is not None:
            if not model_proba.index.equals(processed_data.index):
                self.logger.warning("Model probability index does not match data index. Reindexing probabilities.")
                model_proba = model_proba.reindex(processed_data.index)

            expected_proba_cols = [-1, 0, 1]
            # Ensure probability columns exist and are numeric
            for col in expected_proba_cols:
                if col not in model_proba.columns:
                    model_proba[col] = np.nan
                    self.logger.warning(f"Probability DataFrame missing column {col}. Added with NaN values.")
                else:
                    model_proba[col] = pd.to_numeric(model_proba[col], errors='coerce')
                    if model_proba[col].isnull().any():
                         self.logger.warning(f"NaNs found in probability column {col}. Coerced to numeric.")

            processed_data = processed_data.join(model_proba) # Join probabilities to main DataFrame
            self.logger.debug("Model probabilities aligned and joined to data.")
        else:
            self.logger.warning("No model probabilities provided. Confidence filtering will not be possible.")
            # Add columns with NaNs if not provided, ensuring they exist for later access if needed
            for col in [-1, 0, 1]:
                if col not in processed_data.columns: # Prevent re-adding if already existing via join (from model_proba)
                    processed_data[col] = np.nan

        # --- 5. Final Data Cleaning (Drop NaNs in Critical Columns) ---
        critical_cols: List[str] = required_ohlcv + ['signal']
        if volatility_adjustment_enabled:
            critical_cols.append(atr_col_name)
        if trend_filter_enabled:
            critical_cols.append(ema_col_name)
        if volatility_regime_filter_enabled:
            critical_cols.append(volatility_regime_col_name)

        critical_cols_present = [col for col in critical_cols if col in processed_data.columns]
        initial_rows = len(processed_data)
        processed_data.dropna(subset=critical_cols_present, inplace=True)
        processed_data = processed_data.copy() # Ensure it's a standalone copy after dropna
        rows_removed = initial_rows - len(processed_data)
        if rows_removed > 0:
            self.logger.warning(f"Removed {rows_removed} rows with NaNs in critical columns ({critical_cols_present}) during final cleaning.")

        if processed_data.empty:
            raise ValueError("DataFrame is empty after removing NaNs in critical columns. Cannot proceed.")

        self.logger.info(f"Data preparation complete. Final data shape: {processed_data.shape}")
        return processed_data

    def _calculate_missing_indicator(self, data: pd.DataFrame, indicator_type: str, col_name: str):
        """
        Calculates missing technical indicators (ATR or EMA) using the 'ta' library.
        NOTE: This is a fallback mechanism. The primary responsibility for generating
        technical indicators lies within the FeatureEngineer module.

        Args:
            data (pd.DataFrame): The DataFrame to add the indicator to.
            indicator_type (str): Type of indicator ('ATR' or 'EMA').
            col_name (str): The name for the new indicator column.

        Raises:
            ImportError: If 'ta' library is not available.
            ValueError: If input data is invalid for calculation.
            NotImplementedError: If the indicator type is not supported.
        """
        self.logger.warning(f"Required {indicator_type} column '{col_name}' not found. Attempting fallback calculation...")
        if not TA_AVAILABLE:
            raise ImportError(f"Cannot calculate {indicator_type}: 'ta' library not installed or import failed.")

        try:
            if indicator_type == 'ATR' and AverageTrueRange:
                required_ohlc = ['open', 'high', 'low', 'close']
                # Ensure OHLC columns are numeric before calculation
                if not all(col in data.columns and pd.api.types.is_numeric_dtype(data[col]) for col in required_ohlc):
                    self.logger.error(f"Cannot calculate ATR: Missing or non-numeric OHLC data.")
                    raise ValueError("Invalid OHLC data for ATR calculation.")
                # Extract window from column name, e.g., 'atr_14' -> 14
                window = int(col_name.split('_')[-1]) if '_' in col_name and col_name.split('_')[-1].isdigit() else 14
                indicator = AverageTrueRange(
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    window=window,
                    fillna=False
                )
                data[col_name] = indicator.average_true_range()
                self.logger.info(f"Successfully calculated and added missing {indicator_type} column '{col_name}'.")
            elif indicator_type == 'EMA' and EMAIndicator:
                if 'close' not in data.columns or not pd.api.types.is_numeric_dtype(data['close']):
                    self.logger.error(f"Cannot calculate EMA: Missing or non-numeric close data.")
                    raise ValueError("Invalid close data for EMA calculation.")
                # Extract window from column name, e.g., 'ema_200' -> 200
                window = int(col_name.split('_')[-1]) if '_' in col_name and col_name.split('_')[-1].isdigit() else 200
                indicator = EMAIndicator(
                    close=data['close'],
                    window=window,
                    fillna=False
                )
                data[col_name] = indicator.ema_indicator()
                self.logger.info(f"Successfully calculated and added missing {indicator_type} column '{col_name}'.")
            else:
                raise NotImplementedError(f"Fallback calculation for {indicator_type} not implemented or library component missing.")
        except Exception as e:
            self.logger.error(f"Failed to calculate fallback {indicator_type} '{col_name}': {e}", exc_info=True)
            data[col_name] = np.nan # Assign NaN if calculation fails
            self.logger.critical(f"{indicator_type}-based functionality cannot operate without valid data. Consider disabling or ensuring feature is generated elsewhere.")

    def update_buffer_with_recent_data(self,
                                       current_buffer: pd.DataFrame,
                                       new_data: pd.DataFrame,
                                       buffer_size: int,
                                       latest_timestamp: Optional[pd.Timestamp] = None
                                       ) -> pd.DataFrame:
        """
        Updates the OHLCV data buffer with new incoming data, ensuring it remains
        sorted by time and maintains a fixed maximum size (lookback_bars).
        Drops duplicate entries and handles out-of-order data by re-sorting.

        Args:
            current_buffer (pd.DataFrame): The existing OHLCV data buffer.
            new_data (pd.DataFrame): New OHLCV data (can be single row or multiple).
            buffer_size (int): The maximum number of bars to keep in the buffer.
            latest_timestamp (Optional[pd.Timestamp]): The timestamp of the last known
                                                      latest candle. Used to filter
                                                      already processed data.

        Returns:
            pd.DataFrame: The updated and trimmed data buffer.
        """
        self.logger.debug(f"Updating data buffer. Current size: {len(current_buffer)}, New data size: {len(new_data)}")

        if new_data.empty:
            self.logger.debug("No new data to add to buffer.")
            return current_buffer

        # Ensure new_data index is DatetimeIndex and UTC
        if not isinstance(new_data.index, pd.DatetimeIndex):
            new_data.index = pd.to_datetime(new_data.index, utc=True)
        elif new_data.index.tz is None:
            new_data.index = new_data.index.tz_localize('UTC')
        elif str(new_data.index.tz) != 'UTC':
            new_data.index = new_data.index.tz_convert('UTC')

        # Filter out data that is older than or equal to the latest_timestamp to avoid duplicates
        # and already processed bars in live trading scenarios.
        if latest_timestamp is not None and not current_buffer.empty:
            new_data = new_data[new_data.index > latest_timestamp]
            if new_data.empty:
                self.logger.debug("New data is older than or equal to latest_timestamp; no new unique candles.")
                return current_buffer

        # Concatenate existing buffer and new data
        if current_buffer.empty:
            combined_data = new_data
        else:
            # Ensure current_buffer index is also DatetimeIndex and UTC before combining
            if not isinstance(current_buffer.index, pd.DatetimeIndex):
                current_buffer.index = pd.to_datetime(current_buffer.index, utc=True)
            elif current_buffer.index.tz is None:
                current_buffer.index = current_buffer.index.tz_localize('UTC')
            elif str(current_buffer.index.tz) != 'UTC':
                current_buffer.index = current_buffer.index.tz_convert('UTC')

            combined_data = pd.concat([current_buffer, new_data])

        # Drop duplicates based on index (timestamps)
        # Keep 'last' to prioritize newer data if timestamps are identical (shouldn't happen with proper unique timestamps)
        initial_combined_len = len(combined_data)
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        if len(combined_data) < initial_combined_len:
            self.logger.debug(f"Removed {initial_combined_len - len(combined_data)} duplicate rows from buffer.")

        # Sort by index (time) to ensure correct order
        combined_data.sort_index(inplace=True)

        # Trim the buffer to the desired size (keep the most recent 'buffer_size' bars)
        if len(combined_data) > buffer_size:
            self.logger.debug(f"Trimming buffer from {len(combined_data)} to {buffer_size} bars.")
            return combined_data.tail(buffer_size).copy() # Use .copy() to prevent SettingWithCopyWarning
        else:
            self.logger.debug(f"Buffer size: {len(combined_data)} (no trimming needed).")
            return combined_data.copy() # Return a copy to ensure independent DataFrame
