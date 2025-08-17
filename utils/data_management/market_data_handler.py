# utils/data_management/market_data_handler.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Generator, Tuple, Literal
from datetime import datetime, timezone # For timezone awareness

# Import AppConfig and internal utilities
from config.params import AppConfig, FLOAT_EPSILON
from utils.data_management.data_manager import DataManager
from utils.training.model_trainer import ModelTrainer
from utils.feature_engineering.feature_engineer import FeatureEngineer
from utils.adapters.exchange_interface import ExchangeInterface # For type hinting the adapter
from utils.exceptions import TemporalSafetyError, ExchangeConnectionError, ConfigurationError # Import custom exceptions

logger = logging.getLogger(__name__)

class MarketDataHandler:
    """
    Manages market data for backtesting and live trading.

    Centralizes data loading, feature preparation, model inference,
    and signal integration.
    """

    def __init__(self,
                 app_config: AppConfig,
                 mode: Literal['backtest', 'live'], # New: to differentiate behavior (e.g., for live fetching)
                 symbol: str,
                 interval: str,
                 model_type: str,
                 train_ratio: float = 0.7, # Relevant for backtest mode
                 initial_ohlcv_data: Optional[pd.DataFrame] = None, # For specific backtest/MC scenarios
                 backtest_mode: Optional[Literal['full', 'train', 'test']] = None # ADDED: For data splitting
                 ):
        """
        Initializes the MarketDataHandler.

        Args:
            app_config (AppConfig): Global application configuration.
            mode (Literal['backtest', 'live']): The operating mode of the handler (e.g., affects live fetching).
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '1h').
            model_type (str): The type of ML model used (e.g., 'xgboost', 'lstm').
            train_ratio (float): The ratio of data to use for training (0.0 to 1.0).
                                 Only applicable in "test" or "train" backtest_mode.
            initial_ohlcv_data (Optional[pd.DataFrame]): Optional, pre-loaded OHLCV data.
                                                        If provided, data loading from files is skipped
                                                        in get_processed_data_stream.
            backtest_mode (Optional[Literal['full', 'train', 'test']]): The mode for data splitting
                                                                         when loading from file.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing MarketDataHandler...")

        # Store configuration sections
        self.app_config = app_config # Store the full AppConfig object
        self.exchange_config = app_config.exchange
        self.feature_config = app_config.features
        self.model_config = app_config.model
        self.general_config = app_config.general

        # Store mode-specific parameters
        self.mode = mode # e.g., 'backtest' or 'live'
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.train_ratio = train_ratio
        self.initial_ohlcv_data = initial_ohlcv_data # Data provided at init, if any
        self.backtest_mode = backtest_mode # Store the backtest_mode ('full', 'train', 'test')

        # Initialize internal utilities (instances are created once)
        self.data_manager = DataManager() # Manages raw/processed data files
        self.feature_engineer = FeatureEngineer(config=self.feature_config) # Applies feature engineering
        self.model_trainer = ModelTrainer(model_config=self.model_config) # Loads model and makes predictions

        self.last_processed_live_timestamp: Optional[datetime] = None
        self.logger.info("MarketDataHandler initialized.")


    def _load_data_for_processing(self) -> Optional[pd.DataFrame]:
        """
        Loads and splits historical data (raw OHLCV) from files based on the stored backtest mode.
        This internal method is used when data is NOT provided in-memory (e.g., from file).
        """
        self.logger.info(f"Loading historical data for {self.symbol} {self.interval} (model: {self.model_type}, backtest_mode: {self.backtest_mode})...")
        
        # Load the full raw OHLCV data from disk for feature engineering
        full_data = self.data_manager.load_data(
            symbol=self.symbol,
            interval=self.interval,
            data_type='raw' # CHANGED to 'raw' - we will apply feature engineering here
        )

        if full_data is None or full_data.empty:
            self.logger.error(f"Failed to load raw data for {self.symbol}-{self.interval}. Check data/raw directory.")
            return None
        
        # Ensure data index is DatetimeIndex and UTC timezone
        if not isinstance(full_data.index, pd.DatetimeIndex):
            full_data.index = pd.to_datetime(full_data.index, utc=True)
        if full_data.index.tz is None: # Ensure timezone awareness
            full_data.index = full_data.index.tz_localize(timezone.utc)
        elif full_data.index.tz != timezone.utc: # Convert to UTC if needed
            full_data.index = full_data.index.tz_convert(timezone.utc)

        # Apply train/test split logic based on self.backtest_mode
        if self.backtest_mode == 'full':
            data_to_use = full_data
        elif self.backtest_mode in ['train', 'test']:
            train_size = int(len(full_data) * self.train_ratio)
            if self.backtest_mode == 'train':
                data_to_use = full_data.iloc[:train_size]
            elif self.backtest_mode == 'test':
                data_to_use = full_data.iloc[train_size:]
        else:
            self.logger.error(f"Invalid backtest_mode: {self.backtest_mode}. Using full raw data for feature engineering.")
            data_to_use = full_data
        
        if data_to_use.empty:
            self.logger.warning(f"Data for backtest_mode '{self.backtest_mode}' is empty. Returning None.")
            return None

        self.logger.info(f"Loaded {len(data_to_use)} bars for backtest simulation in mode: {self.backtest_mode}.")
        return data_to_use


    def get_processed_data_stream(
        self,
    ) -> Generator[pd.Series, None, None]:
        """
        Generates a stream of processed historical data, including features,
        model predictions, and probabilities for backtesting.

        Uses the symbol, interval, model_type, backtest_mode, and train_ratio
        set during the handler's initialization.

        Yields:
            pd.Series: A Series for each bar, containing OHLCV, engineered features,
                       signal, and probabilities.
        """
        
        # Determine the source of raw OHLCV data
        raw_ohlcv_data = self.initial_ohlcv_data # Check if data was provided at init (e.g., from MC simulator)
        if raw_ohlcv_data is None:
            # If no initial_ohlcv_data, load raw data from files based on instance attributes
            raw_ohlcv_data = self._load_data_for_processing()
            if raw_ohlcv_data is None or raw_ohlcv_data.empty:
                self.logger.error("No historical raw data available for processing stream.")
                return # Exit generator if no data


        # Ensure data index is DatetimeIndex and UTC timezone
        if not isinstance(raw_ohlcv_data.index, pd.DatetimeIndex):
            raw_ohlcv_data.index = pd.to_datetime(raw_ohlcv_data.index, utc=True)
        if raw_ohlcv_data.index.tz is None: # Ensure timezone awareness
            raw_ohlcv_data.index = raw_ohlcv_data.index.tz_localize(timezone.utc)
        elif raw_ohlcv_data.index.tz != timezone.utc: # Convert to UTC if needed
            raw_ohlcv_data.index = raw_ohlcv_data.index.tz_convert(timezone.utc)
        
        # Load the trained model (or ensure it's loaded)
        try:
            self.model_trainer.load(symbol=self.symbol, interval=self.interval, model_key=self.model_type)
            self.logger.info(f"Model '{self.model_type}' loaded for signal generation.")
        except Exception as e:
            self.logger.critical(f"Failed to load model '{self.model_type}': {e}. Cannot generate signals.", exc_info=True)
            return # Exit generator if model cannot be loaded


        # --- Step 1: Feature Engineering ---
        self.logger.info("Applying feature engineering...")
        # Now, feature_engineer.process() is always called on RAW OHLCV data
        featured_data = self.feature_engineer.process(raw_ohlcv_data)

        # --- Step 2: Clean data for model prediction ---
        # Ensure model feature columns are determined from the loaded model metadata
        model_feature_cols = self.model_trainer.feature_columns_original
        if not model_feature_cols:
            self.logger.warning("Original feature columns not found in loaded model metadata. Using features_to_use from config as fallback.")
            model_feature_cols = self.app_config.model.features_to_use or []
            if not model_feature_cols:
                self.logger.critical("Could not determine original feature columns for the model. Cannot proceed with prediction.")
                return # Exit generator

        # Drop rows with NaNs in any of the model's feature columns before prediction
        initial_featured_len = len(featured_data)
        cleaned_data_for_prediction = featured_data.dropna(subset=model_feature_cols).copy()
        rows_removed_cleaning = initial_featured_len - len(cleaned_data_for_prediction)
        if rows_removed_cleaning > 0:
            self.logger.info(f"Removed {rows_removed_cleaning} rows due to NaNs in model feature columns before prediction.")
        
        if cleaned_data_for_prediction.empty:
            self.logger.warning("Data is empty after cleaning for model prediction. Cannot generate signals.")
            return # Exit generator if no data left for prediction


        # --- Step 3: Generate Predictions and Probabilities ---
        self.logger.info("Generating model predictions and probabilities...")
        predictions = self.model_trainer.predict(cleaned_data_for_prediction).reindex(cleaned_data_for_prediction.index).fillna(0).astype(int)
        probabilities = self.model_trainer.predict_proba(cleaned_data_for_prediction).reindex(cleaned_data_for_prediction.index)

        # --- Step 4: Integrate signals and probabilities back into original data stream ---
        # Create a combined DataFrame for iteration
        # Use a copy to avoid modifying the original dataframes passed in
        combined_data = cleaned_data_for_prediction.copy()
        combined_data['signal'] = predictions
        # Store probabilities as a dictionary in a column, similar to how TradingBot expects it
        combined_data['probabilities'] = combined_data.apply(
            lambda row: {idx: prob for idx, prob in probabilities.loc[row.name].items()}, axis=1
        )

        self.logger.info(f"Generated processed data stream for {len(combined_data)} bars.")

        # Yield each row as a Series for consistent bar-by-bar processing
        for index, row in combined_data.iterrows():
            bar_series = row.copy()
            bar_series.name = index # Ensure the timestamp is the Series index (important for Backtester)
            yield bar_series

    async def get_latest_data(
        self,
        exchange_adapter: ExchangeInterface,
        last_processed_timestamp: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """
        Fetches the latest completed candle from the exchange, processes it,
        and returns it along with the ML signal and probabilities.
        Only returns data for a new, fully completed candle.

        Args:
            exchange_adapter (ExchangeInterface): The live exchange adapter instance.
            last_processed_timestamp (Optional[datetime]): Timestamp of the last candle
                                                            that was successfully processed.

        Returns:
            Optional[pd.Series]: A Series for the latest completed bar, including OHLCV,
                                 features, signal, and probabilities, or None if no new
                                 completed candle is available.
        """
        self.logger.debug(f"Fetching latest data for {self.symbol} {self.interval}...")
        try:
            # 1. Fetch raw OHLCV data (e.g., last 'lookback' candles required by features/model)
            # Determine required lookback for features and model prediction
            # This logic should be derived from feature_config and model_config
            required_lookback = self.feature_config.rolling_window_max_period + \
                                (self.model_config.lstm_params.sequence_length if self.model_config.model_type == 'lstm' else 0)
            
            # Add a buffer
            required_lookback = max(required_lookback, self.general_config.min_historical_data_lookback)
            required_lookback += 5 # Small buffer just in case

            raw_ohlcv_df = await exchange_adapter.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=required_lookback # Fetch enough bars for lookback
            )

            if raw_ohlcv_df is None or raw_ohlcv_df.empty:
                self.logger.warning("No raw OHLCV data fetched from exchange.")
                return None

            # Ensure timezone awareness and sorting
            if not isinstance(raw_ohlcv_df.index, pd.DatetimeIndex):
                raw_ohlcv_df.index = pd.to_datetime(raw_ohlcv_df.index, unit='ms', utc=True) # Assuming timestamp is in ms
            if raw_ohlcv_df.index.tz is None: # Ensure timezone awareness
                raw_ohlcv_df.index = raw_ohlcv_df.index.tz_localize(timezone.utc)
            elif raw_ohlcv_df.index.tz != timezone.utc: # Convert to UTC if needed
                raw_ohlcv_df.index = raw_ohlcv_df.index.tz_convert(timezone.utc)
            
            raw_ohlcv_df = raw_ohlcv_df.sort_index()

            # Identify the latest COMPLETED candle based on the last_processed_timestamp
            latest_completed_candle = raw_ohlcv_df.iloc[-2] # Second to last candle is typically the last completed one
            latest_completed_candle_timestamp = latest_completed_candle.name

            if last_processed_timestamp is not None and latest_completed_candle_timestamp <= last_processed_timestamp:
                self.logger.debug(f"Latest completed candle at {latest_completed_candle_timestamp} already processed or no new candle.")
                return None # No new completed candle to process


            # 2. Apply feature engineering to the raw data (full history needed for correct features)
            featured_data = self.feature_engineer.process(raw_ohlcv_df.copy())

            # 3. Load the model and make prediction on the latest data
            self.model_trainer.load(symbol=self.symbol, interval=self.interval, model_key=self.model_type)
            
            # Get the input data for the model (latest features)
            # Ensure model feature columns are determined from the loaded model metadata
            model_feature_cols = self.model_trainer.feature_columns_original
            if not model_feature_cols:
                self.logger.warning("Original feature columns not found in loaded model metadata for live prediction. Using features_to_use from config as fallback.")
                model_feature_cols = self.app_config.model.features_to_use or []
                if not model_feature_cols:
                    raise RuntimeError("Could not determine original feature columns for the model. Cannot proceed with live prediction.")

            # Prepare data for prediction: ensure lookback sequence if LSTM, and drop NaNs
            # Take a slice of data that corresponds to the model's expected input length
            # For LSTM, it's `sequence_length` bars. For others, it's just the latest bar.
            if self.model_config.model_type == 'lstm':
                sequence_length = self.model_config.lstm_params.sequence_length
                # Ensure we have enough data points for the sequence
                if len(featured_data) < sequence_length:
                    self.logger.warning(f"Not enough data points ({len(featured_data)}) for LSTM sequence length {sequence_length}. Skipping prediction.")
                    return None
                model_input_data = featured_data[model_feature_cols].iloc[-sequence_length:].copy()
            else:
                model_input_data = featured_data[model_feature_cols].iloc[[-1]].copy() # Just the latest bar for non-LSTMs

            # Drop any NaNs in the final model input slice (e.g., from features that couldn't be computed for latest bars)
            model_input_data.dropna(inplace=True)
            if model_input_data.empty:
                self.logger.warning("Model input data is empty after cleaning NaNs. Cannot generate live signal.")
                return None

            live_signal = self.model_trainer.predict(model_input_data).iloc[-1]
            live_probabilities_raw = self.model_trainer.predict_proba(model_input_data)
            
            # Convert raw probabilities to dictionary
            live_probabilities = {}
            if hasattr(live_probabilities_raw, 'shape') and live_probabilities_raw.shape[1] == 3:
                # Assuming class_labels mapping is in ModelTrainer for consistency
                class_labels = getattr(self.model_trainer, 'label_mapping', {0:0, 1:1, 2:-1}) # Default mapping
                last_proba_row = live_probabilities_raw.iloc[-1]
                live_probabilities = {class_labels.get(i, i): float(prob) for i, prob in enumerate(last_proba_row)}
            elif isinstance(live_probabilities_raw, pd.DataFrame):
                live_probabilities = live_probabilities_raw.iloc[-1].dropna().to_dict()
            else:
                self.logger.warning("Could not parse live model probabilities into standard dict format. Probabilities will be empty.")


            # 5. Integrate Signals into the latest completed bar's data
            # Combine the original OHLCV with the features, signal, and probabilities
            # We want the second to last bar (latest_completed_candle) combined with its features/signal/proba
            latest_processed_bar = featured_data.loc[[latest_completed_candle_timestamp]].copy()
            latest_processed_bar['signal'] = int(live_signal) # Apply the signal to the latest bar
            latest_processed_bar['probabilities'] = [live_probabilities] # Store as list to avoid Series/Dict issues


            self.logger.debug(f"Processed new live candle at {latest_completed_candle_timestamp}. Signal: {int(live_signal)}")

            return latest_processed_bar.iloc[0] # Return as a Series

        except TemporalSafetyError as e:
            self.logger.critical(f"Temporal safety violation detected in MarketDataHandler: {e}", exc_info=True)
            raise # Re-raise to be caught by bot's main loop
        except ExchangeConnectionError as e:
            self.logger.error(f"Exchange connection error in MarketDataHandler: {e}", exc_info=True)
            raise # Re-raise to be caught by bot's main loop
        except Exception as e:
            self.logger.critical(f"Unexpected error in get_latest_data: {e}", exc_info=True)
            raise # Re-raise to be caught by bot's main loop

