# utils/data_management/market_data_handler.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Generator, Tuple, Literal
from datetime import datetime, timezone

# Import AppConfig and internal utilities
from config.params import AppConfig, FLOAT_EPSILON
from utils.data_management.data_manager import DataManager
from utils.training.model_trainer import ModelTrainer
from utils.feature_engineering.feature_engineer import FeatureEngineer
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.exceptions import TemporalSafetyError, ExchangeConnectionError, ConfigurationError

logger = logging.getLogger(__name__)

class MarketDataHandler:
    """Manages market data for backtesting and live trading, handling loading, feature engineering, and model inference."""

    def __init__(self,
                 app_config: AppConfig,
                 mode: Literal['backtest', 'live'],
                 symbol: str,
                 interval: str,
                 model_type: str,
                 train_ratio: float = 0.7,
                 initial_ohlcv_data: Optional[pd.DataFrame] = None,
                 backtest_mode: Optional[Literal['full', 'train', 'test']] = None
                 ):
        """Initializes MarketDataHandler.

        Args:
            app_config (AppConfig): Global configuration.
            mode (Literal['backtest', 'live']): Operation mode.
            symbol (str): Trading pair (e.g., 'BTCUSDT').
            interval (str): Data interval (e.g., '1h').
            model_type (str): ML model type.
            train_ratio (float): Train data ratio (0.0-1.0), for backtest.
            initial_ohlcv_data (Optional[pd.DataFrame]): Pre-loaded OHLCV data.
            backtest_mode (Optional[Literal['full', 'train', 'test']]): Data splitting mode.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing MarketDataHandler...")

        self.app_config = app_config
        self.exchange_config = app_config.exchange
        self.feature_config = app_config.features
        self.model_config = app_config.model
        self.general_config = app_config.general

        self.mode = mode
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.train_ratio = train_ratio
        self.initial_ohlcv_data = initial_ohlcv_data
        self.backtest_mode = backtest_mode

        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer(config=self.feature_config)
        self.model_trainer = ModelTrainer(model_config=self.model_config)

        self.last_processed_live_timestamp: Optional[datetime] = None
        self.logger.info("MarketDataHandler initialized.")


    def _load_data_for_processing(self) -> Optional[pd.DataFrame]:
        """Loads and splits historical data based on backtest mode."""
        self.logger.info(f"Loading historical data for {self.symbol} {self.interval} ({self.model_type}, mode: {self.backtest_mode})")

        full_data = self.data_manager.load_dataframe(
            data_type='raw',
            symbol=self.symbol,
            interval=self.interval
        )

        if full_data is None or full_data.empty:
            self.logger.error(f"Failed to load raw data for {self.symbol}-{self.interval}. Check data/raw directory.")
            return None

        # Ensure timezone-aware index
        if not isinstance(full_data.index, pd.DatetimeIndex):
            full_data.index = pd.to_datetime(full_data.index, utc=True)
        if full_data.index.tz is None:
            full_data.index = full_data.index.tz_localize(timezone.utc)
        elif full_data.index.tz != timezone.utc:
            full_data.index = full_data.index.tz_convert(timezone.utc)

        if self.backtest_mode == 'full':
            data_to_use = full_data
        elif self.backtest_mode in ['train', 'test']:
            train_size = int(len(full_data) * self.train_ratio)
            if train_size == 0 and len(full_data) > 0:
                self.logger.warning(f"Train size for {self.backtest_mode} mode is 0. Using minimal data.")
                data_to_use = full_data.iloc[:1] if len(full_data) > 0 else pd.DataFrame()
            elif train_size == len(full_data) and len(full_data) > 0:
                self.logger.warning(f"Train size for {self.backtest_mode} mode is full data length. Adjusting test set to include last element.")
                data_to_use = full_data.iloc[:]
            else:
                if self.backtest_mode == 'train':
                    data_to_use = full_data.iloc[:train_size]
                else:
                    data_to_use = full_data.iloc[train_size:]
        else:
            self.logger.error(f"Invalid backtest_mode: {self.backtest_mode}. Using full raw data.")
            data_to_use = full_data

        if data_to_use.empty:
            self.logger.warning(f"Data for backtest_mode '{self.backtest_mode}' is empty. Returning None.")
            return None

        self.logger.info(f"Loaded {len(data_to_use)} bars for backtest simulation in mode: {self.backtest_mode}.")
        return data_to_use


    def get_processed_data_stream(self) -> Generator[pd.Series, None, None]:
        """Generates a stream of processed data (OHLCV + features + signal + probabilities) for backtesting."""
        self.logger.info("Generating processed data stream...")
        raw_ohlcv_data = self.initial_ohlcv_data
        if raw_ohlcv_data is None:
            raw_ohlcv_data = self._load_data_for_processing()
            if raw_ohlcv_data is None or raw_ohlcv_data.empty:
                self.logger.error("No historical raw data available for stream.")
                return

        # Ensure timezone-aware index
        if not isinstance(raw_ohlcv_data.index, pd.DatetimeIndex):
            raw_ohlcv_data.index = pd.to_datetime(raw_ohlcv_data.index, utc=True)
        if raw_ohlcv_data.index.tz is None:
            raw_ohlcv_data.index = raw_ohlcv_data.index.tz_localize(timezone.utc)
        elif raw_ohlcv_data.index.tz != timezone.utc:
            raw_ohlcv_data.index = raw_ohlcv_data.index.tz_convert(timezone.utc)

        try:
            self.model_trainer.load(symbol=self.symbol, interval=self.interval, model_type=self.model_type)
            self.logger.info(f"Model '{self.model_type}' loaded for signal generation.")
        except Exception as e:
            self.logger.critical(f"Failed to load model '{self.model_type}': {e}. Cannot generate signals.", exc_info=True)
            return

        self.logger.info("Applying feature engineering...")
        featured_data = self.feature_engineer.process(raw_ohlcv_data)

        model_feature_cols = self.model_trainer.feature_columns_original
        if not model_feature_cols:
            self.logger.warning("Original feature columns not found. Using config fallback.")
            model_feature_cols = self.app_config.model.features_to_use or []
            if not model_feature_cols:
                self.logger.critical("Could not determine model features. Cannot proceed with prediction.")
                return

        initial_featured_len = len(featured_data)
        cleaned_data_for_prediction = featured_data.dropna(subset=model_feature_cols).copy()
        rows_removed_cleaning = initial_featured_len - len(cleaned_data_for_prediction)
        if rows_removed_cleaning > 0:
            self.logger.info(f"Removed {rows_removed_cleaning} rows due to NaNs in model features.")

        if cleaned_data_for_prediction.empty:
            self.logger.warning("Data empty after cleaning. Cannot generate signals.")
            return

        self.logger.info("Generating model predictions and probabilities...")
        predictions = self.model_trainer.predict(cleaned_data_for_prediction).reindex(cleaned_data_for_prediction.index).fillna(0).astype(int)
        probabilities_df = self.model_trainer.predict_proba(cleaned_data_for_prediction)

        # Handle the case where predict_proba might return None
        if probabilities_df is not None and not probabilities_df.empty: # ADDED check for None
            self.logger.debug(f"Probabilities DataFrame columns from ModelTrainer BEFORE renaming: {probabilities_df.columns.tolist()}")

            renamed_cols = {
                'proba_-1': -1,
                'proba_0': 0,
                'proba_1': 1
            }
            columns_to_rename = {k: v for k, v in renamed_cols.items() if k in probabilities_df.columns}

            if columns_to_rename:
                probabilities_df = probabilities_df.rename(columns=columns_to_rename)
            else:
                integer_columns_present = all(col in probabilities_df.columns for col in [-1, 0, 1])
                if not integer_columns_present:
                    self.logger.warning("Neither 'proba_X' nor integer columns found. Potential downstream issues.")


            self.logger.debug(f"Probabilities DataFrame columns AFTER renaming: {probabilities_df.columns.tolist()}")

            for label in [-1, 0, 1]:
                if label not in probabilities_df.columns:
                    probabilities_df[label] = 0.0

            probabilities_df = probabilities_df[[-1, 0, 1]]

            combined_data = cleaned_data_for_prediction.copy()
            combined_data['signal'] = predictions
            combined_data['probabilities'] = probabilities_df.apply(lambda row: row.to_dict(), axis=1) # Moved this inside the if block
        else:
            self.logger.warning("Probabilities DataFrame is None or empty. Skipping probability assignment.")
            combined_data = cleaned_data_for_prediction.copy()
            combined_data['signal'] = predictions
            combined_data['probabilities'] = [None] * len(combined_data) # Ensure 'probabilities' column exists even if empty

        self.logger.info(f"Generated processed data stream for {len(combined_data)} bars.")

        for index, row in combined_data.iterrows():
            bar_series = row.copy()
            bar_series.name = index
            yield bar_series

    async def get_latest_data(
        self,
        exchange_adapter: ExchangeInterface,
        last_processed_timestamp: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """Fetches, processes, and returns the latest completed candle with ML signal and probabilities for live trading."""
        self.logger.debug(f"Fetching latest data for {self.symbol} {self.interval}...")
        try:
            # Determine required lookback for features and LSTM
            required_lookback = self.general_config.historical_data_lookback
            self.logger.info(f"Required lookback for {self.symbol} {self.interval}: {required_lookback} bars.")

            raw_ohlcv_df = await exchange_adapter.fetch_recent_candles(
                symbol=self.symbol,
                interval=self.interval,
                limit=required_lookback
            )

            if raw_ohlcv_df is None or raw_ohlcv_df.empty:
                self.logger.warning("No raw OHLCV data fetched from exchange.")
                return None

            # Ensure timezone-aware index
            if not isinstance(raw_ohlcv_df.index, pd.DatetimeIndex):
                raw_ohlcv_df.index = pd.to_datetime(raw_ohlcv_df.index, unit='ms', utc=True)
            if raw_ohlcv_df.index.tz is None:
                raw_ohlcv_df.index = raw_ohlcv_df.index.tz_localize(timezone.utc)
            elif raw_ohlcv_df.index.tz != timezone.utc:
                raw_ohlcv_df.index = raw_ohlcv_df.index.tz_convert(timezone.utc)

            raw_ohlcv_df = raw_ohlcv_df.sort_index()

            # The latest completed candle is usually the second-to-last
            latest_completed_candle = raw_ohlcv_df.iloc[-2]
            latest_completed_candle_timestamp = latest_completed_candle.name

            if last_processed_timestamp is not None and latest_completed_candle_timestamp <= last_processed_timestamp:
                self.logger.debug(f"Latest completed candle at {latest_completed_candle_timestamp} already processed or no new candle.")
                return None

            featured_data = self.feature_engineer.process(raw_ohlcv_df.copy())

            self.model_trainer.load(symbol=self.symbol, interval=self.interval, model_type=self.model_type)

            model_feature_cols = self.model_trainer.feature_columns_original
            if not model_feature_cols:
                self.logger.warning("Original feature columns not found. Using config fallback.")
                model_feature_cols = self.app_config.model.features_to_use or []
                if not model_feature_cols:
                    raise RuntimeError("Could not determine model features. Cannot proceed with live prediction.")

            # Prepare data for model inference based on model type
            if self.model_type == 'lstm':
                sequence_length = self.model_config.lstm_params.sequence_length_bars
                model_input_data = featured_data[model_feature_cols].iloc[-sequence_length:].copy()
                model_input_data.dropna(inplace=True)
                # Defensive: Check again after NaN removal
                if model_input_data.empty or len(model_input_data) < sequence_length:
                    self.logger.warning(f"Model input data empty or insufficient ({len(model_input_data)}) for LSTM sequence length {sequence_length}. Skipping prediction.")
                    return None
            else:
                model_input_data = featured_data[model_feature_cols].iloc[[-1]].copy()
                model_input_data.dropna(inplace=True)
                if model_input_data.empty:
                    self.logger.warning("Model input data empty after cleaning. Cannot generate live signal.")
                    return None

            live_signal = self.model_trainer.predict(model_input_data).iloc[-1]
            live_probabilities_raw = self.model_trainer.predict_proba(model_input_data)

            # Check if live_probabilities_raw is None before trying to access .columns
            if live_probabilities_raw is not None and not live_probabilities_raw.empty:
                self.logger.debug(f"Live Probabilities DataFrame columns from ModelTrainer BEFORE renaming: {live_probabilities_raw.columns.tolist()}")

                live_probabilities_dict = {}
                renamed_cols = {
                    'proba_-1': -1,
                    'proba_0': 0,
                    'proba_1': 1
                }
                columns_to_rename = {k: v for k, v in renamed_cols.items() if k in live_probabilities_raw.columns}

                if columns_to_rename:
                    live_probabilities_raw = live_probabilities_raw.rename(columns=columns_to_rename)
                else:
                    integer_columns_present = all(col in live_probabilities_raw.columns for col in [-1, 0, 1])
                    if not integer_columns_present:
                        self.logger.warning("Neither 'proba_X' nor integer columns found. Potential downstream issues.")

                self.logger.debug(f"Live Probabilities DataFrame columns AFTER renaming: {live_probabilities_raw.columns.tolist()}")

                for label in [-1, 0, 1]:
                    if label not in live_probabilities_raw.columns:
                        live_probabilities_raw[label] = 0.0

                live_probabilities_dict = live_probabilities_raw.iloc[-1].reindex([-1, 0, 1]).fillna(0.0).to_dict()
            else:
                self.logger.warning("Could not parse live model probabilities into dict. Probabilities will be empty.")
                live_probabilities_dict = {} # Ensure it's an empty dict if None or empty

            latest_processed_bar = featured_data.loc[[latest_completed_candle_timestamp]].copy()
            latest_processed_bar['signal'] = int(live_signal)
            latest_processed_bar['probabilities'] = [live_probabilities_dict]

            self.logger.debug(f"Processed new live candle at {latest_completed_candle_timestamp}. Signal: {int(live_signal)}")

            return latest_processed_bar.iloc[0]

        except TemporalSafetyError as e:
            self.logger.critical(f"Temporal safety violation: {e}", exc_info=True)
            raise
        except ExchangeConnectionError as e:
            self.logger.error(f"Exchange connection error: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.critical(f"Unexpected error in get_latest_data: {e}", exc_info=True)
            raise
