# utils/data_manager.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import joblib # Joblib is often used for saving/loading models and potentially metadata
import sys # Import sys for checking modules
from typing import Optional, Any, Dict, Union # Import Dict and Union
from matplotlib.figure import Figure
import json # Import json for metadata and evaluation results

# Set up logging for the data manager
logger = logging.getLogger(__name__)

# Import paths configuration
try:
    from config.paths import PATHS, PATH_CONFIG
except ImportError:
    # Define a basic fallback if paths.py is missing
    logger.error("config.paths not found. Using basic fallback paths. Data loading/saving may fail.")


# --- Conditional Import for TensorFlow and Keras ---
# Need to install tensorflow: pip install tensorflow
try:
    import tensorflow as tf
    # Check for GPU availability and log TensorFlow version only once per module load
    tf_version = getattr(tf, '__version__', 'unknown')
    logger.info(f"TensorFlow (version {tf_version}) imported successfully in DataManager.")
    if tf.config.list_physical_devices('GPU'):
        logger.info("GPU is available and enabled for TensorFlow.")
    else:
        logger.info("GPU is not available or not enabled for TensorFlow.")

    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not found. Keras model saving/loading will not be available in DataManager.")
    tf = None
    TF_AVAILABLE = False
except Exception as e:
    # Catch other potential errors during TF import (e.e.g., DLL issues)
    logger.error(f"Error importing TensorFlow/Keras in DataManager: {e}", exc_info=True)
    tf = None
    TF_AVAILABLE = False


class DataManager:
    """
    Manages loading and saving of data files and model artifacts for the trading project.
    Centralizes file path construction based on symbol, interval, and data type.
    Uses configuration from config/paths.py.
    Supports saving/loading DataFrames (to parquet) and model artifacts (using joblib).
    """
    
    # Mapping of data_type keys to directory path keys in PATHS for load_data/save_data
    _DIRECTORY_KEY_MAP = {
        'raw': 'raw_data_dir',
        'processed': 'processed_data_dir',
        'labeled': 'labeled_data_dir',
        'backtesting_results': 'backtesting_results_dir',
        'live_trading_results': 'live_trading_results_dir',
    }

    # Mapping of artifact_type keys to directory path keys in PATHS for load_model_artifact/save_model_artifact
    _ARTIFACT_DIRECTORY_KEY_MAP = {
        'pipeline': 'trained_models_dir',
        'metadata': 'trained_models_dir',
        'model': 'trained_models_dir', # For Keras models (.keras)
        'preprocessor': 'trained_models_dir', # For separate preprocessors (joblib)
        'evaluation': 'model_analysis_dir', # Added for model evaluation results
        'labeling_analysis': 'labeling_analysis_dir', # Added for labeling analysis results
        'model_analysis': 'model_analysis_dir', # Added for model analysis results (pkl) - Redundant but kept for clarity
        'backtesting_analysis': 'backtesting_analysis_dir', # Added for backtesting analysis results
        'live_trading_analysis': 'live_trading_analysis_dir', # Added for live trading analysis results
        'best_model_weights': 'trained_models_dir', # NEW: For best model weights
        'model_weights': 'trained_models_dir', # NEW: General for model weights
    }

    # Define which artifact types require a model_key subfolder
    ARTIFACT_TYPES_WITH_SUBFOLDER = ['pipeline', 'metadata', 'model', 'preprocessor',
                                     'evaluation', 'model_analysis', 'backtesting_analysis',
                                     'live_trading_analysis', 'best_model_weights', 'model_weights'] # Added new types here

    # Define the data_type key format for model-related artifacts when calling get_file_path
    # This maps the 'artifact_type' (e.g., 'best_model_weights') to the base 'data_type' in the filename
    _MODEL_ARTIFACT_DATA_TYPE_FORMAT = {
        'pipeline': 'model_pipeline',
        'metadata': 'model_metadata',
        'model': 'model', # Keras model uses 'model' as the base name
        'preprocessor': 'preprocessor', # Use 'preprocessor' as the base name
        'evaluation': 'evaluation', # Use 'evaluation' as the base name
        'model_analysis': 'model_analysis', # Use 'model_analysis' as the base name
        'backtesting_analysis': 'backtesting_analysis', # Use 'backtesting_analysis' as the base name
        'live_trading_analysis': 'live_trading_analysis', # Use 'live_trading_analysis' as the base name
        'labeling_analysis': 'labeling_analysis', # Use 'labeling_analysis' as the base name
        'best_model_weights': 'model', # NEW: Treat best weights as a 'model' artifact
        'model_weights': 'model', # NEW: Treat general weights as a 'model' artifact
    }


    def __init__(self):
        """Initializes DataManager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # These are kept as-is for backward compatibility with older scripts
        self.paths = PATHS
        # This is the new, preferred configuration object for refactored scripts
        self.path_config = PATH_CONFIG
        self._ensure_base_dirs()

    def _ensure_base_dirs(self):
        """Ensures that the fundamental directories defined in PATH_CONFIG exist."""
        for key, directory in self.path_config.get('directories', {}).items():
            if key.endswith('_base'): # Only create base directories
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.logger.error(f"Could not create base directory at {directory}: {e}", exc_info=True)

    # ==============================================================================
    # --- NEW, PREFERRED METHODS (For Labeling, Backtesting, MC, and future use) ---
    # ==============================================================================

    def _get_run_dir(self, base_dir_key: str, run_dir_pattern_key: str, **kwargs) -> Path:
        """Generic helper to create and get a unique directory for a specific run."""
        base_dir = Path(self.path_config['directories'][base_dir_key])
        run_dir_pattern = self.path_config['patterns'][run_dir_pattern_key]
        safe_kwargs = {k: str(v).replace('/', '_').replace(':', '_') for k, v in kwargs.items()}
        run_dir = base_dir / run_dir_pattern.format(**safe_kwargs)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def get_labeling_analysis_dir(self, symbol: str, interval: str) -> Path:
        """Gets the unique directory for a specific labeling analysis run."""
        return self._get_run_dir('labeling', 'labeling_run_dir', symbol=symbol, interval=interval)
    
    def get_backtesting_dir(self, model_type: str, symbol: str, interval: str) -> Path:
        """Gets the unique directory for a specific backtesting run."""
        return self._get_run_dir('backtesting', 'backtesting_run_dir', model_type=model_type, symbol=symbol, interval=interval)

    def get_monte_carlo_dir(self, model_type: str, symbol: str, interval: str, timestamp: str, mode: str, num_simulations: int) -> Path:
        """Gets the unique, timestamped directory for a Monte Carlo analysis run."""
        return self._get_run_dir(
            'monte_carlo', 'monte_carlo_run_dir',
            model_type=model_type, symbol=symbol, interval=interval,
            timestamp=timestamp, mode=mode, num_simulations=num_simulations
        )

    def save_dataframe(self, df: pd.DataFrame, data_type: str, **kwargs):
        """Saves a DataFrame (e.g., raw, processed, labeled) to a parquet file."""
        directory = Path(self.path_config['directories'][data_type])
        pattern = self.path_config['patterns'][f"{data_type}_data"]
        safe_kwargs = {k: str(v).replace('/', '_') for k, v in kwargs.items()}
        filename = pattern.format(**safe_kwargs)
        file_path = directory / filename
        self.logger.info(f"Saving '{data_type}' dataframe to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, index=True)

    def load_dataframe(self, data_type: str, **kwargs) -> Optional[pd.DataFrame]:
        """Loads a DataFrame (e.g., raw, processed, labeled) from a parquet file."""
        directory = Path(self.path_config['directories'][data_type])
        pattern = self.path_config['patterns'][f"{data_type}_data"]
        safe_kwargs = {k: str(v).replace('/', '_') for k, v in kwargs.items()}
        filename = pattern.format(**safe_kwargs)
        file_path = directory / filename
        if not file_path.exists():
            self.logger.warning(f"Dataframe file not found: {file_path}")
            return None
        self.logger.info(f"Loading '{data_type}' dataframe from: {file_path}")
        return pd.read_parquet(file_path)

    def save_analysis_plot(self, fig: Figure, run_dir: Path, plot_pattern_key: str, **kwargs):
        """Saves an analysis plot to a specified run directory using a specified pattern key."""
        filename = self.path_config['patterns'][plot_pattern_key].format(**kwargs)
        path = run_dir / filename
        self.logger.info(f"Saving plot with pattern '{plot_pattern_key}' to: {path}")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        
    def save_analysis_table(self, df: pd.DataFrame, run_dir: Path, table_pattern_key: str, **kwargs):
        """Saves an analysis table (DataFrame) to a specified run directory as a CSV."""
        filename = self.path_config['patterns'][table_pattern_key].format(**kwargs)
        path = run_dir / filename
        self.logger.info(f"Saving table with pattern '{table_pattern_key}' to: {path}")
        df.to_csv(path, index=True)

    def save_backtest_results(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame, metrics: Dict, model_type: str, symbol: str, interval: str):
        """Saves all artifacts from a single backtest run to its dedicated directory."""
        run_dir = self.get_backtesting_dir(model_type, symbol, interval)
        self.logger.info(f"Saving backtest results to directory: {run_dir}")
        trades_df.to_parquet(run_dir / self.path_config['patterns']['backtest_trades'])
        equity_df.to_parquet(run_dir / self.path_config['patterns']['backtest_equity'])
        sanitized_metrics = self._sanitize_for_json(metrics)
        with open(run_dir / self.path_config['patterns']['backtest_metrics_json'], 'w') as f:
            json.dump(sanitized_metrics, f, indent=4)
        pd.DataFrame([sanitized_metrics]).to_csv(run_dir / self.path_config['patterns']['backtest_metrics_csv'], index=False)
        self.logger.info("Saved trades, equity, and metrics for backtest run.")

    def _sanitize_for_json(self, data: Any) -> Any:
        if isinstance(data, dict): return {k: self._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list): return [self._sanitize_for_json(i) for i in data]
        if isinstance(data, np.integer): return int(data)
        if isinstance(data, np.floating): return float(data)
        if isinstance(data, np.ndarray): return data.tolist()
        return data
    
    
    # ==============================================================================
    # --- NEW METHODS (Refactored for Model Training) ---
    # ==============================================================================
    def get_model_dir(self, model_type: str, symbol: str, interval: str) -> Path:
        """Gets the unique directory for a specific model's trained artifacts."""
        return self._get_run_dir('models_base', 'model_run_dir', model_type=model_type, symbol=symbol, interval=interval)

    def get_model_analysis_dir(self, model_type: str, symbol: str, interval: str) -> Path:
        """Gets the unique directory for a specific model's analysis results."""
        return self._get_run_dir('model_analysis', 'model_run_dir', model_type=model_type, symbol=symbol, interval=interval)

    def save_evaluation_results(self, results: Dict, model_type: str, symbol: str, interval: str):
        analysis_dir = self.get_model_analysis_dir(model_type, symbol, interval)
        filename = self.path_config['patterns']['model_evaluation']
        path = analysis_dir / filename
        self.logger.info(f"Saving evaluation results to: {path}")
        with open(path, 'w') as f:
            json.dump(self._sanitize_for_json(results), f, indent=4)
            
    def save_feature_importance(self, df: pd.DataFrame, model_type: str, symbol: str, interval: str):
        analysis_dir = self.get_model_analysis_dir(model_type, symbol, interval)
        filename = self.path_config['patterns']['model_feature_importance']
        path = analysis_dir / filename
        self.logger.info(f"Saving feature importance table to: {path}")
        df.to_csv(path, index=False)

    def save_model_plot(self, fig: Figure, plot_type: str, model_type: str, symbol: str, interval: str):
        analysis_dir = self.get_model_analysis_dir(model_type, symbol, interval)
        filename = self.path_config['patterns']['model_plot'].format(plot_type=plot_type)
        path = analysis_dir / filename
        self.logger.info(f"Saving plot '{plot_type}' to: {path}")
        fig.savefig(path, dpi=150, bbox_inches='tight')


    # ==============================================================================
    # --- ORIGINAL METHODS (Unchanged for Backward Compatibility) ---
    # ==============================================================================
    
    def get_file_path(self, symbol: str, interval: str, data_type: str, name_suffix: str = '', model_key: Optional[str] = None) -> Path:
        """
        Constructs a standardized file path for data or model artifacts.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): Data interval (e.g., '1h', '5m').
            data_type (str): Type of data/artifact. This should be one of the *values*
                             in _MODEL_ARTIFACT_DATA_TYPE_FORMAT (e.g., 'model', 'model_metadata', 'evaluation')
                             or a key in _DIRECTORY_KEY_MAP (e.g., 'raw', 'processed').
            name_suffix (str): Optional suffix to add to the filename before the extension.
                                This is for additional descriptors, e.g., '_best_model_weights'.
            model_key (Optional[str]): Required for certain artifact types (those in ARTIFACT_TYPES_WITH_SUBFOLDER)
                                       to create a subfolder.

        Returns:
            Path: The constructed file path.

        Raises:
            ValueError: If data_type is unsupported or model_key is missing for a required type.
        """
        directory_key = None

        # Determine the directory key based on the provided data_type
        # First, check if it's a direct data type mapping (raw, processed, etc.)
        if data_type in self._DIRECTORY_KEY_MAP:
            directory_key = self._DIRECTORY_KEY_MAP.get(data_type)
        # Next, check if it's one of the *values* used in _MODEL_ARTIFACT_DATA_TYPE_FORMAT
        # (e.g., 'model', 'model_metadata'). We need to find the corresponding *key*
        # in _ARTIFACT_DIRECTORY_KEY_MAP via a reverse lookup.
        elif data_type in self._MODEL_ARTIFACT_DATA_TYPE_FORMAT.values():
            # Find the artifact_type (key in _MODEL_ARTIFACT_DATA_TYPE_FORMAT)
            # that maps to this data_type value.
            # Note: This assumes uniqueness of values in _MODEL_ARTIFACT_DATA_TYPE_FORMAT for this reverse lookup.
            # If multiple artifact_types map to the same data_type, this picks the first.
            # For our current use case, this is fine as they should map to the same directory.
            artifact_type_from_data_type = next((k for k, v in self._MODEL_ARTIFACT_DATA_TYPE_FORMAT.items() if v == data_type), None)
            if artifact_type_from_data_type and artifact_type_from_data_type in self._ARTIFACT_DIRECTORY_KEY_MAP:
                directory_key = self._ARTIFACT_DIRECTORY_KEY_MAP.get(artifact_type_from_data_type)
            else:
                raise ValueError(f"Internal error: Could not map data_type '{data_type}' to a valid artifact type directory.")
        else:
            all_supported_keys = list(self._DIRECTORY_KEY_MAP.keys()) + list(self._MODEL_ARTIFACT_DATA_TYPE_FORMAT.values())
            raise ValueError(f"Unsupported data_type: '{data_type}'. Supported types: {all_supported_keys}")


        base_dir = self.paths.get(directory_key)
        if not base_dir:
            raise ValueError(f"Directory path not configured for data_type: '{data_type}' (key: '{directory_key}')")

        # Ensure base_dir is a Path object
        if not isinstance(base_dir, Path):
             base_dir = Path(base_dir)
             self.paths[directory_key] = base_dir # Update in case it was a string


        # Determine if a model_key subfolder is required based on the original artifact type
        # We need to consider *which* artifact type this data_type corresponds to.
        # Check all artifact_types that map to this data_type in _MODEL_ARTIFACT_DATA_TYPE_FORMAT
        artifact_types_for_data_type = [k for k, v in self._MODEL_ARTIFACT_DATA_TYPE_FORMAT.items() if v == data_type]
        
        needs_subfolder = any(at in self.ARTIFACT_TYPES_WITH_SUBFOLDER for at in artifact_types_for_data_type)


        if needs_subfolder:
            if model_key is None:
                raise ValueError(f"model_key is required for data_type '{data_type}' as it corresponds to a subfoldered artifact type.")

            # Ensure the model_key is lowercase and safe for filenames/folders
            model_key_safe = model_key.lower().replace(' ', '_')
            base_dir = base_dir / model_key_safe
            # The directory will be created when saving, but we need the path here.


        # Construct filename
        safe_symbol = symbol.replace('/', '_').upper()
        safe_interval = interval.replace(':', '_')

        extension = '.pkl' # Default for artifacts and general objects
        if data_type in [self._MODEL_ARTIFACT_DATA_TYPE_FORMAT['model']]: # Specifically check for 'model' as the base type for .keras
             extension = '.keras'
        elif data_type in self._DIRECTORY_KEY_MAP.keys(): # Raw, processed, labeled, results
             extension = '.parquet'
        # All other 'data_type' values from _MODEL_ARTIFACT_DATA_TYPE_FORMAT default to .pkl


        # Construct the filename pattern: SYMBOL_INTERVAL_datatype_suffix.extension
        # The data_type itself (e.g., 'model', 'model_metadata') is part of the filename
        # If name_suffix is provided, it's appended directly
        filename_core = f"{safe_symbol}_{safe_interval}_{data_type}"
        if name_suffix:
            filename_core += name_suffix

        filename = f"{filename_core}{extension}"


        # Ensure the filename is safe (remove any remaining invalid characters if necessary)
        filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '.', '-'])

        return base_dir / filename


    def load_data(self, symbol: str, interval: str, data_type: str, name_suffix: str = '') -> Optional[pd.DataFrame]:
        """
        Loads data (DataFrame) from a file using the constructed path.
        Assumes data is saved in parquet format.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            data_type (str): Type of data ('raw', 'processed', 'labeled', 'backtesting_results', 'live_trading_results').
                             Must be a key in _DIRECTORY_KEY_MAP.
            name_suffix (str): Optional suffix to add to the filename before the extension.
                                Used for labeled data to specify the strategy.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame, or None if file not found or error occurs.
        """
        # Ensure data_type is one that should be loaded as a DataFrame
        if data_type not in self._DIRECTORY_KEY_MAP.keys():
             self.logger.error(f"Unsupported data_type '{data_type}' for load_data (DataFrame) method. Supported types: {list(self._DIRECTORY_KEY_MAP.keys())}")
             return None # Or raise ValueError

        try:
            # Use get_file_path with the appropriate data_type and name_suffix
            file_path = self.get_file_path(symbol=symbol, interval=interval, data_type=data_type, name_suffix=name_suffix)
            self.logger.info(f"Attempting to load data (DataFrame) from {file_path}")

            if not file_path.exists():
                self.logger.warning(f"Data file not found: {file_path}")
                return None

            # Always read parquet without assuming an index initially
            df = pd.read_parquet(file_path)
            self.logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")

            # --- Explicitly handle index setting based on data_type ---
            if data_type == 'raw':
                if 'open_time' in df.columns:
                    try:
                        # Convert 'open_time' to datetime, ensuring UTC
                        df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
                        # Set 'open_time' as the index, dropping the original column
                        df = df.set_index('open_time', drop=True)
                        # Ensure the index name is 'timestamp' for consistency
                        df.index.name = 'timestamp'
                        self.logger.info(f"Successfully set 'open_time' as DatetimeIndex and named it 'timestamp' for raw data.")
                    except Exception as e:
                        self.logger.error(f"Failed to convert 'open_time' to DatetimeIndex for raw data: {e}", exc_info=True)
                        raise ValueError(f"Failed to set correct DatetimeIndex for raw data from 'open_time' column.")
                else:
                    self.logger.error(f"Raw data for '{symbol}_{interval}' does not contain an 'open_time' column. Cannot set proper DatetimeIndex.")
                    raise ValueError(f"Missing 'open_time' column in raw data for '{symbol}_{interval}'.")
            else: # For 'processed', 'labeled', 'backtesting_results', 'live_trading_results'
                # For other data types, ensure the index is a DatetimeIndex and named 'timestamp'
                if not isinstance(df.index, pd.DatetimeIndex):
                    self.logger.warning(f"Loaded data for '{data_type}' does not have a DatetimeIndex. Attempting conversion.")
                    try:
                        # If the index is not a DatetimeIndex, try to convert it.
                        # This assumes the index *itself* contains the timestamp information.
                        df.index = pd.to_datetime(df.index, utc=True)
                        df.index.name = 'timestamp'
                        self.logger.info(f"Successfully converted existing index to DatetimeIndex and named it 'timestamp' for '{data_type}'.")
                    except Exception as e:
                        self.logger.error(f"Failed to convert existing index to DatetimeIndex for '{data_type}': {e}", exc_info=True)
                        raise ValueError(f"Failed to set correct DatetimeIndex for '{data_type}' data from existing index.")
                elif df.index.name != 'timestamp':
                    # If it's already a DatetimeIndex but not named 'timestamp', rename it
                    df.index.name = 'timestamp'
                    self.logger.info(f"Existing DatetimeIndex renamed to 'timestamp' for '{data_type}'.")

            return df
        except FileNotFoundError:
            self.logger.warning(f"Data file not found during load_data: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
            # Catch and re-raise any other exceptions during loading
            raise Exception(f"An unexpected error occurred during data loading from {file_path}: {e}")


    def save_data(self, df_to_save: Union[pd.DataFrame, pd.Series], symbol: str, interval: str, data_type: str, name_suffix: str = ''):
        """
        Saves data (DataFrame or Series) to a file using the constructed path.
        Assumes data is saved to parquet.

        Args:
            df_to_save (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to save.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            data_type (str): Type of data ('raw', 'processed', 'labeled', 'backtesting_results', 'live_trading_results').
                             Must be a key in _DIRECTORY_KEY_MAP.
            name_suffix (str): Optional suffix to add to the filename.

        Raises:
            ValueError: If data_type is unsupported for this method or input type is incorrect.
            OSError: If there's an error saving the file.
        """
        # Ensure data_type is one that this method is intended for (DataFrame/Series to parquet)
        if data_type not in self._DIRECTORY_KEY_MAP.keys():
             self.logger.error(f"Unsupported data_type '{data_type}' for save_data (DataFrame/Series) method. Supported types: {list(self._DIRECTORY_KEY_MAP.keys())}")
             raise ValueError(f"Unsupported data_type '{data_type}' for save_data (DataFrame/Series) method.")

        # Ensure the input is a DataFrame or Series
        if not isinstance(df_to_save, (pd.DataFrame, pd.Series)):
             self.logger.error(f"Input for save_data must be a pandas DataFrame or Series, but got {type(df_to_save)}.")
             raise TypeError("Input for save_data must be a pandas DataFrame or Series.")


        try:
            # Use get_file_path with the appropriate data_type (will have .parquet extension)
            file_path = self.get_file_path(symbol=symbol, interval=interval, data_type=data_type, name_suffix=name_suffix)

            self.logger.info(f"Attempting to save data to {file_path}")

            # Ensure the parent directory exists before saving
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to parquet, explicitly saving the index
            df_to_save.to_parquet(file_path, index=True)
            self.logger.info(f"Successfully saved data to {file_path}.")

        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}", exc_info=True)
            raise OSError(f"Failed to save data to {file_path}: {e}")


    def load_model_artifact(self, symbol: str, interval: str, model_key: str, artifact_type: str) -> Any:
        """
        Loads a model artifact (pipeline, metadata, Keras model, preprocessor, analysis results)
        using joblib.load or tf.keras.models.load_model.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            model_key (str): Key for the model configuration.
            artifact_type (str): Type of artifact ('pipeline', 'metadata', 'model',
                                 'preprocessor', 'evaluation', 'labeling_analysis',
                                 'model_analysis', 'backtesting_analysis', 'live_trading_analysis',
                                 'best_model_weights', 'model_weights').
                                 Must be a key in _ARTIFACT_DIRECTORY_KEY_MAP.

        Returns:
            Any: The loaded artifact.

        Raises:
            ValueError: If artifact_type is invalid or path construction fails.
            FileNotFoundError: If the artifact file is not found.
            Exception: For errors during loading.
        """
        # Ensure artifact_type is supported for this method
        if artifact_type not in self._ARTIFACT_DIRECTORY_KEY_MAP.keys():
             logger.error(f"Invalid artifact_type '{artifact_type}'. Must be one of: {list(self._ARTIFACT_DIRECTORY_KEY_MAP.keys())}")
             raise ValueError(f"Invalid artifact_type '{artifact_type}'. Must be one of: {list(self._ARTIFACT_DIRECTORY_KEY_MAP.keys())}")

        # --- CORRECTED: Use the formatted data_type key for get_file_path ---
        data_type_for_path = self._MODEL_ARTIFACT_DATA_TYPE_FORMAT.get(artifact_type)
        if not data_type_for_path:
             # This should not happen if _ARTIFACT_DIRECTORY_KEY_MAP and _MODEL_ARTIFACT_DATA_TYPE_FORMAT are aligned
             logger.error(f"Internal error: Could not find formatted data_type for artifact_type '{artifact_type}'.")
             raise ValueError(f"Internal error: Unsupported artifact_type '{artifact_type}'.")

        # --- Construct and pass the name_suffix which will include model_key and artifact_type for clarity ---
        # This must match how save_model_artifact constructs the suffix
        name_suffix = f'_{model_key}_{artifact_type}'


        file_path = self.get_file_path(symbol=symbol, interval=interval, data_type=data_type_for_path,
                                       name_suffix=name_suffix, # Pass the constructed suffix
                                       model_key=model_key) # model_key is handled by get_file_path now

        self.logger.info(f"Attempting to load model {artifact_type} from {file_path}")

        if not file_path.exists():
            self.logger.error(f"Model artifact file not found: {file_path}")
            raise FileNotFoundError(f"Model artifact file not found: {file_path}")

        try:
            # Check if it's a Keras model file based on extension
            if file_path.suffix == '.keras':
                 if not TF_AVAILABLE:
                      raise ImportError("TensorFlow is required to load Keras model but is not installed.")
                 # Now it's safe to call load_model because TF_AVAILABLE is True
                 artifact = tf.keras.models.load_model(file_path) # type: ignore
                 self.logger.info(f"Successfully loaded Keras model from {file_path}.")
            else:
                 # Load using joblib for other artifact types (.pkl)
                 artifact = joblib.load(file_path)
                 self.logger.info(f"Successfully loaded model {artifact_type} from {file_path}.")

            return artifact

        except FileNotFoundError:
            # Re-raise FileNotFoundError as it's specific and handled by caller
            raise
        except ImportError as e:
             self.logger.error(f"Import error loading artifact from {file_path}: {e}", exc_info=True)
             raise # Re-raise import errors
        except Exception as e:
            self.logger.error(f"Error loading model {artifact_type} from {file_path}: {e}", exc_info=True)
            raise Exception(f"Failed to load model {artifact_type} from {file_path}: {e}")


    def save_model_artifact(self, artifact: Any, symbol: str, interval: str, model_key: str, artifact_type: str):
        """
        Saves a model artifact (pipeline, metadata, Keras model, preprocessor, analysis results)
        using joblib.dump or model.save.

        Args:
            artifact (Any): The artifact to save.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            model_key (str): Key for the model configuration.
            artifact_type (str): Type of artifact ('pipeline', 'metadata', 'model',
                                 'preprocessor', 'evaluation', 'labeling_analysis',
                                 'model_analysis', 'backtesting_analysis', 'live_trading_analysis',
                                 'best_model_weights', 'model_weights').
                                 Must be a key in _ARTIFACT_DIRECTORY_KEY_MAP.

        Raises:
            ValueError: If artifact_type is invalid or path construction fails.
            OSError: If there's an error saving the file.
            Exception: For other errors during saving.
        """
        # Ensure artifact_type is supported for this method
        if artifact_type not in self._ARTIFACT_DIRECTORY_KEY_MAP.keys():
             logger.error(f"Invalid artifact_type '{artifact_type}'. Must be one of: {list(self._ARTIFACT_DIRECTORY_KEY_MAP.keys())}")
             raise ValueError(f"Invalid artifact_type '{artifact_type}'. Must be one of: {list(self._ARTIFACT_DIRECTORY_KEY_MAP.keys())}")

        # --- CORRECTED: Use the formatted data_type key for get_file_path ---
        data_type_for_path = self._MODEL_ARTIFACT_DATA_TYPE_FORMAT.get(artifact_type)
        if not data_type_for_path:
             # This should not happen if _ARTIFACT_DIRECTORY_KEY_MAP and _MODEL_ARTIFACT_DATA_TYPE_FORMAT are aligned
             logger.error(f"Internal error: Could not find formatted data_type for artifact_type '{artifact_type}'.")
             raise ValueError(f"Internal error: Unsupported artifact_type '{artifact_type}'.")

        # --- Construct and pass the name_suffix which will include model_key and artifact_type for clarity ---
        # This must match how load_model_artifact will construct the suffix
        name_suffix = f'_{model_key}_{artifact_type}'


        file_path = self.get_file_path(symbol=symbol, interval=interval, data_type=data_type_for_path,
                                       name_suffix=name_suffix, # Pass the constructed suffix
                                       model_key=model_key) # model_key is handled by get_file_path now

        self.logger.info(f"Attempting to save model {artifact_type} to {file_path}")

        # Ensure the parent directory (including the model_key subfolder if applicable) exists before saving
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Check if the artifact has a 'save' method (like Keras models)
            if hasattr(artifact, 'save') and file_path.suffix == '.keras':
                 # Save Keras model
                 # Check if TensorFlow is available before attempting to save a Keras model
                 if not TF_AVAILABLE:
                      raise ImportError("TensorFlow is required to save Keras model but is not installed.")
                 artifact.save(file_path)
                 self.logger.info(f"Successfully saved Keras model to {file_path}.")
            else:
                 # Save using joblib for other artifact types (.pkl)
                 joblib.dump(artifact, file_path)
                 self.logger.info(f"Successfully saved model {artifact_type} to {file_path}.")

        except Exception as e:
            self.logger.error(f"Error saving model {artifact_type} to {file_path}: {e}", exc_info=True)
            raise OSError(f"Failed to save model {artifact_type} to {file_path}: {e}")

