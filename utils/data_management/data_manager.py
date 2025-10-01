# utils/data_manager.py
import datetime
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
import dataclasses # Import dataclasses for the helper function
import sqlite3 # Add this database module for transactional state management

# Set up logging for the data manager
logger = logging.getLogger(__name__)

# Import paths configuration
try:
    from config.params import AppConfig
    from config.paths import PATH_CONFIG
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
    
def _dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively converts a dataclass object to a dictionary, ready for YAML/JSON serialization.
    Handles nested dataclasses, lists of dataclasses, and dicts of dataclasses.
    """
    if dataclasses.is_dataclass(obj):
        # For dataclasses, convert to dict and recurse on values
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    elif isinstance(obj, list):
        # For lists, recurse on each item
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        # For dicts, recurse on each value
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        # For all other types (int, str, float, bool, etc.), return as is
        return obj


class StateDBManager:
    """
    Manages the bot's state persistence in a transactional SQLite database.
    This replaces the file-based state saving to ensure atomicity.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_db_and_table_exist()
        
        
    def _ensure_db_and_table_exist(self):
        """Connects to the database and creates the state table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    state_data TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"Database for session {self.db_path} ensured to be present.")
        except sqlite3.Error as e:
            logger.critical(f"Database error during setup: {e}")
            raise

    def save_bot_state(self, state: Dict[str, Any]):
        """
        Saves the bot's entire state to the database, overwriting any previous state.
        This is a single, atomic transaction.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            state_json = json.dumps(state)

            conn.execute("BEGIN TRANSACTION")
            cursor.execute("DELETE FROM bot_state")
            cursor.execute("INSERT INTO bot_state (id, state_data) VALUES (?, ?)", (1, state_json))
            conn.commit()
            logger.info("Bot state successfully saved to the database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to save bot state to database. Rolling back transaction: {e}")
            conn.rollback()
        finally:
            conn.close()

    def load_bot_state(self) -> Dict[str, Any]:
        """
        Loads the bot's state from the database. Returns an empty dict if no state is found.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT state_data FROM bot_state WHERE id = 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                state_json = result[0]
                return json.loads(state_json)
            else:
                logger.warning(f"No previous bot state found at {self.db_path}. Starting from a clean slate.")
                return {}
        except sqlite3.Error as e:
            logger.error(f"Failed to load bot state from database: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode bot state from database: {e}")
            return {}

class DataManager:
    """
    Manages loading and saving of data files and model artifacts for the trading project.
    Centralizes file path construction based on symbol, interval, and data type.
    Uses configuration from config/paths.py.
    Supports saving/loading DataFrames (to parquet) and model artifacts (using joblib).
    """

    def __init__(self):
        """Initializes DataManager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
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

    def get_monte_carlo_dir(self, model_type: str, symbol: str, interval: str, mode: str, num_simulations: int) -> Path:
        """Gets the unique directory for a Monte Carlo analysis run."""
        # FIXED: Removed timestamp from arguments as it's no longer used in the path pattern
        return self._get_run_dir(
            'monte_carlo', 'monte_carlo_run_dir',
            model_type=model_type, symbol=symbol, interval=interval,
            mode=mode, num_simulations=num_simulations
        )

    def get_live_trading_dir(self, model_type: str, symbol: str, interval: str) -> Path:
        """Gets the unique directory for a specific live trading run."""
        return self._get_run_dir('live_trading', 'live_trading_run_dir', model_type=model_type, symbol=symbol, interval=interval)

    def load_dataframe(self, data_type: str, **kwargs) -> Optional[pd.DataFrame]:
        # Path logic (modern)
        directory = Path(self.path_config['directories'][data_type])
        pattern = self.path_config['patterns'][f"{data_type}_data"]
        safe_kwargs = {k: str(v).replace('/', '_') for k, v in kwargs.items()}
        filename = pattern.format(**safe_kwargs)
        file_path = directory / filename
        if not file_path.exists():
            self.logger.warning(f"Dataframe file not found: {file_path}")
            return None
        self.logger.info(f"Loading '{data_type}' dataframe from: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"Index name: {df.index.name}")

        # Robust index handling
        if data_type == 'raw':
            # Case 1: index is 'open_time'
            if df.index.name == 'open_time':
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = 'timestamp' # changed from 'open_time'
            # Case 2: index is not 'open_time', but column exists
            elif 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
                df = df.set_index('open_time', drop=True)
                df.index.name = 'timestamp' # changed from 'open_time'
            # Case 3: index is not 'open_time' and column doesn't exist
            else:
                self.logger.error("Raw data does not have 'open_time' index or column!")
                raise ValueError("Missing 'open_time' index or column in raw data.")
        else:
            # For other data types: ensure index is DatetimeIndex and name is 'timestamp'
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            if df.index.name != 'timestamp':
                df.index.name = 'timestamp'
        return df

    def save_dataframe(self, df: pd.DataFrame, data_type: str, **kwargs):
        directory = Path(self.path_config['directories'][data_type])
        pattern = self.path_config['patterns'][f"{data_type}_data"]
        safe_kwargs = {k: str(v).replace('/', '_') for k, v in kwargs.items()}
        filename = pattern.format(**safe_kwargs)
        file_path = directory / filename
        self.logger.info(f"Saving '{data_type}' dataframe to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, index=True)

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
        
        # Create a copy to avoid modifying the original DataFrame in memory
        trades_to_save = trades_df.copy()

        # FIX: Convert model_probabilities column to JSON string before saving
        if 'model_probabilities' in trades_to_save.columns:
            self.logger.info("Converting 'model_probabilities' column to JSON strings for Parquet compatibility.")
            # Use a safe conversion that handles dicts, NaNs, and other types
            trades_to_save['model_probabilities'] = trades_to_save['model_probabilities'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

        trades_to_save.to_parquet(run_dir / self.path_config['patterns']['backtest_trades'])
        equity_df.to_parquet(run_dir / self.path_config['patterns']['backtest_equity'])
        
        sanitized_metrics = self._sanitize_for_json(metrics)
        with open(run_dir / self.path_config['patterns']['backtest_metrics_json'], 'w') as f:
            json.dump(sanitized_metrics, f, indent=4)
        pd.DataFrame([sanitized_metrics]).to_csv(run_dir / self.path_config['patterns']['backtest_metrics_csv'], index=False)
        self.logger.info("Saved trades, equity, and metrics for backtest run.")

    def save_backtest_artifacts(self, model_type: str, symbol: str, interval: str, trades_df: pd.DataFrame,
                                equity_df: pd.DataFrame, metrics_dict: Dict, plots_dict: Dict, app_config: AppConfig):
        """
        Saves all artifacts from a single backtest run, including plots and config.
        This is the new orchestrator-friendly method.
        """
        run_dir = self.get_backtesting_dir(model_type, symbol, interval)
        self.logger.info(f"Saving all backtest artifacts to directory: {run_dir}")

        # 1. Save core results (trades, equity, metrics) using existing method
        self.save_backtest_results(trades_df, equity_df, metrics_dict, model_type, symbol, interval)

        # 2. Save Plots
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        for plot_name, fig in plots_dict.items():
            try:
                # Using a generic plot pattern key from paths.py
                self.save_analysis_plot(fig, plots_dir, "backtest_plot", plot_type=plot_name)
            except Exception as e:
                self.logger.error(f"Failed to save plot '{plot_name}': {e}", exc_info=True)
            finally:
                # Ensure figure is closed to free memory, even if saving fails
                import matplotlib.pyplot as plt
                plt.close(fig)

        # 3. Save Config for reproducibility
        try:
            import yaml
            # FIXED: Use the new robust helper function instead of dataclasses.asdict
            config_dict = _dataclass_to_dict(app_config)
            config_path = run_dir / "config_used.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Saved run configuration to {config_path}")
        except ImportError:
            self.logger.warning("`PyYAML` is not installed. Skipping config save. To install: pip install PyYAML")
        except Exception as e:
            self.logger.error(f"Failed to save config.yaml: {e}", exc_info=True)

    def save_monte_carlo_artifacts(self, model_type: str, symbol: str, interval: str, mode: str, num_simulations: int,
                                   summary_tables: Dict[str, pd.DataFrame], plots_dict: Dict[str, Figure], app_config: AppConfig):
        """
        Saves all artifacts from a Monte Carlo analysis run to a unique directory.
        """
        # FIXED: Removed timestamp generation as it's no longer used in the path.
        run_dir = self.get_monte_carlo_dir(model_type, symbol, interval, mode, num_simulations)
        self.logger.info(f"Saving all Monte Carlo artifacts to directory: {run_dir}")

        # 1. Save summary tables
        table_pattern_map = {
            'all_simulation_metrics': 'mc_raw_metrics',
            'summary_stats': 'mc_summary_stats'
        }
        for table_name, df in summary_tables.items():
            if not df.empty:
                pattern_key = table_pattern_map.get(table_name)
                if pattern_key:
                    self.save_analysis_table(df, run_dir, pattern_key)
                else:
                    self.logger.warning(f"No pattern defined for saving table '{table_name}'. Skipping.")

        # 2. Save Plots
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        for plot_name, fig in plots_dict.items():
            try:
                self.save_analysis_plot(fig, plots_dir, "mc_plot", plot_type=plot_name)
            except Exception as e:
                self.logger.error(f"Failed to save plot '{plot_name}': {e}", exc_info=True)
            finally:
                import matplotlib.pyplot as plt
                plt.close(fig)

        # 3. Save Config for reproducibility
        try:
            import yaml
            config_dict = _dataclass_to_dict(app_config)
            config_path = run_dir / "config_used.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Saved run configuration to {config_path}")
        except ImportError:
            self.logger.warning("`PyYAML` is not installed. Skipping config save. To install: pip install PyYAML")
        except Exception as e:
            self.logger.error(f"Failed to save config.yaml: {e}", exc_info=True)

    def save_bot_state(self, state: Dict[str, Any], model_type: str, symbol: str, interval: str):
        """Saves the bot's current state to a transactional SQLite database."""
        run_dir = self.get_live_trading_dir(model_type, symbol, interval)
        filename = self.path_config['patterns']['live_trading_state']
        file_path = run_dir / filename
        
        self.logger.info(f"Saving bot state to: {file_path}")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Instantiate the new transactional manager and save the state
            state_db_manager = StateDBManager(file_path)
            sanitized_state = self._sanitize_for_json(state)
            state_db_manager.save_bot_state(sanitized_state)
            self.logger.info("Successfully saved bot state.")
        except Exception as e:
            self.logger.error(f"Failed to save bot state to {file_path}: {e}", exc_info=True)
            raise OSError(f"Failed to save bot state to {file_path}: {e}")

    def load_bot_state(self, model_type: str, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Loads the bot's state from a transactional SQLite database."""
        run_dir = self.get_live_trading_dir(model_type, symbol, interval)
        filename = self.path_config['patterns']['live_trading_state']
        file_path = run_dir / filename
        
        if not file_path.exists():
            self.logger.warning(f"Bot state file not found at: {file_path}. Will start with initial capital.")
            return None
        
        self.logger.info(f"Loading bot state from: {file_path}")
        try:
            # Instantiate the new transactional manager and load the state
            state_db_manager = StateDBManager(file_path)
            state = state_db_manager.load_bot_state()
            self.logger.info("Successfully loaded bot state.")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load bot state from {file_path}: {e}", exc_info=True)
            return None
            
    def _sanitize_for_json(self, data: Any) -> Any:
        if isinstance(data, dict): return {k: self._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list): return [self._sanitize_for_json(i) for i in data]
        if isinstance(data, np.integer): return int(data)
        if isinstance(data, np.floating): return float(data)
        if isinstance(data, np.ndarray): return data.tolist()
        if isinstance(data, pd.Timestamp): return data.isoformat()
        if isinstance(data, datetime.datetime): return data.isoformat()
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
        
    # =========================================================================
    # --- New: Configuration Management ---
    # =========================================================================
    def save_app_config(self, config_data: Dict[str, Any], model_type: str, symbol: str, interval: str) -> Path:
        """
        Saves the application configuration dictionary to a JSON file
        in the live trading run directory.

        Args:
            config_data: The dictionary containing the configuration.
            model_type (str): The type of model (e.g., 'random_forest').
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '5m').
        
        Returns:
            The full Path object of the saved file.
        """
        run_dir = self.get_live_trading_dir(model_type, symbol, interval)
        filename_pattern = self.path_config['patterns']['bot_config_file']
        filename = filename_pattern.format(symbol=symbol, interval=interval)
        file_path = run_dir / filename
        
        self.logger.info(f"Saving user configuration to: {file_path}")
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.logger.info(f"Successfully saved configuration file.")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save configuration file to {file_path}: {e}", exc_info=True)
            raise


