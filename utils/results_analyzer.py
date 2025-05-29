# utils/results_analyser.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json # Import json for loading metrics saved by backtester


# Assume config.paths is available
try:
    from config.paths import PATHS
except ImportError:
    print("Error: config/paths.py not found. Please ensure it exists.")
    # Define dummy paths for basic functionality if paths.py is missing
    # These fallbacks should match the expected keys and structure from paths.py
    PATHS = {
        "backtesting_results_dir": Path("./results/backtesting"),
        "live_trading_results_dir": Path("./results/live_trading"),
        "analysis_dir": Path("./results/analysis"), # Base analysis directory
        "backtesting_analysis_dir": Path("./results/analysis/backtesting"), # Corrected key
        "live_trading_analysis_dir": Path("./results/analysis/live_trading"), # Corrected key
        # CORRECTED FILE PATTERN KEYS - ENSURED THESE MATCH THE `paths.py` FILE
        "backtesting_trades_pattern": "{symbol}_{interval}_{model_type}_trades.parquet",
        "backtesting_equity_pattern": "{symbol}_{interval}_{model_type}_equity.parquet",
        "backtesting_metrics_pattern": "{symbol}_{interval}_{model_type}_metrics.json",
        "live_trading_trades_pattern": "{symbol}_{interval}_{model_type}_trades.parquet", # Assuming parquet for consistency, but bot saves CSV
        "live_trading_equity_pattern": "{symbol}_{interval}_{model_type}_equity.parquet", # This file might not exist for live results
        "live_trading_capital_state_pattern": "{symbol}_{interval}_{model_type}_capital_state.json", # Added key for capital state
        # Simplified analysis file patterns
        "analysis_plot_pattern": "{symbol}_{interval}_{model_type}_{analysis_type}.png",
        "analysis_table_pattern": "{symbol}_{interval}_{model_type}_{analysis_type}.csv",
        "logs_dir": Path("./logs"),
    }
    # Ensure default directories exist for fallback
    for key, path in PATHS.items():
        if isinstance(path, Path) and any(dir_suffix in key for dir_suffix in ["_dir", "Dir"]):
             try: path.mkdir(parents=True, exist_ok=True)
             except Exception: pass # Ignore errors if path creation fails in fallback


# Set up logger for this module
logger = logging.getLogger(__name__)

# Define a small epsilon for float comparisons
FLOAT_EPSILON = 1e-6

class ResultsAnalyser:
    """
    Analyzes backtesting or live trading results (trades and equity curve).
    Calculates core performance metrics and generates essential plots.
    For live results, can calculate equity curve from trade history if equity file is missing.
    """

    def __init__(self,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 results_type: str = "backtest", # "backtest" or "live"
                 results_dir: Optional[Path] = None, # Optional override for base results directory
                 analysis_dir: Optional[Path] = None, # Optional override for analysis save directory
                 paths: Dict[str, Any] = PATHS):
        """
        Initializes the ResultsAnalyser.

        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            interval (str): Data interval (e.g., '5m').
            model_type (str): Identifier for the model type (e.g., 'xgboost').
            results_type (str): Type of results to analyze ('backtest' or 'live'). Defaults to 'backtest'.
            results_dir (Optional[Path]): Specific directory containing the results files.
                                           If None, determined from paths config based on results_type.
            analysis_dir (Optional[Path]): Specific directory to save analysis output.
                                           If None, determined from paths config based on results_type.
            paths (Dict[str, Any]): Dictionary of paths configuration. Defaults to PATHS from config.paths.
        """
        logger.info(f"Initializing ResultsAnalyser for {symbol} {interval} ({model_type}) {results_type} results...")
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.results_type = results_type.lower()
        self.paths = paths # Store the provided paths dictionary

        # --- Determine Base Results Directory ---
        if results_dir:
            self.base_results_dir = Path(results_dir)
            logger.info(f"Using specified base results directory: {self.base_results_dir}")
        else:
            if self.results_type == 'backtest':
                results_key = "backtesting_results_dir"
            elif self.results_type == 'live':
                results_key = "live_trading_results_dir"
            else:
                raise ValueError(f"Invalid results_type: {self.results_type}. Must be 'backtest' or 'live'.")

            self.base_results_dir = self.paths.get(results_key)
            if not self.base_results_dir:
                raise ValueError(f"Results directory key '{results_key}' not found in paths configuration.")
            self.base_results_dir = Path(self.base_results_dir)
            logger.info(f"Using configured base results directory: {self.base_results_dir}")


        # --- Determine Analysis Save Directory ---
        if analysis_dir:
            self.analysis_save_dir = Path(analysis_dir)
            logger.info(f"Using specified analysis save directory: {self.analysis_save_dir}")
        else:
            if self.results_type == 'backtest':
                # Use the correct key 'backtesting_analysis_dir' from paths.py
                analysis_key = "backtesting_analysis_dir"
            elif self.results_type == 'live':
                # Use the correct key 'live_trading_analysis_dir' from paths.py
                analysis_key = "live_trading_analysis_dir"
            else:
                 raise ValueError(f"Invalid results_type: {self.results_type}. Cannot determine analysis directory.")

            # Get the base analysis directory from paths
            base_analysis_dir = self.paths.get(analysis_key)
            if not base_analysis_dir:
                 base_analysis_dir = self.paths.get("analysis_dir", Path("./results/analysis"))
                 logger.warning(f"Specific analysis directory key '{analysis_key}' not found. Using general analysis directory: {base_analysis_dir}")

            # Make it model-specific
            self.analysis_save_dir = Path(base_analysis_dir) / self.model_type # ADDED / self.model_type
            logger.info(f"Using configured analysis save directory: {self.analysis_save_dir}")


        # Ensure analysis directory exists
        try:
            self.analysis_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Analysis results will be saved to: {self.analysis_save_dir}")
        except Exception as e:
             logger.error(f"Failed to create analysis directory {self.analysis_save_dir}: {e}", exc_info=True)
             raise IOError(f"Could not create analysis directory: {self.analysis_save_dir}") from e


        # --- Determine File Paths using Patterns ---
        if self.results_type == 'backtest':
            # FIX: Use the correct, pluralized key names from paths.py
            trades_file_key = "backtesting_trades_pattern"
            equity_file_key = "backtesting_equity_pattern"
            metrics_file_key = "backtesting_metrics_pattern"
            capital_state_key = None # No separate capital state file for backtest
        elif self.results_type == 'live':
            trades_file_key = "live_trading_trades_pattern"
            equity_file_key = "live_trading_equity_pattern"
            metrics_file_key = "live_trading_metrics_pattern" # Assuming a similar pattern exists
            capital_state_key = "live_trading_capital_state_pattern" # Key for bot's capital state

        else:
             raise ValueError(f"Invalid results_type: {self.results_type}. Cannot determine file patterns.")


        try:
            # Retrieve the patterns using the keys
            trades_file_pattern = self.paths.get(trades_file_key)
            equity_file_pattern = self.paths.get(equity_file_key)
            metrics_file_pattern = self.paths.get(metrics_file_key)
            capital_state_pattern = self.paths.get(capital_state_key) if capital_state_key else None


            if not trades_file_pattern:
                 raise ValueError(f"Missing trades file pattern key '{trades_file_key}' in paths configuration.")

            # Equity file pattern is optional for live results if we calculate from trades
            if self.results_type == 'backtest' and not equity_file_pattern:
                 raise ValueError(f"Missing equity file pattern key '{equity_file_key}' in paths configuration for backtest results.")
            # For live, equity_file_pattern can be None if the file doesn't exist

            # Format the file paths using the retrieved patterns
            # Ensure symbol and interval are safe for filenames
            safe_symbol = self.symbol.replace('/', '')
            safe_interval = self.interval.replace(':', '_')

            self.trades_file_path = self.base_results_dir / trades_file_pattern.format(
                symbol=safe_symbol, interval=safe_interval, model_type=self.model_type)

            # Equity file path is determined only if the pattern exists (mandatory for backtest)
            self.equity_file_path = self.base_results_dir / equity_file_pattern.format(
                symbol=safe_symbol, interval=safe_interval, model_type=self.model_type) if equity_file_pattern else None


            # Determine the path for the backtester/live-saved metrics file
            self.saved_metrics_file_path = self.base_results_dir / metrics_file_pattern.format(
                 symbol=safe_symbol,
                 interval=safe_interval,
                 model_type=self.model_type,
                 # analysis_type="summary_metrics" # Removed this, as metrics_pattern itself should not contain {analysis_type}
            ) if metrics_file_pattern else None
            # The backtester saves metrics with a simple filename like 'ADAUSDT_5m_xgboost_metrics.json'
            # The metrics_pattern in paths.py should reflect this.
            # So, the format string should only use symbol, interval, model_type.
            # The `analysis_plot_pattern` and `analysis_table_pattern` are for the *output* of `ResultsAnalyser`.


            if self.saved_metrics_file_path: logger.info(f"Expecting saved metrics file at: {self.saved_metrics_file_path}")
            else: logger.warning(f"Metrics file pattern key '{metrics_file_key}' not found. Cannot load saved metrics.")


            # Determine the path for the capital state file (only for live results)
            self.capital_state_file_path = self.base_results_dir / capital_state_pattern.format(
                 symbol=safe_symbol, interval=safe_interval, model_type=self.model_type
            ) if capital_state_pattern else None
            if self.capital_state_file_path: logger.info(f"Expecting capital state file at: {self.capital_state_file_path}")


        except KeyError as e:
            raise ValueError(f"Missing path pattern key in configuration: {e}")
        except Exception as e:
             raise ValueError(f"Error formatting file paths: {e}")


        # --- DataFrames ---
        self.trade_history_df: Optional[pd.DataFrame] = None
        self.equity_df: Optional[pd.DataFrame] = None # Will be loaded or calculated
        self.metrics: Dict[str, Any] = {} # Calculated metrics
        self.saved_metrics: Dict[str, Any] = {} # Metrics loaded from backtester/live file
        self.initial_capital: Optional[float] = None # Will be loaded from equity or capital state


        logger.info(f"ResultsAnalyser initialized for {self.symbol} {self.interval} ({self.model_type}) {self.results_type} results.")

    def _load_data(self):
        """Loads trade history and equity curve data, or calculates equity for live results."""
        try:
            # First, try to load the parquet file with model_type in the name (what backtester saves)
            # This is a a robust attempt to match what backtester.py saves.
            safe_symbol = self.symbol.replace('/', '')
            safe_interval = self.interval.replace(':', '_')
            expected_parquet_filename = f"{safe_symbol}_{safe_interval}_{self.model_type}_trades.parquet"
            parquet_path = self.base_results_dir / expected_parquet_filename

            if parquet_path.exists():
                logger.info(f"Loading trade history from {parquet_path} (Parquet, model-specific)...")
                self.trade_history_df = pd.read_parquet(parquet_path)
            else:
                # Fallback to the pattern defined in paths, which might be CSV or different naming
                logger.warning(f"Model-specific Parquet file not found at {parquet_path}. Falling back to loading from {self.trades_file_path}...")
                if not self.trades_file_path.exists():
                     raise FileNotFoundError(f"Trade history file not found: {self.trades_file_path}")

                if self.trades_file_path.suffix.lower() == '.csv':
                     # Use 'latin1' encoding as a common fallback for problematic CSVs
                     self.trade_history_df = pd.read_csv(self.trades_file_path, keep_default_na=False, encoding='latin1')
                     logger.info(f"Loaded CSV with 'latin1' encoding from {self.trades_file_path}.")
                elif self.trades_file_path.suffix.lower() == '.parquet':
                     self.trade_history_df = pd.read_parquet(self.trades_file_path)
                     logger.info(f"Loaded Parquet from {self.trades_file_path}.")
                else:
                     raise ValueError(f"Unsupported trade history file format: {self.trades_file_path.suffix}")


            # Ensure correct data types and required columns exist
            required_trade_cols = ['entry_time', 'exit_time', 'net_pnl', 'total_fees', 'direction', 'exit_reason']
            if not all(col in self.trade_history_df.columns for col in required_trade_cols):
                 missing = [col for col in required_trade_cols if col not in self.trade_history_df.columns]
                 logger.error(f"Loaded trade history is missing required columns: {missing}")
                 # Attempt to proceed but log a warning if critical columns are missing
                 # raise ValueError(f"Trade history is missing required columns: {missing}") # Option to raise error

            # Ensure datetime columns are timezone-aware (UTC)
            for col in ['entry_time', 'exit_time']:
                 if col in self.trade_history_df.columns:
                      # Convert to datetime, coercing errors to NaT
                      self.trade_history_df[col] = pd.to_datetime(self.trade_history_df[col], errors='coerce', utc=True)
                      # Drop rows where critical time columns are NaT
                      if col in ['entry_time', 'exit_time']: # Only drop if these specific columns are NaT
                           initial_rows = len(self.trade_history_df)
                           self.trade_history_df.dropna(subset=[col], inplace=True)
                           if len(self.trade_history_df) < initial_rows:
                                logger.warning(f"Dropped rows from trade history due to NaT in '{col}'.")


            # Ensure numeric columns are correct type
            for col in ['net_pnl', 'total_fees']:
                 if col in self.trade_history_df.columns:
                      self.trade_history_df[col] = pd.to_numeric(self.trade_history_df[col], errors='coerce')
                      # Drop rows with NaN in critical numeric columns
                      initial_rows = len(self.trade_history_df)
                      self.trade_history_df.dropna(subset=[col], inplace=True)
                      if len(self.trade_history_df) < initial_rows:
                           logger.warning(f"Dropped rows from trade history due to NaN in '{col}'.")


            # Calculate gross PnL using the corrected column names
            if 'net_pnl' in self.trade_history_df.columns and 'total_fees' in self.trade_history_df.columns:
                 self.trade_history_df['gross_pnl'] = self.trade_history_df['net_pnl'] + self.trade_history_df['total_fees']
            else:
                 logger.warning("Cannot calculate 'gross_pnl': Missing 'net_pnl' or 'total_fees' columns.")
                 self.trade_history_df['gross_pnl'] = np.nan # Add column with NaNs

            # Sort trades by exit time to ensure correct equity calculation order
            if 'exit_time' in self.trade_history_df.columns:
                 self.trade_history_df = self.trade_history_df.sort_values(by='exit_time').reset_index(drop=True)
                 logger.debug("Trade history sorted by exit_time.")


            logger.info(f"Trade history loaded. Shape: {self.trade_history_df.shape}")

        except FileNotFoundError as fnf_error:
             logger.error(fnf_error)
             raise
        except ValueError as ve:
             logger.error(ve)
             raise
        except Exception as e:
            logger.error(f"Error loading trade history from {self.trades_file_path}: {e}", exc_info=True)
            raise

        # --- Attempt to load Equity Curve or Capital State ---
        # Construct the expected parquet path for equity curve
        expected_equity_parquet_filename = f"{safe_symbol}_{safe_interval}_{self.model_type}_equity.parquet"
        equity_parquet_path = self.base_results_dir / expected_equity_parquet_filename

        if equity_parquet_path.exists():
            # Load equity curve from parquet file (standard for backtest)
            try:
                logger.info(f"Loading equity curve from {equity_parquet_path} (Parquet, model-specific)...")
                self.equity_df = pd.read_parquet(equity_parquet_path)

                # Ensure index is DatetimeIndex and 'equity' column exists
                if not isinstance(self.equity_df.index, pd.DatetimeIndex):
                     logger.warning("Equity curve index is not DatetimeIndex. Attempting conversion and setting as index.")
                     # Assuming the timestamp column is named 'timestamp' if not the index, or the index name is 'timestamp'
                     if self.equity_df.index.name != 'timestamp' and 'timestamp' in self.equity_df.columns:
                          self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'], errors='coerce', utc=True)
                          self.equity_df.dropna(subset=['timestamp'], inplace=True)
                          self.equity_df = self.equity_df.set_index('timestamp')
                     else:
                          # Attempt to convert the existing index if no timestamp column or index is named 'timestamp'
                          try:
                               self.equity_df.index = pd.to_datetime(self.equity_df.index, errors='coerce', utc=True)
                               # If index name is not None, drop NaT based on index
                               if self.equity_df.index.name is not None:
                                    self.equity_df.dropna(subset=[self.equity_df.index.name], inplace=True)
                               else:
                                    # If index has no name, check for NaT directly in the index
                                    self.equity_df = self.equity_df[pd.notna(self.equity_df.index)]

                          except Exception as e:
                               logger.error(f"Failed to convert equity curve index to DatetimeIndex: {e}")
                               raise ValueError("Equity curve data could not be configured with a DatetimeIndex.") from e


                # Check for the correct column name 'equity'
                if 'equity' not in self.equity_df.columns:
                     raise ValueError("Equity curve DataFrame must contain an 'equity' column.")
                self.equity_df['equity'] = pd.to_numeric(self.equity_df['equity'], errors='coerce')
                self.equity_df.dropna(subset=['equity'], inplace=True) # Drop rows with NaN equity

                if not self.equity_df.index.is_monotonic_increasing:
                     logger.warning("Equity curve index is not sorted chronologically. Sorting...")
                     self.equity_df = self.equity_df.sort_index()

                # Set initial capital from the first equity value
                if not self.equity_df.empty:
                    self.initial_capital = self.equity_df['equity'].iloc[0]
                    logger.info(f"Equity curve loaded. Shape: {self.equity_df.shape}. Initial Capital: {self.initial_capital:.2f}")
                else:
                    logger.warning("Equity curve file loaded but is empty.")
                    self.equity_df = None # Treat as not loaded
                    self.initial_capital = None

            except FileNotFoundError: # Should not happen here due to exists() check, but for safety
                 self.equity_df = None
                 self.initial_capital = None
                 logger.warning(f"Equity file not found at {equity_parquet_path}. Will attempt to calculate from trades.")
            except Exception as e:
                self.equity_df = None
                self.initial_capital = None
                logger.error(f"Error loading equity curve from {equity_parquet_path}: {e}", exc_info=True)
                logger.warning("Will attempt to calculate equity from trades.")

        else: # If parquet file with model_type was not found, try the fallback path (which might be CSV)
            logger.warning(f"Model-specific Parquet equity file not found at {equity_parquet_path}. Falling back to loading from {self.equity_file_path}...")
            if self.equity_file_path and self.equity_file_path.exists():
                 try:
                      if self.equity_file_path.suffix.lower() == '.csv':
                           self.equity_df = pd.read_csv(self.equity_file_path, encoding='latin1')
                           logger.info(f"Loaded CSV equity with 'latin1' encoding from {self.equity_file_path}.")
                      elif self.equity_file_path.suffix.lower() == '.parquet':
                           self.equity_df = pd.read_parquet(self.equity_file_path)
                           logger.info(f"Loaded Parquet equity from {self.equity_file_path}.")
                      else:
                           raise ValueError(f"Unsupported equity file format: {self.equity_file_path.suffix}")

                      # Standard validation after loading equity
                      if not isinstance(self.equity_df.index, pd.DatetimeIndex):
                           if 'timestamp' in self.equity_df.columns:
                                self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'], errors='coerce', utc=True)
                                self.equity_df.dropna(subset=['timestamp'], inplace=True)
                                self.equity_df = self.equity_df.set_index('timestamp')
                           else:
                                self.equity_df.index = pd.to_datetime(self.equity_df.index, errors='coerce', utc=True)
                                self.equity_df = self.equity_df[pd.notna(self.equity_df.index)] # Drop NaT if index has no name

                      if 'equity' not in self.equity_df.columns:
                           raise ValueError("Equity curve DataFrame must contain an 'equity' column.")
                      self.equity_df['equity'] = pd.to_numeric(self.equity_df['equity'], errors='coerce')
                      self.equity_df.dropna(subset=['equity'], inplace=True)

                      if not self.equity_df.index.is_monotonic_increasing:
                           self.equity_df = self.equity_df.sort_index()

                      if not self.equity_df.empty:
                           self.initial_capital = self.equity_df['equity'].iloc[0]
                           logger.info(f"Equity curve loaded. Shape: {self.equity_df.shape}. Initial Capital: {self.initial_capital:.2f}")
                      else:
                           logger.warning("Equity curve file loaded but is empty.")
                           self.equity_df = None
                           self.initial_capital = None

                 except Exception as e:
                      self.equity_df = None
                      self.initial_capital = None
                      logger.error(f"Error loading equity curve from fallback path {self.equity_file_path}: {e}", exc_info=True)
                      logger.warning("Will attempt to calculate equity from trades if live results and capital state available.")
            else:
                 logger.warning("Equity file not found at primary or fallback path.")
                 self.equity_df = None
                 self.initial_capital = None


        # If equity file was not loaded, attempt to load initial capital and calculate equity from trades
        if self.equity_df is None and self.results_type == 'live' and self.capital_state_file_path and self.capital_state_file_path.exists():
             try:
                  logger.info(f"Equity file not found. Attempting to load initial capital from {self.capital_state_file_path}...")
                  with open(self.capital_state_file_path, 'r') as f:
                       capital_state = json.load(f)
                  loaded_capital = capital_state.get('internal_capital') # Key used by trading_bot
                  if loaded_capital is not None and isinstance(loaded_capital, (int, float)) and loaded_capital > 0:
                       self.initial_capital = float(loaded_capital)
                       logger.info(f"Loaded initial capital: {self.initial_capital:.2f}")
                       # Now calculate equity curve from trade history
                       self._calculate_equity_from_trades()
                  else:
                       logger.warning(f"Capital state file found ({self.capital_state_file_path}) but 'internal_capital' was invalid ({loaded_capital}). Cannot calculate equity curve.")
                       self.initial_capital = None
             except FileNotFoundError:
                  logger.warning(f"Capital state file not found at {self.capital_state_file_path}. Cannot calculate equity curve.")
                  self.initial_capital = None
             except json.JSONDecodeError as e:
                  logger.error(f"Error decoding capital state JSON file: {e}. Cannot calculate equity curve.", exc_info=True)
                  self.initial_capital = None
             except Exception as e:
                  logger.error(f"Error loading capital state or calculating equity from trades: {e}", exc_info=True)
                  self.initial_capital = None

        elif self.equity_df is None and self.results_type == 'live' and (not self.capital_state_file_path or not self.capital_state_file_path.exists()):
             logger.warning("Equity file not found and capital state file is missing or path not configured. Cannot calculate equity curve.")
             self.initial_capital = None


        # --- Load Backtester-saved Metrics (if available) ---
        if self.saved_metrics_file_path and self.saved_metrics_file_path.exists():
             try:
                  logger.info(f"Loading saved metrics from {self.saved_metrics_file_path}...")
                  # Assumes metrics are saved as a JSON dictionary
                  with open(self.saved_metrics_file_path, 'r') as f:
                       self.saved_metrics = json.load(f)
                  logger.info("Saved metrics loaded.")
             except Exception as e:
                  logger.warning(f"Failed to load saved metrics: {e}")
                  self.saved_metrics = {} # Ensure it's an empty dict if loading fails
        else:
             logger.info("No saved metrics file found or path not configured. Will calculate metrics from data.")


    def _calculate_equity_from_trades(self):
        """Calculates the equity curve from trade history and initial capital."""
        if self.trade_history_df is None or self.trade_history_df.empty:
             logger.warning("Trade history is empty. Cannot calculate equity curve from trades.")
             self.equity_df = None
             return
        if self.initial_capital is None or self.initial_capital <= 0:
             logger.warning("Initial capital is not set or invalid. Cannot calculate equity curve from trades.")
             self.equity_df = None
             return

        logger.info("Calculating equity curve from trade history...")

        # Ensure trade history is sorted by exit time
        if 'exit_time' not in self.trade_history_df.columns or self.trade_history_df['exit_time'].isnull().any():
             logger.error("Trade history is missing valid 'exit_time' column. Cannot calculate equity curve.")
             self.equity_df = None
             return

        # Calculate cumulative net PnL
        # Use the corrected column name 'net_pnl'
        self.trade_history_df['cumulative_net_pnl'] = self.trade_history_df['net_pnl'].cumsum()

        # Calculate equity after each trade
        self.trade_history_df['equity_after_trade'] = self.initial_capital + self.trade_history_df['cumulative_net_pnl']

        # Create an equity DataFrame indexed by exit time
        # Include initial capital point at the time of the first trade's entry (or first exit if entry missing)
        first_trade_time = self.trade_history_df['entry_time'].iloc[0] if 'entry_time' in self.trade_history_df.columns and pd.notna(self.trade_history_df['entry_time'].iloc[0]) else self.trade_history_df['exit_time'].iloc[0]

        equity_points = [(first_trade_time, self.initial_capital)] if pd.notna(first_trade_time) else []

        # Add equity points after each trade closure
        for index, row in self.trade_history_df.iterrows():
             if pd.notna(row['exit_time']):
                  equity_points.append((row['exit_time'], row['equity_after_trade']))

        if not equity_points:
             logger.warning("No valid equity points generated from trade history.")
             self.equity_df = None
             return

        # Create the equity DataFrame
        self.equity_df = pd.DataFrame(equity_points, columns=['timestamp', 'equity'])
        self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'], utc=True)
        self.equity_df = self.equity_df.set_index('timestamp')

        # Remove duplicate index entries, keeping the last (most recent equity)
        self.equity_df = self.equity_df[~self.equity_df.index.duplicated(keep='last')]

        # Sort by index (timestamp)
        self.equity_df = self.equity_df.sort_index()

        logger.info(f"Equity curve calculated from trades. Shape: {self.equity_df.shape}")


    def _calculate_metrics(self):
        """Calculates core performance metrics."""
        if self.trade_history_df is None or self.equity_df is None:
            logger.error("Data not loaded. Cannot calculate metrics.")
            return

        logger.info("Calculating performance metrics...")
        try:
            # --- Basic Metrics ---
            if self.equity_df.empty:
                 logger.warning("Equity curve is empty. Cannot calculate most metrics.")
                 self.metrics = {"Error": "Equity curve is empty."}
                 return

            # Use the first and last values from the 'equity' column
            # If initial_capital was loaded from file, use that, otherwise use first equity point
            initial_capital = self.initial_capital if self.initial_capital is not None else self.equity_df['equity'].iloc[0]
            final_equity = self.equity_df['equity'].iloc[-1]
            # Use the corrected column names for PnL and fees
            total_net_pnl_from_trades = self.trade_history_df['net_pnl'].sum() if not self.trade_history_df.empty else 0.0
            total_commission = self.trade_history_df['total_fees'].sum() if not self.trade_history_df.empty else 0.0
            # Use the calculated 'gross_pnl' column
            gross_pnl = self.trade_history_df['gross_pnl'].sum() if not self.trade_history_df.empty else 0.0


            # --- Consistency Check ---
            # Check consistency between final equity calculated from trades and sum of net PnL
            # This check is relevant when equity is calculated from trades
            if self.equity_df is not None and 'equity_after_trade' in self.trade_history_df.columns and not self.trade_history_df.empty:
                 calculated_final_equity_from_trades = self.trade_history_df['equity_after_trade'].iloc[-1]
                 pnl_sum_check_diff = abs(calculated_final_equity_from_trades - final_equity)
                 if pnl_sum_check_diff > FLOAT_EPSILON:
                      logger.warning(f"Calculated final equity from trades ({calculated_final_equity_from_trades:.4f}) differs from final equity in equity_df ({final_equity:.4f}). Difference: {pnl_sum_check_diff:.4f}")
                      # Use the final equity from the equity_df for metrics if there's a discrepancy
                      # Or decide which one is more reliable based on context
                 # The original consistency check (Equity Change vs Sum Trades) is still relevant
                 # for backtest results or if equity is loaded from file.
                 equity_change = final_equity - initial_capital
                 pnl_diff = abs(equity_change - total_net_pnl_from_trades)
                 if pnl_diff > FLOAT_EPSILON and (abs(initial_capital) > FLOAT_EPSILON and pnl_diff / abs(initial_capital) > FLOAT_EPSILON):
                      logger.warning(f"Potential Discrepancy Detected between Equity Change and Sum of Trade Net PnL!")
                      logger.warning(f"  Equity Change (Final - Initial): {equity_change:.4f}")
                      logger.warning(f"  Sum of Trade Net PnLs:         {total_net_pnl_from_trades:.4f}")
                      logger.warning(f"  Difference:                    {pnl_diff:.4f}")
                      logger.warning("  This suggests a possible issue in how equity or PnL was logged during the backtest/trading.")
                 else:
                      logger.info("Equity change and sum of trade net PnL are consistent (within tolerance).")
                 consistency_status = f"{pnl_diff:.4f}" if pnl_diff > FLOAT_EPSILON and (abs(initial_capital) > FLOAT_EPSILON and pnl_diff / abs(initial_capital) > FLOAT_EPSILON) else "Consistent"
            else:
                 # If equity was loaded from file or trade history is empty, perform standard check
                 equity_change = final_equity - initial_capital
                 pnl_diff = abs(equity_change - total_net_pnl_from_trades)
                 if pnl_diff > FLOAT_EPSILON and (abs(initial_capital) > FLOAT_EPSILON and pnl_diff / abs(initial_capital) > FLOAT_EPSILON):
                      logger.warning(f"Potential Discrepancy Detected between Equity Change and Sum of Trade Net PnL!")
                      logger.warning(f"  Equity Change (Final - Initial): {equity_change:.4f}")
                      logger.warning(f"  Sum of Trade Net PnLs:         {total_net_pnl_from_trades:.4f}")
                      logger.warning(f"  Difference:                    {pnl_diff:.4f}")
                      logger.warning("  This suggests a possible issue in how equity or PnL was logged during the backtest/trading.")
                 else:
                      logger.info("Equity change and sum of trade net PnL are consistent (within tolerance).")
                 consistency_status = f"{pnl_diff:.4f}" if pnl_diff > FLOAT_EPSILON and (abs(initial_capital) > FLOAT_EPSILON and pnl_diff / abs(initial_capital) > FLOAT_EPSILON) else "Consistent"

            # --- End Consistency Check ---


            num_trades = len(self.trade_history_df)
            self.metrics["num_trades"] = num_trades # Ensure num_trades is set here

            if num_trades == 0:
                logger.warning("No trades found. Most performance metrics cannot be calculated.")
                self.metrics = {
                    "Initial Capital": initial_capital,
                    "Final Equity": final_equity,
                    "Total Net PnL (Sum Trades)": total_net_pnl_from_trades,
                    "Equity Change (Final-Initial)": equity_change,
                    "Gross PnL (Sum Trades)": gross_pnl,
                    "Total Fees Paid": total_commission,
                    "Number of Trades": 0,
                    "Equity/PnL Discrepancy": consistency_status
                }
                return # Exit calculation early


            # --- Trade-Based Metrics ---
            # Use the corrected column name 'net_pnl'
            wins = self.trade_history_df[self.trade_history_df['net_pnl'] > FLOAT_EPSILON] # Use epsilon for win check
            losses = self.trade_history_df[self.trade_history_df['net_pnl'] <= FLOAT_EPSILON] # Use epsilon for loss check
            num_wins = len(wins)
            num_losses = num_trades - num_wins
            win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0

            total_profit = wins['net_pnl'].sum()
            total_loss = abs(losses['net_pnl'].sum())

            profit_factor = total_profit / total_loss if total_loss > FLOAT_EPSILON else (np.inf if total_profit > FLOAT_EPSILON else 0) # Use epsilon

            avg_pnl_per_trade = total_net_pnl_from_trades / num_trades

            # --- Duration Metrics ---
            # Calculate duration using the loaded datetime columns
            if 'entry_time' in self.trade_history_df.columns and 'exit_time' in self.trade_history_df.columns:
                 self.trade_history_df['duration'] = self.trade_history_df['exit_time'] - self.trade_history_df['entry_time']
                 self.trade_history_df['duration_minutes'] = self.trade_history_df['duration'].dt.total_seconds() / 60.0
                 avg_holding_duration_minutes = self.trade_history_df['duration_minutes'].mean()
            elif 'holding_duration_bars' in self.trade_history_df.columns:
                 # Fallback to bars if time columns are missing, convert bars to minutes
                 interval_minutes = self._interval_to_minutes(self.interval)
                 if interval_minutes is not None:
                      avg_holding_duration_minutes = self.trade_history_df['holding_duration_bars'].mean() * interval_minutes
                 else:
                      logger.warning("Cannot calculate trade duration in minutes: Missing time columns and failed to parse interval.")
                      avg_holding_duration_minutes = np.nan
            else:
                 logger.warning("Cannot calculate trade duration: Missing time columns and holding bars column.")
                 avg_holding_duration_minutes = np.nan


            # --- Equity/Drawdown Metrics ---
            if not self.equity_df.index.is_monotonic_increasing:
                 self.equity_df = self.equity_df.sort_index()

            # Use the 'equity' column
            self.equity_df['cumulative_max'] = self.equity_df['equity'].cummax()
            # Handle potential division by zero if cumulative_max is 0
            cumulative_max_safe = self.equity_df['cumulative_max'].replace(0, np.nan)
            self.equity_df['drawdown'] = (self.equity_df['equity'] - self.equity_df['cumulative_max']) / cumulative_max_safe
            max_drawdown = abs(self.equity_df['drawdown'].min()) * 100 if not self.equity_df['drawdown'].dropna().empty else 0


            # --- Time-Based Metrics (Simplified CAGR) ---
            if len(self.equity_df) < 2 or self.equity_df.index.min() is pd.NaT or self.equity_df.index.max() is pd.NaT:
                 logger.warning("Not enough valid equity points (<2) or invalid timestamps to calculate time-based metrics (CAGR).")
                 cagr = np.nan
            else:
                 total_duration = self.equity_df.index[-1] - self.equity_df.index[0]
                 total_duration_years = total_duration.total_seconds() / (365.25 * 24 * 60 * 60)
                 # Handle division by zero or non-positive initial capital
                 if total_duration_years > FLOAT_EPSILON and initial_capital > FLOAT_EPSILON:
                      cagr = ((final_equity / initial_capital) ** (1 / total_duration_years) - 1) * 100
                 else:
                      cagr = np.nan if initial_capital <= FLOAT_EPSILON else 0 # CAGR is 0 if no growth, NaN if initial capital is zero/negative


            # --- Store Metrics ---
            self.metrics = {
                "Initial Capital": f"{initial_capital:.2f}",
                "Final Equity": f"{final_equity:.2f}",
                "Total Net PnL (Sum Trades)": f"{total_net_pnl_from_trades:.2f}",
                "Equity Change (Final-Initial)": f"{equity_change:.2f}",
                "Gross PnL (Sum Trades)": f"{gross_pnl:.2f}",
                "Total Fees Paid": f"{total_commission:.2f}",
                "Number of Trades": num_trades,
                "Number of Wins": num_wins if num_trades > 0 else 0,
                "Number of Losses": num_losses if num_trades > 0 else 0,
                "Win Rate (%)": f"{win_rate:.2f}",
                "Profit Factor": f"{profit_factor:.2f}" if np.isfinite(profit_factor) else ("Inf" if profit_factor == np.inf else "NaN"),
                "Avg PnL per Trade": f"{avg_pnl_per_trade:.4f}",
                "Max Drawdown (%)": f"{max_drawdown:.2f}",
                "CAGR (%)": f"{cagr:.2f}" if np.isfinite(cagr) else "NaN",
                "Avg Holding Duration (minutes)": f"{avg_holding_duration_minutes:.2f}" if pd.notna(avg_holding_duration_minutes) else "NaN",
                "Equity/PnL Discrepancy": consistency_status # Use the calculated status
            }
            logger.info("Performance metrics calculated.")

            # Optionally, compare calculated metrics with saved metrics if loaded
            if self.saved_metrics:
                 logger.info("Comparing calculated metrics with saved metrics:")
                 for key, calc_value_str in self.metrics.items():
                      # Normalize keys for comparison (remove spaces, convert to lower)
                      saved_key_norm = key.replace(" ", "").lower()
                      found_saved_key = None
                      # Find the corresponding key in saved metrics (case-insensitive, ignoring spaces)
                      for saved_key in self.saved_metrics.keys():
                           if saved_key.replace(" ", "").lower() == saved_key_norm:
                                found_saved_key = saved_key
                                break

                      if found_saved_key:
                           saved_value = self.saved_metrics[found_saved_key]
                           try:
                                # Attempt to convert both calculated and saved values to float for comparison
                                # Handle percentage strings, Inf, NaN
                                calc_value = float(str(calc_value_str).replace('%', '').replace('Inf', str(np.inf)).replace('-Inf', str(-np.inf)).replace('NaN', str(np.nan)).strip())
                                saved_value_float = float(str(saved_value).replace('%', '').replace('Inf', str(np.inf)).replace('-Inf', str(-np.inf)).replace('NaN', str(np.nan)).strip())

                                if np.isclose(calc_value, saved_value_float, atol=FLOAT_EPSILON, equal_nan=True):
                                     logger.info(f"  Metric '{key}': Consistent (Calculated: {calc_value_str}, Saved: {saved_value})")
                                else:
                                     logger.warning(f"  Metric '{key}': Discrepancy (Calculated: {calc_value_str}, Saved: {saved_value})")
                           except ValueError:
                                # If conversion to float fails, compare as strings
                                if calc_value_str == str(saved_value):
                                     logger.info(f"  Metric '{key}': Consistent (Calculated: {calc_value_str}, Saved: {saved_value})")
                                else:
                                     logger.warning(f"  Metric '{key}': Discrepancy (Calculated: {calc_value_str}, Saved: {saved_value})")
                      else:
                            logger.debug(f"  Metric '{key}': Not found in saved metrics.")

                 logger.info("-" * 52)

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            self.metrics = {"Error": f"Failed to calculate metrics: {e}"}

    @staticmethod
    def _interval_to_minutes(interval: str) -> Optional[float]:
        """Converts interval string (e.g., '5m', '1h') to minutes."""
        interval = interval.lower()
        try:
            if 'm' in interval:
                return float(interval.replace('m', ''))
            elif 'h' in interval:
                return float(interval.replace('h', '').replace('H', '')) * 60
            elif 'd' in interval:
                return float(interval.replace('d', '').replace('D', '')) * 24 * 60
            elif 'w' in interval:
                return float(interval.replace('w', '').replace('W', '')) * 7 * 24 * 60
            # Added 'M' for monthly intervals
            elif 'M' in interval:
                 # Assuming 30 days for a month for simplicity, or could return None/raise error
                 logger.warning("Monthly interval 'M' is approximate (30 days) for minute conversion.")
                 return float(interval.replace('M', '').replace('m', '')) * 30 * 24 * 60 if interval.replace('M', '').isdigit() else 30 * 24 * 60 # Handle '1M' or 'M'
            else: # Assume minutes if no unit specified and it's numeric
                 return float(interval)
        except ValueError:
            logger.warning(f"Could not parse interval '{interval}' to minutes.")
            return None
        except Exception as e:
             logger.warning(f"An unexpected error occurred parsing interval '{interval}': {e}")
             return None


    def _generate_plots(self):
        """Generates and saves essential analysis plots."""
        if self.trade_history_df is None or self.equity_df is None:
            logger.error("Data not loaded. Cannot generate plots.")
            return
        if self.equity_df.empty and self.trade_history_df.empty:
             logger.warning("Both equity and trade data are empty. Skipping plot generation.")
             return

        logger.info("Generating analysis plots...")
        sns.set_theme(style="darkgrid")

        plot_context = f"{self.symbol}_{self.interval}_{self.model_type}"

        # --- 1. Equity Curve ---
        if not self.equity_df.empty:
            try:
                plt.figure(figsize=(12, 6))
                # Use the 'equity' column
                self.equity_df['equity'].plot(title=f'Equity Curve - {plot_context}')
                plt.ylabel("Equity")
                plt.xlabel("Timestamp")
                plt.grid(True)
                # Use the correct pattern key 'analysis_plot_pattern'
                plot_filename = self.paths.get("analysis_plot_pattern", "{symbol}_{interval}_{model_type}_equity_curve.png").format(
                    symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type="equity_curve")
                save_path = self.analysis_save_dir / plot_filename
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Equity curve plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate equity curve plot: {e}", exc_info=True)
        else:
             logger.warning("Equity curve data is empty. Skipping equity curve plot.")


        # --- 2. Drawdown Plot ---
        # Check if equity_df is not empty and 'drawdown' column exists and has non-NaN values
        if not self.equity_df.empty and 'drawdown' in self.equity_df.columns and not self.equity_df['drawdown'].dropna().empty:
            try:
                plt.figure(figsize=(12, 6))
                # Plot drawdown as percentage
                (self.equity_df['drawdown'] * 100).plot(title=f'Drawdown (%) - {plot_context}', kind='area', alpha=0.5, color='red')
                plt.ylabel("Drawdown (%)")
                plt.xlabel("Timestamp")
                plt.grid(True)
                # Use the correct pattern key 'analysis_plot_pattern'
                plot_filename = self.paths.get("analysis_plot_pattern", "{symbol}_{interval}_{model_type}_drawdown.png").format(
                    symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type="drawdown")
                save_path = self.analysis_save_dir / plot_filename
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Drawdown plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate drawdown plot: {e}", exc_info=True)
        else:
             logger.warning("Drawdown data is not available or empty. Skipping drawdown plot.")


        # --- 3. PnL Distribution ---
        # Check if trade_history_df is not empty and 'net_pnl' column exists and has non-NaN values
        if not self.trade_history_df.empty and 'net_pnl' in self.trade_history_df.columns and not self.trade_history_df['net_pnl'].dropna().empty:
            try:
                plt.figure(figsize=(10, 6))
                # Use the corrected column name 'net_pnl'
                sns.histplot(self.trade_history_df['net_pnl'], bins=50) # Removed kde=True for simplicity
                plt.title(f'Trade Net PnL Distribution - {plot_context}')
                plt.xlabel("Net PnL per Trade")
                plt.ylabel("Frequency")
                plt.grid(True, axis='y')
                # Use the correct pattern key 'analysis_plot_pattern'
                plot_filename = self.paths.get("analysis_plot_pattern", "{symbol}_{interval}_{model_type}_pnl_distribution.png").format(
                    symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type="pnl_distribution")
                save_path = self.analysis_save_dir / plot_filename
                plt.savefig(save_path)
                plt.close()
                logger.info(f"PnL distribution plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate PnL distribution plot: {e}", exc_info=True)
        else:
             logger.warning("Trade net PnL data is not available or empty. Skipping PnL distribution plot.")

        # --- 4. PnL Distribution by Exit Reason ---
        # Check if trade_history_df is not empty and required columns exist
        if not self.trade_history_df.empty and 'net_pnl' in self.trade_history_df.columns and 'exit_reason' in self.trade_history_df.columns:
            self._plot_exit_reason_pnl(plot_context)
        else:
             logger.warning("Trade data is missing 'net_pnl' or 'exit_reason' for exit reason analysis plot.")

        # --- 5. Trade Frequency by Exit Reason (New Plot) ---
        # Check if trade_history_df is not empty and 'exit_reason' column exists
        if not self.trade_history_df.empty and 'exit_reason' in self.trade_history_df.columns:
            self._plot_exit_reason_frequency(plot_context)
        else:
             logger.warning("Trade data is missing 'exit_reason' for exit reason frequency plot.")


        logger.info("Analysis plots generated.")

    def _plot_exit_reason_pnl(self, plot_context: str):
        """
        Generates a plot (e.g., box plot) showing Net PnL distribution by exit reason.
        """
        logger.info("Generating Net PnL distribution plot by exit reason...")
        try:
            # Filter out trades with missing or invalid exit reasons if necessary
            # Assuming 'exit_reason' is a string or categorical column
            plot_data = self.trade_history_df.dropna(subset=['net_pnl', 'exit_reason']).copy()

            if plot_data.empty:
                logger.warning("No valid trade data with exit reasons for PnL by exit reason plot. Skipping.")
                return

            plt.figure(figsize=(12, 7))

            # Use seaborn.boxplot for distribution visualization
            # Order the boxes by median PnL for better comparison (optional)
            order = plot_data.groupby('exit_reason')['net_pnl'].median().sort_values(ascending=False).index

            sns.boxplot(data=plot_data, x='exit_reason', y='net_pnl', order=order)

            plt.title(f'Net PnL Distribution by Exit Reason - {plot_context}')
            plt.xlabel("Exit Reason")
            plt.ylabel("Net PnL per Trade")
            plt.xticks(rotation=45, ha='right') # Rotate labels for readability
            plt.grid(True, axis='y')
            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Use the correct pattern key 'analysis_plot_pattern'
            plot_filename = self.paths.get("analysis_plot_pattern", "{symbol}_{interval}_{model_type}_pnl_by_exit_reason.png").format(
                symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type="pnl_by_exit_reason")
            save_path = self.analysis_save_dir / plot_filename
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Net PnL by exit reason plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to generate Net PnL by exit reason plot: {e}", exc_info=True)

    def _plot_exit_reason_frequency(self, plot_context: str):
        """
        Generates a histogram or bar plot showing the frequency of trades per exit reason.
        """
        logger.info("Generating trade frequency plot by exit reason...")
        try:
            # Count occurrences of each exit reason
            # Use value_counts to get the frequency of each unique value in 'exit_reason'
            exit_reason_counts = self.trade_history_df['exit_reason'].value_counts().reset_index()
            exit_reason_counts.columns = ['exit_reason', 'count']

            if exit_reason_counts.empty:
                logger.warning("No trade data with exit reasons for frequency plot. Skipping.")
                return

            plt.figure(figsize=(10, 6))

            # Use seaborn.barplot for frequency visualization
            # Order the bars by count (optional, but often helpful)
            order = exit_reason_counts.sort_values(by='count', ascending=False)['exit_reason']

            sns.barplot(data=exit_reason_counts, x='exit_reason', y='count', order=order)

            plt.title(f'Trade Frequency by Exit Reason - {plot_context}')
            plt.xlabel("Exit Reason")
            plt.ylabel("Number of Trades")
            plt.xticks(rotation=45, ha='right') # Rotate labels for readability
            plt.grid(True, axis='y')
            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Use the correct pattern key 'analysis_plot_pattern'
            plot_filename = self.paths.get("analysis_plot_pattern", "{symbol}_{interval}_{model_type}_exit_reason_frequency.png").format(
                symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type="exit_reason_frequency")
            save_path = self.analysis_save_dir / plot_filename
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Trade frequency by exit reason plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to generate trade frequency by exit reason plot: {e}", exc_info=True)


    def _save_metrics(self):
        """Saves the calculated metrics to a CSV file."""
        if not self.metrics:
            logger.warning("No metrics calculated to save.")
            return

        try:
            metrics_df = pd.DataFrame([self.metrics]) # Convert dict to DataFrame row
            # Use analysis_table_pattern with analysis_type="summary_metrics"
            filename = self.paths.get("analysis_table_pattern", "{symbol}_{interval}_{model_type}_summary_metrics.csv").format(
                symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type="summary_metrics")
            save_path = self.analysis_save_dir / filename

            # Use mode='a' to append and header=False if file exists for easier comparison across runs
            file_exists = save_path.exists()
            metrics_df.to_csv(save_path, index=False, header=not file_exists, mode='a')

            logger.info(f"Performance metrics saved to {save_path}")
        except KeyError as e:
             logger.error(f"Missing path pattern key for saving metrics: {e}")
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}", exc_info=True)


    def run_analysis(self):
        """Runs the full analysis pipeline: load data, calculate metrics, generate plots, save results."""
        logger.info(f"--- Running Analysis for {self.symbol} {self.interval} ({self.model_type}) {self.results_type} results ---")
        try:
            self._load_data()
            # Check if equity was loaded or calculated successfully
            if self.equity_df is None:
                 logger.error("Could not load or calculate equity curve. Analysis aborted.")
                 return # Abort if equity is not available

            self._calculate_metrics()

            # Log the summary metrics
            logger.info(f"Analysis Summary for {self.symbol} {self.interval} ({self.model_type}) {self.results_type}:")
            if self.metrics: # FIX: Changed 'metrics' to 'self.metrics'
                 for key, value in self.metrics.items():
                      logger.info(f"  {key}: {value}")
            else:
                 logger.info("  No metrics were calculated.")
            logger.info("-" * 52)

            self._save_metrics()
            self._generate_plots() # This method now includes the new exit reason frequency plot

        except (FileNotFoundError, ValueError, IOError) as e:
             logger.error(f"Analysis aborted due to error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        finally:
            logger.info("--- Analysis Complete ---")

