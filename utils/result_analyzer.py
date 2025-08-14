# utils/result_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json # Import json for loading metrics saved by backtester
from datetime import datetime, timezone # For current timestamp in save paths
import re # Added for pattern matching


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
        "live_trading_trades_pattern": "{symbol}_{interval}_{model_type}_trades.parquet",
        "live_trading_equity_pattern": "{symbol}_{interval}_{model_type}_equity.parquet",
        "live_trading_capital_state_pattern": "{symbol}_{interval}_{model_type}_capital_state.json", # Added key for capital state
        # Simplified analysis file patterns for plots and tables
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

class ResultAnalyser: # Class name changed from ResultsAnalyser to ResultAnalyser
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
        Initializes the ResultAnalyser.

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
        logger.info(f"Initializing ResultAnalyser for {symbol} {interval} ({model_type}) {results_type} results...")
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

            # --- IMPORTANT: Logic to find the LATEST file based on pattern matching ---
            # This handles the timestamp in the filename dynamically
            self.trades_file_path = self._find_latest_file_matching_pattern(
                base_dir=self.base_results_dir,
                file_pattern_template=trades_file_pattern,
                symbol=safe_symbol,
                interval=safe_interval,
                model_type=self.model_type
            )
            if not self.trades_file_path:
                raise FileNotFoundError(f"No trade history file found matching pattern '{trades_file_pattern}' for {safe_symbol}_{safe_interval}_{self.model_type} in {self.base_results_dir}.")


            self.equity_file_path = None
            if equity_file_pattern:
                self.equity_file_path = self._find_latest_file_matching_pattern(
                    base_dir=self.base_results_dir,
                    file_pattern_template=equity_file_pattern,
                    symbol=safe_symbol,
                    interval=safe_interval,
                    model_type=self.model_type
                )
                if not self.equity_file_path:
                    logger.warning(f"No equity curve file found matching pattern '{equity_file_pattern}' for {safe_symbol}_{safe_interval}_{self.model_type} in {self.base_results_dir}. Will attempt to calculate if trade data and initial capital available.")


            self.saved_metrics_file_path = None
            if metrics_file_pattern:
                self.saved_metrics_file_path = self._find_latest_file_matching_pattern(
                    base_dir=self.base_results_dir,
                    file_pattern_template=metrics_file_pattern,
                    symbol=safe_symbol,
                    interval=safe_interval,
                    model_type=self.model_type
                )
                if not self.saved_metrics_file_path:
                    logger.warning(f"No saved metrics file found matching pattern '{metrics_file_pattern}' for {safe_symbol}_{safe_interval}_{self.model_type} in {self.base_results_dir}. Will calculate metrics from data.")


            self.capital_state_file_path = None
            if capital_state_pattern and self.results_type == 'live': # Only for live results
                self.capital_state_file_path = self._find_latest_file_matching_pattern(
                    base_dir=self.base_results_dir,
                    file_pattern_template=capital_state_pattern,
                    symbol=safe_symbol,
                    interval=safe_interval,
                    model_type=self.model_type
                )
                if not self.capital_state_file_path:
                    logger.warning(f"No capital state file found matching pattern '{capital_state_pattern}' for {safe_symbol}_{safe_interval}_{self.model_type} in {self.base_results_dir}. Initial capital might be derived from equity curve or default.")


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


        logger.info(f"ResultAnalyser initialized for {self.symbol} {self.interval} ({self.model_type}) {self.results_type} results.")

    def _find_latest_file_matching_pattern(self, base_dir: Path, file_pattern_template: str, **kwargs) -> Optional[Path]:
        """
        Finds the latest file in a directory that matches a given pattern.
        The pattern can contain placeholders like {symbol}, {interval}, {model_type}, {timestamp}.
        This function assumes the timestamp part of the filename is in YYYYMMDD_HHMMSS format
        and is the primary differentiator for 'latest'.
        """
        if not base_dir.is_dir():
            logger.warning(f"Base directory for pattern matching does not exist: {base_dir}")
            return None

        # Build a regex pattern from the file_pattern_template
        # Replace known placeholders with regex capture groups (.*?) for any characters
        # For timestamp, use a specific pattern if possible, otherwise (.*?)
        pattern_str = file_pattern_template.format(
            symbol=kwargs.get('symbol', '.*?'),
            interval=kwargs.get('interval', '.*?'),
            model_type=kwargs.get('model_type', '.*?'),
            timestamp=r'(\d{8}_\d{6})' # Specific regex for YYYYMMDD_HHMMSS
        )
        # Handle the case where the pattern might not have a timestamp, or has a different extension
        pattern_str = pattern_str.replace('.parquet', r'\.parquet').replace('.json', r'\.json').replace('.csv', r'\.csv')
        # Escape any remaining special regex characters that are literal in filenames
        pattern_str = pattern_str.replace('.', r'\.') # Escape literal dots
        pattern_str = pattern_str.replace('(', r'\(').replace(')', r'\)') # Escape parentheses if they might be in symbol/interval

        # For files that *don't* have a timestamp (e.g., if you have `my_file.json`),
        # the above timestamp replacement might fail. A more robust way:
        # If '{timestamp}' in the template:
        if '{timestamp}' in file_pattern_template:
            # Prepare a glob pattern first to find all relevant files quickly
            glob_pattern = file_pattern_template.format(
                symbol=kwargs.get('symbol', '*'),
                interval=kwargs.get('interval', '*'),
                model_type=kwargs.get('model_type', '*'),
                timestamp='*' # Use glob * for the timestamp part
            )
        else:
            # If no timestamp placeholder, the template is the exact filename (except for specific wildcards)
            glob_pattern = file_pattern_template.format(
                symbol=kwargs.get('symbol', '*'),
                interval=kwargs.get('interval', '*'),
                model_type=kwargs.get('model_type', '*')
            )
            # If no timestamp and no model_type etc., it could just be a fixed name
            # In this case, glob_pattern is the exact filename to look for.

        matching_files = list(base_dir.glob(glob_pattern))
        
        if not matching_files:
            logger.debug(f"No files found matching glob pattern: {glob_pattern} in {base_dir}")
            return None

        # If timestamp is part of the pattern, sort by it
        if '{timestamp}' in file_pattern_template:
            # Extract timestamp from filename and find the latest
            files_with_timestamps = []
            for f in matching_files:
                # Attempt to extract timestamp (assuming YYYYMMDD_HHMMSS format directly before extension)
                # Regex example: ^.*_(\d{8}_\d{6})\.
                match = re.search(r'(\d{8}_\d{6})\.(parquet|json|csv)$', f.name)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        dt_obj = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        files_with_timestamps.append((dt_obj, f))
                    except ValueError:
                        logger.debug(f"Could not parse timestamp from filename: {f.name}")
                        continue
                else:
                    logger.debug(f"Filename does not contain expected timestamp format: {f.name}")

            if files_with_timestamps:
                # Sort by datetime object (most recent first)
                files_with_timestamps.sort(key=lambda x: x[0], reverse=True)
                latest_file = files_with_timestamps[0][1]
                logger.info(f"Found latest file matching pattern: {latest_file}")
                return latest_file
            else:
                logger.warning(f"Could not find any files with parseable timestamps matching pattern in {base_dir}. Returning None.")
                return None
        else:
            # If no timestamp, assume there's only one relevant file or return the first one found by glob
            logger.info(f"Returning first file found (no timestamp in pattern): {matching_files[0]}")
            return matching_files[0]
            

    def _load_data(self):
        """
        Loads trade history and equity curve data.
        Prioritizes loading saved metrics first for initial capital.
        Then attempts to load equity curve, if not found, calculates it from trades.
        """
        # Moved loading saved_metrics here to ensure self.initial_capital is potentially set early.
        if self.saved_metrics_file_path and self.saved_metrics_file_path.exists():
             try:
                  logger.info(f"Loading saved metrics from {self.saved_metrics_file_path}...")
                  with open(self.saved_metrics_file_path, 'r') as f:
                       self.saved_metrics = json.load(f)

                  # Prioritize initial_capital from saved_metrics
                  if 'initial_capital' in self.saved_metrics:
                       try:
                           self.initial_capital = float(str(self.saved_metrics['initial_capital']).replace(',', ''))
                           logger.info(f"Initial capital set from saved metrics: {self.initial_capital:.2f}")
                       except ValueError:
                           logger.warning(f"Could not parse 'initial_capital' from saved metrics ({self.saved_metrics['initial_capital']}).")
                           self.initial_capital = None
                  else:
                       logger.warning("'initial_capital' not found in saved metrics. Will attempt to derive from equity curve or capital state.")

                  logger.info("Saved metrics loaded.")
             except Exception as e:
                  logger.warning(f"Failed to load saved metrics: {e}. Proceeding without them.")
                  self.saved_metrics = {}
        else:
             logger.info("No saved metrics file found or path not configured. Will calculate metrics from data.")


        # --- Load Trade History ---
        try:
            if not self.trades_file_path.exists():
                raise FileNotFoundError(f"Trade history file not found: {self.trades_file_path}")

            if self.trades_file_path.suffix.lower() == '.csv':
                 self.trade_history_df = pd.read_csv(self.trades_file_path, keep_default_na=False, encoding='latin1')
                 logger.info(f"Loaded CSV trade history from {self.trades_file_path}.")
            elif self.trades_file_path.suffix.lower() == '.parquet':
                 self.trade_history_df = pd.read_parquet(self.trades_file_path)
                 logger.info(f"Loaded Parquet trade history from {self.trades_file_path}.")
            else:
                 raise ValueError(f"Unsupported trade history file format: {self.trades_file_path.suffix}")

            # Ensure required columns for PnL calculations
            required_trade_cols = ['net_pnl', 'total_fees', 'entry_time', 'exit_time', 'direction', 'exit_reason', 'initial_margin', 'bars_held'] # Added initial_margin, bars_held
            for col in required_trade_cols:
                if col not in self.trade_history_df.columns:
                    logger.warning(f"Trade history missing expected column: '{col}'. Some analysis may be affected.")
                    self.trade_history_df[col] = np.nan # Add column with NaNs to prevent KeyError later

            # Convert datetime columns
            for col in ['entry_time', 'exit_time']:
                self.trade_history_df[col] = pd.to_datetime(self.trade_history_df[col], errors='coerce', utc=True)
            self.trade_history_df.dropna(subset=['exit_time'], inplace=True) # Drop trades with invalid exit times

            # Convert numeric columns
            numeric_cols = ['net_pnl', 'total_fees', 'initial_margin', 'bars_held']
            for col in numeric_cols:
                self.trade_history_df[col] = pd.to_numeric(self.trade_history_df[col], errors='coerce')
            self.trade_history_df.dropna(subset=['net_pnl', 'total_fees'], inplace=True) # Ensure these critical columns are numeric

            # Calculate gross PnL
            self.trade_history_df['gross_pnl'] = self.trade_history_df['net_pnl'] + self.trade_history_df['total_fees']

            # Sort trades by exit time
            self.trade_history_df = self.trade_history_df.sort_values(by='exit_time').reset_index(drop=True)
            logger.info(f"Trade history loaded. Shape: {self.trade_history_df.shape}")

        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
            raise # Re-raise for pipeline to handle
        except ValueError as ve:
            logger.error(ve)
            raise # Re-raise for pipeline to handle
        except Exception as e:
            logger.error(f"Error loading trade history from {self.trades_file_path}: {e}", exc_info=True)
            raise # Re-raise for pipeline to handle


        # --- Attempt to load Equity Curve ---
        if self.equity_file_path and self.equity_file_path.exists():
            try:
                logger.info(f"Loading equity curve from {self.equity_file_path}...")
                self.equity_df = pd.read_parquet(self.equity_file_path) # Assuming parquet for equity from simulator
                
                if not isinstance(self.equity_df.index, pd.DatetimeIndex):
                    # Attempt to convert existing index to DatetimeIndex
                    self.equity_df.index = pd.to_datetime(self.equity_df.index, errors='coerce', utc=True)
                    self.equity_df = self.equity_df[pd.notna(self.equity_df.index)] # Drop rows with NaT index

                if 'equity' not in self.equity_df.columns:
                    # If 'equity' column is missing but there's only one column, rename it.
                    if len(self.equity_df.columns) == 1:
                        self.equity_df.rename(columns={self.equity_df.columns[0]: 'equity'}, inplace=True)
                    else:
                        raise ValueError("Equity curve DataFrame must contain an 'equity' column.")
                
                self.equity_df['equity'] = pd.to_numeric(self.equity_df['equity'], errors='coerce')
                self.equity_df.dropna(subset=['equity'], inplace=True)
                self.equity_df = self.equity_df.sort_index()

                if self.initial_capital is None and not self.equity_df.empty:
                    self.initial_capital = self.equity_df['equity'].iloc[0]
                    logger.info(f"Initial Capital set from equity curve: {self.initial_capital:.2f}")

                logger.info(f"Equity curve loaded. Shape: {self.equity_df.shape}.")

            except Exception as e:
                self.equity_df = None
                logger.warning(f"Error loading equity curve from {self.equity_file_path}: {e}. Will attempt to calculate if needed.")
        else:
            logger.info("No dedicated equity curve file found. Will attempt to calculate from trades.")

        # --- If equity curve not loaded, try to calculate it from trades ---
        if self.equity_df is None and not self.trade_history_df.empty:
            # If initial capital not set from saved metrics, try to get from capital state for live or default
            if self.initial_capital is None:
                if self.results_type == 'live' and self.capital_state_file_path and self.capital_state_file_path.exists():
                    try:
                        with open(self.capital_state_file_path, 'r') as f:
                            capital_state = json.load(f)
                        loaded_capital = capital_state.get('current_balance') # Use 'current_balance' from BotStateHandler
                        if loaded_capital is not None and isinstance(loaded_capital, (int, float)):
                            self.initial_capital = float(loaded_capital)
                            logger.info(f"Loaded initial capital from bot capital state file: {self.initial_capital:.2f}")
                        else:
                            logger.warning(f"Bot capital state file found but 'current_balance' was invalid or missing ({loaded_capital}). Cannot use for initial capital.")
                    except Exception as e:
                        logger.warning(f"Error loading initial capital from bot capital state: {e}")
                
                # Final fallback for initial capital if still None
                if self.initial_capital is None:
                    # Default if no other source
                    self.initial_capital = 10000.0 # A reasonable default if nothing found
                    logger.warning(f"Initial capital could not be determined. Defaulting to {self.initial_capital:.2f}.")

            if self.initial_capital is not None and self.initial_capital > 0:
                self._calculate_equity_from_trades()
            else:
                logger.error("Initial capital is zero or negative. Cannot calculate equity curve from trades.")
                self.equity_df = None


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

        # Fill forward any gaps in the equity curve if needed (e.g., if a bar has no trade)
        # This is typically done if you want an equity curve per bar, but here we plot only at trade events
        # If we wanted per-bar equity, we'd need the OHLCV data passed here. For now, it's point-in-time.

        logger.info(f"Equity curve calculated from trades. Shape: {self.equity_df.shape}")


    def _calculate_metrics(self):
        """Calculates core performance metrics."""
        if self.trade_history_df is None or self.equity_df is None:
            logger.error("Data not loaded. Cannot calculate metrics.")
            from config.params import app_config # Import here for default capital
            initial_cap_fallback = getattr(app_config.strategy, 'initial_capital', 10000.0)
            self.metrics = {
                "initial_capital": initial_cap_fallback,
                "final_capital": initial_cap_fallback,
                "net_profit": 0.0,
                "return_on_capital_pct": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "total_profit_trades": 0,
                "total_loss_trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "avg_profit_per_trade": 0.0,
                "avg_loss_per_trade": 0.0,
                "avg_pnl_per_trade": 0.0,
                "edge_expected_value": 0.0, # NEW
                "total_fees": 0.0,
                "avg_holding_duration_bars": 0.0,
                "equity_pnl_discrepancy": "No data", # String status
                "config_symbol": self.symbol,
                "config_interval": self.interval,
                "config_model_type": self.model_type,
            }
            return

        logger.info("Calculating performance metrics...")
        try:
            # --- Basic Metrics ---
            if self.equity_df.empty:
                 logger.warning("Equity curve is empty. Cannot calculate most metrics.")
                 self.metrics = {"Error": "Equity curve is empty."}
                 return

            from config.params import app_config # Ensure app_config is accessible for config parameters
            
            initial_capital = self.initial_capital # Use the initial_capital stored in the instance
            final_capital = self.equity_df['equity'].iloc[-1]
            total_net_pnl_from_trades = self.trade_history_df['net_pnl'].sum() if not self.trade_history_df.empty else 0.0
            total_fees = self.trade_history_df['total_fees'].sum() if not self.trade_history_df.empty else 0.0
            gross_profit_loss = self.trade_history_df['gross_pnl'].sum() if not self.trade_history_df.empty else 0.0


            # --- Consistency Check ---
            # Check consistency between final equity calculated from sum of PnLs vs. direct equity curve end
            equity_change_from_curve = final_capital - initial_capital
            pnl_sum_check_diff = abs(equity_change_from_curve - total_net_pnl_from_trades)

            if pnl_sum_check_diff > FLOAT_EPSILON:
                logger.warning(f"Potential Discrepancy Detected between Equity Change and Sum of Trade Net PnL!")
                logger.warning(f"  Equity Change (Final - Initial): {equity_change_from_curve:.4f}")
                logger.warning(f"  Sum of Trade Net PnLs:         {total_net_pnl_from_trades:.4f}")
                logger.warning(f"  Difference:                    {pnl_sum_check_diff:.4f}")
                logger.warning("  This suggests a possible issue in how equity or PnL was logged during the backtest/trading.")
                consistency_status = f"{pnl_sum_check_diff:.4f} (Discrepancy)"
            else:
                logger.info("Equity change and sum of trade net PnL are consistent (within tolerance).")
                consistency_status = "Consistent"

            # --- End Consistency Check ---

            num_trades = len(self.trade_history_df)
            
            if num_trades == 0:
                logger.warning("No trades found. Most performance metrics cannot be calculated.")
                self.metrics = {
                    "initial_capital": initial_capital,
                    "final_capital": final_capital,
                    "net_profit": total_net_pnl_from_trades,
                    "return_on_capital_pct": (total_net_pnl_from_trades / initial_capital) * 100 if initial_capital > FLOAT_EPSILON else 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "total_profit_trades": 0,
                    "total_loss_trades": 0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                    "avg_profit_per_trade": 0.0,
                    "avg_loss_per_trade": 0.0,
                    "avg_pnl_per_trade": 0.0,
                    "edge_expected_value": 0.0, # NEW
                    "total_fees": total_fees,
                    "avg_holding_duration_bars": 0.0,
                    "equity_pnl_discrepancy": consistency_status,
                    "config_symbol": self.symbol,
                    "config_interval": self.interval,
                    "config_model_type": self.model_type,
                }
                return # Exit calculation early


            # --- Trade-Based Metrics ---
            wins = self.trade_history_df[self.trade_history_df['net_pnl'] > FLOAT_EPSILON]
            losses = self.trade_history_df[self.trade_history_df['net_pnl'] <= FLOAT_EPSILON]
            num_wins = len(wins)
            num_losses = num_trades - num_wins # Correctly counts non-wins as losses for binary outcome

            win_rate = (num_wins / num_trades)

            total_profit_pnl = wins['net_pnl'].sum()
            total_loss_pnl = losses['net_pnl'].sum() # This will be negative or zero

            profit_factor = abs(total_profit_pnl / total_loss_pnl) if total_loss_pnl < -FLOAT_EPSILON else (np.inf if total_profit_pnl > FLOAT_EPSILON else 0)

            avg_pnl_per_trade = total_net_pnl_from_trades / num_trades

            avg_win_pnl = wins['net_pnl'].mean() if num_wins > 0 else 0.0
            avg_loss_pnl = losses['net_pnl'].mean() if num_losses > 0 else 0.0 # Will be negative or zero

            # Expected Value (Edge) per trade
            edge_expected_value = (win_rate * avg_win_pnl) + ((1 - win_rate) * avg_loss_pnl)


            # --- Equity/Drawdown Metrics ---
            peak = self.equity_df['equity'].expanding(min_periods=1).max()
            # Handle potential division by zero if peak is 0 for drawdown calculation
            peak_safe = peak.replace(0, np.nan)
            drawdown_series = (self.equity_df['equity'] - peak) / peak_safe
            max_drawdown = abs(drawdown_series.min()) if not drawdown_series.dropna().empty else 0.0
            
            # --- Time-Based Metrics (Annualized) ---
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            if len(self.equity_df) > 1:
                returns = self.equity_df['equity'].pct_change().dropna()
                if not returns.empty:
                    # Determine appropriate annualization factor based on interval
                    interval_in_minutes = self._interval_to_minutes(self.interval)
                    if interval_in_minutes and interval_in_minutes > 0:
                        # Assuming 252 trading days/year, 6.5 trading hours/day = 390 minutes/day
                        # Or 24/7 for crypto
                        # For crypto, often 365 days * 24 hours * 60 minutes
                        # annualization_factor = (365 * 24 * 60) / interval_in_minutes
                        # A simpler approach: bars per year directly from frequency
                        # This needs a more robust freq inference, but for now:
                        # Estimate bars per year: assuming 252 trading days for stocks, or 365 for crypto
                        bars_per_day_map = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '2h': 12, '4h': 6, '8h': 3, '12h': 2, '1d': 1}
                        bars_per_day = bars_per_day_map.get(self.interval, 1) # Default to 1 bar per day if not found
                        annualization_factor = bars_per_day * 365 # Crypto 24/7


                    # This needs a more robust freq inference, but for now:
                    if interval_in_minutes:
                         if interval_in_minutes < 60: # Less than hourly (e.g., 5m, 15m)
                              annualization_factor = (365 * 24 * 60) / interval_in_minutes # Bars in a crypto year
                         elif interval_in_minutes == 60: # Hourly
                              annualization_factor = 365 * 24
                         elif interval_in_minutes == 24 * 60: # Daily
                              annualization_factor = 365
                         else: # Larger intervals
                              annualization_factor = 365 / (interval_in_minutes / (24 * 60)) # Days / (minutes per bar / minutes per day)
                    else:
                         annualization_factor = 252 # Default to 252 trading days for safety if interval unknown

                    if annualization_factor <= FLOAT_EPSILON: annualization_factor = 1.0 # Prevent division by zero

                    mean_return_ann = returns.mean() * annualization_factor
                    std_dev_returns_ann = returns.std() * np.sqrt(annualization_factor)

                    if std_dev_returns_ann > FLOAT_EPSILON:
                        sharpe_ratio = mean_return_ann / std_dev_returns_ann # Assuming risk-free rate is 0 for simplicity
                    
                    negative_returns_only = returns[returns < 0]
                    downside_std_dev_ann = negative_returns_only.std() * np.sqrt(annualization_factor) if not negative_returns_only.empty else 0.0

                    if downside_std_dev_ann > FLOAT_EPSILON:
                        sortino_ratio = mean_return_ann / downside_std_dev_ann
            
            # Average Holding Duration in Bars (from trade_history_df)
            avg_holding_duration_bars = self.trade_history_df['bars_held'].mean() if 'bars_held' in self.trade_history_df.columns and not self.trade_history_df['bars_held'].dropna().empty else np.nan


            # --- Final Metrics Dictionary ---
            # Ensure keys match those expected by MonteCarloAnalyzer or for general reporting
            self.metrics = {
                "initial_capital": initial_capital,
                "final_capital": final_capital,
                "net_profit": total_net_pnl_from_trades,
                "return_on_capital_pct": (total_net_pnl_from_trades / initial_capital) * 100 if initial_capital > FLOAT_EPSILON else 0.0,
                "max_drawdown": max_drawdown,
                "total_trades": num_trades,
                "win_rate": win_rate, # As a fraction for consistency, can convert to % for display
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "total_profit_trades": num_wins,
                "total_loss_trades": num_losses,
                "gross_profit": total_profit_pnl, # Sum of positive net_pnl
                "gross_loss": total_loss_pnl, # Sum of negative net_pnl
                "avg_profit_per_trade": avg_win_pnl,
                "avg_loss_per_trade": avg_loss_pnl,
                "avg_pnl_per_trade": avg_pnl_per_trade,
                "edge_expected_value": edge_expected_value, # NEW
                "total_fees": total_fees,
                "avg_holding_duration_bars": avg_holding_duration_bars,
                "equity_pnl_discrepancy": consistency_status, # String status
                # Add config parameters for full traceability in analysis reports, especially for Monte Carlo
                "config_symbol": self.symbol,
                "config_interval": self.interval,
                "config_model_type": self.model_type,
                # Add more relevant config parameters from app_config here if needed for analysis reporting
                "config_initial_capital": initial_capital,
                "config_leverage": getattr(app_config.strategy, 'leverage', np.nan),
                "config_risk_per_trade_pct": getattr(app_config.strategy, 'risk_per_trade_pct', np.nan),
                "config_trading_fee_rate": getattr(app_config.strategy, 'trading_fee_rate', np.nan),
                "config_slippage_tolerance_pct": getattr(app_config.strategy, 'slippage_tolerance_fraction', np.nan),
                "config_maintenance_margin_rate": getattr(app_config.backtest, 'maintenance_margin_rate', np.nan),
                "config_liquidation_fee_rate": getattr(app_config.backtest, 'liquidation_fee_rate', np.nan),
                "config_max_concurrent_trades": getattr(app_config.backtest, 'max_concurrent_trades', np.nan),
                "config_exit_on_neutral_signal": getattr(app_config.strategy, 'exit_on_neutral_signal', np.nan),
                "config_allow_long_trades": getattr(app_config.strategy, 'allow_long_trades', np.nan),
                "config_allow_short_trades": getattr(app_config.strategy, 'allow_short_trades', np.nan),
                "config_confidence_filter_enabled": getattr(app_config.strategy, 'confidence_filter_enabled', np.nan),
                "config_confidence_threshold_long_pct": getattr(app_config.strategy, 'confidence_threshold_long_pct', np.nan),
                "config_confidence_threshold_short_pct": getattr(app_config.strategy, 'confidence_threshold_short_pct', np.nan),
                "config_volatility_regime_filter_enabled": getattr(app_config.strategy, 'volatility_regime_filter_enabled', np.nan),
                # Note: volatility_regime_max_holding_bars is a dict, might need special handling for CSV saving if not flattened
                # For now, will convert to string or handle as a dict directly in JSON output
                "config_volatility_regime_max_holding_bars": str(getattr(app_config.strategy.volatility_regime_params, 'max_holding_bars', {})),
                "config_allow_trading_in_volatility_regime": str(getattr(app_config.strategy.volatility_regime_params, 'allow_trading', {})),
                "config_volatility_adjustment_enabled": getattr(app_config.strategy.sltp_params, 'enabled', np.nan),
                "config_volatility_window_bars": getattr(app_config.strategy.sltp_params, 'volatility_window_bars', np.nan),
                "config_fixed_take_profit_pct": getattr(app_config.strategy.sltp_params, 'fixed_take_profit_pct', np.nan),
                "config_fixed_stop_loss_pct": getattr(app_config.strategy.sltp_params, 'fixed_stop_loss_pct', np.nan),
                "config_alpha_take_profit": getattr(app_config.strategy.sltp_params, 'alpha_take_profit', np.nan),
                "config_alpha_stop_loss": getattr(app_config.strategy.sltp_params, 'alpha_stop_loss', np.nan),
                "config_trend_filter_enabled": getattr(app_config.strategy, 'trend_filter_enabled', np.nan),
                "config_trend_filter_ema_period": getattr(app_config.strategy, 'trend_filter_ema_period', np.nan),
                "config_min_quantity": getattr(app_config.exchange.options, 'minQty', np.nan),
                "config_min_notional": getattr(app_config.exchange.options, 'minNotional', np.nan),
                "config_tie_breaker": getattr(app_config.strategy, 'tie_breaker', 'neutral'), # Assuming this default
                "config_random_seed": getattr(app_config.general, 'random_seed', np.nan),
            }
            logger.info("Performance metrics calculated.")

            # Optionally, compare calculated metrics with saved metrics if loaded
            if self.saved_metrics:
                 logger.info("Comparing calculated metrics with saved metrics:")
                 for key_display, calc_value_native in self.metrics.items():
                      # Attempt to find a corresponding key in saved_metrics (more robust matching)
                      saved_value = self.saved_metrics.get(key_display) # Try direct match first
                      
                      if saved_value is None: # If not found by exact match, try normalized
                          normalized_key_display = key_display.replace(" ", "_").lower()
                          for saved_key, val in self.saved_metrics.items():
                              if saved_key.replace(" ", "_").lower() == normalized_key_display:
                                  saved_value = val
                                  break

                      if saved_value is not None:
                           try:
                                # Convert calculated value (which might be float, int, or string from f-string formatting)
                                # to a numeric type for comparison.
                                # Handle string representations of percentages, Inf, NaN
                                calc_value_float = float(str(calc_value_native).replace('%', '').replace('Inf', str(np.inf)).replace('-Inf', str(-np.inf)).replace('NaN', str(np.nan)).strip())
                                saved_value_float = float(str(saved_value).replace('%', '').replace('Inf', str(np.inf)).replace('-Inf', str(-np.inf)).replace('NaN', str(np.nan)).strip())

                                if np.isclose(calc_value_float, saved_value_float, atol=FLOAT_EPSILON, equal_nan=True):
                                     logger.info(f"  Metric '{key_display}': Consistent (Calculated: {calc_value_native}, Saved: {saved_value})")
                                else:
                                     logger.warning(f"  Metric '{key_display}': Discrepancy (Calculated: {calc_value_native}, Saved: {saved_value})")
                           except ValueError:
                                # If conversion to float fails, compare as strings
                                if str(calc_value_native) == str(saved_value):
                                     logger.info(f"  Metric '{key_display}': Consistent (Calculated: {calc_value_native}, Saved: {saved_value})")
                                else:
                                     logger.warning(f"  Metric '{key_display}': Discrepancy (Calculated: {calc_value_native}, Saved: {saved_value})")
                      else:
                            logger.debug(f"  Metric '{key_display}': Not found in saved metrics.")

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
            elif 'M' in interval: # Assuming monthly, approximate to 30 days
                 logger.warning("Monthly interval 'M' is approximated to 30 days for minute conversion.")
                 return float(interval.replace('M', '')) * 30 * 24 * 60 if interval.replace('M', '').isdigit() else 30 * 24 * 60
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
