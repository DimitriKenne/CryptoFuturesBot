# utils/analysis/performance_analyzer.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt # Still needed for plt.figure, plt.plot, plt.close etc.
import seaborn as sns # Still needed for sns.histplot, sns.boxplot, sns.barplot
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
import json # Import json for loading metrics saved by backtester


# Import configurations directly, assuming they are always available.
# Removed fallback logic as per user's request for conciseness.
from config.paths import PATHS
from config.params import AppConfig, FLOAT_EPSILON
from utils.strategy_execution.trade_calculation_helpers import TradeCalculationHelpers

# New imports for modularity
from utils.analysis.metrics_calculator import MetricsCalculator
from utils.analysis.plotting_utils import PlottingUtils


logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Analyzes trading performance from backtesting or live trading results.
    Calculates key metrics, generates equity curves, and plots various performance
    visualizations. Designed to work with standardized trade history and equity curve data.
    """
    def __init__(self,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 initial_capital: float,
                 trade_history_data: Optional[List[Dict[str, Any]]] = None,
                 equity_data: Optional[pd.Series] = None,
                 results_type: Literal['backtest', 'live'] = 'backtest',
                 app_config: Optional[AppConfig] = None # Added app_config for full context
                 ):
        """
        Initializes the PerformanceAnalyzer.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '1h').
            model_type (str): The type of ML model used (e.g., 'xgboost', 'lstm').
            initial_capital (float): The starting capital for the trading simulation.
            trade_history_data (Optional[List[Dict[str, Any]]]): List of dictionaries
                                                                 representing closed trades.
            equity_data (Optional[pd.Series]): Pandas Series representing the equity curve over time.
            results_type (Literal['backtest', 'live']): Indicates if analyzing backtest or live results.
            app_config (Optional[AppConfig]): The global application configuration object.
        """
        self.logger = logging.getLogger(self.__class__.__name__) # Correctly initialize logger
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.initial_capital = initial_capital
        self.results_type = results_type
        self.app_config = app_config

        # Initialize trade_history_df
        self.trade_history_df = pd.DataFrame(trade_history_data) if trade_history_data else pd.DataFrame()

        # Initialize equity_df to always be a DataFrame with an 'equity' column
        if isinstance(equity_data, pd.Series):
            self.equity_df = equity_data.to_frame(name='equity')
        elif isinstance(equity_data, pd.DataFrame) and 'equity' in equity_data.columns:
            self.equity_df = equity_data
        else:
            self.equity_df = pd.DataFrame(columns=['equity']) # Ensure it's an empty DataFrame with the expected column

        self.metrics: Dict[str, Any] = {} # Stores calculated performance metrics

        # Determine output directory based on results_type
        base_analysis_dir_key = "backtesting_analysis_dir" if results_type == 'backtest' else "live_trading_analysis_dir"
        base_analysis_dir = PATHS.get(base_analysis_dir_key)
        if base_analysis_dir is None:
            # If the specific key is not found, fallback to the general analysis_dir
            # This is a soft fallback for missing specific keys, not a full ImportError fallback.
            base_analysis_dir = PATHS.get("analysis_dir", Path("./results/analysis"))
            self.logger.warning(f"Analysis directory key '{base_analysis_dir_key}' not found in PATHS. Using general analysis directory: {base_analysis_dir}")


        # Create a unique subdirectory for results: model_type/symbol_interval
        self.analysis_dir = Path(base_analysis_dir) / self.model_type / f"{self.symbol.replace('/', '_')}_{self.interval}"
        
        self.analysis_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        self.logger.info(f"Using configured analysis save directory: {self.analysis_dir.parent.parent.name}/{self.analysis_dir.parent.name}/{self.analysis_dir.name}")
        self.logger.info(f"Analysis results will be saved to: {self.analysis_dir}")


        # Initialize MetricsCalculator and PlottingUtils
        self.metrics_calculator = MetricsCalculator(app_config=self.app_config, initial_capital=self.initial_capital)
        self.plotting_utils = PlottingUtils(
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type,
            analysis_dir=self.analysis_dir
        )

        self.logger.info(f"PerformanceAnalyzer initialized for {self.symbol} {self.interval} ({self.model_type}) {self.results_type} results.")


    def _load_data(self):
        """
        Loads trade history and equity curve data from files if not already provided.
        """
        self.logger.info("Attempting to load data for analysis...")
        # Determine paths based on results_type
        trades_pattern_key = "backtesting_trades_pattern" if self.results_type == 'backtest' else "live_trading_trades_pattern"
        equity_pattern_key = "backtesting_equity_pattern" if self.results_type == 'backtest' else "live_trading_equity_pattern"
        metrics_pattern_key = "backtesting_metrics_pattern" if self.results_type == 'backtest' else "live_trading_metrics_pattern"

        base_results_dir_key = "backtesting_results_dir" if self.results_type == 'backtest' else "live_trading_results_dir"
        base_results_dir = PATHS.get(base_results_dir_key)

        if base_results_dir is None:
            self.logger.error(f"Base results directory key '{base_results_dir_key}' not found in PATHS. Cannot load data.")
            return

        # Build file paths
        trades_filepath = Path(base_results_dir) / PATHS[trades_pattern_key].format(
            symbol=self.symbol, interval=self.interval, model_type=self.model_type
        )
        equity_filepath = Path(base_results_dir) / PATHS[equity_pattern_key].format(
            symbol=self.symbol, interval=self.interval, model_type=self.model_type
        )
        metrics_filepath = Path(base_results_dir) / PATHS[metrics_pattern_key].format(
            symbol=self.symbol, interval=self.interval, model_type=self.model_type
        )

        # Load trade history
        if self.trade_history_df.empty:
            try:
                self.trade_history_df = pd.read_parquet(trades_filepath)
                # Ensure datetime columns are timezone-aware (UTC) and numeric columns are correct type
                # Updated to match new `Trade` dataclass field names in TradeExecutionEngine output
                for col in ['entry_timestamp', 'exit_timestamp']:
                    if col in self.trade_history_df.columns:
                        self.trade_history_df[col] = pd.to_datetime(self.trade_history_df[col], errors='coerce', utc=True)
                        self.trade_history_df.dropna(subset=[col], inplace=True)
                for col in ['net_pnl', 'total_fees', 'gross_pnl']:
                    if col in self.trade_history_df.columns:
                        self.trade_history_df[col] = pd.to_numeric(self.trade_history_df[col], errors='coerce')
                        self.trade_history_df.dropna(subset=[col], inplace=True)
                if 'exit_timestamp' in self.trade_history_df.columns:
                    self.trade_history_df.set_index('exit_timestamp', inplace=True)
                self.logger.info(f"Loaded {len(self.trade_history_df)} trades from {trades_filepath}")
            except FileNotFoundError:
                self.logger.warning(f"Trade history file not found at {trades_filepath}. Will proceed with available data.")
                self.trade_history_df = pd.DataFrame() # Ensure it's an empty DataFrame
            except Exception as e:
                self.logger.error(f"Error loading trade history from {trades_filepath}: {e}", exc_info=True)
                self.trade_history_df = pd.DataFrame() # Ensure it's an empty DataFrame


        # Load equity curve
        # Corrected logic to ensure self.equity_df is always a DataFrame with 'equity' column
        if self.equity_df.empty or 'equity' not in self.equity_df.columns or self.equity_df['equity'].empty:
            try:
                loaded_equity_data = pd.read_parquet(equity_filepath)
                if isinstance(loaded_equity_data, pd.Series):
                    self.equity_df = loaded_equity_data.to_frame(name='equity')
                elif isinstance(loaded_equity_data, pd.DataFrame) and 'equity' in loaded_equity_data.columns:
                    self.equity_df = loaded_equity_data
                else:
                    raise ValueError("Loaded equity data is not in expected Series or 'equity' column DataFrame format.")
                
                # Ensure index is DatetimeIndex and 'equity' column is numeric
                if not self.equity_df.empty:
                    if not isinstance(self.equity_df.index, pd.DatetimeIndex):
                        self.equity_df.index = pd.to_datetime(self.equity_df.index, errors='coerce', utc=True)
                        self.equity_df.dropna(subset=[self.equity_df.index.name if self.equity_df.index.name else self.equity_df.index], inplace=True) # handle unnamed index

                    if 'equity' in self.equity_df.columns:
                        self.equity_df['equity'] = pd.to_numeric(self.equity_df['equity'], errors='coerce')
                        self.equity_df.dropna(subset=['equity'], inplace=True)
                    else:
                        self.logger.warning("Loaded equity DataFrame does not have an 'equity' column. Resetting to empty.")
                        self.equity_df = pd.DataFrame(columns=['equity']) # Reset if column is missing

                    if not self.equity_df.index.is_monotonic_increasing:
                        self.equity_df = self.equity_df.sort_index()

                self.logger.info(f"Loaded equity curve with {len(self.equity_df)} points from {equity_filepath}")
            except FileNotFoundError:
                self.logger.warning(f"Equity curve file not found at {equity_filepath}. Will try to reconstruct.")
                self.equity_df = pd.DataFrame(columns=['equity']) # Ensure it's an empty DataFrame
            except Exception as e:
                self.logger.error(f"Error loading equity curve from {equity_filepath}: {e}", exc_info=True)
                self.equity_df = pd.DataFrame(columns=['equity']) # Ensure it's an empty DataFrame

        # Load metrics (if they were saved separately by Backtester)
        if not self.metrics: # Only load if metrics are not already populated
            try:
                with open(metrics_filepath, 'r') as f:
                    self.metrics = json.load(f)
                self.logger.info(f"Loaded metrics from {metrics_filepath}")
            except FileNotFoundError:
                self.logger.warning(f"Metrics file not found at {metrics_filepath}.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding metrics JSON from {metrics_filepath}: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error loading metrics from {metrics_filepath}: {e}", exc_info=True)


    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculates a comprehensive set of performance metrics for the trading strategy.
        Delegates to MetricsCalculator.
        """
        self.metrics = self.metrics_calculator.calculate_all_metrics(
            trade_history_df=self.trade_history_df,
            equity_df=self.equity_df,
            symbol=self.symbol,
            interval=self.interval
        )
        return self.metrics

    def _convert_numpy_to_python_types(self, obj):
        """Recursively converts NumPy types in a dictionary or list to Python native types."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python_types(elem) for elem in obj]
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj

    def _save_metrics(self):
        """Saves the calculated performance metrics to a JSON file and CSV."""
        if not self.metrics:
            self.logger.warning("No metrics calculated to save.")
            return

        json_filepath = self.analysis_dir / "summary_metrics.json"
        csv_filepath = self.analysis_dir / "summary_metrics.csv" # Always save a CSV

        try:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = self._convert_numpy_to_python_types(self.metrics)
            with open(json_filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            self.logger.info(f"Summary metrics saved to {json_filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save summary metrics to {json_filepath}: {e}", exc_info=True)

        try:
            # Convert values to strings for CSV to handle mixed types (floats, 'Inf', 'NaN')
            metrics_display = {k: str(v) for k, v in self.metrics.items()}
            metrics_df = pd.DataFrame([metrics_display])
            metrics_df.to_csv(csv_filepath, index=False) # Overwrite or create
            self.logger.info(f"Summary metrics also saved to {csv_filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics to {csv_filepath}: {e}", exc_info=True)


    def _generate_plots(self):
        """Generates and saves various performance plots. Delegates to PlottingUtils."""
        self.logger.info("Generating performance plots...")
        
        # Pass dataframes directly to PlottingUtils methods
        self.plotting_utils.plot_equity_curve(self.equity_df)
        self.plotting_utils.plot_drawdown_curve(self.equity_df)
        self.plotting_utils.plot_trade_pnl_distribution(self.trade_history_df)
        self.plotting_utils.plot_exit_reason_pnl_boxplot(self.trade_history_df)
        self.plotting_utils.plot_exit_reason_frequency_barplot(self.trade_history_df)

        self.logger.info("Performance plots generated.")


    def run_full_analysis(self):
        """Runs the full analysis pipeline: calculate metrics, generate plots, save results."""
        self.logger.info(f"\n{'='*40}\n📊 Running Analysis for {self.symbol} {self.interval} ({self.model_type}) {self.results_type}\n{'='*40}")
        try:
            # If data was not passed to constructor, try to load from files
            if self.trade_history_df.empty or self.equity_df.empty or not self.metrics:
                self._load_data()
            
            # Check if equity was loaded or calculated successfully.
            if self.equity_df.empty or 'equity' not in self.equity_df.columns or self.equity_df['equity'].empty:
                self.logger.error("❌ Could not load or calculate equity curve. Analysis aborted.")
                return

            self.calculate_summary_metrics() # This will populate self.metrics

            # Log the summary metrics
            self.logger.info(f"\n{'-'*30}\n📈 Analysis Summary\n{'-'*30}")
            if self.metrics:
                for key, value in self.metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"{key}: {value:.4f}")
                    else:
                        self.logger.info(f"{key}: {value}")
            else:
                self.logger.info("No metrics were calculated.")
            self.logger.info("=" * 50)

            self._save_metrics()
            self._generate_plots()

        except (FileNotFoundError, ValueError, IOError) as e:
            self.logger.error(f"❌ Analysis aborted due to data loading/processing error: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"❌ An unexpected error occurred during analysis: {e}", exc_info=True)
        finally:
            self.logger.info(f"\n{'='*40}\n✅ Analysis Complete\n{'='*40}")

