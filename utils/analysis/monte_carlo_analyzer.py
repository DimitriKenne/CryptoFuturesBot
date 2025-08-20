# utils/analysis/monte_carlo_analyzer.py

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt # Still needed for plt.figure, plt.plot, plt.close etc.
import seaborn as sns # Still needed for sns.histplot, sns.boxplot, sns.barplot
from pathlib import Path
from typing import Dict, Any, Optional, List

# New imports for modularity
from config.params import AppConfig # Needed for MetricsCalculator initialization
from utils.analysis.metrics_calculator import MetricsCalculator
from utils.analysis.plotting_utils import PlottingUtils
from config.paths import PATHS # For output directory


logger = logging.getLogger(__name__)

class MonteCarloAnalyzer:
    """
    Handles analysis, plotting, and saving of Monte Carlo results.
    Aggregates metrics and equity curves from multiple simulations,
    and compares them to a deterministic baseline.
    """
    def __init__(self, metrics_df: pd.DataFrame, all_equity_curves: List[pd.Series],
                 deterministic_results: Dict[str, Any], output_dir: Path,
                 all_simulated_paths: List[pd.DataFrame], app_config: AppConfig):
        """
        Initializes the MonteCarloAnalyzer.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing summary metrics for each simulation run.
            all_equity_curves (List[pd.Series]): List of Pandas Series, each representing an equity curve from a simulation.
            deterministic_results (Dict[str, Any]): A dictionary containing results from the deterministic backtest.
                                                    Expected keys: 'trades', 'equity_curve', 'metrics', 'ohlcv_data'.
            output_dir (Path): The directory where analysis results and plots will be saved.
            all_simulated_paths (List[pd.DataFrame]): List of DataFrames, each containing simulated OHLCV paths.
            app_config (AppConfig): The global application configuration object.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_df = metrics_df
        self.all_equity_curves = all_equity_curves
        self.deterministic_results = deterministic_results
        self.output_dir = output_dir
        self.all_simulated_paths = all_simulated_paths
        self.app_config = app_config

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"MonteCarloAnalyzer initialized. Results will be saved to: {self.output_dir}")

        # Initialize PlottingUtils with the Monte Carlo specific output directory
        # The symbol, interval, model_type can be extracted from deterministic_results or config
        self.plotting_utils = PlottingUtils(
            symbol=self.deterministic_results.get('config_symbol', 'UNKNOWN'),
            interval=self.deterministic_results.get('config_interval', 'UNKNOWN'),
            model_type=self.app_config.model.model_type, # Use model type from app config
            analysis_dir=self.output_dir # Plots will go into the MC-specific output dir
        )

    def save_summary_stats(self):
        """
        Calculates and saves descriptive statistics for the aggregated performance metrics.
        """
        if self.metrics_df.empty:
            self.logger.warning("Metrics DataFrame is empty. Cannot save summary statistics.")
            return

        summary_stats = self.metrics_df.describe().transpose()
        summary_filepath = self.output_dir / PATHS.get("monte_carlo_summary_stats_pattern", "1_performance_summary_stats.csv")

        try:
            summary_stats.to_csv(summary_filepath)
            self.logger.info(f"Performance summary stats saved to {summary_filepath}")
            self.logger.info(f"\n{summary_stats}") # Log the summary stats
        except Exception as e:
            self.logger.error(f"Error saving performance summary stats: {e}", exc_info=True)

        # Also save the raw metrics DataFrame for detailed inspection
        raw_metrics_filepath = self.output_dir / PATHS.get("monte_carlo_raw_metrics_pattern", "all_simulation_metrics.csv")
        try:
            self.metrics_df.to_csv(raw_metrics_filepath, index=False)
            self.logger.info(f"Raw metrics for all simulations saved to {raw_metrics_filepath}")
        except Exception as e:
            self.logger.error(f"Error saving raw simulation metrics: {e}", exc_info=True)


    def plot_performance_distribution(self, metric: str = 'Total Return (%)'):
        """
        Plots the distribution of a key performance metric from multiple simulations,
        delegating to PlottingUtils.
        """
        det_metric_value = self.deterministic_results.get('metrics', {}).get(metric)
        self.plotting_utils.plot_performance_distribution(
            metrics_df=self.metrics_df,
            metric=metric,
            deterministic_metric_value=det_metric_value # Pass for plotting on the same chart
        )


    def plot_equity_curves(self, num_to_plot=50):
        """
        Plots a sample of simulated equity curves against a deterministic baseline,
        delegating to PlottingUtils.
        """
        det_equity = self.deterministic_results.get('equity_curve')
        self.plotting_utils.plot_equity_curves_comparison(
            all_equity_curves=self.all_equity_curves,
            deterministic_equity=det_equity,
            num_to_plot=num_to_plot
        )

    def plot_risk_reward_scatter(self, x_metric: str = 'Max Drawdown (%)', y_metric: str = 'Total Return (%)'):
        """
        Creates a scatter plot to visualize the risk/reward profile, delegating to PlottingUtils.
        """
        det_x = self.deterministic_results.get('metrics', {}).get(x_metric)
        det_y = self.deterministic_results.get('metrics', {}).get(y_metric)
        self.plotting_utils.plot_risk_reward_scatter(
            metrics_df=self.metrics_df,
            x_metric=x_metric,
            y_metric=y_metric,
            deterministic_x=det_x,
            deterministic_y=det_y
        )

    def plot_simulated_ohlcv_paths(self, num_to_plot=5):
        """
        Plots a sample of simulated OHLCV paths (close prices) for visual inspection, delegating to PlottingUtils.
        """
        det_ohlcv = self.deterministic_results.get('ohlcv_data')
        self.plotting_utils.plot_simulated_ohlcv_paths(
            all_simulated_paths=self.all_simulated_paths,
            deterministic_ohlcv=det_ohlcv,
            num_to_plot=num_to_plot
        )

    def run_full_analysis(self):
        """Runs all analysis and plotting steps to generate a comprehensive Monte Carlo report."""
        self.logger.info("--- Starting Monte Carlo Results Analysis ---")
        self.save_summary_stats()
        self.plot_performance_distribution(metric='Total Return (%)')
        self.plot_equity_curves()
        self.plot_risk_reward_scatter()
        self.plot_simulated_ohlcv_paths(num_to_plot=self.app_config.trading.backtest.monte_carlo_plot_simulations)
        self.logger.info("--- Monte Carlo Results Analysis Complete ---")

