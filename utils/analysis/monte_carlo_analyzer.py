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
            all_equity_curves (List[pd.Series]): List of pandas Series, where each Series represents
                                                  the equity curve of one simulation.
            deterministic_results (Dict[str, Any]): Dictionary containing results from the
                                                     single deterministic backtest (e.g., equity_curve, metrics, ohlcv_data).
            output_dir (Path): The directory where all analysis plots and summary files will be saved.
            all_simulated_paths (List[pd.DataFrame]): List of raw simulated OHLCV DataFrames.
            app_config (AppConfig): The global application configuration object.
        """
        self.metrics_df = metrics_df
        self.all_equity_curves = all_equity_curves
        self.deterministic_results = deterministic_results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_simulated_paths = all_simulated_paths # Store all simulated paths
        self.app_config = app_config # Store app_config for passing to sub-modules

        sns.set_style("darkgrid") # Apply seaborn style

        # Initialize PlottingUtils and MetricsCalculator
        self.plotting_utils = PlottingUtils(
            symbol=self.deterministic_results.get("config_symbol", "UNKNOWN"),
            interval=self.deterministic_results.get("config_interval", "UNKNOWN"),
            model_type=self.deterministic_results.get("config_model_type", "UNKNOWN"),
            analysis_dir=self.output_dir # Monte Carlo analyzer saves directly to its specific output_dir
        )
        # MetricsCalculator might not be directly used for aggregating MC results,
        # but its methods for calculating individual metrics can be reused if needed.
        # For overall MC analysis, we're mostly interested in the distributions of already calculated metrics.
        # However, for consistency, we pass initial capital to it.
        initial_capital_det = self.deterministic_results.get('initial_capital', 0.0)
        self.metrics_calculator = MetricsCalculator(app_config=self.app_config, initial_capital=initial_capital_det)


    def save_summary_stats(self):
        """Calculates and saves descriptive statistics of performance metrics across simulations."""
        # Updated KPI names to match PerformanceAnalyzer output
        kpis = ['Total Return (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Profit Factor', 'Number of Trades', 'Total Net PnL (Sum Trades)']
        summary_stats = pd.DataFrame(index=['mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        for kpi in kpis:
            if kpi in self.metrics_df.columns:
                # Ensure the column is numeric before describing
                numeric_col = pd.to_numeric(self.metrics_df[kpi], errors='coerce').dropna()
                if not numeric_col.empty:
                    summary_stats[kpi] = numeric_col.describe(percentiles=[.25, .5, .75]).loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                else:
                    summary_stats[kpi] = np.nan # Fill with NaN if no numeric data
            else:
                summary_stats[kpi] = np.nan # Fill with NaN if column not found
        
        filepath = self.output_dir / "1_performance_summary_stats.csv"
        summary_stats.to_csv(filepath)
        logger.info(f"Performance summary stats saved to {filepath}")
        logger.info("\n" + summary_stats.to_string(float_format="%.2f"))

        raw_filepath = self.output_dir / "all_simulation_metrics.csv"
        self.metrics_df.to_csv(raw_filepath, index=False)
        logger.info(f"Raw metrics for all simulations saved to {raw_filepath}")

    def plot_performance_distribution(self, metric='Total Return (%)'):
        """Plots the distribution of a key performance metric, delegating to PlottingUtils."""
        det_metric_value = self.deterministic_results.get('metrics', {}).get(metric)
        self.plotting_utils.plot_performance_distribution(
            metrics_df=self.metrics_df,
            metric=metric,
        )
        # Add deterministic line after the general plot (PlottingUtils can't do this directly)
        # This part remains in MonteCarloAnalyzer as it needs info from deterministic_results
        filepath = self.output_dir / self.plotting_utils._get_plot_filename(f"distribution_{metric.replace(' (%)', '').replace(' ', '_').lower()}")
        if filepath.exists() and pd.notna(det_metric_value) and isinstance(det_metric_value, (int, float)):
            try:
                # Reload the plot, add the line, and resave
                fig = plt.imread(filepath) # This reads as an image, need to regenerate the plot
                plt.close(fig) # Close the dummy figure
                
                # Regenerate the plot to add the line
                plot_data = pd.to_numeric(self.metrics_df[metric], errors='coerce').dropna()
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.histplot(plot_data, kde=True, bins=30, stat="density", ax=ax)
                ax.axvline(det_metric_value, color='red', linestyle='--', linewidth=2, label=f'Deterministic Result ({det_metric_value:.2f}%)')
                ax.set_title(f'Distribution of {metric}', fontsize=16)
                ax.set_xlabel(metric)
                ax.set_ylabel('Density')
                ax.legend()
                self.plotting_utils._save_plot(fig, f"distribution_{metric.replace(' (%)', '').replace(' ', '_').lower()}")

            except Exception as e:
                logger.error(f"Error adding deterministic line to performance distribution plot: {e}", exc_info=True)


    def plot_equity_curves(self, num_to_plot=50):
        """Plots a sample of simulated equity curves against the deterministic baseline, delegating to PlottingUtils."""
        det_equity = self.deterministic_results.get('equity_curve')
        self.plotting_utils.plot_equity_curves_comparison(
            all_equity_curves=self.all_equity_curves,
            deterministic_equity=det_equity,
            num_to_plot=num_to_plot
        )

    def plot_risk_reward_scatter(self, x_metric='Max Drawdown (%)', y_metric='Total Return (%)'):
        """Creates a scatter plot to visualize the risk/reward profile, delegating to PlottingUtils."""
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
        logger.info("--- Starting Monte Carlo Results Analysis ---")
        self.save_summary_stats()
        self.plot_performance_distribution(metric='Total Return (%)')
        self.plot_equity_curves()
        self.plot_risk_reward_scatter(x_metric='Max Drawdown (%)', y_metric='Total Return (%)')
        self.plot_simulated_ohlcv_paths()
        logger.info("--- Monte Carlo Results Analysis Complete ---")

