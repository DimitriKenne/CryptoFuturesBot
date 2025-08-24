# utils/strategy_evaluation/monte_carlo_analyzer.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# New imports for modularity
from config.params import AppConfig
from utils.strategy_evaluation.plotting_utils import PlottingUtils


logger = logging.getLogger(__name__)

class MonteCarloAnalyzer:
    """
    Handles analysis of Monte Carlo results. This is a pure analysis engine.
    It aggregates metrics and equity curves from multiple simulations,
    compares them to a deterministic baseline, and returns all generated
    artifacts (tables and plots) without performing any file I/O.
    """
    def __init__(self,
                 app_config: AppConfig,
                 metrics_df: pd.DataFrame,
                 all_equity_curves: List[pd.Series],
                 deterministic_results: Dict[str, Any],
                 all_simulated_paths: List[pd.DataFrame]):
        """
        Initializes the MonteCarloAnalyzer.

        Args:
            app_config (AppConfig): The global application configuration object.
            metrics_df (pd.DataFrame): DataFrame containing summary metrics for each simulation run.
            all_equity_curves (List[pd.Series]): List of Pandas Series, each representing an equity curve.
            deterministic_results (Dict[str, Any]): Dictionary with results from the deterministic backtest.
                                                    Expected keys: 'metrics', 'equity_curve', 'ohlcv_data'.
            all_simulated_paths (List[pd.DataFrame]): List of DataFrames of simulated OHLCV paths.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app_config = app_config
        self.metrics_df = metrics_df
        self.all_equity_curves = all_equity_curves
        self.deterministic_results = deterministic_results
        self.all_simulated_paths = all_simulated_paths

        # Initialize PlottingUtils (now stateless)
        self.plotting_utils = PlottingUtils()
        self.logger.info("MonteCarloAnalyzer initialized.")

    def _calculate_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Calculates and returns descriptive statistics tables for the aggregated performance metrics.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the summary stats and raw metrics DataFrames.
        """
        self.logger.info("Calculating summary statistics tables...")
        if self.metrics_df.empty:
            self.logger.warning("Metrics DataFrame is empty. Returning empty tables.")
            return {'summary_stats': pd.DataFrame(), 'all_simulation_metrics': pd.DataFrame()}

        # Make a copy to avoid modifying the original DataFrame
        metrics_copy = self.metrics_df.copy()

        # Ensure numeric columns are treated as such, coercing errors for robust stats
        numeric_cols = metrics_copy.select_dtypes(include=np.number).columns.tolist()
        for col in metrics_copy.columns:
            if col not in numeric_cols:
                metrics_copy[col] = pd.to_numeric(metrics_copy[col], errors='coerce')

        summary_stats = metrics_copy.describe().transpose()
        
        tables = {
            'summary_stats': summary_stats,
            'all_simulation_metrics': self.metrics_df # Return original for full data
        }
        self.logger.info("Summary tables calculated.")
        return tables

    def _generate_plots(self) -> Dict[str, Any]:
        """
        Generates all Monte Carlo analysis plots by delegating to PlottingUtils.
        This method now correctly unpacks the deterministic results.
        """
        self.logger.info("Generating Monte Carlo analysis plot figures...")
        plots_dict = {}

        # --- CORRECTED: Unpack all deterministic results ---
        det_metrics = self.deterministic_results.get('metrics', {})
        det_equity = self.deterministic_results.get('equity_curve')
        det_ohlcv = self.deterministic_results.get('ohlcv_data')

        # Define the specific metrics we want to plot for clarity
        return_metric = 'Total Return (%)'
        drawdown_metric = 'Max Drawdown (%)'

        # Generate plots using the stateless PlottingUtils, passing the unpacked values
        plots_dict['return_distribution'] = self.plotting_utils.plot_performance_distribution(
            metrics_df=self.metrics_df,
            metric=return_metric,
            deterministic_metric_value=det_metrics.get(return_metric)
        )
        
        plots_dict['equity_curves_comparison'] = self.plotting_utils.plot_equity_curves_comparison(
            all_equity_curves=self.all_equity_curves,
            deterministic_equity=det_equity,
            num_to_plot=self.app_config.trading.backtest.monte_carlo_plot_simulations
        )
        
        plots_dict['risk_reward_scatter'] = self.plotting_utils.plot_risk_reward_scatter(
            metrics_df=self.metrics_df,
            x_metric=drawdown_metric,
            y_metric=return_metric,
            deterministic_x=det_metrics.get(drawdown_metric),
            deterministic_y=det_metrics.get(return_metric)
        )

        plots_dict['simulated_ohlcv_paths'] = self.plotting_utils.plot_simulated_ohlcv_paths(
            all_simulated_paths=self.all_simulated_paths,
            deterministic_ohlcv=det_ohlcv,
            num_to_plot=self.app_config.trading.backtest.monte_carlo_plot_simulations
        )

        self.logger.info(f"{len(plots_dict)} Monte Carlo plot figures generated.")
        return plots_dict

    def generate_analysis_artifacts(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Runs the full analysis pipeline for Monte Carlo results and returns all generated artifacts.
        """
        self.logger.info("--- Starting Monte Carlo Results Analysis ---")
        
        tables = self._calculate_summary_tables()
        plots = self._generate_plots()
        
        self.logger.info("--- Monte Carlo Results Analysis Complete ---")
        return tables, plots