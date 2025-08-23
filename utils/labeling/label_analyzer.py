# utils/labeling/label_analyzer.py

import logging
import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt

# Import refactored components
from .analysis_calculator import AnalysisCalculator
from .analysis_plotter import AnalysisPlotter
from utils.data_management.data_manager import DataManager

class LabelAnalyzer:
    """
    Orchestrates common analyses on labeled dataframes by delegating to
    injected helper components for calculation, plotting, and file management.
    """
    def __init__(
        self,
        data_manager: DataManager,
        calculator: AnalysisCalculator,
        plotter: AnalysisPlotter,
        logger: Optional[logging.Logger] = None
    ):
        self.dm = data_manager
        self.calculator = calculator
        self.plotter = plotter
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("LabelAnalyzer initialized with required components.")

    def perform_all_analyses(
        self,
        df_combined: pd.DataFrame,
        symbol: str,
        interval: str,
        labeling_strategy: str,
        future_horizons: List[int],
        f_window: int
    ):
        self.logger.info(f"--- Starting Common Analyses for {symbol} {interval} [{labeling_strategy}] ---")
        
        common_kwargs = {'symbol': symbol, 'interval': interval, 'labeling_strategy': labeling_strategy}
        
        # Centralized loop for running all analyses modules
        analyses_to_run = [
            (self._analyze_label_distribution, {}),
            (self._analyze_mfe_mae, {'f_window': f_window}),
            (self._analyze_future_returns, {'horizons': future_horizons}),
            (self._analyze_regime_profitability, {'horizons': future_horizons}),
        ]

        for method, params in analyses_to_run:
            try:
                method(df_combined.copy(), **common_kwargs, **params)
            except Exception as e:
                self.logger.error(f"Failed to run analysis '{method.__name__}': {e}", exc_info=True)
        
        self.logger.info(f"--- Completed Common Analyses for {symbol} {interval} [{labeling_strategy}] ---")

    def _analyze_label_distribution(self, df: pd.DataFrame, **kwargs):
        self.logger.info("Analyzing label distribution...")
        if 'label' not in df.columns or df['label'].empty:
            self.logger.warning("Skipping label distribution: 'label' column not found or empty.")
            return

        distribution_df = self.calculator.calculate_label_distribution(df)
        fig = self.plotter.plot_label_distribution(distribution_df, **kwargs)
        
        analysis_dir = self.dm.get_labeling_analysis_dir(symbol=kwargs['symbol'], interval=kwargs['interval'])
        table_kwargs = {'analysis_type': 'label_distribution', 'labeling_strategy': kwargs['labeling_strategy']}
        
        self.dm.save_analysis_table(df=distribution_df, run_dir=analysis_dir, table_pattern_key='labeling_table', **table_kwargs)

        if fig:
            plot_kwargs = {'analysis_type': 'label_distribution', 'labeling_strategy': kwargs['labeling_strategy']}
            self.dm.save_analysis_plot(fig=fig, run_dir=analysis_dir, plot_pattern_key='labeling_plot', **plot_kwargs)
            plt.close(fig)

    def _analyze_mfe_mae(self, df: pd.DataFrame, f_window: int, **kwargs):
        self.logger.info(f"Analyzing MFE/MAE over a {f_window}-bar window...")
        mfe_mae_raw, mfe_mae_summary = self.calculator.calculate_mfe_mae_stats(df, f_window)
        if mfe_mae_raw.empty:
            self.logger.warning("No MFE/MAE results to analyze.")
            return
            
        fig_scatter = self.plotter.plot_mfe_mae_scatter(mfe_mae_raw, **kwargs)
        fig_dist = self.plotter.plot_mfe_mae_distributions(mfe_mae_raw, **kwargs)

        analysis_dir = self.dm.get_labeling_analysis_dir(symbol=kwargs['symbol'], interval=kwargs['interval'])
        strategy_kwarg = {'labeling_strategy': kwargs['labeling_strategy']}

        self.dm.save_analysis_table(df=mfe_mae_raw, run_dir=analysis_dir, table_pattern_key='labeling_table', analysis_type='mfe_mae_raw', **strategy_kwarg)
        self.dm.save_analysis_table(df=mfe_mae_summary, run_dir=analysis_dir, table_pattern_key='labeling_table', analysis_type='mfe_mae_summary', **strategy_kwarg)

        if fig_scatter:
            self.dm.save_analysis_plot(fig=fig_scatter, run_dir=analysis_dir, plot_pattern_key='labeling_plot', analysis_type='mfe_mae_scatter', **strategy_kwarg)
            plt.close(fig_scatter)
        if fig_dist:
            self.dm.save_analysis_plot(fig=fig_dist, run_dir=analysis_dir, plot_pattern_key='labeling_plot', analysis_type='mfe_mae_distributions', **strategy_kwarg)
            plt.close(fig_dist)

    def _analyze_future_returns(self, df: pd.DataFrame, horizons: List[int], **kwargs):
        self.logger.info(f"Analyzing future returns for horizons: {horizons}...")
        returns_summary, returns_raw = self.calculator.calculate_future_returns_by_label(df, horizons)
        if returns_summary.empty:
            self.logger.warning("No future return results to analyze.")
            return

        fig = self.plotter.plot_future_returns(returns_raw, **kwargs)
        analysis_dir = self.dm.get_labeling_analysis_dir(symbol=kwargs['symbol'], interval=kwargs['interval'])
        strategy_kwarg = {'labeling_strategy': kwargs['labeling_strategy']}

        self.dm.save_analysis_table(df=returns_summary, run_dir=analysis_dir, table_pattern_key='labeling_table', analysis_type='future_returns_summary', **strategy_kwarg)

        if fig:
            self.dm.save_analysis_plot(fig=fig, run_dir=analysis_dir, plot_pattern_key='labeling_plot', analysis_type='future_returns_boxplot', **strategy_kwarg)
            plt.close(fig)

    def _analyze_regime_profitability(self, df: pd.DataFrame, horizons: List[int], **kwargs):
        if 'volatility_regime' not in df.columns:
            self.logger.info("Skipping volatility regime analysis: 'volatility_regime' column not found.")
            return
            
        self.logger.info("Analyzing profitability by volatility regime...")
        regime_summary, regime_raw = self.calculator.calculate_profitability_by_regime(df, horizons)
        if regime_summary.empty:
            self.logger.warning("No regime profitability results to analyze.")
            return
            
        figures_to_save = self.plotter.plot_regime_profitability(regime_raw, **kwargs)
        analysis_dir = self.dm.get_labeling_analysis_dir(symbol=kwargs['symbol'], interval=kwargs['interval'])
        strategy_kwarg = {'labeling_strategy': kwargs['labeling_strategy']}

        self.dm.save_analysis_table(df=regime_summary, run_dir=analysis_dir, table_pattern_key='labeling_table', analysis_type='regime_profitability_summary', **strategy_kwarg)

        for plot_type, fig in figures_to_save:
            if fig:
                self.dm.save_analysis_plot(fig=fig, run_dir=analysis_dir, plot_pattern_key='labeling_plot', analysis_type=plot_type, **strategy_kwarg)
                plt.close(fig)