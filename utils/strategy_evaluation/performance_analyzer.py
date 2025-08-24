# utils/analysis/performance_analyzer.py

import pandas as pd
import logging
from typing import Dict, Any, Tuple

# Import configurations and utilities
from config.params import AppConfig
from utils.strategy_evaluation.metrics_calculator import MetricsCalculator
from utils.strategy_evaluation.plotting_utils import PlottingUtils

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Analyzes trading performance from backtesting or live trading results.
    This class is a pure analysis engine. It does not perform any file I/O.
    Its sole responsibility is to take in raw backtest results and return
    calculated metrics and plot figures.
    """
    def __init__(self,
                 app_config: AppConfig,
                 trade_history_df: pd.DataFrame,
                 equity_df: pd.DataFrame,
                 symbol: str,
                 interval: str,
                 model_type: str
                 ):
        """
        Initializes the PerformanceAnalyzer.

        Args:
            app_config (AppConfig): The global application configuration object.
            trade_history_df (pd.DataFrame): DataFrame containing the history of all trades.
            equity_df (pd.DataFrame): DataFrame with a DatetimeIndex and an 'equity' column.
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '1h').
            model_type (str): The type of ML model used (e.g., 'xgboost', 'lstm').
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app_config = app_config
        self.trade_history_df = trade_history_df
        self.equity_df = equity_df
        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type

        # Initialize MetricsCalculator and PlottingUtils
        self.metrics_calculator = MetricsCalculator(
            app_config=self.app_config,
            initial_capital=self.app_config.trading.risk.initial_capital
        )
        self.plotting_utils = PlottingUtils()

        self.logger.info("PerformanceAnalyzer initialized.")

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Delegates to MetricsCalculator to compute all performance metrics."""
        if self.trade_history_df.empty or self.equity_df.empty:
            self.logger.warning("Trade history or equity data is empty. Cannot calculate metrics.")
            return {}
            
        return self.metrics_calculator.calculate_all_metrics(
            trade_history_df=self.trade_history_df,
            equity_df=self.equity_df,
            interval=self.interval
        )

    def _generate_plots(self) -> Dict[str, Any]:
        """
        Generates all performance plots by delegating to PlottingUtils.

        Returns:
            Dict[str, Any]: A dictionary mapping plot names to their Figure objects.
        """
        self.logger.info("Generating performance plot figures...")
        plots_dict = {}
        
        base_title = f"{self.symbol} {self.interval} {self.model_type}"
        
        plots_dict['equity_curve'] = self.plotting_utils.plot_equity_curve(self.equity_df, f'Equity Curve - {base_title}')
        plots_dict['drawdown_curve'] = self.plotting_utils.plot_drawdown_curve(self.equity_df, f'Drawdown Curve - {base_title}')
        plots_dict['pnl_distribution'] = self.plotting_utils.plot_trade_pnl_distribution(self.trade_history_df, f'PnL Distribution - {base_title}')
        plots_dict['exit_reason_boxplot'] = self.plotting_utils.plot_exit_reason_pnl_boxplot(self.trade_history_df, f'PnL by Exit Reason - {base_title}')
        plots_dict['exit_reason_frequency'] = self.plotting_utils.plot_exit_reason_frequency_barplot(self.trade_history_df, f'Exit Reason Frequency - {base_title}')
        
        self.logger.info(f"{len(plots_dict)} plot figures generated.")
        return plots_dict

    def generate_analysis_artifacts(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs the full analysis pipeline and returns all generated artifacts.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                - A dictionary containing all calculated performance metrics.
                - A dictionary mapping plot names to their generated Figure objects.
        """
        self.logger.info("Generating all analysis artifacts (metrics and plots)...")
        
        metrics = self._calculate_metrics()
        plots = self._generate_plots()

        self.logger.info("Analysis artifacts generated successfully.")
        return metrics, plots