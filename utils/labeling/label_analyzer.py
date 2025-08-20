# utils/labeling/label_analyzer.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Module-level logger
logger = logging.getLogger(__name__)

from config.params import app_config, FLOAT_EPSILON
from config.label import LabelConfig

from utils.labeling.analysis_calculator import AnalysisCalculator
from utils.labeling.analysis_plotter import AnalysisPlotter
from utils.labeling.label_generator import LabelGenerator


class LabelAnalyzer:
    """
    Analyzes trading labels and OHLCV data, saving plots and tables.
    Orchestrates AnalysisCalculator for computations and AnalysisPlotter for visualizations.
    """

    def __init__(
        self,
        paths: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        fee: Optional[float] = None,
        slippage: Optional[float] = None,
        f_window: Optional[int] = None,
        labeling_strategy_type: Optional[str] = None,
    ):
        """
        Initializes LabelAnalyzer.

        Args:
            paths (Dict): PATHS dictionary from config/paths.py.
            logger (logging.Logger): Logger instance.
            fee (float): Transaction fee rate (0-1). Resolves from config if None.
            slippage (float): Slippage rate (0-1). Resolves from config if None.
            f_window (int): Forward lookahead window. Used for MFE/MAL. Resolves from config if None.
            labeling_strategy_type (str): Type of labeling strategy being analyzed.
        """
        self.paths = paths
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.fee = fee if fee is not None else app_config.labeling.trading_fee_pct / 100.0
        self.slippage = slippage if slippage is not None else app_config.labeling.slippage_tolerance_pct / 100.0

        self.labeling_strategy_type = (
            labeling_strategy_type if labeling_strategy_type is not None
            else app_config.labeling.labeling_strategy_type
        )
        strategy_config_obj = getattr(app_config.labeling, self.labeling_strategy_type, None)
        if strategy_config_obj:
            self.f_window = f_window if f_window is not None else getattr(strategy_config_obj, "future_return_window", 150)
        else:
            self.f_window = f_window if f_window is not None else 150
            self.logger.warning(f"Strategy config for '{self.labeling_strategy_type}' not found. Using default f_window: {self.f_window}")

        self.plot_pattern = self.paths.get("labeling_analysis_plot_pattern")
        self.table_pattern = self.paths.get("labeling_analysis_table_pattern")
        self.strategy_dir_pattern_str = self.paths.get("labeling_strategy_analysis_dir_pattern")

        if not all([self.plot_pattern, self.table_pattern, self.strategy_dir_pattern_str]):
            self.logger.error("Missing required analysis patterns in PATHS.")
            raise ValueError("Missing required analysis patterns in PATHS.")

        self.calculator = AnalysisCalculator(self.fee, self.slippage)
        self.plotter = AnalysisPlotter(self.logger, self.plot_pattern) # Pass plot_pattern here

        self.logger.info(f"LabelAnalyzer initialized with fee={self.fee:.6f}, slippage={self.slippage:.6f}, f_window={self.f_window}, strategy_type='{self.labeling_strategy_type}'.")


    def analyze_label_distribution(self, df_labeled: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """Analyzes and saves the overall distribution of labels (-1, 0, 1)."""
        self.logger.info(f"Starting Label Distribution Analysis for {symbol.upper()} {interval}...")

        if df_labeled.empty or 'label' not in df_labeled.columns:
            self.logger.warning("Empty or missing 'label' column. Skipping.")
            return

        df_labeled = df_labeled.copy()
        df_labeled['label'] = pd.to_numeric(df_labeled['label'], errors='coerce').fillna(0).astype(int)

        label_counts = df_labeled['label'].value_counts().sort_index()
        total_labels = label_counts.sum()
        label_percentages = (label_counts / total_labels) * 100 if total_labels > 0 else pd.Series(dtype=float)

        distribution_summary = pd.DataFrame({
            'Label': label_counts.index,
            'Count': label_counts.values,
            'Percentage': label_percentages.values
        })

        self.logger.info("\nOverall Label Distribution:")
        self.logger.info(distribution_summary.to_string(index=False))

        analysis_type_suffix = "label_distribution"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            distribution_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved label distribution summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save label distribution summary table: {e}")

        self.plotter.plot_label_distribution(distribution_summary, symbol, interval, output_dir) # Removed plot_pattern arg
        self.logger.info("Label Distribution Analysis complete.")


    def analyze_mfe_mae(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Analyzes Max Favorable Return (MFR) and Max Adverse Loss (MAL)
        following non-zero labels, over the f_window.
        """
        self.logger.info(f"Starting Max Return/Loss within f_window Analysis for {symbol.upper()} {interval}...")

        required_cols = ['open', 'high', 'low', 'close', 'label']
        if not all(col in df_combined.columns for col in required_cols):
            self.logger.error(f"Missing columns: {[col for col in required_cols if col not in df_combined.columns]}. Skipping.")
            return None

        df_combined = df_combined.copy()
        for col in ['open', 'high', 'low', 'close', 'label']:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        df_combined.dropna(subset=['open', 'high', 'low', 'close', 'label'], inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame empty after NaN drop. Skipping.")
            return None

        df_combined['label'] = df_combined['label'].astype(int)

        max_return_loss_data = []
        n = len(df_combined)

        for i in range(n):
            current_label = df_combined['label'].iloc[i]
            if current_label != 0:
                entry_price_ref = df_combined['close'].iloc[i]
                segment_end_iloc = min(i + self.f_window + 1, n)
                price_segment_df = df_combined.iloc[i + 1 : segment_end_iloc].copy()

                if price_segment_df.empty:
                    self.logger.warning(f"No valid price segment for signal at index {df_combined.index[i]}. Skipping.")
                    continue

                max_favorable, max_adverse = self.calculator.calculate_max_return_loss_within_window(
                    price_segment_df, entry_price_ref, current_label
                )
                max_return_loss_data.append({
                    'label': current_label,
                    'max_favorable_return': max_favorable,
                    'max_adverse_loss': max_adverse
                })

        if not max_return_loss_data:
            self.logger.info("No active signals found for Max Return/Loss analysis.")
            return None

        df_max_return_loss = pd.DataFrame(max_return_loss_data)
        df_max_return_loss.dropna(subset=['max_favorable_return', 'max_adverse_loss'], inplace=True)

        if df_max_return_loss.empty:
            self.logger.info("Max Return/Loss DataFrame empty after NaN drop.")
            return None

        self.logger.info("Calculating Max Return/Loss summary statistics...")
        if len(df_max_return_loss) > 0:
            mfr_agg_funcs = ['count', 'mean', 'median']
            mal_agg_funcs = ['count', 'mean', 'median']
            quantiles_to_add = [0.75, 0.90, 0.95]
            for q in quantiles_to_add:
                if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q)):
                     mfr_agg_funcs.append(lambda x, q=q: x.quantile(q))
                     mal_agg_funcs.append(lambda x, q=q: x.quantile(q))
                else:
                     self.logger.warning(f"Not enough data ({df_max_return_loss['max_favorable_return'].count()}) for {q*100}th percentile. Skipping.")

            mfr_summary = df_max_return_loss.groupby('label')['max_favorable_return'].agg(mfr_agg_funcs).reset_index()
            mal_summary = df_max_return_loss.groupby('label')['max_adverse_loss'].agg(mal_agg_funcs).reset_index()

            mfr_summary_cols = ['Label', 'Count', 'Mean Max Favorable Return', 'Median Max Favorable Return']
            mal_summary_cols = ['Label', 'Count', 'Mean Max Adverse Loss', 'Median Max Adverse Loss']

            if any(q == 0.75 for q in quantiles_to_add if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q))):
                mfr_summary_cols.append('75th Percentile Max Favorable Return')
                mal_summary_cols.append('75th Percentile Max Adverse Loss')
            if any(q == 0.90 for q in quantiles_to_add if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q))):
                mfr_summary_cols.append('90th Percentile Max Favorable Return')
                mal_summary_cols.append('90th Percentile Max Adverse Loss')
            if any(q == 0.95 for q in quantiles_to_add if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q))):
                mfr_summary_cols.append('95th Percentile Max Favorable Return')
                mal_summary_cols.append('95th Percentile Max Adverse Loss')

            mfr_summary.columns = mfr_summary_cols[:len(mfr_summary.columns)]
            mal_summary.columns = mal_summary_cols[:len(mal_summary.columns)]
        else:
            self.logger.warning("No Max Return/Loss data for summary statistics.")
            mfr_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean Max Favorable Return', 'Median Max Favorable Return'])
            mal_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean Max Adverse Loss', 'Median Max Adverse Loss'])

        self.logger.info("\nMaximum Favorable Return (within f_window) Summary (%):")
        self.logger.info(mfr_summary.to_string())
        self.logger.info("\nMaximum Adverse Loss (within f_window) Summary (%):")
        self.logger.info(mal_summary.to_string())

        analysis_type_suffix_mfr = "max_favorable_return_fwindow"
        table_path_mfr = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix_mfr
        ).replace(':', '_')
        try:
            mfr_summary.to_csv(table_path_mfr, index=False)
            self.logger.info(f"Saved MFR summary to {table_path_mfr}")
        except Exception as e:
            self.logger.error(f"Failed to save MFR summary table: {e}")

        analysis_type_suffix_mal = "max_adverse_loss_fwindow"
        table_path_mal = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix_mal
        ).replace(':', '_')
        try:
            mal_summary.to_csv(table_path_mal, index=False)
            self.logger.info(f"Saved MAL summary to {table_path_mal}")
        except Exception as e:
            self.logger.error(f"Failed to save MAL summary table: {e}")

        self.plotter.plot_mfr_mae_distributions(df_max_return_loss, symbol, interval, output_dir) # Removed plot_pattern arg
        self.plotter.plot_mfr_mae_scatter(df_max_return_loss, symbol, interval, output_dir) # Removed plot_pattern arg

        self.logger.info("Max Return/Loss (within f_window) Analysis complete.")
        return df_max_return_loss


    def analyze_future_returns(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path, horizons: List[int]):
        """Analyzes future NET returns over specified horizons following non-zero labels."""
        self.logger.info(f"Starting Future NET Returns Analysis for {symbol.upper()} {interval} over horizons {horizons} bars...")

        required_cols = ['close', 'label']
        if not all(col in df_combined.columns for col in required_cols):
            self.logger.error(f"Missing columns: {[col for col in required_cols if col not in df_combined.columns]}. Skipping.")
            return

        df_combined = df_combined.copy()
        for col in ['close', 'label']:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        df_combined.dropna(subset=['close', 'label'], inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame empty after NaN drop. Skipping.")
            return

        df_combined['label'] = df_combined['label'].astype(int)
        returns_data = []

        for horizon in horizons:
            future_close_series = df_combined['close'].shift(-horizon)

            for i in range(len(df_combined)):
                current_label = df_combined['label'].iloc[i]
                current_close = df_combined['close'].iloc[i]
                future_close = future_close_series.iloc[i]

                if current_label != 0 and pd.notna(current_close) and pd.notna(future_close) and abs(current_close) > FLOAT_EPSILON:
                    net_return = self.calculator.calculate_net_return_scalar(current_close, future_close, current_label)
                    if pd.notna(net_return):
                        returns_data.append({
                            'label': current_label,
                            'horizon': horizon,
                            'return_pct': net_return
                        })

        if not returns_data:
            self.logger.info("No non-zero labels with valid future net returns found.")
            return

        df_returns = pd.DataFrame(returns_data)

        self.logger.info("Calculating Future Net Returns summary statistics...")
        if len(df_returns) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
            agg_funcs_with_quantiles = agg_funcs + [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

            returns_summary = df_returns.groupby(['label', 'horizon'])['return_pct'].agg(agg_funcs_with_quantiles).reset_index()
            returns_summary.columns = ['Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile']
        else:
            self.logger.warning("No future returns data for summary statistics.")
            returns_summary = pd.DataFrame(columns=['Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile'])

        self.logger.info("\nFuture Net Returns Summary (%) by Label and Horizon:")
        self.logger.info(returns_summary.to_string())

        analysis_type_suffix = "future_net_returns_summary"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            returns_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved future net returns summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save future net returns summary table: {e}")

        self.plotter.plot_future_returns(df_returns, symbol, interval, output_dir) # Removed plot_pattern arg
        self.logger.info("Future Net Returns Analysis complete.")


    def analyze_regime_profitability(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path, horizons: List[int]):
        """Analyzes future NET returns by volatility regime and label."""
        self.logger.info(f"Starting Volatility Regime Net Profitability Analysis for {symbol.upper()} {interval} over horizons {horizons} bars...")

        required_cols = ['close', 'label', 'volatility_regime']
        if not all(col in df_combined.columns for col in required_cols):
            self.logger.error(f"Missing columns: {[col for col in required_cols if col not in df_combined.columns]}. Skipping.")
            return

        df_combined = df_combined.copy()
        for col in required_cols:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        df_combined.dropna(subset=required_cols, inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame empty after NaN drop. Skipping.")
            return

        df_combined['label'] = df_combined['label'].astype(int)
        df_combined['volatility_regime'] = df_combined['volatility_regime'].astype(int)
        returns_data = []

        for horizon in horizons:
            future_close_series = df_combined['close'].shift(-horizon)

            for i in range(len(df_combined)):
                current_label = df_combined['label'].iloc[i]
                current_close = df_combined['close'].iloc[i]
                future_close = future_close_series.iloc[i]
                volatility_regime: int = df_combined['volatility_regime'].iloc[i]

                if current_label != 0 and pd.notna(current_close) and pd.notna(future_close) and abs(current_close) > FLOAT_EPSILON:
                    net_return = self.calculator.calculate_net_return_scalar(current_close, future_close, current_label)
                    if pd.notna(net_return):
                        returns_data.append({
                            'label': current_label,
                            'volatility_regime': volatility_regime,
                            'horizon': horizon,
                            'return_pct': net_return
                        })

        if not returns_data:
            self.logger.info("No non-zero labels with valid future net returns/volatility regimes found.")
            return

        df_regime_returns = pd.DataFrame(returns_data)

        self.logger.info("Calculating Volatility Regime Net Profitability summary statistics...")
        if len(df_regime_returns) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
            agg_funcs_with_quantiles = agg_funcs + [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

            regime_returns_summary = df_regime_returns.groupby(['volatility_regime', 'label', 'horizon'])['return_pct'].agg(agg_funcs_with_quantiles).reset_index()
            regime_returns_summary.columns = ['Volatility Regime', 'Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile']
        else:
            self.logger.warning("No volatility regime returns data for summary statistics.")
            regime_returns_summary = pd.DataFrame(columns=['Volatility Regime', 'Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile'])

        self.logger.info("\nVolatility Regime Net Profitability Summary (%) by Regime, Label, and Horizon:")
        self.logger.info(regime_returns_summary.to_string())

        analysis_type_suffix = "regime_net_profitability_summary"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            regime_returns_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved volatility regime net profitability summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save volatility regime net profitability summary table: {e}")

        self.plotter.plot_regime_profitability(df_regime_returns, symbol, interval, output_dir) # Removed plot_pattern arg
        self.logger.info("Volatility Regime Net Profitability Analysis complete.")


    def perform_all_analyses(
        self,
        df_combined: pd.DataFrame,
        symbol: str,
        interval: str,
        labeling_strategy: str,
        future_horizons: List[int]):
        """
        Orchestrates common label analyses, saving results to a strategy-specific folder.
        Strategy-specific analysis is triggered by LabelGenerator.
        """
        self.logger.info(f"Performing all common analyses for {symbol} {interval} with strategy '{labeling_strategy}'...")

        base_analysis_dir = Path(self.paths.get("analysis_dir", "./results/analysis"))
        strategy_relative_path = self.strategy_dir_pattern_str.format(labeling_strategy=labeling_strategy)
        analysis_output_dir = base_analysis_dir / strategy_relative_path

        try:
            analysis_output_dir.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Ensured analysis results directory exists: {analysis_output_dir}")
        except OSError as e:
            self.logger.error(f"Error creating analysis results directory {analysis_output_dir}: {e}", exc_info=True)
            raise

        self.analyze_label_distribution(df_combined.copy(), symbol, interval, analysis_output_dir)
        self.analyze_mfe_mae(df_combined.copy(), symbol, interval, analysis_output_dir)
        self.analyze_future_returns(df_combined.copy(), symbol, interval, analysis_output_dir, future_horizons)
        self.analyze_regime_profitability(df_combined.copy(), symbol, interval, analysis_output_dir, future_horizons)

        self.logger.info(f"All common analyses completed for {symbol} {interval} with strategy '{labeling_strategy}'.")
