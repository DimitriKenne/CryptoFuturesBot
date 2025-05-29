# utils/label_analyzer.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List

# Set up a module-level logger
logger = logging.getLogger(__name__)

class LabelAnalyzer:
    """
    Performs various analyses on generated trading labels and processed OHLCV data.
    Results (plots and tables) are saved to strategy-specific subfolders.
    """

    def __init__(self, paths: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the LabelAnalyzer.

        Args:
            paths (Dict[str, Any]): The PATHS dictionary from config/paths.py.
            logger (logging.Logger): A logger instance for logging messages.
        """
        self.paths = paths
        self.logger = logger
        self.logger.info("LabelAnalyzer initialized.")

        # Ensure required patterns are available
        self.plot_pattern = self.paths.get("labeling_analysis_plot_pattern")
        self.table_pattern = self.paths.get("labeling_analysis_table_pattern")
        self.strategy_dir_pattern_str = self.paths.get("labeling_strategy_analysis_dir_pattern")

        if not all([self.plot_pattern, self.table_pattern, self.strategy_dir_pattern_str]):
            self.logger.error("Missing required analysis patterns in PATHS. Check config/paths.py.")
            raise ValueError("Missing required analysis patterns in PATHS.")

    def _calculate_percentage_change_scalar(self, start_price: float, end_price: float) -> float:
        """Calculates percentage change for scalar values, handling potential division by zero."""
        if abs(start_price) < 1e-9: # Use a small epsilon to check for near-zero
            return 0.0 if end_price == start_price else np.nan # Avoid division by zero
        return ((end_price - start_price) / start_price) * 100.0

    def _calculate_mfe_mae(self, df_segment: pd.DataFrame, entry_price: float, trade_type: int) -> Tuple[float, float]:
        """
        Calculates MFE and MAE for a given price segment starting from an entry price.

        Args:
            df_segment (pd.DataFrame): DataFrame slice containing OHLC data for the trade duration.
                                       Must have 'high' and 'low' columns.
            entry_price (float): The price at which the trade is considered entered.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            Tuple[float, float]: (MFE percentage, MAE percentage). Returns (np.nan, np.nan) if segment is empty.
        """
        if df_segment.empty:
            return np.nan, np.nan

        df_segment = df_segment.copy() # Work on a copy to avoid SettingWithCopyWarning

        # Ensure high and low columns are numeric, coercing errors to NaN
        df_segment['high'] = pd.to_numeric(df_segment['high'], errors='coerce')
        df_segment['low'] = pd.to_numeric(df_segment['low'], errors='coerce')

        # Drop rows where high/low became NaN after coercion
        df_segment.dropna(subset=['high', 'low'], inplace=True)

        if df_segment.empty:
             return np.nan, np.nan # Return NaN if segment becomes empty after dropping NaNs


        if trade_type == 1: # Long trade
            # MFE: Max percentage increase from entry_price to highest high in segment
            max_high = df_segment['high'].max()
            mfe_pct = self._calculate_percentage_change_scalar(entry_price, max_high)

            # MAE: Max percentage decrease from entry_price to lowest low in segment
            min_low = df_segment['low'].min()
            mae_pct = self._calculate_percentage_change_scalar(entry_price, min_low)
            mae_pct = abs(mae_pct) # Report MAE as a positive value representing loss magnitude

        elif trade_type == -1: # Short trade
            # MFE: Max percentage decrease from entry_price to lowest low in segment
            min_low = df_segment['low'].min()
            mfe_pct = self._calculate_percentage_change_scalar(entry_price, min_low)
            mfe_pct = abs(mfe_pct) # Report MFE as a positive value representing profit magnitude

            # MAE: Max percentage increase from entry_price to highest high in segment
            max_high = df_segment['high'].max()
            mae_pct = self._calculate_percentage_change_scalar(entry_price, max_high)
            mae_pct = abs(mae_pct) # Report MAE as a positive value representing loss magnitude

        else:
            return np.nan, np.nan # Should not happen with labels 1 or -1

        return mfe_pct, mae_pct


    def analyze_label_streaks(self, df_labeled: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Analyzes the duration of consecutive non-zero label streaks.
        """
        self.logger.info(f"Starting Label Streak Analysis for {symbol.upper()} {interval}...")

        if df_labeled.empty or 'label' not in df_labeled.columns:
            self.logger.warning("Labeled DataFrame is empty or missing 'label' column. Skipping streak analysis.")
            return

        df_labeled = df_labeled.copy() # Work on a copy
        # Ensure label is integer type
        df_labeled['label'] = pd.to_numeric(df_labeled['label'], errors='coerce').fillna(0).astype(int)

        # Identify the start of new streaks (where label changes from previous row)
        streak_starts = (df_labeled['label'] != df_labeled['label'].shift(1)).fillna(True)

        streak_data = []
        current_label = None
        current_start_index = None

        for index, row in df_labeled.iterrows():
            label = row['label']
            if streak_starts.loc[index]:
                if current_label is not None and current_label != 0:
                    try:
                        start_iloc = df_labeled.index.get_loc(current_start_index)
                        end_iloc = df_labeled.index.get_loc(index)
                        streak_length = end_iloc - start_iloc
                        if streak_length > 0:
                             streak_data.append({'label': current_label, 'duration': streak_length})
                    except KeyError:
                        self.logger.warning(f"Could not get integer location for index {current_start_index} or {index}. Skipping streak.")

                current_label = label
                current_start_index = index

        if current_label is not None and current_label != 0:
             try:
                 start_iloc = df_labeled.index.get_loc(current_start_index)
                 streak_length = len(df_labeled) - start_iloc
                 if streak_length > 0:
                     streak_data.append({'label': current_label, 'duration': streak_length})
             except KeyError:
                self.logger.warning(f"Could not get integer location for index {current_start_index} for the last streak. Skipping.")


        if not streak_data:
            self.logger.info("No non-zero label streaks found for analysis.")
            return

        df_streaks = pd.DataFrame(streak_data)
        df_streaks_nonzero = df_streaks[df_streaks['label'] != 0].copy()

        if df_streaks_nonzero.empty:
            self.logger.info("No non-zero label streaks found for duration analysis after filtering.")
            return

        self.logger.info("Calculating streak duration summary statistics...")
        if len(df_streaks_nonzero) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std']
            for q in [0.75, 0.90, 0.95]:
                if len(df_streaks_nonzero) >= (1 / (1 - q)):
                     agg_funcs.append(lambda x, q=q: x.quantile(q))
                else:
                     self.logger.warning(f"Not enough data points ({len(df_streaks_nonzero)}) for {q*100}th percentile calculation. Skipping.")

            streak_summary = df_streaks_nonzero.groupby('label')['duration'].agg(agg_funcs).reset_index()
            col_names = ['Label', 'Count', 'Mean Duration', 'Median Duration', 'Std Dev Duration']
            if 0.75 in [q for q in [0.75, 0.90, 0.95] if len(df_streaks_nonzero) >= (1 / (1 - q))]:
                col_names.append('75th Percentile')
            if 0.90 in [q for q in [0.75, 0.90, 0.95] if len(df_streaks_nonzero) >= (1 / (1 - q))]:
                col_names.append('90th Percentile')
            if 0.95 in [q for q in [0.75, 0.90, 0.95] if len(df_streaks_nonzero) >= (1 / (1 - q))]:
                col_names.append('95th Percentile')

            streak_summary.columns = col_names
        else:
            self.logger.warning("No non-zero streaks to calculate summary statistics.")
            streak_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean Duration', 'Median Duration', 'Std Dev Duration'])


        self.logger.info("\nLabel Streak Duration Summary (in bars):")
        self.logger.info(streak_summary.to_string())

        analysis_type_suffix = "label_streak_duration"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            streak_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved streak duration summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save streak duration summary table: {e}")

        self.logger.info("Generating streak duration distribution plots...")
        try:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sns.histplot(data=df_streaks_nonzero[(df_streaks_nonzero['label'] == 1) & (df_streaks_nonzero['duration'] < df_streaks_nonzero['duration'].quantile(0.99))], x='duration', bins=50, kde=True, color='green')
            plt.title(f'Label 1 (Long) Streak Durations\n{symbol.upper()} {interval}')
            plt.xlabel('Duration (bars)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(1, 2, 2)
            sns.histplot(data=df_streaks_nonzero[(df_streaks_nonzero['label'] == -1) & (df_streaks_nonzero['duration'] < df_streaks_nonzero['duration'].quantile(0.99))], x='duration', bins=50, kde=True, color='red')
            plt.title(f'Label -1 (Short) Streak Durations\n{symbol.upper()} {interval}')
            plt.xlabel('Duration (bars)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_type_suffix = "label_streak_duration_plot"
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved streak duration plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save streak duration plot: {e}")

        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting streak durations: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("Label Streak Analysis complete.")


    def analyze_mfe_mae(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Analyzes Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
        for periods following non-zero labels until the label flips or data ends.

        Assumes df_combined contains 'open', 'high', 'low', 'close' and 'label' columns
        and is indexed by time.
        """
        self.logger.info(f"Starting MFE/MAE Analysis for {symbol.upper()} {interval}...")

        required_cols = ['open', 'high', 'low', 'close', 'label']
        if not all(col in df_combined.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_combined.columns]
            self.logger.error(f"Missing required columns for MFE/MAE analysis: {missing}. Skipping.")
            return

        df_combined = df_combined.copy() # Work on a copy
        # Ensure OHLC and label columns are numeric, coercing errors to NaN
        for col in ['open', 'high', 'low', 'close', 'label']:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Drop rows where critical OHLC or label became NaN
        df_combined.dropna(subset=['open', 'high', 'low', 'close', 'label'], inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame is empty after dropping NaNs in critical columns. Skipping MFE/MAE analysis.")
            return

        # Ensure label is integer type after dropping NaNs
        df_combined['label'] = df_combined['label'].astype(int)


        mfe_mae_data = []
        n = len(df_combined)

        i = 0
        while i < n:
            current_label = df_combined['label'].iloc[i]

            if current_label != 0:
                entry_price_ref = df_combined['close'].iloc[i]

                j = i + 1
                while j < n and df_combined['label'].iloc[j] == current_label:
                    j += 1
                price_segment_df = df_combined.iloc[i+1 : j].copy()

                if not all(col in price_segment_df.columns for col in ['high', 'low']):
                     self.logger.warning(f"Price segment starting at index {df_combined.index[i]} is missing 'high' or 'low' columns. Skipping MFE/MAE for this segment.")
                     mfe_mae_data.append({'label': current_label, 'mfe': np.nan, 'mae': np.nan})
                else:
                     mfe, mae = self._calculate_mfe_mae(price_segment_df, entry_price_ref, current_label)
                     mfe_mae_data.append({'label': current_label, 'mfe': mfe, 'mae': mae})

                i = j

            else:
                i += 1


        if not mfe_mae_data:
            self.logger.info("No non-zero labels found for MFE/MAE analysis.")
            return

        df_mfe_mae = pd.DataFrame(mfe_mae_data)
        df_mfe_mae.dropna(subset=['mfe', 'mae'], inplace=True)

        if df_mfe_mae.empty:
            self.logger.info("MFE/MAE DataFrame is empty after dropping NaNs.")
            return

        self.logger.info("Calculating MFE/MAE summary statistics...")
        if len(df_mfe_mae) > 0:
            mfe_agg_funcs = ['count', 'mean', 'median']
            mae_agg_funcs = ['count', 'mean', 'median']
            for q in [0.75, 0.90, 0.95]:
                if len(df_mfe_mae) >= (1 / (1 - q)):
                     mfe_agg_funcs.append(lambda x, q=q: x.quantile(q))
                     mae_agg_funcs.append(lambda x, q=q: x.quantile(q))
                else:
                     self.logger.warning(f"Not enough data points ({len(df_mfe_mae)}) for {q*100}th percentile MFE/MAE calculation. Skipping.")


            mfe_summary = df_mfe_mae.groupby('label')['mfe'].agg(mfe_agg_funcs).reset_index()
            mae_summary = df_mfe_mae.groupby('label')['mae'].agg(mae_agg_funcs).reset_index()

            mfe_col_names = ['Label', 'Count', 'Mean MFE', 'Median MFE']
            mae_col_names = ['Label', 'Count', 'Mean MAE', 'Median MAE']

            if 0.75 in [q for q in [0.75, 0.90, 0.95] if len(df_mfe_mae) >= (1 / (1 - q))]:
                mfe_col_names.append('75th Percentile MFE')
                mae_col_names.append('75th Percentile MAE')
            if 0.90 in [q for q in [0.75, 0.90, 0.95] if len(df_mfe_mae) >= (1 / (1 - q))]:
                mfe_col_names.append('90th Percentile MFE')
                mae_col_names.append('90th Percentile MAE')
            if 0.95 in [q for q in [0.75, 0.90, 0.95] if len(df_mfe_mae) >= (1 / (1 - q))]:
                mfe_col_names.append('95th Percentile MFE')
                mae_col_names.append('95th Percentile MAE')

            mfe_summary.columns = mfe_col_names
            mae_summary.columns = mae_col_names
        else:
            self.logger.warning("No MFE/MAE data to calculate summary statistics.")
            mfe_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean MFE', 'Median MFE'])
            mae_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean MAE', 'Median MAE'])


        self.logger.info("\nMaximum Favorable Excursion (MFE) Summary (%):")
        self.logger.info(mfe_summary.to_string())

        self.logger.info("\nMaximum Adverse Excursion (MAE) Summary (%):")
        self.logger.info(mae_summary.to_string())

        analysis_type_suffix_mfe = "mfe_summary"
        table_path_mfe = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix_mfe
        ).replace(':', '_')
        try:
            mfe_summary.to_csv(table_path_mfe, index=False)
            self.logger.info(f"Saved MFE summary table to {table_path_mfe}")
        except Exception as e:
            self.logger.error(f"Failed to save MFE summary table: {e}")

        analysis_type_suffix_mae = "mae_summary"
        table_path_mae = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix_mae
        ).replace(':', '_')
        try:
            mae_summary.to_csv(table_path_mae, index=False)
            self.logger.info(f"Saved MAE summary table to {table_path_mae}")
        except Exception as e:
            self.logger.error(f"Failed to save MAE summary table: {e}")


        self.logger.info("Generating MFE/MAE distribution plots...")
        try:
            plt.figure(figsize=(12, 12))

            plt.subplot(2, 2, 1)
            sns.histplot(data=df_mfe_mae[df_mfe_mae['label'] == 1], x='mfe', bins=50, kde=True, color='green')
            plt.title(f'Label 1 (Long) MFE Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('MFE (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 2)
            sns.histplot(data=df_mfe_mae[df_mfe_mae['label'] == 1], x='mae', bins=50, kde=True, color='red')
            plt.title(f'Label 1 (Long) MAE Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('MAE (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 3)
            sns.histplot(data=df_mfe_mae[df_mfe_mae['label'] == -1], x='mfe', bins=50, kde=True, color='green')
            plt.title(f'Label -1 (Short) MFE Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('MFE (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 4)
            sns.histplot(data=df_mfe_mae[df_mfe_mae['label'] == -1], x='mae', bins=50, kde=True, color='red')
            plt.title(f'Label -1 (Short) MAE Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('MAE (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_type_suffix = "mfe_mae_distribution_plot"
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved MFE/MAE distribution plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save MFE/MAE distribution plot: {e}")

        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting MFE/MAE distributions: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("MFE/MAE Analysis complete.")


    def analyze_future_returns(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path, horizons: List[int]):
        """
        Analyzes future percentage returns over specified horizons following non-zero labels.

        Assumes df_combined contains 'close' and 'label' columns and is indexed by time.
        """
        self.logger.info(f"Starting Future Returns Analysis for {symbol.upper()} {interval} over horizons {horizons} bars...")

        required_cols = ['close', 'label']
        if not all(col in df_combined.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_combined.columns]
            self.logger.error(f"Missing required columns for Future Returns analysis: {missing}. Skipping.")
            return

        df_combined = df_combined.copy() # Work on a copy
        # Ensure close and label columns are numeric, coercing errors to NaN
        for col in ['close', 'label']:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Drop rows where critical close or label became NaN
        df_combined.dropna(subset=['close', 'label'], inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame is empty after dropping NaNs in critical columns. Skipping Future Returns analysis.")
            return

        # Ensure label is integer type after dropping NaNs
        df_combined['label'] = df_combined['label'].astype(int)


        returns_data = []

        for horizon in horizons:
            future_close = df_combined['close'].shift(-horizon)
            safe_close = df_combined['close'].replace(0, np.nan)
            returns_pct = ((future_close - df_combined['close']) / safe_close) * 100.0

            for label_val in [-1, 1]:
                returns_for_label = returns_pct[df_combined['label'] == label_val].dropna().tolist()
                for ret in returns_for_label:
                    returns_data.append({'label': label_val, 'horizon': horizon, 'return_pct': ret})


        if not returns_data:
            self.logger.info("No non-zero labels with valid future returns found for analysis.")
            return

        df_returns = pd.DataFrame(returns_data)

        self.logger.info("Calculating Future Returns summary statistics...")
        if len(df_returns) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
            agg_funcs_with_quantiles = agg_funcs + [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

            returns_summary = df_returns.groupby(['label', 'horizon'])['return_pct'].agg(agg_funcs_with_quantiles).reset_index()
            returns_summary.columns = ['Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile']

        else:
            self.logger.warning("No future returns data to calculate summary statistics.")
            returns_summary = pd.DataFrame(columns=['Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile'])


        self.logger.info("\nFuture Returns Summary (%) by Label and Horizon:")
        self.logger.info(returns_summary.to_string())

        analysis_type_suffix = "future_returns_summary"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            returns_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved future returns summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save future returns summary table: {e}")

        self.logger.info("Generating Future Returns distribution plots...")
        try:
            plt.figure(figsize=(14, 8))

            plt.subplot(1, 2, 1)
            sns.boxplot(data=df_returns[(df_returns['label'] == 1) & (df_returns['return_pct'].between(df_returns['return_pct'].quantile(0.005), df_returns['return_pct'].quantile(0.995)))],
                        x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False)
            plt.title(f'Label 1 (Long) Future Return Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Horizon (bars)')
            plt.ylabel('Return (%)')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(1, 2, 2)
            sns.boxplot(data=df_returns[(df_returns['label'] == -1) & (df_returns['return_pct'].between(df_returns['return_pct'].quantile(0.005), df_returns['return_pct'].quantile(0.995)))],
                        x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False)
            plt.title(f'Label -1 (Short) Future Return Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Horizon (bars)')
            plt.ylabel('Return (%)')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_type_suffix = "future_returns_distribution_plot"
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved future returns distribution plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save future returns distribution plot: {e}")

        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting future returns: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("Future Returns Analysis complete.")


    def analyze_regime_profitability(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path, horizons: List[int]):
        """
        Analyzes future percentage returns over specified horizons, grouped by
        volatility regime and label.

        Assumes df_combined contains 'close', 'label', and 'volatility_regime' columns
        and is indexed by time.
        """
        self.logger.info(f"Starting Volatility Regime Profitability Analysis for {symbol.upper()} {interval} over horizons {horizons} bars...")

        required_cols = ['close', 'label', 'volatility_regime']
        if not all(col in df_combined.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_combined.columns]
            self.logger.error(f"Missing required columns for Volatility Regime Profitability analysis: {missing}. Skipping.")
            return

        df_combined = df_combined.copy() # Work on a copy
        # Ensure close, label, and volatility_regime columns are numeric, coercing errors to NaN
        for col in required_cols:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Drop rows where critical columns became NaN
        df_combined.dropna(subset=required_cols, inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame is empty after dropping NaNs in critical columns. Skipping Volatility Regime Profitability analysis.")
            return

        # Ensure label and volatility_regime are integer types after dropping NaNs
        df_combined['label'] = df_combined['label'].astype(int)
        df_combined['volatility_regime'] = df_combined['volatility_regime'].astype(int)


        returns_data = []

        for horizon in horizons:
            future_close = df_combined['close'].shift(-horizon)
            safe_close = df_combined['close'].replace(0, np.nan)
            returns_pct = ((future_close - df_combined['close']) / safe_close) * 100.0

            returns_df = pd.DataFrame({
                'label': df_combined['label'],
                'volatility_regime': df_combined['volatility_regime'],
                'return_pct': returns_pct
            }).dropna(subset=['return_pct'])

            for label_val in [-1, 1]:
                returns_for_label = returns_df[returns_df['label'] == label_val].copy()
                if not returns_for_label.empty:
                    for index, row in returns_for_label.iterrows():
                        returns_data.append({
                            'label': label_val,
                            'volatility_regime': row['volatility_regime'],
                            'horizon': horizon,
                            'return_pct': row['return_pct']
                        })


        if not returns_data:
            self.logger.info("No non-zero labels with valid future returns and volatility regimes found for analysis.")
            return

        df_regime_returns = pd.DataFrame(returns_data)

        self.logger.info("Calculating Volatility Regime Profitability summary statistics...")
        if len(df_regime_returns) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
            agg_funcs_with_quantiles = agg_funcs + [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

            regime_returns_summary = df_regime_returns.groupby(['volatility_regime', 'label', 'horizon'])['return_pct'].agg(agg_funcs_with_quantiles).reset_index()
            regime_returns_summary.columns = ['Volatility Regime', 'Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile']

        else:
            self.logger.warning("No volatility regime returns data to calculate summary statistics.")
            regime_returns_summary = pd.DataFrame(columns=['Volatility Regime', 'Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile'])


        self.logger.info("\nVolatility Regime Profitability Summary (%) by Regime, Label, and Horizon:")
        self.logger.info(regime_returns_summary.to_string())

        analysis_type_suffix = "regime_profitability_summary"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            regime_returns_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved volatility regime profitability summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save volatility regime profitability summary table: {e}")

        self.logger.info("Generating Volatility Regime Profitability plots...")
        try:
            for horizon in horizons:
                plt.figure(figsize=(10, 6))
                plot_data = regime_returns_summary[regime_returns_summary['Horizon'] == horizon].copy()

                if plot_data.empty:
                     self.logger.warning(f"No data for plotting regime profitability for horizon {horizon}. Skipping plot.")
                     plt.close()
                     continue

                pivot_data = plot_data.pivot_table(index='Volatility Regime', columns='Label', values='Mean Return')

                if pivot_data.empty:
                     self.logger.warning(f"Pivot table is empty for horizon {horizon}. Skipping plot.")
                     plt.close()
                     continue

                pivot_data = pivot_data.sort_index()
                pivot_data.plot(kind='bar', colormap='coolwarm', ax=plt.gca())

                plt.title(f'Mean Future Return (%) by Volatility Regime and Label\n{symbol.upper()} {interval}, Horizon: {horizon} bars')
                plt.xlabel('Volatility Regime')
                plt.ylabel('Mean Return (%)')
                plt.xticks(rotation=0)
                plt.grid(axis='y', linestyle='--')
                plt.legend(title='Label', labels=['Short (-1)', 'Long (1)'])
                plt.tight_layout()

                plot_type_suffix = f"regime_profitability_horizon_{horizon}_plot"
                plot_path = output_dir / self.plot_pattern.format(
                    symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
                ).replace(':', '_')
                try:
                    plt.savefig(plot_path, dpi=150)
                    self.logger.info(f"Saved volatility regime profitability plot (Horizon {horizon}) to {plot_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save volatility regime profitability plot (Horizon {horizon}): {e}")


        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting volatility regime profitability: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')


        self.logger.info("Volatility Regime Profitability Analysis complete.")


    def perform_all_analyses(self, df_combined: pd.DataFrame, symbol: str, interval: str, label_strategy: str, future_horizons: List[int]):
        """
        Orchestrates all label analyses and saves results to a strategy-specific folder.

        Args:
            df_combined (pd.DataFrame): Combined DataFrame with OHLCV, features, and 'label' column.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            label_strategy (str): The name of the labeling strategy being analyzed.
            future_horizons (List[int]): List of future horizons for return analysis.
        """
        self.logger.info(f"Performing all analyses for {symbol} {interval} with strategy '{label_strategy}'...")

        # Dynamically create the strategy-specific analysis directory
        analysis_output_dir = Path(self.strategy_dir_pattern_str.format(label_strategy=label_strategy))
        try:
            analysis_output_dir.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Ensured analysis results directory exists: {analysis_output_dir}")
        except OSError as e:
            self.logger.error(f"Error creating analysis results directory {analysis_output_dir}: {e}", exc_info=True)
            raise # Re-raise to stop pipeline if directory cannot be created


        # 1. Label Streak Analysis
        self.analyze_label_streaks(df_combined.copy(), symbol, interval, analysis_output_dir)

        # 2. MFE/MAE Analysis
        self.analyze_mfe_mae(df_combined.copy(), symbol, interval, analysis_output_dir)

        # 3. Future Returns Analysis (General)
        self.analyze_future_returns(df_combined.copy(), symbol, interval, analysis_output_dir, future_horizons)

        # 4. Volatility Regime Profitability Analysis
        self.analyze_regime_profitability(df_combined.copy(), symbol, interval, analysis_output_dir, future_horizons)

        self.logger.info(f"All analyses completed for {symbol} {interval} with strategy '{label_strategy}'.")

