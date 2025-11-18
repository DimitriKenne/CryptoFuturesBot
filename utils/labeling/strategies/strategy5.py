from typing import Any, List, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseLabelingStrategy
from utils.data_management.data_manager import DataManager # Assumed import for utility functions

# --- Helper Functions (Updated) ---

def attach_htf_context(df: pd.DataFrame, htf_interval: str = "1d") -> pd.DataFrame:
    """
    Attach HTF open, close, high, low, and the timestamp of HTF high/low for each LTF bar.
    Assumes df has a DatetimeIndex and LTF OHLC columns.
    """
    # 1. Resample to get HTF OHLC
    htf_ohlc = df.resample(htf_interval).agg(
        htf_open=('open', 'first'),
        htf_close=('close', 'last'),
        htf_high=('high', 'max'),
        htf_low=('low', 'min'),
    ).reset_index()

    htf_ohlc['htf_bar_index'] = np.arange(len(htf_ohlc))

    # 2. Find timestamp of HTF high/low within the HTF bar (for Phase-Awareness)
    # This requires a groupby and idxmax/idxmin operation
    htf_high_ts = df['high'].groupby(pd.Grouper(freq=htf_interval)).idxmax().rename('htf_high_ts')
    htf_low_ts = df['low'].groupby(pd.Grouper(freq=htf_interval)).idxmin().rename('htf_low_ts')
    
    # Merge the timestamps back into the HTF DataFrame
    htf_ohlc = pd.merge(htf_ohlc, htf_high_ts, 
                        left_on='timestamp', right_index=True, how='left')
    htf_ohlc = pd.merge(htf_ohlc, htf_low_ts, 
                        left_on='timestamp', right_index=True, how='left')

    # 3. Merge_asof to propagate HTF data to LTF
    df_sorted = df.sort_index().reset_index().rename(columns={'index': 'timestamp'})
    result = pd.merge_asof(
        df_sorted,
        htf_ohlc,
        left_on='timestamp',
        right_on='timestamp',
        direction='backward'
    )
    result.set_index('timestamp', inplace=True)
    return result


# --- Strategy Class (Updated) ---

class Strategy5(BaseLabelingStrategy):
    """
    HTF-LTF Context Quantile labeling (Phase-Aware).
    
    1. HTF Context: Defined by Open->Close return exceeding a historical quantile threshold.
    2. LTF Label: Applied if the HTF context matches AND the LTF potential return 
       (LTF Open -> HTF High/Low) exceeds the same HTF quantile threshold, 
       AND the target has not yet been hit (Phase-Awareness).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store intermediate data for analysis
        self._intermediate_htf_returns_df = pd.DataFrame()
        
    def _calculate_htf_net_returns(self, df_htf: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculates Open->Close net returns for HTF bars."""
        
        # Long Entry: HTF Open + Fees/Slippage | Long Exit: HTF Close - Fees/Slippage
        long_entry = df_htf['htf_open'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        long_exit = df_htf['htf_close'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        htf_net_return_long = (long_exit - long_entry) / long_entry

        # Short Entry: HTF Open - Fees/Slippage | Short Exit: HTF Close + Fees/Slippage
        short_entry = df_htf['htf_open'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        short_exit = df_htf['htf_close'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        htf_net_return_short = (short_entry - short_exit) / short_entry
        
        return htf_net_return_long, htf_net_return_short
    
    def _calculate_htf_max_excursion_returns(self, df_htf: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates maximum favorable net returns for HTF bars (Open -> High/Low), 
        used to define the magnitude of a significant move for the threshold.
        """
        
        # 1. LONG Max Favorable Excursion (Entry at Open, Exit at High)
        # Long Entry: HTF Open + Fees/Slippage
        long_entry = df_htf['htf_open'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        # Long Exit: HTF High - Fees/Slippage
        long_exit = df_htf['htf_high'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        htf_net_return_long = (long_exit - long_entry) / long_entry

        # 2. SHORT Max Favorable Excursion (Entry at Open, Exit at Low)
        # Short Entry: HTF Open - Fees/Slippage
        short_entry = df_htf['htf_open'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        # Short Exit: HTF Low + Fees/Slippage
        short_exit = df_htf['htf_low'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        htf_net_return_short = (short_entry - short_exit) / short_entry
        
        return htf_net_return_long, htf_net_return_short
        
    def calculate_raw_labels(self, df_original_input: pd.DataFrame) -> pd.DataFrame:
        df = df_original_input.copy()
        
        # 1. Attach HTF Context
        df = attach_htf_context(df, self.config.htf_timeframe)
        
        # --- FIX: Use a robust check for DatetimeIndex ---
        # The original check caused an error with timezone-aware dtypes (datetime64[ns, UTC])
        if not isinstance(df.index, pd.DatetimeIndex):
            # This handles cases where the index is not a DatetimeIndex at all
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')


        # 2. Calculate HTF Net Returns (Open -> high/low for Max Excursion)
        # Group by HTF bar to get a single return per bar
        htf_groups = df.groupby('htf_bar_index').first()
        
        htf_net_return_long, htf_net_return_short = self._calculate_htf_max_excursion_returns(htf_groups) # Use _calculate_htf_net_returns when using Open->Close
        
      # Store for analysis
        self._intermediate_htf_returns_df = pd.DataFrame({
            'Net_Return_Long': htf_net_return_long.reset_index(drop=True),
            'Net_Return_Short': htf_net_return_short.reset_index(drop=True)
        })

        # 3. Calculate Quantile Thresholds from POSITIVE HTF returns
        
        # Filter only POSITIVE returns
        positive_long_returns = htf_net_return_long[htf_net_return_long > 0]
        positive_short_returns = htf_net_return_short[htf_net_return_short > 0]
        
        if positive_long_returns.empty or positive_short_returns.empty:
             self.logger.warning("No positive HTF returns found. Setting thresholds to 0.")
             bullish_threshold = 0
             bearish_threshold = 0
        else:
             bullish_threshold = positive_long_returns.quantile(self.config.bullish_quantile_pct / 100.0)
             bearish_threshold = positive_short_returns.quantile(self.config.bearish_quantile_pct / 100.0)
        
        self.logger.info(f"Calculated Bullish Threshold: {bullish_threshold:.4f} | Bearish Threshold: {bearish_threshold:.4f}")

        # 4. HTF Regime Labeling (The context for LTF trades)
        htf_labels = pd.Series(0, index=htf_groups.index)
        
        # A. Bullish Regime: HTF Long Return >= Threshold
        bullish_regime = htf_net_return_long >= bullish_threshold
        htf_labels[bullish_regime] = 1
        
        # B. Bearish Regime: HTF Short Return >= Threshold
        bearish_regime = htf_net_return_short >= bearish_threshold
        htf_labels[bearish_regime] = -1

        # Propagate HTF label to LTF bars
        ltf_htf_label = df['htf_bar_index'].map(htf_labels).values
        df['htf_label'] = ltf_htf_label

        # 5. LTF Potential Net Return (LTF Open -> HTF High/Low)
        # Entry/Exit includes fees/slippage
        
        # Long Potential (LTF Open -> HTF High)
        ltf_entry_long = df['open'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        ltf_exit_long = df['htf_high'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        ltf_net_return_long = (ltf_exit_long - ltf_entry_long) / ltf_entry_long

        # Short Potential (LTF Open -> HTF Low)
        ltf_entry_short = df['open'] * (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        ltf_exit_short = df['htf_low'] * (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        ltf_net_return_short = (ltf_entry_short - ltf_exit_short) / ltf_entry_short

        # 6. Apply all three filters (Context + Phase-Aware + Potential)
        
        df['label'] = 0
        
        # Phase-aware masks (Filter 2)
        bullish_mask = (df['htf_label'] == 1) & (df.index <= df['htf_high_ts'])
        bearish_mask = (df['htf_label'] == -1) & (df.index <= df['htf_low_ts'])

        # LTF Long Label (Filter 1 & 2 & 3)
        long_condition = bullish_mask & (ltf_net_return_long >= bullish_threshold)
        df.loc[long_condition, 'label'] = 1

        # LTF Short Label (Filter 1 & 2 & 3)
        short_condition = bearish_mask & (ltf_net_return_short >= bearish_threshold)
        df.loc[short_condition, 'label'] = -1

        return df[['label']]

    # --- Strategy-Specific Analysis Implementation (New) ---

    def perform_strategy_specific_analysis(
        self,
        df_original_input: pd.DataFrame,
        plotter: Any,
        calculator: Any
    ) -> List[Tuple[str, Any]]:
        """
        Performs analysis specific to Strategy 5: HTF Net Return Distribution
        Analysis to verify the quantile thresholds used for filtering.
        """
        
        # 1. Check for and process intermediate HTF returns
        df_net_returns = self._intermediate_htf_returns_df.copy()
        
        if df_net_returns.empty or df_net_returns.isnull().all().all():
            self.logger.warning("Intermediate HTF net returns DataFrame is empty or invalid. Re-calculating raw labels to ensure data availability.")
            # Re-run calculation to populate the internal DF (inefficient, but necessary for standalone analysis call)
            # We don't use the output, just the side effect of populating self._intermediate_htf_returns_df
            self.calculate_raw_labels(df_original_input.copy())
            df_net_returns = self._intermediate_htf_returns_df.copy()
            
            if df_net_returns.empty:
                 self.logger.error("HTF net returns still empty after re-calculation. Skipping analysis.")
                 return []
        
        # Filter for positive returns only, as these are used for threshold calculation
        df_positive_returns = pd.DataFrame({
            'Net_Return_Long': df_net_returns['Net_Return_Long'].clip(lower=0),
            'Net_Return_Short': df_net_returns['Net_Return_Short'].clip(lower=0)
        })

        # Convert returns to percentage for better visualization
        df_positive_returns['Net_Return_Long'] *= 100.0
        df_positive_returns['Net_Return_Short'] *= 100.0
        
        analysis_artifacts = []

        # 2. Analysis: Distribution of Positive HTF Net Returns with Thresholds
        plot_specs = [
            {
                'column': 'Net_Return_Long',
                'title': f'HTF Long Return Distribution (Bullish Filter at {self.config.bullish_quantile_pct}th Pct)',
                'color': 'green',
                'xlabel': 'HTF Net Return (%)',
                # Upper threshold for positive returns
                'label_upper_pct': self.config.bullish_quantile_pct, 
            },
            {
                'column': 'Net_Return_Short',
                'title': f'HTF Short Return Distribution (Bearish Filter at {self.config.bearish_quantile_pct}th Pct)',
                'color': 'red',
                'xlabel': 'HTF Net Return (%)',
                # Upper threshold for positive returns
                'label_upper_pct': self.config.bearish_quantile_pct, 
            }
        ]
        
        # Use the consolidated plotting function for side-by-side visualization
        fig_returns = plotter.plot_feature_distributions(
            df_positive_returns, 
            plot_specs=plot_specs, 
            symbol=df_original_input.index.name if df_original_input.index.name else 'Asset',
            interval=getattr(self.config, "htf_timeframe", "HTF")
        )
        
        analysis_artifacts.append(("htf_return_quantile_analysis", fig_returns))

        # 3. Analysis: Final Raw Label Distribution (Requires re-running to get labels)
        try:
            # Re-run to ensure the labels are calculated in the latest state
            raw_labeled_df = self.calculate_raw_labels(df_original_input.copy())
            
            # Assuming 'calculate_label_distribution' exists in AnalysisCalculator
            label_summary = calculator.calculate_label_distribution(raw_labeled_df['label'])
            
            fig_labels = plotter.plot_label_distribution(
                label_summary, 
                symbol=df_original_input.index.name if df_original_input.index.name else 'Asset',
                interval=getattr(self.config, "htf_timeframe", "HTF")
            )
            
            analysis_artifacts.append(("raw_label_distribution", fig_labels))
            
        except Exception as e:
            self.logger.error(f"Error calculating or plotting label distribution: {e}. Returning analysis for HTF returns only.")

        return analysis_artifacts