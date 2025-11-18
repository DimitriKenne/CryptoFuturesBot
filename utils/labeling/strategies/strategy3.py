import pandas as pd
import numpy as np
from .base_strategy import BaseLabelingStrategy
from typing import Dict, Any, List, Tuple

def attach_htf_context(df: pd.DataFrame, htf_interval: str = "1d") -> pd.DataFrame:
    """
    Attach HTF OHLC, bar index, and the timestamp of the next HTF bar start 
    (which acts as the expiry time for the LTF trade).
    """
    # 1. Calculate HTF OHLC and the start of the next bar (expiry time)
    htf_data = df.resample(htf_interval).agg(
        htf_open=('open', 'first'),
        htf_close=('close', 'last'),
        htf_high=('high', 'max'),
        htf_low=('low', 'min'),
        # Get the timestamp of the NEXT bar start, which is the close of the current HTF trade window
        htf_expiry_ts=('close', 'last') 
    ).reset_index(names='htf_bar_start_ts')
    
    # Shift the start time to get the expiry time
    htf_data['htf_expiry_ts'] = htf_data['htf_bar_start_ts'].shift(-1)
    # The last bar's expiry will be NaN, we can fill it with a large value or drop the last bar later.
    htf_data.dropna(subset=['htf_expiry_ts'], inplace=True) 

    # 2. Add HTF bar index and range for the volatility filter
    htf_data['htf_bar_index'] = np.arange(len(htf_data))
    htf_data['htf_range'] = htf_data['htf_high'] - htf_data['htf_low']
    
    # 3. Final preparation and merge_asof
    df_sorted = df.sort_index().reset_index(names='timestamp')
    
    result = pd.merge_asof(
        df_sorted,
        htf_data,
        left_on='timestamp',
        right_on='htf_bar_start_ts',
        direction='backward'
    )
    result.set_index('timestamp', inplace=True)
    
    # Drop the redundant 'htf_bar_start_ts' column from the merge
    result.drop(columns=['htf_bar_start_ts'], inplace=True)
    
    return result


class Strategy3(BaseLabelingStrategy):
    """
    HTF Volatility Filtered Labeling (Fixed R/R, Daily Expiration).
    1. Filter LTF bars based on the volatility of the corresponding HTF bar.
    2. Label trades that hit a fixed TP/SL percentage before the HTF bar closes.
    """

    def __init__(self, config, logger, trading_fee_rate, slippage_tolerance_rate):
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        # Store HTF ranges for analysis
        self._intermediate_htf_ranges = pd.Series(dtype=float) 

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the raw labels (1, -1, 0) for the input LTF DataFrame.
        """
        htf_interval = getattr(self.config, "htf_timeframe", "1d")
        out = attach_htf_context(df, htf_interval=htf_interval)
        out['label'] = 0

        # Load config parameters
        config = self.config # Assuming config is already the correct type
        volatility_quantile_pct = config.volatility_quantile_pct
        take_profit_pct = config.take_profit_pct / 100.0  # Convert to fraction
        stop_loss_pct = config.stop_loss_pct / 100.0      # Convert to fraction
        
        trading_fee = self.trading_fee_rate
        slippage = self.slippage_tolerance_rate

        # --- 1. HTF Volatility Filter ---
        
        # Calculate the Volatility Threshold (e.g., 75th percentile of daily range)
        htf_ranges = out.groupby('htf_bar_index')['htf_range'].first().dropna()
        self._intermediate_htf_ranges = htf_ranges # Store for analysis
        
        if htf_ranges.empty:
            self.logger.warning("HTF ranges are empty. Cannot apply volatility filter.")
            return out[['label']]
            
        volatility_threshold = np.percentile(htf_ranges, volatility_quantile_pct)
        
        # Identify LTF bars corresponding to highly volatile HTF bars
        volatile_htf_bars = htf_ranges[htf_ranges >= volatility_threshold].index
        volatility_mask = out['htf_bar_index'].isin(volatile_htf_bars)
        
        # --- 2. Trade Targets and Barriers ---
        
        # Calculate targets for entry at LTF 'open'
        entry_price = out['open']
        
        # --- Long Trade Targets (Label 1) ---
        # Long Entry Cost: Open price * (1 + fee + slippage)
        long_entry_cost = entry_price * (1 + trading_fee + slippage)
        # TP Price: Entry Price * (1 + TP %)
        long_tp_price = entry_price * (1 + take_profit_pct)
        # SL Price: Entry Price * (1 - SL %)
        long_sl_price = entry_price * (1 - stop_loss_pct)
        
        # --- Short Trade Targets (Label -1) ---
        # Short Entry Proceeds: Open price * (1 - fee - slippage)
        short_entry_proceeds = entry_price * (1 - trading_fee - slippage)
        # TP Price (Exit): Entry Price * (1 - TP %)
        short_tp_price = entry_price * (1 - take_profit_pct)
        # SL Price (Exit): Entry Price * (1 + SL %)
        short_sl_price = entry_price * (1 + stop_loss_pct)


        # --- 3. Labeling Logic (Triple Barrier adapted to single HTF bar) ---
        
        # Group by HTF bar to apply the trade simulation within each day/period
        def apply_labeling(group: pd.DataFrame) -> pd.Series:
            
            # --- Caching targets for this group's LTF bars ---
            # Use LTF High/Low to check for target hits within the *same* LTF bar
            # We look ahead from the current bar's open to the end of the day.
            
            labels = pd.Series(0, index=group.index)
            
            for i in range(len(group)):
                # Only process bars that passed the volatility filter
                if not volatility_mask.loc[group.index[i]]:
                    continue
                
                current_bar = group.iloc[i]
                start_time = current_bar.name
                expiry_time = current_bar['htf_expiry_ts']
                
                # The lookahead window for this specific LTF bar
                lookahead_window = group.loc[(group.index > start_time) & (group.index <= expiry_time)]
                
                # If there are no future bars (the last bar of the day), we cannot assess targets
                if lookahead_window.empty:
                    # Check if the current bar itself hits TP/SL.
                    # This check is complex (intrabar movement) and usually skipped
                    # or handled by setting TP/SL relative to Open/Close.
                    # We will only rely on targets hit by subsequent bars for simplicity and robustness.
                    continue
                    
                # --- Long Trade Check ---
                long_tp_price_i = current_bar['long_tp_price']
                long_sl_price_i = current_bar['long_sl_price']
                
                # Did TP hit before SL? (Check against future Highs/Lows)
                tp_hit_ts_long = lookahead_window.loc[lookahead_window['high'] >= long_tp_price_i].index.min()
                sl_hit_ts_long = lookahead_window.loc[lookahead_window['low'] <= long_sl_price_i].index.min()

                if pd.notna(tp_hit_ts_long) and (pd.isna(sl_hit_ts_long) or tp_hit_ts_long < sl_hit_ts_long):
                    labels.loc[start_time] = 1 # Long profitable
                elif pd.notna(sl_hit_ts_long) and (pd.isna(tp_hit_ts_long) or sl_hit_ts_long < tp_hit_ts_long):
                    # Long unprofitable (SL hit first) - Label as 0
                    pass
                
                
                # --- Short Trade Check ---
                short_tp_price_i = current_bar['short_tp_price']
                short_sl_price_i = current_bar['short_sl_price']
                
                # Did TP hit before SL? (Short TP is a lower price, Short SL is a higher price)
                tp_hit_ts_short = lookahead_window.loc[lookahead_window['low'] <= short_tp_price_i].index.min()
                sl_hit_ts_short = lookahead_window.loc[lookahead_window['high'] >= short_sl_price_i].index.min()

                if pd.notna(tp_hit_ts_short) and (pd.isna(sl_hit_ts_short) or tp_hit_ts_short < sl_hit_ts_short):
                    labels.loc[start_time] = -1 # Short profitable
                elif pd.notna(sl_hit_ts_short) and (pd.isna(tp_hit_ts_short) or sl_hit_ts_short < tp_hit_ts_short):
                    # Short unprofitable (SL hit first) - Label as 0
                    pass

            return labels
        
        # Attach targets to the LTF DataFrame for easier access in the loop
        out['long_tp_price'] = long_tp_price
        out['long_sl_price'] = long_sl_price
        out['short_tp_price'] = short_tp_price
        out['short_sl_price'] = short_sl_price
        
        # Apply the complex labeling logic group-wise (per HTF bar)
        out['label'] = out.groupby('htf_bar_index', group_keys=False).apply(apply_labeling).fillna(0).astype(int)

        # Clean output
        return out[['label']]

    def perform_strategy_specific_analysis(self, df_original_input, plotter, calculator) -> List[Tuple[str, Any]]:
        """
        Analysis for Strategy 3: Plot the distribution of HTF ranges to confirm the volatility filter.
        """
        self.logger.info("Performing strategy-specific analysis for Strategy 3 (Future Range Dominance)...")
        if self._intermediate_htf_ranges.empty:
            self.logger.warning("No intermediate HTF range data found for Strategy 3. Skipping analysis.")
            return []
            
        df_ranges = self._intermediate_htf_ranges.copy()
        
        # The range is in absolute price terms. We'll convert to percentage range for better context.
        # Estimate average price to normalize the range for distribution plot
        avg_price = df_original_input['close'].mean()
        if avg_price > 0:
            df_ranges_pct = (df_ranges / avg_price) * 100.0
        else:
            # Avoid division by zero, set to zero
            df_ranges_pct = df_ranges * 0.0 
            
        df_ranges_pct.name = "HTF_Range_Pct"
        
        # Plot the distribution of HTF ranges
        plot_specs = [
            {
                'column': 'HTF_Range_Pct',
                'title': f'HTF Range Distribution (Filter at {self.config.volatility_quantile_pct}th Percentile)',
                'color': 'skyblue',
                'xlabel': 'HTF Range (%)',
                # This is the feature that requires the upper quantile line
                'label_upper_pct': self.config.volatility_quantile_pct, 
            }
        ]

        # Use the consolidated plotting function
        fig = plotter.plot_feature_distributions(
            df_ranges_pct.to_frame(), 
            plot_specs=plot_specs, 
            symbol=df_original_input.index.name,
            interval=self.config.htf_timeframe
        )
        
        return [('htf_range_distribution', fig)]