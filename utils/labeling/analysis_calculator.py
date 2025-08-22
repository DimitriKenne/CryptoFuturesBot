# utils/labeling/analysis_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional

from config.params import FLOAT_EPSILON

logger = logging.getLogger(__name__)

class AnalysisCalculator:
    """
    Performs specialized calculations for labeling analysis.
    Contains both low-level scalar calculations and high-level methods
    that process entire DataFrames to produce statistical summaries.
    """

    def __init__(self, trading_fee_rate: float, slippage_tolerance_rate: float):
        """Initializes the calculator with transaction cost parameters."""
        self.trading_fee_rate = trading_fee_rate
        self.slippage_tolerance_rate = slippage_tolerance_rate
        logger.debug(f"AnalysisCalculator initialized with fee={self.trading_fee_rate:.6f}, slippage={self.slippage_tolerance_rate:.6f}.")

    # ==============================================================================
    # --- LOW-LEVEL SCALAR & VECTORIZED METHODS ---
    # ==============================================================================

    def calculate_net_return_scalar(self, entry_price: float, exit_price: float, trade_type: int) -> float:
        """Calculates net return for a single trade (percentage), accounting for costs."""
        if abs(entry_price) < FLOAT_EPSILON:
            return np.nan

        cost_factor_long_entry = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)
        revenue_factor_long_exit = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        revenue_factor_short_entry = (1 - self.trading_fee_rate - self.slippage_tolerance_rate)
        cost_factor_short_exit = (1 + self.trading_fee_rate + self.slippage_tolerance_rate)

        if trade_type == 1: # Long
            cost_to_enter = entry_price * cost_factor_long_entry
            revenue_from_exit = exit_price * revenue_factor_long_exit
            return ((revenue_from_exit - cost_to_enter) / cost_to_enter) * 100.0
        elif trade_type == -1: # Short
            revenue_from_enter = entry_price * revenue_factor_short_entry
            cost_to_exit = exit_price * cost_factor_short_exit
            return ((revenue_from_enter - cost_to_exit) / revenue_from_enter) * 100.0
        
        return np.nan

    def _calculate_mfe_mae_for_segment(
        self,
        price_segment: pd.DataFrame,
        entry_price: float,
        trade_type: int
    ) -> Tuple[float, float]:
        """
        Low-level helper to calculate MFE/MAE for a single price segment.
        """
        if price_segment.empty: return np.nan, np.nan

        if trade_type == 1: # Long trade
            favorable_prices = price_segment['high']
            adverse_prices = price_segment['low']
        elif trade_type == -1: # Short trade
            favorable_prices = price_segment['low']
            adverse_prices = price_segment['high']
        else:
            return np.nan, np.nan

        favorable_returns = np.vectorize(self.calculate_net_return_scalar)(entry_price, favorable_prices, trade_type)
        adverse_returns = np.vectorize(self.calculate_net_return_scalar)(entry_price, adverse_prices, trade_type)
        
        mfe = np.nanmax(favorable_returns) if not np.all(np.isnan(favorable_returns)) else 0
        mae = abs(np.nanmin(adverse_returns)) if not np.all(np.isnan(adverse_returns)) else 0

        return mfe, mae

    # ==============================================================================
    # --- NEW: HIGH-LEVEL ORCHESTRATION METHODS ---
    # These methods contain the loops and logic previously in LabelAnalyzer
    # ==============================================================================

    def calculate_label_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the percentage distribution of labels."""
        label_counts = df['label'].value_counts()
        distribution = (label_counts / label_counts.sum() * 100).rename('Percentage').reset_index()
        distribution.columns = ['Label', 'Percentage']
        return distribution

    def calculate_mfe_mae_stats(self, df: pd.DataFrame, f_window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates MFE/MAE statistics for all signals in a DataFrame.
        This method now contains the loop previously in LabelAnalyzer.
        
        Returns:
            A tuple containing:
            - pd.DataFrame: Raw MFE/MAE results for each trade signal.
            - pd.DataFrame: A summary table grouped by label.
        """
        results = []
        for i in range(len(df) - f_window):
            label = df['label'].iloc[i]
            if label == 0:
                continue
            
            entry_price = df['close'].iloc[i]
            segment = df.iloc[i+1 : i+1+f_window]
            mfe, mae = self._calculate_mfe_mae_for_segment(segment, entry_price, label)
            
            if not pd.isna(mfe):
                results.append({'mfe': mfe, 'mae': mae, 'label': label})
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()

        df_mfe_mae_raw = pd.DataFrame(results)
        
        # Create summary table
        summary = df_mfe_mae_raw.groupby('label').agg(
            count=('mfe', 'count'),
            mfe_mean=('mfe', 'mean'),
            mfe_median=('mfe', 'median'),
            mfe_p75=('mfe', lambda x: x.quantile(0.75)),
            mae_mean=('mae', 'mean'),
            mae_median=('mae', 'median'),
            mae_p75=('mae', lambda x: x.quantile(0.75)),
        ).reset_index()

        return df_mfe_mae_raw, summary

    def calculate_future_returns_by_label(self, df: pd.DataFrame, horizons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates future net returns for multiple horizons across all signals.
        
        Returns:
            A tuple containing:
            - pd.DataFrame: A summary table of returns grouped by label and horizon.
            - pd.DataFrame: Raw return results for each signal and horizon.
        """
        results = []
        for h in horizons:
            for i in range(len(df) - h):
                label = df['label'].iloc[i]
                if label == 0:
                    continue
                
                entry_price = df['close'].iloc[i]
                exit_price = df['close'].iloc[i + h]
                ret = self.calculate_net_return_scalar(entry_price, exit_price, label)
                
                if not pd.isna(ret):
                    results.append({'horizon': h, 'return_pct': ret, 'label': label})
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()
            
        df_returns_raw = pd.DataFrame(results)
        
        summary = df_returns_raw.groupby(['label', 'horizon'])['return_pct'].agg(
            ['count', 'mean', 'median', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        ).reset_index()
        summary.columns = ['label', 'horizon', 'count', 'mean', 'median', 'std', 'p25', 'p75']
        
        return summary, df_returns_raw

    def calculate_profitability_by_regime(self, df: pd.DataFrame, horizons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates future returns segmented by volatility regime.
        
        Returns:
            A tuple containing:
            - pd.DataFrame: A summary table grouped by regime, label, and horizon.
            - pd.DataFrame: Raw return results for each signal and horizon.
        """
        if 'volatility_regime' not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
            
        results = []
        for h in horizons:
            for i in range(len(df) - h):
                label = df['label'].iloc[i]
                regime = df['volatility_regime'].iloc[i]
                if label == 0 or pd.isna(regime):
                    continue
                
                entry_price = df['close'].iloc[i]
                exit_price = df['close'].iloc[i + h]
                ret = self.calculate_net_return_scalar(entry_price, exit_price, label)
                
                if not pd.isna(ret):
                    results.append({
                        'horizon': h, 
                        'return_pct': ret, 
                        'label': label, 
                        'volatility_regime': int(regime)
                    })
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()
        
        df_regime_raw = pd.DataFrame(results)
        
        summary = df_regime_raw.groupby(['volatility_regime', 'label', 'horizon'])['return_pct'].agg(
            ['count', 'mean', 'median', 'std']
        ).reset_index()
        
        return summary, df_regime_raw