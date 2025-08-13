# utils/labeling_strategies/strategy4.py

import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

# Import the specific config dataclass for this strategy
from config.label_config_schema import Strategy4Config

class Strategy4(BaseLabelingStrategy):
    """
    Strategy 4: Clustering-Based Labeling.

    This strategy identifies market regimes by applying K-Means clustering
    on a set of pre-selected and PCA-reduced technical features.
    It then maps these identified clusters to trading labels (1 for Buy, -1 for Sell, 0 for Neutral)
    based on the historical median future returns observed within each cluster, now calculated
    with explicit consideration for trading fees and slippage.

    This approach is data-driven, allowing the model to discover patterns in market conditions
    that correlate with significant future price movements, adjusted for realistic transaction costs.
    """

    def __init__(self, config: Strategy4Config, logger: logging.Logger, trading_fee_rate: float, slippage_tolerance_pct: float):
        """
        Initializes Strategy 4 (Clustering-Based Labeling Strategy).

        Args:
            config (Strategy4Config): The configuration dataclass for this strategy.
            logger (logging.Logger): A logger instance.
            trading_fee_rate (float): The transaction fee rate.
            slippage_tolerance_pct (float): The estimated slippage rate.
        """
        # Pass config to the superclass, which now also accepts fee/slippage
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_pct)
        self.logger.info("Strategy 4 (Clustering-Based Labeling) initializing...")
        self._validate_strategy_config()

        # Access parameters directly from the Strategy4Config dataclass
        self.n_clusters = self.config.n_clusters
        self.features_for_clustering = self.config.features_for_clustering
        self.pca_n_components = self.config.pca_n_components
        self.cluster_to_label_mapping = self.config.cluster_to_label_mapping
        self.future_return_window = self.config.future_return_window

        # Scaler, PCA, and KMeans will be initialized and fitted within calculate_raw_labels
        # to ensure temporal safety (only fit on available historical data).
        self.scaler = None
        self.pca = None
        self.kmeans = None

        self.logger.info(f"  Number of Clusters (K): {self.n_clusters}")
        self.logger.info(f"  PCA Components/Variance: {self.pca_n_components}")
        self.logger.info(f"  Cluster to Label Mapping: {self.cluster_to_label_mapping}")
        self.logger.info(f"  Features for Clustering: {self.features_for_clustering[:5]}... (showing first 5)")
        self.logger.info(f"  Trading Fee Rate: {self.trading_fee_rate:.4f}")
        self.logger.info(f"  Slippage Tolerance: {self.slippage_tolerance_pct:.6f}")
        self.logger.info(f"  Future Return Window: {self.future_return_window} bars")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Strategy 4,
        now accessing directly from self.config (Strategy4Config).
        """
        if not isinstance(self.config.n_clusters, int) or self.config.n_clusters <= 1:
            raise ValueError("'n_clusters' must be an integer greater than 1.")
        
        if not isinstance(self.config.features_for_clustering, list) or not self.config.features_for_clustering:
            raise ValueError("'features_for_clustering' must be a non-empty list of strings.")
        
        if not isinstance(self.config.pca_n_components, (int, float)) or (isinstance(self.config.pca_n_components, float) and not (0 < self.config.pca_n_components <= 1)):
            raise ValueError("'pca_n_components' must be a positive integer or a float between 0 and 1.")
        
        if not isinstance(self.config.cluster_to_label_mapping, dict) or not self.config.cluster_to_label_mapping:
            raise ValueError("'cluster_to_label_mapping' must be a non-empty dictionary.")
        
        # Ensure all mapped labels are -1, 0, or 1
        if not all(label in [-1, 0, 1] for label in self.config.cluster_to_label_mapping.values()):
            raise ValueError("All values in 'cluster_to_label_mapping' must be -1, 0, or 1.")

        if not isinstance(self.config.future_return_window, int) or self.config.future_return_window <= 0:
            raise ValueError("'future_return_window' must be a positive integer.")

        self.logger.debug("Strategy 4 config validated.")


    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw labels for Strategy 4 (Clustering-Based Labeling).

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and engineered features,
                               indexed by time. Assumed to be cleaned (no NaNs in OHLCV)
                               by LabelGenerator.

        Returns:
            pd.DataFrame: DataFrame with 'label' column (1, -1, or 0).
                          The index will match the input DataFrame's index.
        """
        self.logger.debug("Calculating raw labels for Strategy 4 (Clustering-Based Labeling).")
        self._validate_input_df(df, ['close'] + self.features_for_clustering)

        df_copy = df.copy() # Work on a copy

        # --- 1. Feature Selection and NaN Handling ---
        # Select only the features for clustering
        features_df = df_copy[self.features_for_clustering].copy()

        # Drop rows with any NaN values in the selected features.
        initial_rows = len(features_df)
        features_df.dropna(inplace=True)
        rows_dropped = initial_rows - len(features_df)

        if rows_dropped > 0:
            self.logger.warning(f"Dropped {rows_dropped} rows due to NaNs in selected features for clustering.")
        
        if features_df.empty:
            self.logger.error("Feature DataFrame is empty after dropping NaNs. Cannot perform clustering.")
            return pd.DataFrame({'label': 0}, index=df.index)

        # Keep track of the index of the rows that remain after NaN removal
        valid_indices = features_df.index

        # --- 2. Feature Scaling ---
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.features_for_clustering, index=valid_indices)
        self.logger.debug(f"Features scaled. Scaled data shape: {X_scaled_df.shape}")

        # --- 3. Apply PCA for Dimensionality Reduction ---
        self.pca = PCA(n_components=self.pca_n_components, random_state=self.config.get('random_seed', 42)) # Access random_seed from config if available, otherwise default
        X_pca = self.pca.fit_transform(X_scaled_df)
        
        pca_component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_component_names, index=valid_indices)
        self.logger.debug(f"PCA applied. Reduced data shape: {X_pca_df.shape}. Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        # --- 4. K-Means Clustering ---
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.config.get('random_seed', 42), n_init=10) # Access random_seed from config if available, otherwise default
        cluster_labels = self.kmeans.fit_predict(X_pca_df)
        
        cluster_labels_series = pd.Series(cluster_labels, index=valid_indices)
        self.logger.debug(f"Clustering complete. Cluster distribution:\n{cluster_labels_series.value_counts().sort_index()}")

        # --- 5. Calculate NET Future Returns for Cluster Interpretation (within strategy) ---
        if 'close' not in df_copy.columns:
            self.logger.error("Missing 'close' column in DataFrame for future return calculation. Cannot interpret clusters based on returns.")
            return pd.DataFrame({'label': 0}, index=df.index)

        close_prices = df_copy.loc[valid_indices, 'close'].copy()
        future_close_prices = close_prices.shift(-self.future_return_window)

        # Adjusted factors for transaction costs, using attributes from base class
        entry_cost_long_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)
        exit_revenue_long_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)

        entry_revenue_short_factor = (1 - self.trading_fee_rate - self.slippage_tolerance_pct)
        exit_cost_short_factor = (1 + self.trading_fee_rate + self.slippage_tolerance_pct)

        safe_current_close = close_prices.replace(0, np.nan) # Avoid division by zero

        net_return_long_potential = (
            (future_close_prices * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) /
            (safe_current_close * entry_cost_long_factor)
        ) * 100.0

        net_return_short_potential = (
            (safe_current_close * entry_revenue_short_factor - future_close_prices * exit_cost_short_factor) /
            (safe_current_close * entry_revenue_short_factor)
        ) * 100.0

        df_for_interpretation = pd.DataFrame({
            'cluster': cluster_labels_series,
            'net_return_long_potential': net_return_long_potential,
            'net_return_short_potential': net_return_short_potential
        }, index=valid_indices)

        df_for_interpretation['effective_future_return_pct'] = np.where(
            df_for_interpretation['net_return_long_potential'] > df_for_interpretation['net_return_short_potential'],
            df_for_interpretation['net_return_long_potential'],
            -df_for_interpretation['net_return_short_potential'] # Invert short potential to reflect price movement direction
        )
        
        df_for_interpretation.dropna(subset=['effective_future_return_pct'], inplace=True)
        
        median_returns_per_cluster_calculated = df_for_interpretation.groupby('cluster')['effective_future_return_pct'].median().sort_values(ascending=False)
        self.logger.debug(f"Median NET future returns per cluster (calculated internally):\n{median_returns_per_cluster_calculated}")

        # --- 6. Apply Label Mapping from Config ---
        raw_labels_series = cluster_labels_series.map(self.cluster_to_label_mapping)

        # --- 7. Align Labels back to Original DataFrame Index ---
        final_labels = pd.DataFrame(0, index=df.index, columns=['label'], dtype=np.int8)
        
        final_labels.loc[raw_labels_series.index, 'label'] = raw_labels_series.astype(np.int8)

        self.logger.debug("Raw labels calculated for Strategy 4 (Clustering-Based Labeling) and aligned to original index.")
        
        return final_labels
