# utils/labeling_strategies/strategy4.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .base_strategy import BaseLabelingStrategy, logger, FLOAT_EPSILON

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

    Parameters (configured via LABELING_CONFIG in params.py):
    - 'n_clusters': The number of clusters (K) for the K-Means algorithm.
    - 'features_for_clustering': A list of original feature names to be used for clustering.
                                 These features will be scaled and PCA-transformed.
    - 'pca_n_components': The number of components or explained variance ratio for PCA.
                          Can be an integer (e.g., 3) or a float between 0 and 1 (e.g., 0.95).
    - 'cluster_to_label_mapping': A dictionary mapping each cluster ID (integer) to a trading label
                                  (-1, 0, or 1). This mapping is derived from the exploratory analysis
                                  of median future returns per cluster.
    - 'trading_fee_rate': The fee rate applied per transaction (e.g., 0.0005 for 0.05%).
    - 'slippage_rate': The estimated slippage rate per transaction (e.g., 0.0001 for 0.01%).
    - 'future_return_window': The number of bars forward to calculate future returns for interpretation.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes Strategy 4 (Clustering-Based Labeling Strategy).
        """
        super().__init__(config, logger)
        self.logger.info("Strategy 4 (Clustering-Based Labeling) initializing...")
        self._validate_strategy_config()

        self.n_clusters = self.config['n_clusters']
        self.features_for_clustering = self.config['features_for_clustering']
        self.pca_n_components = self.config['pca_n_components']
        self.cluster_to_label_mapping = self.config['cluster_to_label_mapping']
        self.trading_fee_rate = self.config.get('trading_fee_rate', 0.0005) # Default to 0.05% (fraction)
        # CHANGE: Load slippage_rate directly as a fraction
        self.slippage_rate = self.config.get('slippage_rate', 0.0001) # Default to 0.01% (fraction)
        self.future_return_window = self.config.get('future_return_window', 50) # Default to 50 bars

        # Scaler and PCA will be initialized and fitted within calculate_raw_labels
        # to ensure temporal safety (only fit on available historical data).
        self.scaler = None
        self.pca = None
        self.kmeans = None

        self.logger.info(f"  Number of Clusters (K): {self.n_clusters}")
        self.logger.info(f"  PCA Components/Variance: {self.pca_n_components}")
        self.logger.info(f"  Cluster to Label Mapping: {self.cluster_to_label_mapping}")
        self.logger.info(f"  Features for Clustering: {self.features_for_clustering[:5]}... (showing first 5)")
        self.logger.info(f"  Trading Fee Rate: {self.trading_fee_rate:.4f}")
        self.logger.info(f"  Slippage Rate: {self.slippage_rate:.6f}") # Log the fraction directly
        self.logger.info(f"  Future Return Window: {self.future_return_window} bars")


    def _validate_strategy_config(self):
        """
        Validates configuration parameters specific to Strategy 4.
        """
        required_keys = ['n_clusters', 'features_for_clustering', 'pca_n_components', 'cluster_to_label_mapping']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key for Strategy 4: '{key}'")

        if not isinstance(self.config['n_clusters'], int) or self.config['n_clusters'] <= 1:
            raise ValueError("'n_clusters' must be an integer greater than 1.")
        
        if not isinstance(self.config['features_for_clustering'], list) or not self.config['features_for_clustering']:
            raise ValueError("'features_for_clustering' must be a non-empty list of strings.")
        
        if not isinstance(self.config['pca_n_components'], (int, float)) or (isinstance(self.config['pca_n_components'], float) and not (0 < self.config['pca_n_components'] <= 1)):
            raise ValueError("'pca_n_components' must be a positive integer or a float between 0 and 1.")
        
        if not isinstance(self.config['cluster_to_label_mapping'], dict) or not self.config['cluster_to_label_mapping']:
            raise ValueError("'cluster_to_label_mapping' must be a non-empty dictionary.")
        
        # Ensure all mapped labels are -1, 0, or 1
        if not all(label in [-1, 0, 1] for label in self.config['cluster_to_label_mapping'].values()):
            raise ValueError("All values in 'cluster_to_label_mapping' must be -1, 0, or 1.")

        # Validate new fee/slippage parameters
        if not isinstance(self.config.get('trading_fee_rate', 0.0), (int, float)) or self.config.get('trading_fee_rate', 0.0) < 0:
            raise ValueError("'trading_fee_rate' must be a non-negative number.")
        # CHANGE: Validate slippage_rate as a fraction (0-1)
        slippage_rate_val = self.config.get('slippage_rate', 0.0)
        if not isinstance(slippage_rate_val, (int, float)) or not (0.0 <= slippage_rate_val < 1.0): # Should be 0 to <1
            raise ValueError("'slippage_rate' must be a non-negative fraction less than 1.")
        if not isinstance(self.config.get('future_return_window', 0), int) or self.config.get('future_return_window', 0) <= 0:
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
        # Validate input df for basic OHLCV columns, as features are derived from them
        self._validate_input_df(df, ['close'] + self.features_for_clustering)

        df_copy = df.copy() # Work on a copy

        # --- 1. Feature Selection and NaN Handling ---
        # Select only the features for clustering
        features_df = df_copy[self.features_for_clustering].copy()

        # Drop rows with any NaN values in the selected features.
        # This is crucial before scaling and PCA/clustering.
        initial_rows = len(features_df)
        features_df.dropna(inplace=True)
        rows_dropped = initial_rows - len(features_df)

        if rows_dropped > 0:
            self.logger.warning(f"Dropped {rows_dropped} rows due to NaNs in selected features for clustering.")
        
        if features_df.empty:
            self.logger.error("Feature DataFrame is empty after dropping NaNs. Cannot perform clustering.")
            # Return a DataFrame with all neutral labels, aligned to original index
            return pd.DataFrame({'label': 0}, index=df.index)

        # Keep track of the index of the rows that remain after NaN removal
        # We will align the labels back to the original df's full index later.
        valid_indices = features_df.index

        # --- 2. Feature Scaling ---
        # Initialize and fit_transform scaler on the current data slice.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.features_for_clustering, index=valid_indices)
        self.logger.debug(f"Features scaled. Scaled data shape: {X_scaled_df.shape}")

        # --- 3. Apply PCA for Dimensionality Reduction ---
        # Initialize and fit_transform PCA on the scaled data.
        self.pca = PCA(n_components=self.pca_n_components, random_state=self.config.get('random_seed', 42))
        X_pca = self.pca.fit_transform(X_scaled_df)
        
        # Convert PCA results back to DataFrame
        pca_component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_component_names, index=valid_indices)
        self.logger.debug(f"PCA applied. Reduced data shape: {X_pca_df.shape}. Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        # --- 4. K-Means Clustering ---
        # Initialize and fit_predict KMeans on the PCA-transformed data.
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.config.get('random_seed', 42), n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_pca_df)
        
        # Create a Series of cluster labels, indexed by the valid_indices
        cluster_labels_series = pd.Series(cluster_labels, index=valid_indices)
        self.logger.debug(f"Clustering complete. Cluster distribution:\n{cluster_labels_series.value_counts().sort_index()}")

        # --- 5. Calculate NET Future Returns for Cluster Interpretation (within strategy) ---
        # Use the 'close' price from the original, aligned DataFrame for return calculation.
        # This part is for *interpreting* the clusters, not for the clustering itself.
        # It replicates the net return calculation from the exploratory notebook.

        # Ensure df_copy has 'close' column before proceeding
        if 'close' not in df_copy.columns:
            self.logger.error("Missing 'close' column in DataFrame for future return calculation.")
            # If close price is critical and missing, we can't calculate returns meaningfully.
            # Assign neutral labels for safety.
            return pd.DataFrame({'label': 0}, index=df.index)

        # Align df_copy to valid_indices, getting only the 'close' prices for calculation
        close_prices = df_copy.loc[valid_indices, 'close'].copy()
        future_close_prices = close_prices.shift(-self.future_return_window)

        # Adjusted factors for transaction costs
        # Use self.trading_fee_rate (fraction) and self.slippage_rate (fraction)
        entry_cost_long_factor = (1 + self.trading_fee_rate + self.slippage_rate)
        exit_revenue_long_factor = (1 - self.trading_fee_rate - self.slippage_rate)

        entry_revenue_short_factor = (1 - self.trading_fee_rate - self.slippage_rate)
        exit_cost_short_factor = (1 + self.trading_fee_rate + self.slippage_rate)

        safe_current_close = close_prices.replace(0, np.nan) # Avoid division by zero

        # Net Return for a theoretical LONG position (positive if profitable)
        net_return_long_potential = (
            (future_close_prices * exit_revenue_long_factor - safe_current_close * entry_cost_long_factor) /
            (safe_current_close * entry_cost_long_factor)
        ) * 100.0

        # Net Return for a theoretical SHORT position (positive if profitable)
        net_return_short_potential = (
            (safe_current_close * entry_revenue_short_factor - future_close_prices * exit_cost_short_factor) /
            (safe_current_close * entry_revenue_short_factor)
        ) * 100.0

        # Create a DataFrame for interpretation, aligned to valid_indices
        df_for_interpretation = pd.DataFrame({
            'cluster': cluster_labels_series,
            'net_return_long_potential': net_return_long_potential,
            'net_return_short_potential': net_return_short_potential
        }, index=valid_indices)

        # Determine the effective 'future_return_pct' for interpretation
        # This represents the best potential profit (long or short) at the future window.
        # Positive value indicates potential long profit, negative indicates potential short profit.
        df_for_interpretation['effective_future_return_pct'] = np.where(
            df_for_interpretation['net_return_long_potential'] > df_for_interpretation['net_return_short_potential'],
            df_for_interpretation['net_return_long_potential'],
            -df_for_interpretation['net_return_short_potential'] # Invert short potential to reflect price movement direction
        )
        
        # Drop NaNs introduced by the shift for future return calculation
        df_for_interpretation.dropna(subset=['effective_future_return_pct'], inplace=True)
        
        # Recalculate median returns for validation/logging if needed.
        # This part is primarily for internal checks/logging within the strategy class
        # if you want to verify the mapping dynamically, but the primary mapping comes
        # from the config based on your comprehensive offline analysis.
        median_returns_per_cluster_calculated = df_for_interpretation.groupby('cluster')['effective_future_return_pct'].median().sort_values(ascending=False)
        self.logger.debug(f"Median NET future returns per cluster (calculated internally):\n{median_returns_per_cluster_calculated}")

        # --- 6. Apply Label Mapping from Config ---
        # Map the cluster labels to trading labels using the mapping from config
        # (which was determined during exploratory analysis with fees/slippage).
        raw_labels_series = cluster_labels_series.map(self.cluster_to_label_mapping)

        # --- 7. Align Labels back to Original DataFrame Index ---
        # Create a DataFrame for labels with the original full index
        final_labels = pd.DataFrame(0, index=df.index, columns=['label'], dtype=np.int8)
        
        # Fill in the labels for the rows that were processed
        # Ensure that raw_labels_series also only contains data for valid_indices and aligns correctly.
        final_labels.loc[raw_labels_series.index, 'label'] = raw_labels_series.astype(np.int8)

        self.logger.debug("Raw labels calculated for Strategy 4 (Clustering-Based Labeling) and aligned to original index.")
        
        return final_labels
