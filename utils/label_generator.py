# utils/label_generator.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type # Import Type for type hinting
import math

# --- Import Strategy Classes ---
# Need to import the concrete strategy classes here
from .labeling_strategies.base_strategy import BaseLabelingStrategy, logger # Import logger from base
from .labeling_strategies.directional_ternary import DirectionalTernaryStrategy
from .labeling_strategies.triple_barrier import TripleBarrierStrategy
from .labeling_strategies.max_return_quantile import MaxReturnQuantileStrategy
from .labeling_strategies.ema_return_percentile import EmaReturnPercentileStrategy

# Define FLOAT_EPSILON here as it's used in strategies and potentially in LabelGenerator
FLOAT_EPSILON = 1e-9

# Map label_type strings to strategy classes
STRATEGY_MAP: Dict[str, Type[BaseLabelingStrategy]] = {
    'directional_ternary': DirectionalTernaryStrategy,
    'triple_barrier': TripleBarrierStrategy,
    'max_return_quantile': MaxReturnQuantileStrategy,
    "ema_return_percentile": EmaReturnPercentileStrategy,
}

class LabelGenerator:
    """
    Generates trading labels by orchestrating different labeling strategies.

    Selects a strategy based on configuration, calculates raw labels, and applies
    label propagation smoothing based on min_holding_period.
    Uses the centralized logging configured by the calling script.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the LabelGenerator with configuration and selects the strategy.

        Args:
            config (Dict[str, Any]): Configuration dictionary from params.py (LABELING_CONFIG).
                                     Must contain 'label_type' and 'min_holding_period',
                                     plus parameters relevant to the selected strategy type.
            logger (logging.Logger): A logger instance for logging messages.
        """
        self.config = config
        self.logger = logger
        self.label_type = self.config.get('label_type')
        self.min_holding_period = self.config.get('min_holding_period', 1) # Default to 1 if not specified

        if self.label_type not in STRATEGY_MAP:
            raise ValueError(f"Unknown label strategy type: '{self.label_type}'. "
                             f"Available strategies are: {', '.join(STRATEGY_MAP.keys())}")

        strategy_class = STRATEGY_MAP[self.label_type]

        try:
            # Pass the relevant subset of the config to the strategy
            # Each strategy will validate its own specific parameters
            self.strategy: BaseLabelingStrategy = strategy_class(config=self.config, logger=self.logger)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred initializing LabelGenerator: {e}")
            raise # Re-raise the exception after logging

        self.logger.info(f"LabelGenerator initialized for '{self.label_type}' strategy.")
        self.logger.info(f"  Min Holding Period (Propagation): {self.min_holding_period} bars")


    @staticmethod
    def get_available_strategies() -> List[str]:
        """Returns a list of available labeling strategy names."""
        return list(STRATEGY_MAP.keys())

    def calculate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates trading labels using the configured strategy and applies label propagation.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and required features,
                               indexed by time.

        Returns:
            pd.DataFrame: DataFrame with the 'label' column (1, -1, or 0) after propagation.
        Raises:
            ValueError: If input DataFrame is invalid or missing required columns.
        """
        self.logger.info(f"Starting label calculation using '{self.label_type}' strategy...")

        # Basic input validation (more detailed validation is in BaseLabelingStrategy._validate_input_df)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input DataFrame is empty or not a pandas DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'open', 'high', 'low', 'close' columns.")

        # Drop rows with any NaN in critical OHLCV columns before passing to strategy
        initial_rows = len(df)
        df_cleaned = df.dropna(subset=['open', 'high', 'low', 'close']).copy()
        if len(df_cleaned) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_cleaned)} rows due to NaNs in OHLCV data before label calculation.")
        if df_cleaned.empty:
            self.logger.error("DataFrame became empty after dropping NaNs in OHLCV data. Cannot generate labels.")
            # Return an empty DataFrame with original index and 0 labels if possible, or just empty
            return pd.DataFrame(index=df.index, data={'label': 0})


        # Log specific info for TripleBarrierStrategy if volatility adjustment is used
        if isinstance(self.strategy, TripleBarrierStrategy) and self.strategy.use_volatility_adjustment and self.strategy.atr_column_name:
            self.logger.info(f"Triple Barrier Strategy is configured to use volatility adjustment with ATR column: {self.strategy.atr_column_name}")


        # Calculate raw labels using the selected strategy
        raw_labeled_df = self.strategy.calculate_raw_labels(df_cleaned)

        # --- CRITICAL FIX: Ensure raw_labeled_df has the correct DatetimeIndex ---
        # This step forces the index to be correct before propagation and saving.
        if not isinstance(raw_labeled_df.index, pd.DatetimeIndex) or not raw_labeled_df.index.equals(df_cleaned.index):
            self.logger.warning(f"Raw labeled DataFrame index is incorrect or does not match cleaned input. Attempting to reindex and align.")
            # Create a new DataFrame with the correct index from df_cleaned
            # and align the 'label' column. Fill NaNs (for dropped rows) with 0 (neutral).
            temp_labels = pd.DataFrame(
                {'label': raw_labeled_df['label'] if 'label' in raw_labeled_df.columns else 0},
                index=df_cleaned.index
            )
            raw_labeled_df = temp_labels.reindex(df_cleaned.index, fill_value=0)
            # Ensure 'label' column is integer type after reindexing
            raw_labeled_df['label'] = pd.to_numeric(raw_labeled_df['label'], errors='coerce').fillna(0).astype(int)
            self.logger.info("Raw labeled DataFrame index successfully reindexed and aligned.")

        # Ensure the raw_labeled_df has a 'label' column (after potential reindexing)
        if 'label' not in raw_labeled_df.columns:
            raise ValueError("Labeling strategy did not return a 'label' column after index alignment attempt.")


        # Apply label propagation (smoothing)
        self.logger.info(f"Applying label propagation with min_holding_period: {self.min_holding_period}")
        final_labeled_df = self._apply_label_propagation(raw_labeled_df)

        self.logger.info("Label calculation and propagation complete.")
        return final_labeled_df


    def _apply_label_propagation(self, df_raw_labeled: pd.DataFrame) -> pd.DataFrame:
        """
        Applies label propagation smoothing based on min_holding_period.
        If a non-zero label (1 or -1) is found, it propagates that label forward for
        min_holding_period bars, unless a new, conflicting label appears.

        Args:
            df_raw_labeled (pd.DataFrame): DataFrame with raw 'label' column.
                                          Assumed to be indexed by the cleaned data index.

        Returns:
            pd.DataFrame: DataFrame with labels smoothed by propagation.
                          Will have the same index as df_raw_labeled.
        """
        self.logger.debug(f"Applying label propagation with min_hold={self.min_holding_period}...")

        if df_raw_labeled.empty or 'label' not in df_raw_labeled.columns:
            self.logger.warning("Raw labeled DataFrame is empty or missing 'label' column. Skipping propagation.")
            return df_raw_labeled # Return as is

        # Ensure label column is integer type for comparison
        df_raw_labeled['label'] = pd.to_numeric(df_raw_labeled['label'], errors='coerce').fillna(0).astype(int)

        n = len(df_raw_labeled)
        propagated_labels = df_raw_labeled['label'].copy()

        i = 0
        while i < n:
            current_label = propagated_labels.iloc[i] # This is the label at the current bar 'i'

            if current_label != 0:
                # Determine the end of the propagation window
                propagation_end_limit = min(i + self.min_holding_period, n)

                # Find the actual end of propagation, considering conflicting signals
                actual_propagation_end_iloc = propagation_end_limit
                for j in range(i + 1, propagation_end_limit):
                    original_label_at_j = df_raw_labeled['label'].iloc[j]
                    if original_label_at_j != 0 and original_label_at_j != current_label:
                        # Found a new, conflicting signal. Propagation stops BEFORE this bar 'j'.
                        actual_propagation_end_iloc = j
                        break
                
                # Propagate the current_label from i+1 up to actual_propagation_end_iloc - 1
                # (i.e., for the bars that should receive the propagated label)
                if actual_propagation_end_iloc > i + 1: # Ensure there are bars to propagate to
                    propagated_labels.iloc[i+1 : actual_propagation_end_iloc] = current_label
                
                # Move the outer loop index 'i' to the point where the next distinct signal starts.
                # If a conflicting signal was found at 'j', the next iteration should start at 'j'.
                # Otherwise, it should start after the full propagation period.
                i = actual_propagation_end_iloc # This will be the start of the next segment to process

            else: # If current_label is 0 (neutral), just move to the next bar
                i += 1
            
            # The outer loop's increment for 'i' is handled by the 'i = actual_propagation_end_iloc' or 'i += 1'
            # statement inside the loop. No need for an extra 'i += 1' at the end of the while loop.
            # Removing the `i += 1` at the end of the while loop to prevent double incrementing.
            # This is critical for correctness.
            pass # Removed the extra i += 1

        df_raw_labeled['label'] = propagated_labels
        return df_raw_labeled


    # Removed the analyze_labels method as LabelAnalyzer is now the dedicated analysis component.
    # def analyze_labels(self, df_labeled: pd.DataFrame, output_dir: Path, plot_filename_pattern: str, table_filename_pattern: str, symbol: str, interval: str):
    #     """
    #     Performs basic analysis on the generated labels (distribution).
    #     (This method is now redundant with LabelAnalyzer and can be removed or simplified if only for quick internal checks).
    #     """
    #     # ... (original analyze_labels content) ...
