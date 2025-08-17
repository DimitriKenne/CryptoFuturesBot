# utils/label_generator.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union
import math
import copy
import sys
import importlib

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Import LabelConfig and DEFAULT_LABEL_CONFIG from the new schema ---
from config.label import LabelConfig, DEFAULT_LABEL_CONFIG

# --- Import Base Strategy and its logger ---
from utils.labeling.strategies.base_strategy import BaseLabelingStrategy, logger

# Import FLOAT_EPSILON from the central constants file (config.params)
from config.params import FLOAT_EPSILON

# --- Dynamically Map Labeling Strategy Names to Strategy Classes ---
LABELING_STRATEGY_MAP: Dict[str, Type[BaseLabelingStrategy]] = {}
for i in range(1, 5):  # Adjust range if you have more strategies
    strategy_key = f'labeling_strategy_{i}'
    module_path = f'utils.labeling.strategies.strategy{i}'
    class_name = f'Strategy{i}'
    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        LABELING_STRATEGY_MAP[strategy_key] = strategy_class
        logger.debug(f"Dynamically loaded {class_name} for key '{strategy_key}'.")
    except ImportError as e:
        logger.warning(f"Could not dynamically load module '{module_path}': {e}. Skipping this strategy.")
    except AttributeError as e:
        logger.warning(f"Could not find class '{class_name}' in module '{module_path}': {e}. Skipping this strategy.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading strategy '{strategy_key}': {e}", exc_info=True)

class LabelGenerator:
    """
    Generates trading labels by orchestrating different labeling strategies.

    Selects a strategy based on configuration, calculates raw labels, and applies
    label propagation smoothing based on min_holding_period.
    Uses the centralized logging configured by the calling script.
    """

    def __init__(self, config: Optional[Union[LabelConfig, Dict[str, Any]]] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the LabelGenerator with configuration and selects the labeling strategy.

        Args:
            config (Optional[Union[LabelConfig, Dict[str, Any]]]): Configuration for labeling.
                If None, defaults to a deep copy of DEFAULT_LABEL_CONFIG.
                If a dictionary is passed, it will be converted to a LabelConfig object.
                If a LabelConfig instance is passed, it will be deep copied.
            logger (Optional[logging.Logger]): A logger instance for logging messages.
                If None, a default logger will be used.
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        # Convert the input config to a LabelConfig dataclass instance.
        if config is None:
            self._label_config: LabelConfig = copy.deepcopy(DEFAULT_LABEL_CONFIG)
        elif isinstance(config, dict):
            self._label_config: LabelConfig = LabelConfig(**copy.deepcopy(config))
        elif isinstance(config, LabelConfig):
            self._label_config: LabelConfig = copy.deepcopy(config)
        else:
            raise TypeError("Config must be a LabelConfig instance or a dictionary, not " + str(type(config)))

        self.labeling_strategy_type = self._label_config.labeling_strategy_type
        self.min_holding_period = self._label_config.min_holding_period

        if self.labeling_strategy_type not in LABELING_STRATEGY_MAP:
            raise ValueError(f"Unknown labeling strategy type: '{self.labeling_strategy_type}'. "
                             f"Available strategies are: {', '.join(LABELING_STRATEGY_MAP.keys())}")

        strategy_class = LABELING_STRATEGY_MAP[self.labeling_strategy_type]

        try:
            # Pass the specific strategy's configuration object (already a dataclass instance)
            strategy_specific_config_obj = getattr(self._label_config, self.labeling_strategy_type)
            self.labeling_strategy: BaseLabelingStrategy = strategy_class(
                config=strategy_specific_config_obj,
                logger=self.logger,
                trading_fee_rate=self._label_config.trading_fee_rate,
                slippage_tolerance_pct=self._label_config.slippage_tolerance_pct
            )
        except Exception as e:
            self.logger.error(f"Error initializing LabelGenerator for strategy '{self.labeling_strategy_type}': {e}", exc_info=True)
            raise

        self.logger.info(f"LabelGenerator initialized for '{self.labeling_strategy_type}' strategy.")
        self.logger.info(f"  Min Holding Period (Propagation): {self.min_holding_period} bars")

    @staticmethod
    def get_available_labeling_strategies() -> List[str]:
        """Returns a list of available labeling strategy names."""
        return list(LABELING_STRATEGY_MAP.keys())

    def calculate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates trading labels using the configured labeling strategy and applies label propagation.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and required features, indexed by time.

        Returns:
            pd.DataFrame: DataFrame with the 'label' column (1, -1, or 0) after propagation.
        Raises:
            ValueError: If input DataFrame is invalid or missing required columns.
        """
        self.logger.info(f"Starting label calculation using '{self.labeling_strategy_type}' strategy...")

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input DataFrame is empty or not a pandas DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'open', 'high', 'low', 'close' columns.")

        initial_rows = len(df)
        df_cleaned = df.dropna(subset=['open', 'high', 'low', 'close']).copy()
        if len(df_cleaned) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(df_cleaned)} rows due to NaNs in OHLCV data before label calculation.")
        if df_cleaned.empty:
            self.logger.error("DataFrame became empty after dropping NaNs in OHLCV data. Cannot generate labels.")
            return pd.DataFrame(index=df.index, data={'label': 0})

        raw_labeled_df = self.labeling_strategy.calculate_raw_labels(df_cleaned)

        if not isinstance(raw_labeled_df.index, pd.DatetimeIndex) or not raw_labeled_df.index.equals(df_cleaned.index):
            self.logger.warning(f"Raw labeled DataFrame index is incorrect or does not match cleaned input. Attempting to reindex and align.")
            temp_labels = pd.DataFrame(
                {'label': raw_labeled_df['label'] if 'label' in raw_labeled_df.columns else 0},
                index=df_cleaned.index
            )
            raw_labeled_df = temp_labels.reindex(df_cleaned.index, fill_value=0)
            raw_labeled_df['label'] = pd.to_numeric(raw_labeled_df['label'], errors='coerce').fillna(0).astype(int)
            self.logger.info("Raw labeled DataFrame index successfully reindexed and aligned.")

        if 'label' not in raw_labeled_df.columns:
            raise ValueError("Labeling strategy did not return a 'label' column after index alignment attempt.")

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

        Returns:
            pd.DataFrame: DataFrame with labels smoothed by propagation.
        """
        self.logger.debug(f"Applying label propagation with min_hold={self.min_holding_period}...")

        if df_raw_labeled.empty or 'label' not in df_raw_labeled.columns:
            self.logger.warning("Raw labeled DataFrame is empty or missing 'label' column. Skipping propagation.")
            return df_raw_labeled

        df_raw_labeled['label'] = pd.to_numeric(df_raw_labeled['label'], errors='coerce').fillna(0).astype(int)

        n = len(df_raw_labeled)
        propagated_labels = df_raw_labeled['label'].copy()

        i = 0
        while i < n:
            current_label = propagated_labels.iloc[i]
            if current_label != 0:
                propagation_end_limit = min(i + self.min_holding_period, n)
                actual_propagation_end_iloc = propagation_end_limit
                for j in range(i + 1, propagation_end_limit):
                    original_label_at_j = df_raw_labeled['label'].iloc[j]
                    if original_label_at_j != 0 and original_label_at_j != current_label:
                        actual_propagation_end_iloc = j
                        break
                if actual_propagation_end_iloc > i + 1:
                    propagated_labels.iloc[i+1 : actual_propagation_end_iloc] = current_label
                i = actual_propagation_end_iloc
            else:
                i += 1
        df_raw_labeled['label'] = propagated_labels
        return df_raw_labeled
