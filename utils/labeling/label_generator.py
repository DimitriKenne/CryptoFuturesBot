# utils/labeling/label_generator.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union
import importlib
import matplotlib.pyplot as plt

# --- Import our refactored components and base classes ---
from .strategies.base_strategy import BaseLabelingStrategy
from .analysis_calculator import AnalysisCalculator
from .analysis_plotter import AnalysisPlotter
from utils.data_management.data_manager import DataManager

# --- Type Hinting Imports ---
from matplotlib.figure import Figure

# Setup logger for this module
logger = logging.getLogger(__name__)

# Dynamically Map Labeling Strategy Names to Strategy Classes
LABELING_STRATEGY_MAP: Dict[str, Type[BaseLabelingStrategy]] = {}
for i in range(1, 5):
    strategy_key = f'labeling_strategy_{i}'
    module_path = f'utils.labeling.strategies.strategy{i}'
    class_name = f'Strategy{i}'
    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        LABELING_STRATEGY_MAP[strategy_key] = strategy_class
        logger.debug(f"Dynamically loaded {class_name} for key '{strategy_key}'.")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load strategy for '{strategy_key}': {e}. Skipping.")
    except Exception as e:
        logger.error(f"An unexpected error occurred loading strategy '{strategy_key}': {e}", exc_info=True)


class LabelGenerator:
    """
    Orchestrates label generation using an injected strategy object.
    """
    def __init__(
        self,
        labeling_strategy: BaseLabelingStrategy,
        min_holding_period: int,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.labeling_strategy = labeling_strategy
        self.min_holding_period = min_holding_period
        
        self.logger.info(f"LabelGenerator initialized with '{self.labeling_strategy.__class__.__name__}' strategy.")
        self.logger.info(f"  Min Holding Period (Propagation): {self.min_holding_period} bars")

    @staticmethod
    def get_available_labeling_strategies() -> List[str]:
        return list(LABELING_STRATEGY_MAP.keys())

    def calculate_labels(
        self,
        df: pd.DataFrame,
        dm: DataManager,
        plotter: AnalysisPlotter,
        calculator: AnalysisCalculator,
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        self.logger.info(f"Starting label calculation using '{self.labeling_strategy.__class__.__name__}'...")

        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame is empty or does not have a DatetimeIndex.")

        # 1. Calculate raw labels
        raw_labeled_df = self.labeling_strategy.calculate_raw_labels(df.copy())

        # 2. Perform and save strategy-specific analysis
        self.logger.info("Performing strategy-specific analysis...")
        try:
            artifacts = self.labeling_strategy.perform_strategy_specific_analysis(
                df_original_input=df.copy(), plotter=plotter, calculator=calculator
            )
            
            # Get the single analysis directory for this asset.
            analysis_dir = dm.get_labeling_analysis_dir(symbol=symbol, interval=interval)
            
            for analysis_type, artifact in artifacts:
                if isinstance(artifact, Figure):
                    plot_kwargs = {
                        'analysis_type': analysis_type,
                        'labeling_strategy': self.labeling_strategy.__class__.__name__
                    }
                    dm.save_analysis_plot(
                        fig=artifact,
                        run_dir=analysis_dir,
                        plot_pattern_key='labeling_plot',
                        **plot_kwargs
                    )
                    plt.close(artifact)
                else:
                    self.logger.warning(f"Unknown artifact type '{type(artifact)}' for analysis '{analysis_type}'. Skipping save.")
        except Exception as e:
            self.logger.error(f"Error during strategy-specific analysis: {e}", exc_info=True)

        # 3. Apply label propagation
        self.logger.info(f"Applying label propagation with min_holding_period: {self.min_holding_period}")
        final_labeled_df = self._apply_label_propagation(raw_labeled_df)

        # 4. Ensure last bars are neutral to prevent look-ahead bias
        if self.min_holding_period > 0 and len(final_labeled_df) >= self.min_holding_period:
            final_labeled_df.iloc[-self.min_holding_period:, final_labeled_df.columns.get_loc('label')] = 0

        self.logger.info("Label calculation and propagation complete.")
        return final_labeled_df

    def _apply_label_propagation(self, df_raw_labeled: pd.DataFrame) -> pd.DataFrame:
        if 'label' not in df_raw_labeled.columns:
            self.logger.error("Raw labeled DataFrame missing 'label' column. Cannot apply propagation.");
            return df_raw_labeled

        propagated_labels = df_raw_labeled['label'].copy().fillna(0).astype(int)
        n = len(propagated_labels)
        i = 0
        while i < n:
            current_label = propagated_labels.iloc[i]
            if current_label != 0:
                end_limit = min(i + self.min_holding_period, n)
                for j in range(i + 1, end_limit):
                    if propagated_labels.iloc[j] != 0 and propagated_labels.iloc[j] != current_label:
                        end_limit = j
                        break
                propagated_labels.iloc[i:end_limit] = current_label
                i = end_limit
            else:
                i += 1
        
        df_raw_labeled['label'] = propagated_labels
        return df_raw_labeled