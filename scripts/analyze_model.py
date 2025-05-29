#!/usr/bin/env python3
"""
analyse_model.py

Loads a trained model pipeline, features from processed data, and labels from
labeled data, merges them, evaluates the model's performance on a test set,
and generates analysis metrics and plots for a ternary classification model (-1, 0, 1).
Includes enhanced analysis based on label generator diagnostics and feature importance.
Logs are saved to a single file in the logs directory specified in paths.py,
with clear markers for each analysis run, using the centralized rotating logger.

**FIXED**: Updated DataManager calls to use the new parameterized methods (load_data, load_model_artifact).
**FIXED**: Removed manual path construction for data and model files, relying on DataManager.
**FIXED**: Corrected the check for infinite values by using np.isinf() on each numeric column Series.
**FIXED**: Removed redundant plt.legend() call for probability plot to allow seaborn to handle legend labels.
**MODIFIED**: Added support for analyzing LSTM models, including loading and probability prediction.
**MODIFIED**: Adjusted feature importance analysis to handle models without standard importance attributes (like LSTM).
**FIXED**: Aligned data and model loading paths with keys defined in paths.py.
**FIXED**: Corrected LSTM prediction logic for the test set to predict for each sequence.
**FIXED**: Imported Counter from collections to resolve NameError.
**FIXED**: Removed kde=True from sns.histplot to prevent ValueError with limited unique data points.
**FIXED**: Added a more robust check before plotting probability histogram to ensure both correct/incorrect predictions are present in the data subset being plotted.
**FIXED**: Added type and column validation before prediction to prevent ValueError in ColumnTransformer when input is not a DataFrame.
**FIXED**: Added missing imports for `datetime`, `joblib`, and `to_categorical`.
**FIXED**: Corrected setup_rotating_logging keyword argument name.
**ADDED**: More robust error handling and logging.
**FIXED**: Corrected ModelTrainer initialization call to pass config dictionary.
**FIXED**: Use trainer.feature_columns_original to select features for X_test.
**FIXED**: Corrected ModelTrainer.predict and predict_proba to pass original DataFrame to scikit-learn pipeline.
**FIXED**: Removed 'labels' argument from balanced_accuracy_score call for compatibility with older scikit-learn versions.
**FIXED**: Use DataManager.save_model_artifact to save evaluation results dictionary.
**ADDED**: Definitions for plotting functions (confusion matrix, feature importance, probability distributions, ROC AUC, Precision-Recall).
**FIXED**: Corrected sns.barplot call for feature importance to resolve FutureWarning (again, ensuring it's in this version).
**FIXED**: Refined DataManager loading calls within ModelTrainer to ensure correct artifact types and names are used.
**FIXED**: Ensured consistent use of `all_expected_labels` for metrics and plotting functions.
**FIXED**: Added checks for minimum unique classes before attempting ROC/PR plots.
**FIXED**: Added checks for probability columns existence before plotting histograms.
**FIXED**: Corrected variable name from 'data_manager' to 'dm' where the DataManager instance is used.
**ADDED**: Handling for NaN values in predictions before calculating evaluation metrics.
**FIXED**: Passed `symbol`, `interval`, and `model_key` explicitly to plotting functions to resolve `NameError`.
**FIXED**: Corrected the call to `setup_rotating_logging` to match its signature in `logger_config.py`.
**FIXED**: Passed `y_proba_df`, `y_test_evaluated`, and `all_expected_labels` from `evaluate_model` to `analyse_model_pipeline` and then to plotting functions.
**FIXED**: Passed `dm` and `train_ratio` to `evaluate_model`.
**FIXED**: Added `plot_calibration_curve` function definition.
**REVERTED**: Removed manual reshaping for LSTM in `evaluate_model` and instead pass original `X_test` to `ModelTrainer`'s `predict` and `predict_proba` methods, as `ModelTrainer` handles internal data preparation.
**FIXED**: Modified `load_trained_model_and_preprocessor` to return the `ModelTrainer` instance directly, allowing `evaluate_model` to call its methods.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter
import time
import copy
from datetime import datetime # Import datetime

import pandas as pd
import numpy as np

# Import scikit-learn and imblearn components for analysis/evaluation
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, accuracy_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score)
from sklearn.utils import column_or_1d
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve # Import calibration_curve

# Conditional import for TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model # type: ignore
    from tensorflow.keras.utils import to_categorical # type: ignore
    _TF_AVAILABLE = True
    _TF_GPU_AVAILABLE = tf.config.list_physical_devices('GPU')
    if _TF_GPU_AVAILABLE:
        print(f"TensorFlow GPU detected: {_TF_GPU_AVAILABLE}")
        logging.info(f"TensorFlow GPU detected: {_TF_GPU_AVAILABLE}")
    else:
        print("TensorFlow GPU not detected.")
        logging.info("TensorFlow GPU not detected.")
except ImportError:
    _TF_AVAILABLE = False
    _TF_GPU_AVAILABLE = False
    print("TensorFlow not installed. LSTM model analysis will be skipped.")
    logging.warning("TensorFlow not installed. LSTM model analysis will be skipped.")


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    from config.params import MODEL_CONFIG, GENERAL_CONFIG, LABELING_CONFIG, FEATURE_CONFIG
    from utils.data_manager import DataManager
    from utils.logger_config import setup_rotating_logging
    from utils.model_trainer import ModelTrainer
    from utils.exceptions import TemporalSafetyError, ModelAnalysisError # Import ModelAnalysisError
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules: {e}")
    sys.exit(1)

# Set up logging for this script
log_filepath = PATHS['logs_dir'] / f"{Path(__file__).stem}.log"
# Corrected call to setup_rotating_logging
logger = setup_rotating_logging(
    Path(__file__).stem, # Pass the base filename as the first positional argument
    log_level=logging.INFO # Use the correct keyword argument for level
)

# Custom exception for analysis errors (now imported from exceptions.py)
# class ModelAnalysisError(Exception):
#     """Custom exception for errors during model analysis."""
#     pass

def log_analysis_start_end(func):
    """Decorator to log the start and end of analysis functions."""
    def wrapper(*args, **kwargs):
        logger.info(f"--- Starting {func.__name__.replace('_', ' ').title()} ---")
        try:
            result = func(*args, **kwargs)
            logger.info(f"--- Finished {func.__name__.replace('_', ' ').title()} ---")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__.replace('_', ' ').title()}: {e}", exc_info=True)
            raise ModelAnalysisError(f"Failed during {func.__name__.replace('_', ' ').title()}") from e
    return wrapper

# --- Plotting Libraries (Conditional on PLOT_AVAILABLE) ---
# Import matplotlib and seaborn unconditionally, then use PLOT_AVAILABLE to guard calls.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.switch_backend('Agg') # Use Agg backend for matplotlib to prevent display issues in environments without GUI
    PLOT_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib or Seaborn not found. Plotting will be skipped. Install using 'pip install matplotlib seaborn'.")
    PLOT_AVAILABLE = False
    # No need to set plt = None or sns = None here, as PLOT_AVAILABLE will guard their use.


# Helper to clean data (copied from train_model for consistency)
def clean_data(df: pd.DataFrame, cols_to_check: List[str]) -> pd.DataFrame:
    """
    Removes rows with NaN or infinite values in the specified columns.
    Ensures the input DataFrame is not modified in place.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols_to_check (List[str]): List of columns to check for NaNs/Infs.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing NaNs/Infs removed
                      from the specified columns.
    """
    initial_rows = len(df)
    df_cleaned = df.copy() # Work on a copy

    # Ensure cols_to_check are actually in df_cleaned after the copy
    cols_to_check_present = [col for col in cols_to_check if col in df_cleaned.columns]
    if not cols_to_check_present:
         logger.warning("None of the specified columns to check for NaNs are present in the DataFrame.")
         return df_cleaned # Return copy if no columns to check

    df_cleaned.dropna(subset=cols_to_check_present, inplace=True)


    # Check for Infinite values in numeric specified columns
    numeric_cols_to_check = [col for col in cols_to_check_present if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col])]

    for col in numeric_cols_to_check:
         if np.isinf(df_cleaned[col]).any():
              logger.warning(f"Infinite values found in column '{col}'. Removing rows with Inf.")
              df_cleaned = df_cleaned[~np.isinf(df_cleaned[col])]


    removed_rows = initial_rows - len(df_cleaned)
    if removed_rows > 0:
         logger.info(f"Removed {removed_rows} rows with NA or Inf values in specified columns.")

    return df_cleaned


# --- Plotting Functions (Always defined, but check PLOT_AVAILABLE internally) ---

def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: Path, symbol: str, interval: str, model_key: str, title: str = 'Confusion Matrix'):
    """
    Plots the confusion matrix using seaborn.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping confusion matrix plot: Plotting libraries not available.")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # Use the passed symbol, interval, model_key for the title
    plt.title(f'Confusion Matrix for {model_key.replace("_", " ").title()} ({symbol} {interval})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved confusion matrix plot to {save_path}")
    except Exception as e:
        logger.error(f"Error saving confusion matrix plot to {save_path}: {e}", exc_info=True)
    finally:
        plt.close() # Close the figure to free memory


def plot_feature_importance(importances: Dict[str, float], save_path: Path, symbol: str, interval: str, model_key: str, title: str = 'Feature Importance', top_n: int = 20):
    """
    Plots the top N feature importances.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping feature importance plot: Plotting libraries not available.")
        return
    if not importances:
         logger.warning("No feature importances data to plot.")
         return

    importance_series = pd.Series(importances).sort_values(ascending=False)
    top_importances = importance_series.head(top_n)

    plt.figure(figsize=(10, max(6, len(top_importances) * 0.4)))
    sns.barplot(x=top_importances.values, y=top_importances.index,
                hue=top_importances.index, # Explicitly map hue to 'Feature' for distinct colors
                palette='viridis',
                legend=False) # Remove legend if hue is used for color mapping

    # Use the passed symbol, interval, model_key for the title
    plt.title(f'Feature Importance (Top {top_n})\n{symbol.upper()} {interval} ({model_key})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved feature importance plot to {save_path}")
    except Exception as e:
        logger.error(f"Error saving feature importance plot to {save_path}: {e}", exc_info=True)
    finally:
        plt.close()


def plot_probability_histograms(y_true: pd.Series, y_proba: np.ndarray, classes: list, save_path: Path, symbol: str, interval: str, model_key: str, title: str = 'Prediction Probability Distributions'):
    """
    Plots histograms of predicted probabilities for each class, separated by true label.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping probability histograms plot: Plotting libraries not available.")
        return
    if y_true.empty or y_proba.shape[0] == 0 or y_proba.shape[1] != len(classes):
         logger.warning("Insufficient data or incorrect shape for probability histogram plotting.")
         return

    y_true_df = y_true.to_frame(name='true_label')
    proba_df = pd.DataFrame(y_proba, index=y_true_df.index, columns=classes)
    merged_df = pd.concat([y_true_df, proba_df], axis=1)

    if merged_df.empty:
        logger.warning("Merged data for probability histogram is empty.")
        return

    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5), sharey=True)
    if len(classes) == 1:
         axes = [axes]

    for i, class_label in enumerate(classes):
        proba_column = class_label
        if proba_column not in merged_df.columns:
             logger.warning(f"Probability column for class {class_label} not found in DataFrame. Skipping plot for this class.")
             continue

        subset_data = merged_df[proba_column].dropna()
        if subset_data.empty:
             logger.warning(f"Skipping probability histogram for class {class_label}: No non-NaN probability data.")
             axes[i].set_title(f'Probabilities for Class {class_label}\n(No Data)')
             axes[i].set_xlabel(f'Predicted Probability ({class_label})')
             axes[i].set_ylabel('Density')
             continue

        sns.histplot(data=merged_df, x=proba_column, hue='true_label', ax=axes[i], stat='density', common_norm=False, bins=30, palette='viridis')
        axes[i].set_title(f'Probabilities for Class {class_label}')
        axes[i].set_xlabel(f'Predicted Probability ({class_label})')
        axes[i].set_ylabel('Density')
        if axes[i].get_legend() is None:
             axes[i].legend(title='True Label')
        else:
             axes[i].get_legend().set_title('True Label')

    # Use the passed symbol, interval, model_key for the main title
    fig.suptitle(f'Prediction Probability Distribution\n{symbol.upper()} {interval} ({model_key})', y=1.02)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved probability distribution plots to {save_path}")
    except Exception as e:
        logger.error(f"Error saving probability distribution plots to {save_path}: {e}", exc_info=True)
    finally:
        plt.close(fig)


def plot_roc_auc(y_true: pd.Series, y_proba: np.ndarray, classes: list, output_dir: Path, symbol: str, interval: str, model_key: str):
    """
    Plots ROC curves and calculates AUC for each class in a multi-class setting.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping ROC AUC plot: Plotting libraries not available.")
        return
    if y_true.empty or y_proba.shape[0] == 0 or y_proba.shape[1] != len(classes):
         logger.warning("Insufficient data or incorrect shape for ROC AUC plotting.")
         return
    unique_true_classes = np.unique(y_true.dropna())
    if len(unique_true_classes) < 2:
         logger.warning(f"Skipping ROC AUC plot: Need at least two unique classes in true labels. Found: {unique_true_classes}")
         return

    sorted_classes = sorted(classes)
    y_true_clean = y_true.dropna()
    y_proba_clean = pd.DataFrame(y_proba, index=y_true.index, columns=sorted_classes).loc[y_true_clean.index].values

    if y_true_clean.empty or y_proba_clean.shape[0] == 0:
         logger.warning("Skipping ROC AUC plot: No non-NaN data points after cleaning true labels.")
         return

    y_true_bin = label_binarize(y_true_clean, classes=sorted_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, class_label in enumerate(sorted_classes):
        if y_true_bin.shape[1] > i and np.any(y_true_bin[:, i] == 1):
             if y_proba_clean.shape[1] > i:
                  fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba_clean[:, i])
                  roc_auc[i] = auc(fpr[i], tpr[i])
             else:
                  logger.warning(f"Skipping ROC AUC calculation for class {class_label}: Probability array does not have enough columns.")
                  fpr[i], tpr[i], roc_auc[i] = None, None, None
        else:
             logger.warning(f"Skipping ROC AUC calculation for class {class_label}: No positive samples in true labels for this class.")
             fpr[i], tpr[i], roc_auc[i] = None, None, None

    plt.figure(figsize=(8, 6))

    for i, class_label in enumerate(sorted_classes):
        if roc_auc.get(i) is not None:
             plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_label} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve\n{symbol.upper()} {interval} ({model_key})')
    plt.legend(loc="lower right")
    plt.tight_layout()

    safe_interval = interval.replace(':', '_')
    roc_auc_plot_path = output_dir / f"{symbol.upper()}_{safe_interval}_{model_key}_roc_auc_curve.png".replace(':', '_')
    try:
        plt.savefig(roc_auc_plot_path, dpi=150)
        logger.info(f"Saved ROC AUC plot to {roc_auc_plot_path}")
    except Exception as e:
        logger.error(f"Error saving ROC AUC plot to {roc_auc_plot_path}: {e}", exc_info=True)
    finally:
        plt.close()


def plot_precision_recall_curve(y_true: pd.Series, y_proba: np.ndarray, classes: list, output_dir: Path, symbol: str, interval: str, model_key: str):
    """
    Plots Precision-Recall curves for each class in a multi-class setting.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping Precision-Recall plot: Plotting libraries not available.")
        return
    if y_true.empty or y_proba.shape[0] == 0 or y_proba.shape[1] != len(classes):
         logger.warning("Insufficient data or incorrect shape for Precision-Recall plotting.")
         return
    unique_true_classes = np.unique(y_true.dropna())
    if len(unique_true_classes) < 2:
         logger.warning(f"Skipping Precision-Recall plot: Need at least two unique classes in true labels. Found: {unique_true_classes}")
         return

    sorted_classes = sorted(classes)
    y_true_clean = y_true.dropna()
    y_proba_clean = pd.DataFrame(y_proba, index=y_true.index, columns=sorted_classes).loc[y_true_clean.index].values

    if y_true_clean.empty or y_proba_clean.shape[0] == 0:
         logger.warning("Skipping Precision-Recall plot: No non-NaN data points after cleaning true labels.")
         return

    y_true_bin = label_binarize(y_true_clean, classes=sorted_classes)


    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, class_label in enumerate(sorted_classes):
         if y_true_bin.shape[1] > i and np.any(y_true_bin[:, i] == 1):
              if y_proba_clean.shape[1] > i:
                   precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba_clean[:, i])
                   average_precision[i] = average_precision_score(y_true_bin[:, i], y_proba_clean[:, i])
              else:
                   logger.warning(f"Skipping Precision-Recall calculation for class {class_label}: Probability array does not have enough columns.")
                   precision[i], recall[i], average_precision[i] = None, None, None
         else:
              logger.warning(f"Skipping Precision-Recall calculation for class {class_label}: No positive samples in true labels for this class.")
              precision[i], recall[i], average_precision[i] = None, None, None


    plt.figure(figsize=(8, 6))

    for i, class_label in enumerate(sorted_classes):
        if average_precision.get(i) is not None:
             plt.plot(recall[i], precision[i], label=f'Precision-Recall curve of class {sorted_classes[i]} (area = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n{symbol.upper()} {interval} ({model_key})')
    plt.legend(loc="lower left")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()

    safe_interval = interval.replace(':', '_')
    pr_curve_plot_path = output_dir / f"{symbol.upper()}_{safe_interval}_{model_key}_precision_recall_curve.png".replace(':', '_')
    try:
        plt.savefig(pr_curve_plot_path, dpi=150)
        logger.info(f"Saved Precision-Recall plot to {pr_curve_plot_path}")
    except Exception as e:
        logger.error(f"Error saving Precision-Recall plot to {pr_curve_plot_path}: {e}", exc_info=True)
    finally:
        plt.close()


def plot_calibration_curve(y_true: pd.Series, y_proba: np.ndarray, classes: list, output_dir: Path, symbol: str, interval: str, model_key: str):
    """
    Plots calibration curves (reliability diagrams) for each class.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping calibration plot: Plotting libraries not available.")
        return
    if y_true.empty or y_proba.shape[0] == 0 or y_proba.shape[1] != len(classes):
         logger.warning("Insufficient data or incorrect shape for calibration plotting.")
         return
    unique_true_classes = np.unique(y_true.dropna())
    if len(unique_true_classes) < 2:
         logger.warning(f"Skipping calibration plot: Need at least two unique classes in true labels. Found: {unique_true_classes}")
         return

    sorted_classes = sorted(classes)
    y_true_clean = y_true.dropna()
    y_proba_clean = pd.DataFrame(y_proba, index=y_true.index, columns=sorted_classes).loc[y_true_clean.index].values

    if y_true_clean.empty or y_proba_clean.shape[0] == 0:
         logger.warning("Skipping calibration plot: No non-NaN data points after cleaning true labels.")
         return

    y_true_bin = label_binarize(y_true_clean, classes=sorted_classes)

    plt.figure(figsize=(8, 8))
    for i, class_label in enumerate(sorted_classes):
        if y_true_bin.shape[1] > i and np.any(y_true_bin[:, i] == 1): # Ensure there are positive samples for this class
            if y_proba_clean.shape[1] > i:
                prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_proba_clean[:, i], n_bins=10)
                plt.plot(prob_pred, prob_true, "s-", label=f"Class {class_label}")
            else:
                logger.warning(f"Skipping calibration curve for class {class_label}: Probability array does not have enough columns.")
        else:
            logger.warning(f"Skipping calibration curve for class {class_label}: No positive samples in true labels for this class.")


    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f'Calibration Plot for {model_key.replace("_", " ").title()} ({symbol} {interval})')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    safe_interval = interval.replace(':', '_')
    calibration_plot_path = output_dir / f"{symbol.upper()}_{safe_interval}_{model_key}_calibration_plot.png".replace(':', '_')
    try:
        plt.savefig(calibration_plot_path, dpi=150)
        logger.info(f"Saved calibration plot to {calibration_plot_path}")
    except Exception as e:
        logger.error(f"Error saving calibration plot to {calibration_plot_path}: {e}", exc_info=True)
    finally:
        plt.close()


@log_analysis_start_end
def load_and_prepare_data(symbol: str, interval: str, model_key: str, train_ratio: float, data_manager: DataManager) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Loads processed and labeled data, merges them, and splits into train/test sets.
    Handles feature selection and NaN removal.
    """
    logger.info(f"Loading and preparing data for {symbol} {interval} with train_ratio={train_ratio}...")

    try:
        # Load processed data (features)
        processed_data = data_manager.load_data(
            symbol=symbol,
            interval=interval,
            data_type='processed'
        )
        logger.info(f"Loaded processed data. Shape: {processed_data.shape}")

        # Load labeled data (labels)
        labeled_data = data_manager.load_data(
            symbol=symbol,
            interval=interval,
            data_type='labeled'
        )
        logger.info(f"Loaded labeled data. Shape: {labeled_data.shape}")

        # Ensure indexes are aligned
        combined_df = pd.merge(processed_data, labeled_data, left_index=True, right_index=True, how='inner')
        logger.info(f"Merged processed and labeled data. Combined shape: {combined_df.shape}")

        # Get the list of features to use based on the model_key
        model_config = MODEL_CONFIG.get(model_key, {})
        # Prioritize features_to_use explicitly defined in MODEL_CONFIG
        selected_features = model_config.get('features_to_use', [])

        if not selected_features:
            logger.warning(f"No specific 'features_to_use' defined for model_key '{model_key}' in MODEL_CONFIG. Inferring features from processed data.")
            # If no specific features are defined, assume all columns in processed_data are features
            # EXCEPT for known non-feature columns (OHLCV, volume, open_time, label, etc.)
            # This list should ideally be consistent with what's dropped in train_model.py's load_and_split_data
            # Let's define a comprehensive list of columns that are *never* features
            non_feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'open_time', # Explicitly exclude open_time
                'label', # The target label itself
                'vol_adj' # If this is a temporary column from labeling
            ]
            
            # Filter out columns that are in the processed_data but are known non-features
            selected_features = [col for col in processed_data.columns if col not in non_feature_columns]

            if not selected_features:
                logger.critical("No valid features found in processed data after excluding known non-feature columns. Please check feature generation and configuration.")
                raise ModelAnalysisError("No features found for analysis.")
            else:
                logger.info(f"Inferred {len(selected_features)} features from processed data: {selected_features[:5]}... (and more)")
        else:
            # Ensure specified features actually exist in the DataFrame
            missing_features = [f for f in selected_features if f not in combined_df.columns]
            if missing_features:
                logger.critical(f"Specified features for model '{model_key}' are missing from the data: {missing_features}")
                raise ModelAnalysisError(f"Missing features in data: {missing_features}")
            logger.info(f"Using specified {len(selected_features)} features for model '{model_key}'.")


        # Drop rows with any NaN values in selected features or the label column
        initial_rows = combined_df.shape[0]
        # Make a copy to avoid SettingWithCopyWarning if combined_df is a slice
        df_cleaned = combined_df[selected_features + ['label']].dropna().copy()
        rows_after_nan_drop = df_cleaned.shape[0]
        if initial_rows - rows_after_nan_drop > 0:
            logger.warning(f"Dropped {initial_rows - rows_after_nan_drop} rows due to NaN values in features or labels. Remaining rows: {rows_after_nan_drop}")

        if df_cleaned.empty:
            logger.critical("DataFrame is empty after dropping NaN values. Cannot proceed with analysis.")
            raise ModelAnalysisError("Empty DataFrame after NaN removal.")

        # Check for infinite values in features (should not be an issue with correct feature engineering)
        for col in selected_features:
            if np.isinf(df_cleaned[col]).any():
                logger.critical(f"Infinite values detected in feature column: {col}. Please check feature engineering logic.")
                # Replace inf with NaN and drop, or handle appropriately
                df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_cleaned.dropna(subset=[col], inplace=True)
                logger.warning(f"Removed rows with infinite values in {col}. Current shape: {df_cleaned.shape}")

        if df_cleaned.empty:
            logger.critical("DataFrame is empty after handling infinite values. Cannot proceed with analysis.")
            raise ModelAnalysisError("Empty DataFrame after infinite value handling.")

        # Split into features (X) and labels (y)
        X = df_cleaned[selected_features]
        y = df_cleaned['label']

        # Convert label to integer type, if it's not already
        if not pd.api.types.is_integer_dtype(y):
            y = y.astype(int)
            logger.info("Label column cast to integer type.")

        # Filter out labels that are not -1, 0, or 1 if they somehow exist
        valid_labels = [-1, 0, 1]
        original_y_count = len(y)
        y = y[y.isin(valid_labels)]
        X = X.loc[y.index] # Keep only features corresponding to valid labels
        if len(y) < original_y_count:
            logger.warning(f"Removed {original_y_count - len(y)} rows with invalid labels (not -1, 0, or 1). Remaining rows: {len(y)}")

        if X.empty or y.empty:
            logger.critical("DataFrame is empty after filtering invalid labels. Cannot proceed with analysis.")
            raise ModelAnalysisError("Empty DataFrame after label filtering.")

        # Train-test split using time-series split
        n_samples = len(X)
        if n_samples < 2:
            logger.critical(f"Not enough samples ({n_samples}) to perform train-test split. Need at least 2.")
            raise ModelAnalysisError("Insufficient data for train-test split.")

        split_index = int(n_samples * train_ratio)
        if split_index == 0: # Ensure at least one training sample
            split_index = 1
        if split_index == n_samples: # Ensure at least one test sample
            split_index = n_samples - 1
            if split_index == 0: # Handle case where n_samples is 1
                 logger.critical(f"Not enough samples ({n_samples}) to create a test set. Adjust train_ratio or provide more data.")
                 raise ModelAnalysisError("Insufficient data for train-test split.")


        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        if X_test.empty or y_test.empty:
            logger.critical(f"Test set is empty after split with train_ratio={train_ratio}. Consider adjusting train_ratio.")
            raise ModelAnalysisError("Empty test set after split.")

        logger.info(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Check label distribution in training set for potential imbalance issues
        train_label_counts = Counter(y_train)
        logger.info(f"Training label distribution: {train_label_counts}")
        if any(count == 0 for count in train_label_counts.values()):
            logger.warning("One or more classes have zero samples in the training set. This might cause issues for model training.")


        # Check label distribution in test set
        test_label_counts = Counter(y_test)
        logger.info(f"Test label distribution: {test_label_counts}")
        if any(count == 0 for count in test_label_counts.values()):
             logger.warning("One or more classes have zero samples in the test set. Evaluation metrics might be misleading.")


        # Pass the original feature columns for consistent preprocessing/prediction
        feature_columns_original = selected_features

        return X_train, X_test, y_train, y_test, feature_columns_original

    except Exception as e:
        logger.critical(f"Error during data loading and preparation: {e}", exc_info=True)
        raise ModelAnalysisError("Failed to load and prepare data.") from e

@log_analysis_start_end
def load_trained_model_and_preprocessor(symbol: str, interval: str, model_key: str, data_manager: DataManager) -> ModelTrainer:
    """
    Loads the trained ModelTrainer instance.
    Returns: The loaded ModelTrainer instance.
    """
    logger.info(f"Loading trained ModelTrainer instance for {model_key} ({symbol} {interval})...")
    try:
        model_config_for_trainer = MODEL_CONFIG.get(model_key, {})
        if not model_config_for_trainer:
            logger.error(f"Model configuration for '{model_key}' not found in MODEL_CONFIG.")
            raise ModelAnalysisError(f"Model configuration for '{model_key}' not found.")

        # Initialize ModelTrainer with the model-specific configuration
        trainer = ModelTrainer(config=model_config_for_trainer)
        # Call the load method with symbol, interval, and model_key
        trainer.load(symbol=symbol, interval=interval, model_key=model_key)

        logger.info(f"ModelTrainer instance loaded successfully for {model_key}.")
        return trainer

    except FileNotFoundError as e:
        logger.critical(f"Model or preprocessor not found for {model_key} ({symbol} {interval}). Please ensure the model has been trained and saved: {e}")
        raise ModelAnalysisError(f"Model or preprocessor not found.") from e
    except Exception as e:
        logger.critical(f"Error loading trained model or preprocessor: {e}", exc_info=True)
        raise ModelAnalysisError("Failed to load trained model or preprocessor.") from e


@log_analysis_start_end
def evaluate_model(trainer: ModelTrainer, X_test: pd.DataFrame, y_test: pd.Series, model_key: str,
                   output_dir: Path, analysis_table_pattern: str,
                   symbol: str, interval: str, dm: DataManager, train_ratio: float) -> Tuple[np.ndarray, pd.Series, pd.DataFrame, List[int]]: # Added return types
    """
    Evaluates the model's performance on the test set and saves metrics.
    Handles different model types (sklearn, LSTM) by leveraging ModelTrainer's methods.
    Returns: Tuple of (y_pred_evaluated, y_test_evaluated, y_proba_df, all_expected_labels)
    """
    logger.info(f"Evaluating model '{model_key}' on the test set...")

    if X_test.empty or y_test.empty:
        logger.warning("Test set is empty. Skipping model evaluation.")
        # Return empty/None values if no evaluation is performed
        return np.array([]), pd.Series(), pd.DataFrame(), []

    # Make predictions using the trainer's predict method
    y_pred_series = None
    try:
        logger.info(f"Making predictions with {model_key} model...")
        y_pred_series = trainer.predict(X_test)
        if not isinstance(y_pred_series, pd.Series):
             # If trainer.predict returns numpy array, convert to Series with correct index
             y_pred_series = pd.Series(y_pred_series, index=X_test.index, name='prediction')
        logger.info(f"{model_key} predictions made.")
    except Exception as e:
        logger.critical(f"Error during prediction with {model_key} model: {e}", exc_info=True)
        raise ModelAnalysisError(f"Prediction failed for {model_key}.") from e

    if y_pred_series is None or y_pred_series.empty:
        logger.critical("No predictions were generated. Cannot evaluate model.")
        raise ModelAnalysisError("No predictions to evaluate.")

    # Get predicted probabilities for plotting
    y_proba_df = pd.DataFrame() # Initialize as empty
    try:
        if hasattr(trainer, 'predict_proba') and callable(trainer.predict_proba):
            logger.info(f"Getting probability predictions with {model_key} model...")
            y_proba_df_raw = trainer.predict_proba(X_test)
            if y_proba_df_raw is not None and not y_proba_df_raw.empty:
                # Ensure y_proba_df has the same index as X_test
                y_proba_df = y_proba_df_raw.reindex(X_test.index)
                logger.info(f"{model_key} probability predictions made.")
            else:
                logger.warning(f"trainer.predict_proba for {model_key} returned empty or None.")
        else:
            logger.warning(f"Model type '{model_key}' or its trainer instance does not support predict_proba.")
    except Exception as e:
        logger.error(f"Error getting predicted probabilities for {model_key}: {e}", exc_info=True)
        logger.warning("Predicted probabilities not available for plotting due to error.")


    # --- Handle NaNs in Predictions Before Evaluation ---
    nan_predictions_mask = y_pred_series.isna()
    if nan_predictions_mask.any():
        num_nan_predictions = nan_predictions_mask.sum()
        logger.warning(f"Found {num_nan_predictions} NaN values in predictions.")
        logger.warning("Removing corresponding samples from test set for evaluation.")

        y_test_evaluated = y_test[~nan_predictions_mask]
        y_pred_evaluated = y_pred_series[~nan_predictions_mask]

        if not y_proba_df.empty:
            y_proba_df = y_proba_df[~nan_predictions_mask] # Filter probabilities as well

        logger.info(f"Evaluation will be performed on {len(y_test_evaluated)} samples after removing NaNs.")
    else:
        y_test_evaluated = y_test
        y_pred_evaluated = y_pred_series
        logger.info("No NaN values found in predictions. Evaluating on the full test set.")

    # Ensure y_pred_evaluated is a numpy array for sklearn metrics
    y_pred_evaluated_np = y_pred_evaluated.to_numpy()

    # Determine all expected labels (e.g., [-1, 0, 1])
    # Use trainer.classes if available, otherwise infer from data
    all_expected_labels = trainer.classes if hasattr(trainer, 'classes') and trainer.classes is not None else sorted(list(set(y_test_evaluated.unique()).union(set(y_pred_evaluated.unique()))))
    # FIX: Use len() to check if the list/array is empty, which is robust for both lists and numpy arrays.
    if len(all_expected_labels) == 0: # Fallback if no unique labels are found after cleaning
        all_expected_labels = [-1, 0, 1]
    
    # Ensure probabilities DataFrame has columns named after the classes
    if not y_proba_df.empty and y_proba_df.shape[1] == len(all_expected_labels):
        y_proba_df.columns = all_expected_labels
    elif not y_proba_df.empty:
        logger.warning(f"Probability DataFrame has {y_proba_df.shape[1]} columns, but {len(all_expected_labels)} expected labels. Cannot assign column names reliably.")


    # --- 5. Evaluate Model ---
    logger.info("Evaluating model performance on the test set...")

    y_test_1d = column_or_1d(y_test_evaluated)
    y_pred_1d = column_or_1d(y_pred_evaluated_np) # Use the numpy array version

    try:
        overall_accuracy = accuracy_score(y_test_1d, y_pred_1d)
        bal_acc = balanced_accuracy_score(y_test_1d, y_pred_1d)
        class_report = classification_report(y_test_1d, y_pred_1d, labels=all_expected_labels, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test_1d, y_pred_1d, labels=all_expected_labels)

        logger.info(f"Evaluation complete.")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise ModelAnalysisError(f"Evaluation failed: {e}") from e


    # --- 6. Log and Store Evaluation Results ---
    logger.info("--- Evaluation Metrics (Test Set) ---")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test_1d, y_pred_1d, labels=all_expected_labels, zero_division=0))
    logger.info("Confusion Matrix:\n" + str(conf_matrix))


    evaluation_results = {
        'overall_accuracy': overall_accuracy,
        'balanced_accuracy': bal_acc,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'evaluated_set_shape': y_test_evaluated.shape,
        'evaluated_set_index_range': (str(y_test_evaluated.index.min()) if not y_test_evaluated.empty else 'N/A',
                                      str(y_test_evaluated.index.max()) if not y_test_evaluated.empty else 'N/A'),
        'evaluated_set_label_distribution': y_test_evaluated.value_counts(normalize=True).sort_index().to_dict() if not y_test_evaluated.empty else {},
        'num_samples_removed_for_evaluation': num_nan_predictions if nan_predictions_mask.any() else 0,
        'model_key': model_key,
        'symbol': symbol,
        'interval': interval,
        'train_ratio': train_ratio,
        'timestamp': datetime.now().isoformat()
    }

    try:
        dm.save_model_artifact(
            artifact=evaluation_results,
            symbol=symbol,
            interval=interval,
            model_key=model_key,
            artifact_type='evaluation'
        )
        logger.info(f"Evaluation results saved successfully using DataManager.")
    except Exception as e:
        logger.error(f"Error saving evaluation results using DataManager: {e}", exc_info=True)
        logger.warning("Saving evaluation results failed. Analysis plots will still be attempted.")

    # Return the necessary values for plotting
    return y_pred_evaluated_np, y_test_evaluated, y_proba_df, all_expected_labels


@log_analysis_start_end
def analyze_feature_importance(trainer: ModelTrainer, feature_columns_original: List[str], model_key: str, output_dir: Path, analysis_table_pattern: str, analysis_plot_pattern: str, symbol: str, interval: str):
    """
    Analyzes and visualizes feature importance for tree-based models.
    Skips for LSTM models.
    """
    logger.info(f"Analyzing feature importance for {model_key}...")

    # Feature importance is typically for tree-based models. Skip for LSTM.
    if model_key == 'lstm':
        logger.info("Skipping feature importance analysis for LSTM model as it's not directly applicable in the same way as tree-based models.")
        return

    importance_data = None
    final_model = None

    if hasattr(trainer, 'pipeline') and trainer.pipeline is not None and len(trainer.pipeline.steps) > 0:
        final_model = trainer.pipeline.steps[-1][1]
    elif hasattr(trainer, 'model') and trainer.model is not None:
        final_model = trainer.model

    if final_model is not None:
        if hasattr(final_model, 'feature_importances_'):
            importance_data = final_model.feature_importances_
            feature_names = feature_columns_original # If preprocessor is just scaling, original names are fine
        elif hasattr(final_model, 'coef_'):
            # For linear models, coefficients can indicate importance
            # For multi-class, coef_ is (n_classes, n_features)
            # Take the sum of absolute coefficients for a simple overall importance
            importance_data = np.sum(np.abs(final_model.coef_), axis=0)
            feature_names = feature_columns_original
        else:
            logger.warning(f"Model type '{model_key}' does not have direct 'feature_importances_' or 'coef_' attribute. Skipping feature importance analysis.")
            return
    else:
        logger.warning(f"No final model found in trainer for '{model_key}'. Skipping feature importance analysis.")
        return


    if importance_data is None or len(importance_data) == 0:
        logger.warning("No feature importance data available.")
        return

    # Create a DataFrame for importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_data
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Save feature importance to a CSV
    safe_interval = interval.replace(':', '_')
    importance_filepath = output_dir / analysis_table_pattern.format(
        symbol=symbol.upper(), # Use actual symbol
        interval=safe_interval, # Use actual interval
        model_type=model_key, # FIX: Changed from model_key to model_type
        analysis_type=f"{model_key}_feature_importance"
    )

    try:
        importance_filepath.parent.mkdir(parents=True, exist_ok=True)
        feature_importance_df.to_csv(importance_filepath.with_suffix('.csv'), index=False)
        logger.info(f"Feature importance saved to {importance_filepath.with_suffix('.csv')}")
    except Exception as e:
        logger.error(f"Error saving feature importance: {e}", exc_info=True)


    # Plot feature importance
    plt.figure(figsize=(10, max(6, len(feature_importance_df) * 0.4))) # Adjust figure size dynamically
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df,
                hue='Feature', # Explicitly map hue to 'Feature'
                palette='viridis',
                legend=False) # Remove legend if hue is used for color mapping

    # Use the passed symbol, interval, model_key for the title
    plt.title(f'Feature Importance (Top {min(20, len(feature_importance_df))})\n{symbol.upper()} {interval} ({model_key})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    importance_plot_path = output_dir / analysis_plot_pattern.format(
        symbol=symbol.upper(), interval=safe_interval, model_type=model_key, # FIX: Changed from model_key to model_type
        analysis_type=f"{model_key}_feature_importance"
    )
    plt.savefig(importance_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance plot to {importance_plot_path}")


def analyse_model_pipeline(symbol: str, interval: str, model_key: str, train_ratio: float):
    """
    Main pipeline function to perform comprehensive model analysis.
    """
    logger.info(f"--- Starting Model Analysis Pipeline for {model_key} ({symbol} {interval}) ---")
    start_time = time.time()
    current_run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Defined here

    dm = DataManager() # Initialize DataManager

    # Define paths for saving analysis results
    try:
        analysis_output_dir = dm.get_file_path(
            symbol=symbol,
            interval=interval,
            data_type='model_analysis',
            model_key=model_key,
            name_suffix='_plots_temp' # Temporary suffix, actual path is parent
        ).parent
        analysis_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured model analysis results directory exists: {analysis_output_dir}")
    except Exception as e:
        logger.error(f"Error ensuring model analysis output directory exists: {e}", exc_info=True)
        raise ModelAnalysisError(f"Failed to create analysis output directory: {e}")


    try:
        # --- 1. Load Data using the new DataManager methods ---
        logger.info("Loading and merging features and labels data using DataManager...")

        df_features = dm.load_data(symbol=symbol, interval=interval, data_type='processed')
        if df_features is None or df_features.empty:
            raise ModelAnalysisError(f"Failed to load processed data for {symbol} {interval}.")

        df_labels = dm.load_data(symbol=symbol, interval=interval, data_type='labeled')
        if df_labels is None or df_labels.empty:
            raise ModelAnalysisError(f"Failed to load labeled data for {symbol} {interval}.")

        if len(df_labels.columns) > 1:
             logger.warning(f"Labeled data file for {symbol}@{interval} contains more than one column. Assuming the first column '{df_labels.columns[0]}' is the label.")
             df_labels = df_labels[[df_labels.columns[0]]]


        df_merged = pd.merge(df_features, df_labels, left_index=True, right_index=True, how='inner')

        if df_merged.empty:
             logger.error("Merged features and labels DataFrame is empty.")
             raise ModelAnalysisError("Merged features and labels DataFrame is empty.")


        logger.info(f"Successfully merged features and labels. Merged shape: {df_merged.shape}")

    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        logger.error("Please ensure you have run the feature engineering and labeling scripts first.")
        raise ModelAnalysisError(f"Data loading failed: {e}") from e
    except Exception as e:
        logger.error(f"Error loading or merging data: {e}", exc_info=True)
        raise ModelAnalysisError(f"Data loading failed: {e}") from e


    # --- 2. Load Trained ModelTrainer instance ---
    logger.info(f"Loading trained ModelTrainer instance for '{model_key}'...")
    try:
        # Load the ModelTrainer instance directly
        trainer = load_trained_model_and_preprocessor(symbol, interval, model_key, dm)

        logger.info(f"ModelTrainer instance for '{model_key}' loaded successfully.")

        original_feature_columns = trainer.feature_columns_original

        if original_feature_columns is None or not original_feature_columns:
             logger.error("Original feature columns could not be loaded from the trainer metadata.")
             raise ModelAnalysisError("Original feature columns not available after loading model.")

        logger.info(f"Using original feature columns loaded from trainer instance: {original_feature_columns}")

        label_column = df_labels.columns[0]


    except FileNotFoundError as e:
        logger.error(f"Trained model file not found by DataManager: {e}")
        logger.error(f"Please ensure you have run the training script for {symbol}@{interval} with model '{model_key}', and that ModelTrainer saves correctly using DataManager.")
        raise ModelAnalysisError(f"Model loading failed: {e}") from e
    except Exception as e:
        logger.error(f"Error loading trained model using DataManager: {e}", exc_info=True)
        raise ModelAnalysisError(f"Model loading failed: {e}") from e


    # --- 3. Prepare Data for Analysis (Test Set) ---
    logger.info("Preparing data for analysis (test set)...")
    columns_to_check_for_nan = original_feature_columns + [label_column]
    df_cleaned = clean_data(df_merged, columns_to_check_for_nan)

    if df_cleaned.empty:
        logger.error("No data remaining after cleaning NaN/Inf values. Cannot perform analysis.")
        raise ModelAnalysisError("No data remaining after cleaning.")

    logger.info(f"Data cleaned successfully. Cleaned shape: {df_cleaned.shape}")

    split_index = int(len(df_cleaned) * train_ratio)

    if split_index >= len(df_cleaned):
         logger.error(f"Train ratio {train_ratio} is too high. No data left for test set after split.")
         raise ModelAnalysisError(f"Train ratio {train_ratio} is too high, test set is empty.")

    df_test = df_cleaned.iloc[split_index:].copy()

    if df_test.empty:
         logger.error("Test set is empty after splitting.")
         raise ModelAnalysisError("Test set is empty after splitting.")

    X_test = df_test[original_feature_columns]
    y_test = df_test[label_column]

    logger.info(f"Test set created. Shape: {df_test.shape}")
    logger.info(f"Test set index range: {df_test.index.min()} to {df_test.index.max()}")

    if not y_test.empty:
        logger.info("Test set label distribution:")
        logger.info(y_test.value_counts(normalize=True).sort_index().to_string())
    else:
        logger.warning("Test set is empty, cannot log label distribution.")


    # --- 4. Make Predictions and Evaluate Model ---
    # Call evaluate_model and capture its returns
    y_pred_evaluated_np, y_test_evaluated, y_proba_df, all_expected_labels = evaluate_model(
        trainer, X_test, y_test, model_key,
        analysis_output_dir, PATHS['analysis_table_pattern'], # Use analysis_table_pattern
        symbol, interval, dm, train_ratio # Pass dm and train_ratio here
    )

    if y_test_evaluated.empty:
        logger.warning("No data available for plotting after evaluation. Skipping plots.")
        return # Exit if no data to plot


    # --- 7. Perform Detailed Analysis and Plotting ---
    logger.info("Performing detailed analysis and plotting...")

    try:
        plots_base_dir = analysis_output_dir # Already defined above

        plots_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured plots directory exists: {plots_base_dir}")

        safe_interval = interval.replace(':', '_')
        conf_matrix_plot_path = plots_base_dir / PATHS['analysis_plot_pattern'].format(
            symbol=symbol.upper(), interval=safe_interval, model_type=model_key, analysis_type="confusion_matrix" # FIX: Changed model_key to model_type
        )
        # Feature importance plot path is dynamically generated in analyze_feature_importance
        # Proba plot path is dynamically generated in plot_probability_histograms
        # ROC AUC plot path is dynamically generated in plot_roc_auc
        # PR Curve plot path is dynamically generated in plot_precision_recall_curve
        # Calibration plot path is dynamically generated in plot_calibration_curve


        # Pass symbol, interval, model_key to the plotting function
        # Re-calculate confusion matrix for plotting from the returned evaluated predictions
        conf_matrix_for_plot = confusion_matrix(y_test_evaluated, y_pred_evaluated_np, labels=all_expected_labels)
        plot_confusion_matrix(conf_matrix_for_plot, all_expected_labels, conf_matrix_plot_path, symbol, interval, model_key)


        # Feature Importance Plot
        # Pass the loaded trainer instance to this function
        analyze_feature_importance(trainer, original_feature_columns, model_key,
                                   plots_base_dir, PATHS['analysis_table_pattern'],
                                   PATHS['analysis_plot_pattern'], symbol, interval)


        if not y_proba_df.empty: # Check if y_proba_df is not empty
             # Pass the collected y_test_evaluated, y_proba_df.values, and all_expected_labels
             plot_probability_histograms(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir / PATHS['analysis_plot_pattern'].format(symbol=symbol.upper(), interval=safe_interval, model_type=model_key, analysis_type="probability_distributions"), symbol, interval, model_key) # FIX: Changed model_key to model_type
             plot_roc_auc(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir, symbol, interval, model_key)
             plot_precision_recall_curve(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir, symbol, interval, model_key)
             plot_calibration_curve(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir, symbol, interval, model_key)
        else:
             logger.info(f"Skipping probability-based plots: y_proba_df is empty for model type '{model_key}'.")


    except Exception as e:
         logger.error(f"Error setting up or generating plots: {e}", exc_info=True)


    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- Model Analysis Run Finished ({current_run_time_str}) in {duration:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a trained machine learning model.")
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT, ADAUSDT)."
    )
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval (e.e.g., 5m, 1h, 1d).'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_CONFIG.keys()),
        help=f"Model type to analyze. Must be one of: {list(MODEL_CONFIG.keys())}"
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=GENERAL_CONFIG.get('train_test_split_ratio', 0.8),
        help=f"Ratio of data used for training (0.0 to 1.0 exclusive). Default: {GENERAL_CONFIG.get('train_test_split_ratio', 0.8)}"
    )

    args = parser.parse_args()

    if not (0 < args.train_ratio < 1):
         logger.error(f"Invalid --train_ratio value: {args.train_ratio}. Must be between 0.0 and 1.0 (exclusive).")
         sys.exit(1)

    try:
        analyse_model_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            model_key=args.model,
            train_ratio=args.train_ratio
        )
    except ModelAnalysisError:
        sys.exit(1)
    except SystemExit:
        pass
    except Exception:
        logger.exception("Model analysis script terminated due to an unhandled error.")
        sys.exit(1)

    """
    Usage example:

    Analyze the trained RandomForest model for ADAUSDT 5m data:
        python scripts/analyze_model.py --symbol ADAUSDT --interval 5m --model random_forest

    Ensure you have run the train_model.py script successfully for the specified
    symbol, interval, and model before running this analysis script.
    The script will load data from data/processed/ and data/labeled/ and
    the trained model and metadata from models/trained_models/, all managed by DataManager.
    Analysis results (metrics and plots) will be saved to results/analysis/model_analysis/.
    """
