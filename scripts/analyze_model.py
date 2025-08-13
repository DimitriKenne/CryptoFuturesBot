#!/usr/bin/env python3
"""
analyse_model.py

Loads a trained model pipeline, features from processed data, and labels from
labeled data, merges them, evaluates the model's performance on a test set,
and generates analysis metrics and plots for a ternary classification model (-1, 0, 1).
Includes enhanced analysis based on label generator diagnostics and feature importance.
Logs are saved to a single file in the logs directory specified in paths.py,
with clear markers for each analysis run, using the centralized rotating logger.

MODIFIED: Updated to use `app_config` for all configuration access, ensuring consistency.
MODIFIED: `load_trained_model_and_preprocessor` now initializes `ModelTrainer` without
          explicit config, as `trainer.load()` handles loading the saved configuration.
MODIFIED: Feature selection in `load_and_prepare_data` now explicitly uses the
          `original_feature_columns` from the loaded `ModelTrainer` instance,
          guaranteeing analysis is done on the features the model was trained on.
MODIFIED: Simplified default argument parsing to rely on `app_config`.
MODIFIED: Refined logic for handling empty dataframes and potential NaN predictions
          before evaluation and plotting.
MODIFIED: Ensured `all_expected_labels` consistently represents [-1, 0, 1] for metrics and plots.
MODIFIED: Updated plotting functions to correctly receive and use `all_expected_labels`,
          and to handle cases where probability data might be empty.
MODIFIED: Removed redundant `val_ratio_check` as this script focuses on analysis
          and does not perform model training or validation splitting in the same way.
FIXED: Ensured `sys.path.insert` is correctly placed relative to `PROJECT_ROOT`.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter
import time
import copy
from datetime import datetime

import pandas as pd
import numpy as np

# Import scikit-learn components for analysis/evaluation
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.utils import column_or_1d
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# Conditional import for TensorFlow/Keras for LSTM (now handled via app_config.model)
# We still need the raw import for type hinting/checking.
try:
    import tensorflow as tf # type: ignore
    _TF_IMPORTED = True
except ImportError:
    _TF_IMPORTED = False

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    from config.params import app_config # Use app_config as the central config object
    from utils.data_manager import DataManager
    from utils.logger_config import setup_rotating_logging
    from utils.training.model_trainer import ModelTrainer
    from utils.exceptions import TemporalSafetyError, ModelAnalysisError
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Configuration file not found: {e}. Ensure config/params.py and config/paths.py exist.", file=sys.stderr)
    sys.exit(1)
except AttributeError as e:
    print(f"CRITICAL ERROR: Configuration object missing expected attribute or key: {e}. Check config/params.py and config/paths.py.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred during initial imports or configuration loading: {e}", file=sys.stderr)
    sys.exit(1)


# Set up logging for this script
log_filepath = PATHS['logs_dir'] / f"{Path(__file__).stem}.log"
logger = setup_rotating_logging(
    Path(__file__).stem,
    log_level=logging.INFO
)

# Log TensorFlow status from app_config
if app_config.model.TF_AVAILABLE:
    logger.info(f"TensorFlow (version {getattr(app_config.model.tf, '__version__', 'unknown')}) imported successfully via app_config.model.tf.")
    if app_config.model.tf.config.list_physical_devices('GPU'):
        logger.info("GPU is available and enabled for TensorFlow.")
    else:
        logger.info("GPU is not available or not enabled for TensorFlow.")
else:
    logger.warning("TensorFlow not found. LSTM model analysis will be skipped.")


# --- Plotting Libraries (Conditional on PLOT_AVAILABLE) ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.switch_backend('Agg')
    PLOT_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib or Seaborn not found. Plotting will be skipped. Install using 'pip install matplotlib seaborn'.")
    PLOT_AVAILABLE = False


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
    df_cleaned = df.copy()

    cols_to_check_present = [col for col in cols_to_check if col in df_cleaned.columns]
    if not cols_to_check_present:
        logger.warning("None of the specified columns to check for NaNs are present in the DataFrame. Returning original DataFrame copy.")
        return df_cleaned

    df_cleaned.dropna(subset=cols_to_check_present, inplace=True)

    numeric_cols_to_check = [col for col in cols_to_check_present if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col])]

    for col in numeric_cols_to_check:
        if np.isinf(df_cleaned[col]).any():
            logger.warning(f"Infinite values found in column '{col}'. Replacing with NaN and dropping rows.")
            df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_cleaned.dropna(subset=[col], inplace=True)

    removed_rows = initial_rows - len(df_cleaned)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows with NA or Inf values in specified columns. Remaining rows: {len(df_cleaned)}")

    return df_cleaned


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


# --- Plotting Functions (Always defined, but check PLOT_AVAILABLE internally) ---

def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: Path, symbol: str, interval: str, model_key: str):
    """
    Plots the confusion matrix using seaborn.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping confusion matrix plot: Plotting libraries not available.")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
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
        plt.close()

def plot_feature_importance(importances: Dict[str, float], save_path: Path, symbol: str, interval: str, model_key: str, top_n: int = 20):
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

    if top_importances.empty:
        logger.warning("No top feature importances found after filtering.")
        return

    plt.figure(figsize=(10, max(6, len(top_importances) * 0.4)))
    sns.barplot(x=top_importances.values, y=top_importances.index,
                hue=top_importances.index, # Explicitly map hue to 'Feature' for distinct colors
                palette='viridis',
                legend=False)
    plt.title(f'Feature Importance (Top {min(top_n, len(top_importances))})\n{symbol.upper()} {interval} ({model_key.replace("_", " ").title()})')
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


def plot_probability_histograms(y_true: pd.Series, y_proba: np.ndarray, classes: list, save_path: Path, symbol: str, interval: str, model_key: str):
    """
    Plots histograms of predicted probabilities for each class, separated by true label.
    """
    if not PLOT_AVAILABLE:
        logger.warning("Skipping probability histograms plot: Plotting libraries not available.")
        return
    if y_true.empty or y_proba.shape[0] == 0 or y_proba.shape[1] != len(classes):
        logger.warning(f"Insufficient data or incorrect shape for probability histogram plotting. y_true_shape: {y_true.shape}, y_proba_shape: {y_proba.shape}, classes_len: {len(classes)}.")
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
        if i >= proba_df.shape[1]: # Ensure the probability column exists
            logger.warning(f"Probability column for class {class_label} (index {i}) not found in y_proba_df. Skipping plot for this class.")
            continue

        proba_column = classes[i] # Use the actual class label as column name

        subset_data = merged_df[[proba_column, 'true_label']].dropna()
        if subset_data.empty:
            logger.warning(f"Skipping probability histogram for class {class_label}: No non-NaN probability data or true labels.")
            # Set up empty plot for visual consistency
            axes[i].set_title(f'Probabilities for Class {class_label}\n(No Data)')
            axes[i].set_xlabel(f'Predicted Probability ({class_label})')
            axes[i].set_ylabel('Density')
            continue

        sns.histplot(data=subset_data, x=proba_column, hue='true_label', ax=axes[i], stat='density', common_norm=False, bins=30, palette='viridis')
        axes[i].set_title(f'Probabilities for Class {class_label}')
        axes[i].set_xlabel(f'Predicted Probability ({class_label})')
        axes[i].set_ylabel('Density')
        if axes[i].get_legend() is None:
            axes[i].legend(title='True Label')
        else:
            axes[i].get_legend().set_title('True Label')

    fig.suptitle(f'Prediction Probability Distribution\n{symbol.upper()} {interval} ({model_key.replace("_", " ").title()})', y=1.02)
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
        logger.warning(f"Insufficient data or incorrect shape for ROC AUC plotting. y_true_shape: {y_true.shape}, y_proba_shape: {y_proba.shape}, classes_len: {len(classes)}.")
        return
    unique_true_classes = np.unique(y_true.dropna())
    if len(unique_true_classes) < 2:
        logger.warning(f"Skipping ROC AUC plot: Need at least two unique classes in true labels. Found: {unique_true_classes}")
        return

    sorted_classes = sorted(classes)
    y_true_clean = y_true.dropna()

    # Ensure y_proba_clean aligns with y_true_clean's index, and only take the values
    y_proba_df_temp = pd.DataFrame(y_proba, index=y_true.index, columns=sorted_classes)
    y_proba_clean = y_proba_df_temp.loc[y_true_clean.index].values

    if y_true_clean.empty or y_proba_clean.shape[0] == 0:
        logger.warning("Skipping ROC AUC plot: No non-NaN data points after cleaning true labels or probabilities.")
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
    plt.title(f'Receiver Operating Characteristic (ROC) Curve\n{symbol.upper()} {interval} ({model_key.replace("_", " ").title()})')
    plt.legend(loc="lower right")
    plt.tight_layout()

    safe_interval = interval.replace(':', '_')
    roc_auc_plot_path = output_dir / f"{symbol.upper()}_{safe_interval}_{model_key}_roc_auc_curve.png"
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
        logger.warning(f"Insufficient data or incorrect shape for Precision-Recall plotting. y_true_shape: {y_true.shape}, y_proba_shape: {y_proba.shape}, classes_len: {len(classes)}.")
        return
    unique_true_classes = np.unique(y_true.dropna())
    if len(unique_true_classes) < 2:
        logger.warning(f"Skipping Precision-Recall plot: Need at least two unique classes in true labels. Found: {unique_true_classes}")
        return

    sorted_classes = sorted(classes)
    y_true_clean = y_true.dropna()
    y_proba_df_temp = pd.DataFrame(y_proba, index=y_true.index, columns=sorted_classes)
    y_proba_clean = y_proba_df_temp.loc[y_true_clean.index].values

    if y_true_clean.empty or y_proba_clean.shape[0] == 0:
        logger.warning("Skipping Precision-Recall plot: No non-NaN data points after cleaning true labels or probabilities.")
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
            plt.plot(recall[i], precision[i], label=f'Precision-Recall curve of class {class_label} (area = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n{symbol.upper()} {interval} ({model_key.replace("_", " ").title()})')
    plt.legend(loc="lower left")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()

    safe_interval = interval.replace(':', '_')
    pr_curve_plot_path = output_dir / f"{symbol.upper()}_{safe_interval}_{model_key}_precision_recall_curve.png"
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
        logger.warning(f"Insufficient data or incorrect shape for calibration plotting. y_true_shape: {y_true.shape}, y_proba_shape: {y_proba.shape}, classes_len: {len(classes)}.")
        return
    unique_true_classes = np.unique(y_true.dropna())
    if len(unique_true_classes) < 2:
        logger.warning(f"Skipping calibration plot: Need at least two unique classes in true labels. Found: {unique_true_classes}")
        return

    sorted_classes = sorted(classes)
    y_true_clean = y_true.dropna()
    y_proba_df_temp = pd.DataFrame(y_proba, index=y_true.index, columns=sorted_classes)
    y_proba_clean = y_proba_df_temp.loc[y_true_clean.index].values


    if y_true_clean.empty or y_proba_clean.shape[0] == 0:
        logger.warning("Skipping calibration plot: No non-NaN data points after cleaning true labels or probabilities.")
        return

    y_true_bin = label_binarize(y_true_clean, classes=sorted_classes)

    plt.figure(figsize=(8, 8))
    for i, class_label in enumerate(sorted_classes):
        if y_true_bin.shape[1] > i and np.any(y_true_bin[:, i] == 1):
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
    calibration_plot_path = output_dir / f"{symbol.upper()}_{safe_interval}_{model_key}_calibration_plot.png"
    try:
        plt.savefig(calibration_plot_path, dpi=150)
        logger.info(f"Saved calibration plot to {calibration_plot_path}")
    except Exception as e:
        logger.error(f"Error saving calibration plot to {calibration_plot_path}: {e}", exc_info=True)
    finally:
        plt.close()


@log_analysis_start_end
def load_and_prepare_data(symbol: str, interval: str, train_ratio: float, data_manager: DataManager, feature_columns_original: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads processed and labeled data, merges them, and splits into train/test sets.
    Handles feature selection and NaN removal using the original features from the loaded model.
    """
    logger.info(f"Loading and preparing data for {symbol} {interval} with train_ratio={train_ratio}...")

    try:
        processed_data = data_manager.load_data(symbol=symbol, interval=interval, data_type='processed')
        if processed_data is None or processed_data.empty:
            raise ModelAnalysisError(f"Failed to load processed data for {symbol} {interval}.")
        logger.info(f"Loaded processed data. Shape: {processed_data.shape}")

        labeled_data = data_manager.load_data(symbol=symbol, interval=interval, data_type='labeled')
        if labeled_data is None or labeled_data.empty:
            raise ModelAnalysisError(f"Failed to load labeled data for {symbol} {interval}.")
        logger.info(f"Loaded labeled data. Shape: {labeled_data.shape}")

        if len(labeled_data.columns) > 1:
            logger.warning(f"Labeled data file for {symbol}@{interval} contains more than one column. Assuming the first column '{labeled_data.columns[0]}' is the label.")
            labeled_data = labeled_data[[labeled_data.columns[0]]]

        combined_df = pd.merge(processed_data, labeled_data, left_index=True, right_index=True, how='inner')
        if combined_df.empty:
            logger.error("Merged features and labels DataFrame is empty.")
            raise ModelAnalysisError("Merged features and labels DataFrame is empty.")
        logger.info(f"Merged processed and labeled data. Combined shape: {combined_df.shape}")

        # Use the `feature_columns_original` list passed from the loaded ModelTrainer
        selected_features = feature_columns_original
        missing_features = [f for f in selected_features if f not in combined_df.columns]
        if missing_features:
            logger.critical(f"Features required by the trained model are missing from the data: {missing_features}")
            raise ModelAnalysisError(f"Missing features in data: {missing_features}")
        logger.info(f"Using {len(selected_features)} features as per the loaded model's configuration.")

        label_column = labeled_data.columns[0] # Assuming label column is the first/only column in labeled_data

        # Drop rows with any NaN values in selected features or the label column
        initial_rows = combined_df.shape[0]
        cols_to_check_for_nan = selected_features + [label_column]
        df_cleaned = clean_data(combined_df, cols_to_check_for_nan)

        if df_cleaned.empty:
            logger.critical("DataFrame is empty after dropping NaN/Inf values. Cannot proceed with analysis.")
            raise ModelAnalysisError("Empty DataFrame after NaN/Inf removal.")
        logger.info(f"Data cleaned successfully. Cleaned shape: {df_cleaned.shape}")

        X = df_cleaned[selected_features]
        y = df_cleaned[label_column]

        if not pd.api.types.is_integer_dtype(y):
            y = y.astype(int)
            logger.info("Label column cast to integer type.")

        valid_labels = [-1, 0, 1]
        original_y_count = len(y)
        y = y[y.isin(valid_labels)]
        X = X.loc[y.index]
        if len(y) < original_y_count:
            logger.warning(f"Removed {original_y_count - len(y)} rows with invalid labels (not -1, 0, or 1). Remaining rows: {len(y)}")

        if X.empty or y.empty:
            logger.critical("DataFrame is empty after filtering invalid labels. Cannot proceed with analysis.")
            raise ModelAnalysisError("Empty DataFrame after label filtering.")

        n_samples = len(X)
        if n_samples < 2:
            logger.critical(f"Not enough samples ({n_samples}) to perform train-test split. Need at least 2.")
            raise ModelAnalysisError("Insufficient data for train-test split.")

        split_index = int(n_samples * train_ratio)
        if split_index == 0:
            split_index = 1
        if split_index == n_samples:
            split_index = n_samples - 1
            if split_index == 0:
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

        train_label_counts = Counter(y_train)
        logger.info(f"Training label distribution:\n{train_label_counts}")
        if any(count == 0 for count in train_label_counts.values()):
            logger.warning("One or more classes have zero samples in the training set. This might affect the representativeness of your test set.")

        test_label_counts = Counter(y_test)
        logger.info(f"Test label distribution:\n{test_label_counts}")
        if any(count == 0 for count in test_label_counts.values()):
            logger.warning("One or more classes have zero samples in the test set. Evaluation metrics might be misleading.")

        logger.info("Data loading and splitting complete.")
        return X_train, X_test, y_train, y_test

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
        # Initialize ModelTrainer without explicit config initially.
        # The .load() method will read the saved ModelConfig from the metadata.
        trainer = ModelTrainer()
        trainer.load(symbol=symbol, interval=interval, model_key=model_key)

        logger.info(f"ModelTrainer instance for {model_key} loaded successfully.")
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
                   symbol: str, interval: str, dm: DataManager, train_ratio: float) -> Tuple[np.ndarray, pd.Series, pd.DataFrame, List[int]]:
    """
    Evaluates the model's performance on the test set and saves metrics.
    Handles different model types (sklearn, LSTM) by leveraging ModelTrainer's methods.
    Returns: Tuple of (y_pred_evaluated, y_test_evaluated, y_proba_df, all_expected_labels)
    """
    logger.info(f"Evaluating model '{model_key}' on the test set...")

    if X_test.empty or y_test.empty:
        logger.warning("Test set is empty. Skipping model evaluation.")
        return np.array([]), pd.Series(dtype=int), pd.DataFrame(), trainer.classes.tolist() # Ensure classes are returned

    # Make predictions using the trainer's predict method
    y_pred_series: Optional[pd.Series] = None
    try:
        logger.info(f"Making predictions with {model_key} model...")
        y_pred_series = trainer.predict(X_test)
        if not isinstance(y_pred_series, pd.Series):
             # This case handles when trainer.predict returns a numpy array for special cases (e.g., LSTM alignment)
             # and ensures it's converted to a Series with the correct index.
             # However, trainer.predict is designed to return pd.Series already.
             # This check is mostly for robustness.
             y_pred_series = pd.Series(y_pred_series, index=X_test.index, name='prediction')
        logger.info(f"{model_key} predictions made. Shape: {y_pred_series.shape}")
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
                # Reindex to ensure alignment and consistent length
                y_proba_df = y_proba_df_raw.reindex(y_pred_series.index) # Align with actual predictions
                logger.info(f"{model_key} probability predictions made. Shape: {y_proba_df.shape}")
            else:
                logger.warning(f"trainer.predict_proba for {model_key} returned empty or None.")
        else:
            logger.warning(f"Model type '{model_key}' or its trainer instance does not support predict_proba.")
    except Exception as e:
        logger.error(f"Error getting predicted probabilities for {model_key}: {e}", exc_info=True)
        logger.warning("Predicted probabilities not available for plotting due to error.")


    # --- Handle NaNs in Predictions Before Evaluation ---
    # Ensure y_pred_series and y_test are aligned on index before checking NaNs
    # This is critical if ModelTrainer.predict returns a series with a different index subset
    # (e.g., for LSTM where predictions start after sequence_length)
    common_index_for_eval = y_pred_series.dropna().index.intersection(y_test.index)
    
    if common_index_for_eval.empty:
        logger.warning("No common valid index between predictions and true labels after removing NaNs. Skipping evaluation.")
        return np.array([]), pd.Series(dtype=int), pd.DataFrame(), trainer.classes.tolist()

    y_test_evaluated = y_test.loc[common_index_for_eval]
    y_pred_evaluated = y_pred_series.loc[common_index_for_eval]

    if not y_proba_df.empty:
        y_proba_df = y_proba_df.loc[common_index_for_eval] # Filter probabilities as well

    num_nan_predictions_removed = len(y_test) - len(y_test_evaluated)
    if num_nan_predictions_removed > 0:
        logger.warning(f"Removed {num_nan_predictions_removed} samples due to NaN predictions or index mismatch for evaluation.")
    logger.info(f"Evaluation will be performed on {len(y_test_evaluated)} samples after alignment and NaN removal.")

    # Ensure y_pred_evaluated is a numpy array for sklearn metrics
    y_pred_evaluated_np = y_pred_evaluated.to_numpy()

    # Determine all expected labels (e.g., [-1, 0, 1]) - should always be from trainer.classes
    all_expected_labels = trainer.classes.tolist()
    if not all_expected_labels: # Fallback if trainer.classes is empty somehow
        all_expected_labels = [-1, 0, 1]
        logger.warning(f"trainer.classes was empty; falling back to default expected labels: {all_expected_labels}")

    # Ensure probabilities DataFrame columns match expected labels if not empty
    if not y_proba_df.empty and y_proba_df.shape[1] == len(all_expected_labels):
        y_proba_df.columns = all_expected_labels
    elif not y_proba_df.empty:
        logger.warning(f"Probability DataFrame has {y_proba_df.shape[1]} columns, but {len(all_expected_labels)} expected labels. Cannot assign column names reliably for plotting.")


    # --- 5. Evaluate Model ---
    logger.info("Evaluating model performance on the aligned test set...")

    try:
        overall_accuracy = accuracy_score(y_test_evaluated, y_pred_evaluated_np)
        bal_acc = balanced_accuracy_score(y_test_evaluated, y_pred_evaluated_np)
        class_report = classification_report(y_test_evaluated, y_pred_evaluated_np, labels=all_expected_labels, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test_evaluated, y_pred_evaluated_np, labels=all_expected_labels)

        logger.info(f"Evaluation complete.")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise ModelAnalysisError(f"Evaluation failed: {e}") from e


    # --- 6. Log and Store Evaluation Results ---
    logger.info("--- Evaluation Metrics (Test Set) ---")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test_evaluated, y_pred_evaluated_np, labels=all_expected_labels, zero_division=0))
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
        'num_samples_removed_for_evaluation': num_nan_predictions_removed,
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
            artifact_type='evaluation' # Consistent artifact type name
        )
        logger.info(f"Evaluation results saved successfully using DataManager.")
    except Exception as e:
        logger.error(f"Error saving evaluation results using DataManager: {e}", exc_info=True)
        logger.warning("Saving evaluation results failed. Analysis plots will still be attempted.")

    return y_pred_evaluated_np, y_test_evaluated, y_proba_df, all_expected_labels


@log_analysis_start_end
def analyze_feature_importance(trainer: ModelTrainer, model_key: str, output_dir: Path, analysis_table_pattern: str, symbol: str, interval: str):
    """
    Analyzes and visualizes feature importance for tree-based models.
    Skips for LSTM models.
    """
    logger.info(f"Analyzing feature importance for {model_key}...")

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
        feature_names_for_importance = trainer.feature_columns_processed
        if feature_names_for_importance is None or not feature_names_for_importance:
            logger.error("trainer.feature_columns_processed is None or empty. Cannot determine feature names for importance. Skipping feature importance plot.")
            return

        if hasattr(final_model, 'feature_importances_'):
            importance_data = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'):
            # For linear models, coefficients can indicate importance
            # For multi-class, coef_ is (n_classes, n_features)
            importance_data = np.sum(np.abs(final_model.coef_), axis=0)
        else:
            logger.warning(f"Model type '{model_key}' does not have direct 'feature_importances_' or 'coef_' attribute. Skipping feature importance analysis.")
            return
    else:
        logger.warning(f"No final model found in trainer for '{model_key}'. Skipping feature importance analysis.")
        return

    if importance_data is None or len(importance_data) == 0:
        logger.warning("No feature importance data available after extraction.")
        return

    if len(importance_data) != len(feature_names_for_importance):
        logger.error(f"CRITICAL: Mismatch in lengths for feature importance: Importance data length ({len(importance_data)}) vs Feature names length ({len(feature_names_for_importance)}). Skipping feature importance plot.")
        return

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names_for_importance,
        'Importance': importance_data
    }).sort_values(by='Importance', ascending=False)

    safe_interval = interval.replace(':', '_')
    importance_filepath = output_dir / analysis_table_pattern.format(
        symbol=symbol.upper(),
        interval=safe_interval,
        model_type=model_key,
        analysis_type="feature_importance"
    )

    try:
        importance_filepath.parent.mkdir(parents=True, exist_ok=True)
        feature_importance_df.to_csv(importance_filepath.with_suffix('.csv'), index=False)
        logger.info(f"Feature importance saved to {importance_filepath.with_suffix('.csv')}")
    except Exception as e:
        logger.error(f"Error saving feature importance table: {e}", exc_info=True)

    # Plot feature importance using the dedicated plotting function
    importance_plot_path = output_dir / PATHS['analysis_plot_pattern'].format(
        symbol=symbol.upper(), interval=safe_interval, model_type=model_key,
        analysis_type="feature_importance"
    )
    plot_feature_importance(feature_importance_df.set_index('Feature')['Importance'].to_dict(),
                            importance_plot_path, symbol, interval, model_key)


def analyse_model_pipeline(symbol: str, interval: str, model_key: str, train_ratio: float):
    """
    Main pipeline function to perform comprehensive model analysis.
    """
    logger.info(f"--- Starting Model Analysis Pipeline for {model_key} ({symbol} {interval}) ---")
    start_time = time.time()
    current_run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dm = DataManager()

    try:
        analysis_output_dir = dm.get_file_path(
            symbol=symbol,
            interval=interval,
            data_type='model_analysis',
            model_key=model_key,
            name_suffix='_plots_temp'
        ).parent
        analysis_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured model analysis results directory exists: {analysis_output_dir}")
    except Exception as e:
        logger.error(f"Error ensuring model analysis output directory exists: {e}", exc_info=True)
        raise ModelAnalysisError(f"Failed to create analysis output directory: {e}")

    # --- Load Trained ModelTrainer instance FIRST ---
    logger.info(f"Loading trained ModelTrainer instance for '{model_key}'...")
    trainer: Optional[ModelTrainer] = None
    try:
        trainer = load_trained_model_and_preprocessor(symbol, interval, model_key, dm)
        logger.info(f"ModelTrainer instance for '{model_key}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load trained ModelTrainer instance. Analysis cannot proceed. Error: {e}", exc_info=True)
        raise

    if trainer.feature_columns_original is None or not trainer.feature_columns_original:
        logger.error("Original feature columns could not be loaded from the trainer metadata. Cannot proceed with analysis.")
        raise ModelAnalysisError("Original feature columns not available after loading model.")
    original_feature_columns = trainer.feature_columns_original
    logger.info(f"Using original feature columns loaded from trainer instance: {original_feature_columns}")


    # --- Load and Prepare Data for Analysis (Test Set) ---
    logger.info("Preparing data for analysis (test set)...")
    try:
        # Pass the original_feature_columns to load_and_prepare_data
        _, X_test, _, y_test = load_and_prepare_data(
            symbol, interval, train_ratio, dm, original_feature_columns
        )
    except Exception as e:
        logger.error(f"Failed to load or prepare data for analysis. Error: {e}", exc_info=True)
        raise


    # --- Make Predictions and Evaluate Model ---
    y_pred_evaluated_np, y_test_evaluated, y_proba_df, all_expected_labels = evaluate_model(
        trainer, X_test, y_test, model_key,
        analysis_output_dir, PATHS['analysis_table_pattern'],
        symbol, interval, dm, train_ratio
    )

    if y_test_evaluated.empty:
        logger.warning("No data available for plotting after evaluation. Skipping plots.")
        return

    # --- Perform Detailed Analysis and Plotting ---
    logger.info("Performing detailed analysis and plotting...")

    try:
        plots_base_dir = analysis_output_dir
        plots_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured plots directory exists: {plots_base_dir}")

        safe_interval = interval.replace(':', '_')

        # Confusion Matrix Plot
        conf_matrix_plot_path = plots_base_dir / PATHS['analysis_plot_pattern'].format(
            symbol=symbol.upper(), interval=safe_interval, model_type=model_key, analysis_type="confusion_matrix"
        )
        conf_matrix_for_plot = confusion_matrix(y_test_evaluated, y_pred_evaluated_np, labels=all_expected_labels)
        plot_confusion_matrix(conf_matrix_for_plot, all_expected_labels, conf_matrix_plot_path, symbol, interval, model_key)


        # Feature Importance Plot
        analyze_feature_importance(trainer, model_key,
                                   plots_base_dir, PATHS['analysis_table_pattern'],
                                   symbol, interval)

        # Probability-based plots
        if not y_proba_df.empty and y_proba_df.shape[0] > 0:
            logger.info("Generating probability-based plots...")
            proba_plot_path = plots_base_dir / PATHS['analysis_plot_pattern'].format(
                symbol=symbol.upper(), interval=safe_interval, model_type=model_key, analysis_type="probability_distributions"
            )
            plot_probability_histograms(y_test_evaluated, y_proba_df.values, all_expected_labels, proba_plot_path, symbol, interval, model_key)

            plot_roc_auc(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir, symbol, interval, model_key)
            plot_precision_recall_curve(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir, symbol, interval, model_key)
            plot_calibration_curve(y_test_evaluated, y_proba_df.values, all_expected_labels, plots_base_dir, symbol, interval, model_key)
        else:
            logger.info(f"Skipping probability-based plots: No valid probability data (y_proba_df is empty or has no data) for model type '{model_key}'.")


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
        help='Time interval (e.g., 5m, 1h, 1d).'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['random_forest', 'xgboost', 'lstm'], # Explicitly define choices
        help=f"Model type to analyze. Available: ['random_forest', 'xgboost', 'lstm']"
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        # Use app_config for default train_test_split_ratio
        default=app_config.model.train_test_split_ratio,
        help=f"Ratio of data used for training (0.0 to 1.0 exclusive) to determine the test set for analysis. Default: {app_config.model.train_test_split_ratio}"
    )

    args = parser.parse_args()

    if not (0 < args.train_ratio < 1):
        logger.error(f"Invalid --train_ratio value: {args.train_ratio}. Must be between 0.0 and 1.0 (exclusive).")
        sys.exit(1)

    # Removed val_ratio_check as it's not relevant for analysis script which only uses train_ratio to define test set.

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
