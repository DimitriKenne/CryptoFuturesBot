#!/usr/bin/env python3
"""
analyse_model.py

Loads a trained model and performs a comprehensive analysis, saving all artifacts
to a structured directory managed by the DataManager.
"""
import json
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter
from itertools import cycle
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.switch_backend('Agg')
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import PATH_CONFIG
from config.params import app_config
from config.validator import validate_config
from utils.data_management.data_manager import DataManager
from utils.logger_config import setup_rotating_logging
from utils.training.model_trainer import ModelTrainer
from utils.exceptions import ModelAnalysisError

# --- Logging and Validation ---
logger = setup_rotating_logging(Path(__file__).stem, log_level=logging.INFO)
validate_config(app_config)
logger.info("Configuration validated successfully.")

def clean_data(df: pd.DataFrame, cols_to_check: List[str]) -> pd.DataFrame:
    """Removes rows with NA or Inf values from the specified columns."""
    initial_rows = len(df)
    cols_present = [col for col in cols_to_check if col in df.columns]
    df_cleaned = df.dropna(subset=cols_present).copy()
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
    df_cleaned.dropna(inplace=True)
    removed_rows = initial_rows - len(df_cleaned)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows with NA/Inf values.")
    return df_cleaned

def log_analysis_start_end(func):
    """Decorator to log the start and end of an analysis function."""
    def wrapper(*args, **kwargs):
        logger.info(f"--- Starting: {func.__name__.replace('_', ' ').title()} ---")
        try:
            result = func(*args, **kwargs)
            logger.info(f"--- Finished: {func.__name__.replace('_', ' ').title()} ---")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise ModelAnalysisError(f"Failed during {func.__name__}") from e
    return wrapper

# --- Plotting Functions ---

def plot_confusion_matrix(cm: np.ndarray, classes: list, dm: DataManager, model_type: str, symbol: str, interval: str):
    """Generates and saves a confusion matrix plot using DataManager."""
    if not PLOT_AVAILABLE: return
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    dm.save_model_plot(plt.gcf(), 'confusion_matrix', model_type, symbol, interval)
    plt.close()

def plot_feature_importance(importances_df: pd.DataFrame, dm: DataManager, model_type: str, symbol: str, interval: str, top_n: int = 20):
    """Generates and saves a feature importance plot using DataManager."""
    if not PLOT_AVAILABLE or importances_df.empty: return
    top_importances = importances_df.head(top_n)
    plt.figure(figsize=(10, max(6, len(top_importances) * 0.4)))
    sns.barplot(x='Importance', y='Feature', data=top_importances, palette='viridis', hue='Feature', legend=False)
    plt.title(f'Feature Importance (Top {len(top_importances)})')
    plt.tight_layout()
    dm.save_model_plot(plt.gcf(), 'feature_importance', model_type, symbol, interval)
    plt.close()

def plot_roc_auc(y_test: pd.Series, y_proba: pd.DataFrame, classes: list, dm: DataManager, model_type: str, symbol: str, interval: str):
    """Generates and saves an ROC/AUC plot using DataManager."""
    if not PLOT_AVAILABLE: return
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.to_numpy().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', color='deeppink', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    dm.save_model_plot(plt.gcf(), 'roc_auc_curve', model_type, symbol, interval)
    plt.close()

def plot_precision_recall(y_test: pd.Series, y_proba: pd.DataFrame, classes: list, dm: DataManager, model_type: str, symbol: str, interval: str):
    """Generates and saves a Precision-Recall curve plot using DataManager."""
    if not PLOT_AVAILABLE: return
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    precision, recall, average_precision = dict(), dict(), dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba.iloc[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_proba.iloc[:, i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['navy', 'turquoise', 'darkorange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR curve of class {classes[i]} (AP={average_precision[i]:0.2f})')

    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    dm.save_model_plot(plt.gcf(), 'precision_recall_curve', model_type, symbol, interval)
    plt.close()

def plot_calibration_curve(y_test: pd.Series, y_proba: pd.DataFrame, classes: list, dm: DataManager, model_type: str, symbol: str, interval: str):
    """Generates and saves a calibration curve plot using DataManager."""
    if not PLOT_AVAILABLE: return
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for i, class_label in enumerate(classes):
        prob_pos = y_proba.iloc[:, i]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test == class_label, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Class {class_label}")
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=f"Class {class_label}", histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    dm.save_model_plot(plt.gcf(), 'calibration_curve', model_type, symbol, interval)
    plt.close()

@log_analysis_start_end
def load_and_prepare_data(symbol: str, interval: str, train_ratio: float, dm: DataManager, feature_columns_original: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads and splits the test dataset for analysis."""
    processed_data = dm.load_dataframe(data_type='processed', symbol=symbol, interval=interval)
    if processed_data is None or processed_data.empty:
        raise FileNotFoundError(f"Processed feature data not found for {symbol} {interval}.")
    
    labeled_data = dm.load_dataframe(data_type='labeled', symbol=symbol, interval=interval)
    if labeled_data is None or labeled_data.empty:
        raise FileNotFoundError(f"Labeled data not found for {symbol} {interval}.")

    combined_df = pd.merge(processed_data, labeled_data, left_index=True, right_index=True, how='inner')
    label_column = labeled_data.columns[0]
    df_cleaned = clean_data(combined_df, feature_columns_original + [label_column])
    X = df_cleaned[feature_columns_original]
    y = df_cleaned[label_column].astype(int)
    split_index = int(len(X) * train_ratio)
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]
    return X_test, y_test

@log_analysis_start_end
def evaluate_model(trainer: ModelTrainer, X_test: pd.DataFrame, y_test: pd.Series, model_type: str, symbol: str, interval: str, dm: DataManager, train_ratio: float) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], List[int]]:
    """Evaluates the model and saves metrics using DataManager."""
    if X_test.empty:
        return pd.Series(dtype=int), pd.Series(dtype=int), None, trainer.classes.tolist()
    
    y_pred_series = trainer.predict(X_test)
    y_proba_df = trainer.predict_proba(X_test)

    common_index = y_pred_series.dropna().index.intersection(y_test.index)
    y_test_evaluated = y_test.loc[common_index]
    y_pred_evaluated = y_pred_series.loc[common_index]
    
    if y_proba_df is not None: y_proba_df = y_proba_df.loc[common_index]
    
    all_expected_labels = trainer.classes.tolist()
    
    # Generate string report for logging and dict report for saving
    report_str = classification_report(y_test_evaluated, y_pred_evaluated, labels=all_expected_labels, zero_division=0)
    report_dict = classification_report(y_test_evaluated, y_pred_evaluated, labels=all_expected_labels, output_dict=True, zero_division=0)

    evaluation_results = {
        'overall_accuracy': accuracy_score(y_test_evaluated, y_pred_evaluated),
        'balanced_accuracy': balanced_accuracy_score(y_test_evaluated, y_pred_evaluated),
        'classification_report': report_dict,
        'confusion_matrix': confusion_matrix(y_test_evaluated, y_pred_evaluated, labels=all_expected_labels).tolist(),
        'model_type': model_type, 'symbol': symbol, 'interval': interval, 'train_ratio': train_ratio, 'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Model evaluation complete. Balanced Accuracy: {evaluation_results['balanced_accuracy']:.4f}")
    logger.info(f"Classification Report:\n{report_str}")
    
    dm.save_evaluation_results(evaluation_results, model_type, symbol, interval)
    
    return y_pred_evaluated, y_test_evaluated, y_proba_df, all_expected_labels

@log_analysis_start_end
def analyze_feature_importance(trainer: ModelTrainer, dm: DataManager, model_type: str, symbol: str, interval: str):
    """Analyzes and saves feature importance using DataManager."""
    if trainer.model_type == 'lstm':
        logger.info("Feature importance analysis is not applicable for LSTM models. Skipping.")
        return
    
    if not hasattr(trainer.model, 'feature_importances_'):
        logger.warning(f"The loaded model of type '{trainer.model_type}' does not have 'feature_importances_'. Skipping analysis.")
        return

    importances = getattr(trainer.model, 'feature_importances_', None)
    if importances is None or trainer.feature_columns_processed is None:
        logger.warning("Could not retrieve feature importances from the model. Skipping analysis.")
        return
        
    feature_importance_df = pd.DataFrame({
        'Feature': trainer.feature_columns_processed,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    if not feature_importance_df.empty:
        dm.save_feature_importance(feature_importance_df, model_type, symbol, interval)
        plot_feature_importance(feature_importance_df, dm, model_type, symbol, interval)

def analyse_model_pipeline(symbol: str, interval: str, model_type: str, train_ratio: float):
    """Main pipeline to orchestrate the model analysis."""
    logger.info("======================================================================")
    logger.info(f"--- Starting Model Analysis Pipeline ---")
    logger.info(f"Run Parameters: Model={model_type}, Symbol={symbol}, Interval={interval}")
    logger.info("======================================================================")
    dm = DataManager()
    
    logger.info(f"Loading trained ModelTrainer instance for '{model_type}'...")
    trainer = ModelTrainer()
    trainer.load(symbol=symbol, interval=interval, model_type=model_type)
    logger.info(f"Successfully loaded trainer. Confirmed model type: '{trainer.model_type}'")
    
    if not trainer.feature_columns_original:
        raise ModelAnalysisError("Missing feature columns in loaded model metadata.")

    try:
        X_test, y_test = load_and_prepare_data(symbol, interval, train_ratio, dm, trainer.feature_columns_original)
        logger.info(f"Test data prepared successfully. Shape: {X_test.shape}")
    except FileNotFoundError as e:
        logger.error(f"Data loading failed for analysis: {e}")
        sys.exit(1)

    y_pred, y_test_eval, y_proba_df, labels = evaluate_model(
        trainer, X_test, y_test, model_type, symbol, interval, dm, train_ratio
    )

    if not y_test_eval.empty:
        conf_matrix = confusion_matrix(y_test_eval, y_pred, labels=labels)
        plot_confusion_matrix(conf_matrix, labels, dm, model_type, symbol, interval)
        
        analyze_feature_importance(trainer, dm, model_type, symbol, interval)
        
        if y_proba_df is not None:
            logger.info("Generating additional model analysis plots...")
            plot_roc_auc(y_test_eval, y_proba_df, labels, dm, model_type, symbol, interval)
            plot_precision_recall(y_test_eval, y_proba_df, labels, dm, model_type, symbol, interval)
            plot_calibration_curve(y_test_eval, y_proba_df, labels, dm, model_type, symbol, interval)
            logger.info("Additional plots generated successfully.")
        else:
            logger.warning("Probability predictions not available, skipping ROC, Precision-Recall, and Calibration plots.")
    else:
        logger.warning("Test set was empty after processing. No plots or further analysis will be generated.")
        
    logger.info("======================================================================")
    logger.info("--- Model Analysis Run Finished Successfully ---")
    logger.info("======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a trained machine learning model.")
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument(
        '--interval', type=str, required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
    )
    parser.add_argument(
        '--model_type', type=str, required=True,
        choices=['random_forest', 'xgboost', 'lstm'],
    )
    parser.add_argument('--train_ratio', type=float, default=app_config.model.train_test_split_ratio)
    
    args = parser.parse_args()

    analyse_model_pipeline(
        symbol=args.symbol,
        interval=args.interval,
        model_type=args.model_type,
        train_ratio=args.train_ratio
    )