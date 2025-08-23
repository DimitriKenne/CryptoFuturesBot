# utils/training/model_trainer.py

import json
import logging
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING, Union
from dataclasses import fields, is_dataclass, asdict

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, accuracy_score)

from pandas import Int8Dtype

from pathlib import Path
from datetime import datetime, timezone
import copy

# --- Add project root to Python path for imports ---
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary items from params.py (the central AppConfig) and config schemas
try:
    from config.params import app_config
    from config.paths import PATH_CONFIG
    from config.model import ModelConfig, LSTMParams, RandomForestParams, XGBoostParams
    from utils.data_management.data_manager import DataManager

    # Import the new modular training utilities
    from .preprocessor_builder import PreprocessorBuilder
    from .model_builder import ModelBuilder
    from .data_sequencer import DataSequencer

    TF_AVAILABLE = app_config.model.TF_AVAILABLE
    tf = app_config.model.tf
except ImportError as e:
    logging.error(f"Failed to import necessary modules for ModelTrainer: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error during imports/config loading: {e}")
    raise

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle special types like numpy arrays and non-serializable objects.
    """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        try:
            return super().default(o)
        except TypeError:
            return str(o)

def config_to_dict(config_obj):
    """
    Recursively converts a dataclass instance to a dictionary, ensuring serializability.
    """
    if is_dataclass(config_obj):
        return {f.name: config_to_dict(getattr(config_obj, f.name)) for f in fields(config_obj)}
    return config_obj


class ModelTrainer:
    """
    Handles training, evaluation, and saving of different trading models.
    """

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initializes the ModelTrainer with the model configuration.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ModelTrainer initializing...")

        if model_config is None:
            self._model_config: ModelConfig = copy.deepcopy(app_config.model)
        elif isinstance(model_config, ModelConfig):
            self._model_config: ModelConfig = copy.deepcopy(model_config)
        else:
            raise TypeError(f"Config must be a ModelConfig instance, not {type(model_config)}")

        self.model_type = self._model_config.model_type
        self.features_to_use = self._model_config.features_to_use

        if self.model_type == 'lstm' and not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but is not installed.")

        self.model: Optional[Union['BaseEstimator', 'tf.keras.models.Model']] = None
        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_columns_processed: Optional[List[str]] = None
        self.feature_columns_original: Optional[List[str]] = None
        self.training_history: Optional[Dict[str, Any]] = None

        self.label_map: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self.inverse_label_map: Dict[int, int] = {0: -1, 1: 0, 2: 1}
        self.classes: np.ndarray = np.array([-1, 0, 1])

        self.preprocessor_builder = PreprocessorBuilder(
            scaler_type=self._model_config.scaler_type,
            pca_enabled=self._model_config.pca_enabled,
            pca_n_components=self._model_config.pca_n_components,
            features_to_use=self.features_to_use
        )
        self.model_builder = ModelBuilder(tf_available=TF_AVAILABLE, tf_module=tf)

        if self.model_type == 'lstm':
            self.sequence_length = self._model_config.lstm_params.sequence_length_bars
            self.data_sequencer = DataSequencer(sequence_length=self.sequence_length, tf_module=tf)
        else:
            self.sequence_length = 1
            self.data_sequencer = None

        self.dm = DataManager()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, symbol: Optional[str] = None, interval: Optional[str] = None):
        """
        Trains the model using the provided training data.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty.")
        if len(X_train) != len(y_train):
            raise ValueError("Training features and labels have different lengths.")

        self.logger.info(f"Starting training for {self.model_type} model...")
        self.feature_columns_original = X_train.columns.tolist()
        self.preprocessor = self.preprocessor_builder.build_and_fit(X_train)
        self.feature_columns_processed = self.preprocessor_builder.processed_feature_names

        if not self.feature_columns_processed:
            raise RuntimeError("No features were selected by the preprocessor.")

        if self.model_type == 'lstm':
            if not TF_AVAILABLE or self.data_sequencer is None:
                raise RuntimeError("TensorFlow/DataSequencer not available for LSTM training.")
            self._train_lstm(X_train, y_train, X_val, y_val, symbol, interval)
        else:
            self._train_sklearn(X_train, y_train, X_val, y_val)

    def _train_sklearn(self, X_train, y_train, X_val, y_val):
        steps = [('preprocessor', self.preprocessor)]
        model_params = getattr(self._model_config, f"{self.model_type}_params")
        
        self.model = self.model_builder.build_model(
            model_type=self.model_type,
            model_params=model_params,
            general_config_random_seed=app_config.general.random_seed,
            general_config_n_processors=app_config.general.n_processors
        )

        balancing_strategy = getattr(model_params, 'class_balancing', None)
        if balancing_strategy == 'undersampling':
            steps.append(('sampler', RandomUnderSampler(random_state=app_config.general.random_seed)))
        elif balancing_strategy == 'oversampling':
            steps.append(('sampler', SMOTE(random_state=app_config.general.random_seed)))
        
        steps.append(('model', self.model))
        self.pipeline = Pipeline(steps)
        
        fit_kwargs = {}
        if self.model_type == 'xgboost' and model_params.early_stopping_rounds is not None:
            if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
                X_val_processed = self.preprocessor_builder.transform(X_val)
                y_val_mapped = y_val.map(self.label_map)
                fit_kwargs['model__eval_set'] = [(X_val_processed, y_val_mapped)]
            else:
                self.logger.warning("XGBoost early stopping rounds specified, but no validation set provided.")

        self.pipeline.fit(X_train, y_train.map(self.label_map), **fit_kwargs)
        self.logger.info(f"{self.model_type} pipeline training complete.")

    def _train_lstm(self, X_train, y_train, X_val, y_val, symbol, interval):
        lstm_params: LSTMParams = self._model_config.lstm_params
        
        # --- IMPROVED LOGGING: Confirm dense layer configuration ---
        if lstm_params.dense_units and lstm_params.dense_units > 0:
            self.logger.info(f"LSTM architecture will include a final dense layer with {lstm_params.dense_units} units.")
        else:
            self.logger.info("LSTM architecture will not include an additional final dense layer.")

        self.model = self.model_builder.build_model(
            model_type='lstm',
            n_features=len(self.feature_columns_processed),
            model_params=lstm_params,
            general_config_random_seed=app_config.general.random_seed,
            general_config_n_processors=app_config.general.n_processors
        )

        X_train_scaled = self.preprocessor_builder.transform(X_train)
        y_train_mapped = y_train.map(self.label_map).values
        X_train_seq, y_train_seq_one_hot = self.data_sequencer.create_sequences(X_train_scaled, y_train_mapped)

        val_data = None
        if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
            X_val_scaled = self.preprocessor_builder.transform(X_val)
            y_val_mapped = y_val.map(self.label_map).values
            X_val_seq, y_val_seq_one_hot = self.data_sequencer.create_sequences(X_val_scaled, y_val_mapped)
            if X_val_seq.shape[0] > 0:
                val_data = (X_val_seq, y_val_seq_one_hot)

        callbacks = []
        monitor_metric = 'val_loss' if val_data or (lstm_params.validation_split > 0) else 'loss'
        if lstm_params.early_stopping_patience is not None and lstm_params.early_stopping_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=lstm_params.early_stopping_patience, restore_best_weights=True))
        
        if symbol and interval:
            model_dir = self.dm.get_model_dir(self.model_type, symbol, interval)
            checkpoint_path = model_dir / PATH_CONFIG['patterns']['model_keras']
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor=monitor_metric, save_best_only=True, save_weights_only=False))

        history = self.model.fit(
            X_train_seq, y_train_seq_one_hot,
            epochs=lstm_params.epochs,
            batch_size=lstm_params.batch_size,
            validation_data=val_data,
            validation_split=lstm_params.validation_split if val_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )
        self.training_history = history.history
        self.logger.info("LSTM model training complete.")

        # --- IMPROVED LOGGING: Post-training analysis ---
        self.logger.info("--- LSTM Training Summary ---")
        history_data = history.history
        
        # Find the best epoch
        best_epoch_index = np.argmin(history_data[monitor_metric])
        best_epoch_num = best_epoch_index + 1
        best_score = history_data[monitor_metric][best_epoch_index]
        self.logger.info(f"Best model found at epoch {best_epoch_num} with {monitor_metric}: {best_score:.4f}")
        
        # Log final epoch metrics for overfitting diagnosis
        final_epoch_index = len(history_data['loss']) - 1
        final_train_loss = history_data['loss'][final_epoch_index]
        final_train_acc = history_data.get('accuracy', [0])[final_epoch_index]
        
        log_message = (
            f"Final Epoch ({final_epoch_index + 1}) Metrics:\n"
            f"  Training Loss:   {final_train_loss:.4f}\n"
            f"  Training Acc:    {final_train_acc:.4f}"
        )
        if 'val_loss' in history_data:
            final_val_loss = history_data['val_loss'][final_epoch_index]
            final_val_acc = history_data.get('val_accuracy', [0])[final_epoch_index]
            log_message += (
                f"\n  Validation Loss: {final_val_loss:.4f}\n"
                f"  Validation Acc:  {final_val_acc:.4f}"
            )
        self.logger.info(log_message)
        self.logger.info("---------------------------")


    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the trained model on the test data.
        """
        if X_test.empty or y_test.empty:
            raise ValueError("Test data is empty.")
        if len(X_test) != len(y_test):
            raise ValueError("Test features and labels have different lengths.")

        self.logger.info(f"Evaluating {self.model_type} model on test set...")
        y_test_mapped = y_test.map(self.label_map).values

        if self.model_type == 'lstm':
            if self.model is None or self.preprocessor is None or self.data_sequencer is None:
                raise RuntimeError("LSTM model or components not available for evaluation.")
            
            X_test_scaled = self.preprocessor_builder.transform(X_test)
            X_test_seq, y_test_seq_one_hot = self.data_sequencer.create_sequences(X_test_scaled, y_test_mapped)

            if X_test_seq.shape[0] == 0:
                return {"note": "No test sequences generated for evaluation."}

            loss, accuracy = self.model.evaluate(X_test_seq, y_test_seq_one_hot, verbose=0)
            y_pred_proba = self.model.predict(X_test_seq)
            y_pred_mapped = np.argmax(y_pred_proba, axis=1)
            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).values
            y_test_aligned_original = y_test.iloc[self.sequence_length - 1:].values

            results = {
                "loss": loss, "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy_score(y_test_aligned_original, y_pred_original),
                "classification_report": classification_report(y_test_aligned_original, y_pred_original, labels=self.classes, zero_division=0, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test_aligned_original, y_pred_original, labels=self.classes)
            }
        else: # Scikit-learn models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline not available for evaluation.")
            
            y_pred_mapped = self.pipeline.predict(X_test)
            y_pred_original = pd.Series(y_pred_mapped).map(self.inverse_label_map).values
            
            results = {
                "overall_accuracy": accuracy_score(y_test.values, y_pred_original),
                "balanced_accuracy": balanced_accuracy_score(y_test.values, y_pred_original),
                "classification_report": classification_report(y_test.values, y_pred_original, labels=self.classes, zero_division=0, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test.values, y_pred_original, labels=self.classes)
            }
        
        # --- IMPROVED LOGGING: Print evaluation results ---
        self.logger.info("--- Test Set Evaluation Results ---")
        self.logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        if 'overall_accuracy' in results: self.logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        if 'loss' in results: self.logger.info(f"Loss: {results['loss']:.4f}")
        
        report_str = classification_report(
            y_test_aligned_original if self.model_type == 'lstm' else y_test.values,
            y_pred_original,
            labels=self.classes,
            zero_division=0
        )
        self.logger.info(f"Classification Report:\n{report_str}")
        
        cm_df = pd.DataFrame(results['confusion_matrix'], index=[f"True_{c}" for c in self.classes], columns=[f"Pred_{c}" for c in self.classes])
        self.logger.info(f"Confusion Matrix:\n{cm_df}")
        self.logger.info("---------------------------------")

        # Convert numpy types for JSON serialization
        results['confusion_matrix'] = results['confusion_matrix'].tolist()
        return results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions on new data using the trained model pipeline.
        """
        if X.empty:
            self.logger.warning("Input data for prediction is empty. Returning empty Series.")
            return pd.Series(dtype=Int8Dtype())

        self.logger.info(f"Making predictions with {self.model_type} model...")
        if self.model_type == 'lstm':
            if self.model is None or self.preprocessor is None or self.data_sequencer is None:
                raise RuntimeError("LSTM model or components not available for prediction.")
            
            X_scaled = self.preprocessor_builder.transform(X)
            X_sequences, _ = self.data_sequencer.create_sequences(X_scaled, np.zeros(len(X_scaled)))

            if X_sequences.shape[0] == 0:
                return pd.Series(np.nan, index=X.index, dtype=float).astype(Int8Dtype())

            y_pred_mapped = np.argmax(self.model.predict(X_sequences), axis=1)
            y_pred_values = pd.Series(y_pred_mapped).map(self.inverse_label_map).values
            pred_index = X.index[self.sequence_length - 1:]
            
            return pd.Series(y_pred_values, index=pred_index, name='predictions').reindex(X.index).astype(Int8Dtype())
        else: # Scikit-learn models
            if self.pipeline is None:
                raise RuntimeError("Model pipeline not available for prediction.")
            
            y_pred_mapped = self.pipeline.predict(X)
            return pd.Series(pd.Series(y_pred_mapped).map(self.inverse_label_map).values, index=X.index, dtype=Int8Dtype())

    def predict_proba(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Makes probability predictions on new data using the trained model.
        """
        if X.empty:
            self.logger.warning("Input data for probability prediction is empty. Returning None.")
            return None

        self.logger.info(f"Getting probability predictions with {self.model_type} model...")
        if self.model_type == 'lstm':
            if self.model is None or self.preprocessor is None or self.data_sequencer is None:
                raise RuntimeError("LSTM model or components not available for probability prediction.")
            
            X_scaled = self.preprocessor_builder.transform(X)
            X_sequences, _ = self.data_sequencer.create_sequences(X_scaled, np.zeros(len(X_scaled)))
            
            if X_sequences.shape[0] == 0:
                return None
            
            y_pred_proba = self.model.predict(X_sequences)
            pred_index = X.index[self.sequence_length - 1:]
            
            return pd.DataFrame(y_pred_proba, index=pred_index, columns=['proba_-1', 'proba_0', 'proba_1']).reindex(X.index)
        else: # Scikit-learn models
            if self.pipeline is None or not hasattr(self.pipeline, 'predict_proba'):
                self.logger.info(f"Model type '{self.model_type}' or its pipeline does not support predict_proba.")
                return None
            
            y_pred_proba = self.pipeline.predict_proba(X)
            return pd.DataFrame(y_pred_proba, index=X.index, columns=['proba_-1', 'proba_0', 'proba_1'])

    def save(self, symbol: str, interval: str, saved_by: str):
        """Saves the trained model and all related artifacts to a structured directory."""
        if self.model is None and self.pipeline is None:
            raise RuntimeError("No model or pipeline has been trained to save.")
        
        self.logger.info(f"Saving trained {self.model_type} model and artifacts for {symbol.upper()} {interval}...")
        model_dir = self.dm.get_model_dir(self.model_type, symbol, interval)
        self.logger.info(f"Artifacts will be saved in: {model_dir}")

        patterns = PATH_CONFIG['patterns']
        
        metadata = {
            'model_type': self.model_type,
            'feature_columns_processed': self.feature_columns_processed,
            'feature_columns_original': self.feature_columns_original,
            'label_map': self.label_map,
            'inverse_label_map': self.inverse_label_map,
            'classes': self.classes.tolist(),
            'model_config': config_to_dict(self._model_config),
            'training_history': self.training_history,
            'save_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'saved_by': saved_by,
        }
        with open(model_dir / patterns['model_metadata'], 'w') as f:
            json.dump(metadata, f, indent=4, cls=CustomJSONEncoder)
        self.logger.info("Model metadata saved successfully.")

        if self.model_type == 'lstm':
            if self.model:
                self.model.save(model_dir / patterns['model_keras'])
                self.logger.info("Keras model saved successfully.")
            if self.preprocessor:
                joblib.dump(self.preprocessor, model_dir / patterns['model_preprocessor'])
                self.logger.info("Preprocessor saved successfully.")
        else:
            if self.pipeline:
                joblib.dump(self.pipeline, model_dir / patterns['model_pipeline'])
                self.logger.info("Scikit-learn pipeline saved successfully.")

        self.logger.info(f"All {self.model_type} artifacts saved successfully.")

    def load(self, symbol: str, interval: str, model_type: str) -> 'ModelTrainer':
        self.logger.info(f"Loading trained {model_type} model for {symbol.upper()} {interval}...")
        self.model_type = model_type
        
        model_dir = self.dm.get_model_dir(self.model_type, symbol, interval)
        if not model_dir.exists(): raise FileNotFoundError(f"Model directory not found: {model_dir}")

        patterns = PATH_CONFIG['patterns']
        
        with open(model_dir / patterns['model_metadata'], 'r') as f: metadata = json.load(f)
        
        config_dict = metadata.get('model_config')
        if config_dict:
            if 'lstm_params' in config_dict: config_dict['lstm_params'] = LSTMParams(**config_dict['lstm_params'])
            if 'random_forest_params' in config_dict: config_dict['random_forest_params'] = RandomForestParams(**config_dict['random_forest_params'])
            if 'xgboost_params' in config_dict: config_dict['xgboost_params'] = XGBoostParams(**config_dict['xgboost_params'])
            
            init_fields = {f.name for f in fields(ModelConfig) if f.init}
            self._model_config = ModelConfig(**{k: v for k, v in config_dict.items() if k in init_fields})
        
        self.preprocessor_builder = PreprocessorBuilder(
            scaler_type=self._model_config.scaler_type,
            pca_enabled=self._model_config.pca_enabled,
            pca_n_components=self._model_config.pca_n_components,
            features_to_use=self._model_config.features_to_use
        )
        self.model_builder = ModelBuilder(tf_available=TF_AVAILABLE, tf_module=tf)

        if self.model_type == 'lstm':
            if not TF_AVAILABLE: raise ImportError("TensorFlow required for LSTM models.")
            self.sequence_length = self._model_config.lstm_params.sequence_length_bars
            self.data_sequencer = DataSequencer(sequence_length=self.sequence_length, tf_module=tf)
        else:
            self.data_sequencer = None

        self.feature_columns_processed = metadata.get('feature_columns_processed')
        self.feature_columns_original = metadata.get('feature_columns_original')
        self.training_history = metadata.get('training_history')
        self.label_map = {int(k): v for k, v in metadata.get('label_map', {}).items()}
        self.inverse_label_map = {int(k): v for k, v in metadata.get('inverse_label_map', {}).items()}
        self.classes = np.array(metadata.get('classes', [-1, 0, 1]))
        self.logger.info("Metadata, configuration, and components re-initialized.")

        if self.model_type == 'lstm':
            self.model = tf.keras.models.load_model(model_dir / patterns['model_keras'])
            self.preprocessor = joblib.load(model_dir / patterns['model_preprocessor'])
            self.preprocessor_builder.preprocessor = self.preprocessor
        else:
            self.pipeline = joblib.load(model_dir / patterns['model_pipeline'])
            self.model = self.pipeline.named_steps['model']
            self.preprocessor = self.pipeline.named_steps['preprocessor']
            self.preprocessor_builder.preprocessor = self.preprocessor

        self.logger.info(f"ModelTrainer instance for {self.model_type} loaded successfully.")
        return self