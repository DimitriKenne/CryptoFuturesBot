# utils/training/preprocessor_builder.py

from collections import Counter
import logging
from typing import List, Optional, Any, Dict, Tuple, Union # Added Union for pca_n_components type

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# No longer need to import DimensionalityReductionConfig here as it's not used directly
# from config.model_config_schema import DimensionalityReductionConfig

logger = logging.getLogger(__name__)


class PreprocessorBuilder:
    """
    Builds and manages the preprocessing pipeline (ColumnTransformer) for features.
    Handles scaling (StandardScaler, MinMaxScaler) and optional PCA dimensionality reduction.
    """
    def __init__(
        self,
        scaler_type: Optional[str] = None,
        pca_enabled: bool = False, # Changed from pca_config
        pca_n_components: Optional[Union[int, float]] = None, # Changed from pca_config
        features_to_use: Optional[List[str]] = None,
        # ADDED: Parameter for class balancing strategy
        class_balancing_strategy: Optional[str] = None,
        random_seed: int = 42 # Added for reproducible sampling
    ):
        """
        Initializes the PreprocessorBuilder.

        Args:
            scaler_type (Optional[str]): Type of scaler to use ('standard', 'minmax', None).
            pca_enabled (bool): Whether to apply PCA dimensionality reduction.
            pca_n_components (Optional[Union[int, float]]): Number of PCA components or variance explained.
            features_to_use (Optional[List[str]]): A list of feature column names to use.
                                                  If None, all numeric columns in X are used.
            class_balancing_strategy (Optional[str]): Strategy for class balancing ('oversampling', 'undersampling', None).
                                                      Used for models that handle balancing as a preprocessing step.
            random_seed (int): Random seed for reproducible sampling.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.scaler_type = scaler_type
        self.pca_enabled = pca_enabled
        self.pca_n_components = pca_n_components
        self.features_to_use = features_to_use
        self.class_balancing_strategy = class_balancing_strategy # Store the strategy
        self.random_seed = random_seed
        self.preprocessor: Optional[ColumnTransformer] = None # Will store the ColumnTransformer
        self.sampler: Optional[Union[SMOTE, RandomUnderSampler]] = None # Will store the sampler
        self.processed_feature_names: Optional[List[str]] = None

        self.logger.info(f"PreprocessorBuilder initialized with scaler: {self.scaler_type}, PCA enabled: {self.pca_enabled} (n_components: {self.pca_n_components}), Class Balancing: {self.class_balancing_strategy}")

    def build_and_fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ColumnTransformer:
        """
        Creates and fits a ColumnTransformer for preprocessing.
        Applies StandardScaler/MinMaxScaler to all numeric features in the specified subset or all numeric features.
        Optionally includes a PCA step after scaling if PCA is enabled.

        Args:
            X (pd.DataFrame): The input DataFrame containing features.
            y (Optional[pd.Series]): The target series, required if class balancing is enabled.

        Returns:
            ColumnTransformer: The fitted preprocessor.

        Raises:
            ValueError: If specified features are not found in X, or y is None when balancing is enabled.
        """
        if X.empty:
            self.logger.warning("Input DataFrame for preprocessor is empty. Cannot fit preprocessor.")
            self.preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
            self.processed_feature_names = []
            return self.preprocessor

        # Check for y if balancing is enabled
        if self.class_balancing_strategy and y is None:
            raise ValueError("Target series 'y' must be provided if class balancing is enabled.")

        # If a feature subset is provided, select only those columns
        if self.features_to_use is not None:
            missing_features = [feat for feat in self.features_to_use if feat not in X.columns]
            if missing_features:
                error_msg = f"Specified features not found in input data for preprocessor: {missing_features}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            X_subset = X[self.features_to_use].copy()
            self.logger.info(f"Preprocessor will be fitted on the specified feature subset: {self.features_to_use}")
        else:
            X_subset = X.select_dtypes(include=np.number).copy()
            self.logger.info("Preprocessor will be fitted on all numeric features in the input data.")


        # Select numeric features from the (potentially subsetted) DataFrame
        numeric_features = X_subset.select_dtypes(include=np.number).columns.tolist()

        if not numeric_features:
            self.logger.warning("No numeric features found in the input DataFrame (or subset) for preprocessing. Returning a passthrough transformer.")
            self.preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
            self.processed_feature_names = []
            return self.preprocessor

        # Define scaler step
        scaler_step: Optional[tuple] = None
        if self.scaler_type == 'standard':
            scaler_step = ('scaler', StandardScaler())
            self.logger.info("Using StandardScaler for numeric feature scaling.")
        elif self.scaler_type == 'minmax':
            scaler_step = ('scaler', MinMaxScaler())
            self.logger.info("Using MinMaxScaler for numeric feature scaling.")
        else:
            self.logger.info("No scaler type specified or supported. Skipping scaling step.")

        numeric_transformer_steps = [scaler_step] if scaler_step else []

        # Add PCA step if enabled
        if self.pca_enabled: # Use direct pca_enabled flag
            if self.pca_n_components is None:
                 self.logger.warning("PCA enabled but 'pca_n_components' is None. Defaulting to 0.95 variance explained.")
                 pca_params = {'n_components': 0.95} # Fallback to a default if not set
            else:
                 pca_params = {'n_components': self.pca_n_components}

            self.logger.info(f"Adding PCA step with params: {pca_params}")
            pca = PCA(**pca_params)
            numeric_transformer_steps.append(('pca', pca))


        numeric_transformer = Pipeline(steps=numeric_transformer_steps)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )

        self.logger.info("Fitting preprocessor...")
        self.preprocessor.fit(X_subset[numeric_features])
        self.logger.info("Preprocessor fitted.")

        try:
            self.processed_feature_names = self.preprocessor.get_feature_names_out().tolist()
            self.logger.info(f"Feature columns after preprocessing: {self.processed_feature_names}")

            if self.pca_enabled and 'pca' in self.preprocessor.named_transformers_['num'].named_steps:
                # Access the fitted PCA model through the pipeline structure
                fitted_pca = self.preprocessor.named_transformers_['num'].named_steps['pca']
                actual_n_components = fitted_pca.n_components_
                self.logger.info(f"PCA reduced features to {actual_n_components} components.")
                self.logger.info(f"Explained variance ratio: {fitted_pca.explained_variance_ratio_.sum():.4f}")

        except AttributeError:
            self.logger.warning("get_feature_names_out not available. Assuming feature columns are the original numeric features.")
            self.processed_feature_names = numeric_features

        return self.preprocessor

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input DataFrame using the fitted preprocessor.

        Args:
            X (pd.DataFrame): The input DataFrame containing features.

        Returns:
            np.ndarray: The transformed feature array.

        Raises:
            RuntimeError: If the preprocessor has not been fitted.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted. Call build_and_fit() first.")

        # If a feature subset was used during fit, ensure consistent columns for transform
        if self.features_to_use is not None:
            missing_features = [feat for feat in self.features_to_use if feat not in X.columns]
            if missing_features:
                raise ValueError(f"Input data for transformation is missing features that were used during fitting: {missing_features}")
            X_transform_subset = X[self.features_to_use].copy()
        else:
            X_transform_subset = X.select_dtypes(include=np.number).copy()

        self.logger.info("Transforming data using fitted preprocessor...")
        X_transformed = self.preprocessor.transform(X_transform_subset)
        self.logger.info("Data transformed.")
        return X_transformed

    def fit_resample(self, X_transformed: np.ndarray, y_mapped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies class balancing (oversampling or undersampling) to the transformed data.
        This method is intended to be called after feature transformation,
        but before sequence creation for LSTM models.

        Args:
            X_transformed (np.ndarray): The feature array after preprocessing (scaling, PCA).
            y_mapped (np.ndarray): The target array (labels mapped to 0, 1, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The resampled feature and target arrays.

        Raises:
            RuntimeError: If class balancing strategy is invalid.
        """
        if self.class_balancing_strategy == 'oversampling':
            self.logger.info(f"Applying Oversampling (SMOTE) for training data. Original counts: {Counter(y_mapped)}")
            self.sampler = SMOTE(random_state=self.random_seed)
            X_resampled, y_resampled = self.sampler.fit_resample(X_transformed, y_mapped)
            self.logger.info(f"Oversampling applied. New counts: {Counter(y_resampled)}")
            return X_resampled, y_resampled
        elif self.class_balancing_strategy == 'undersampling':
            self.logger.info(f"Applying Undersampling (RandomUnderSampler) for training data. Original counts: {Counter(y_mapped)}")
            self.sampler = RandomUnderSampler(random_state=self.random_seed)
            X_resampled, y_resampled = self.sampler.fit_resample(X_transformed, y_mapped)
            self.logger.info(f"Undersampling applied. New counts: {Counter(y_resampled)}")
            return X_resampled, y_resampled
        elif self.class_balancing_strategy is None:
            self.logger.info("No class balancing strategy specified or applied for training data.")
            return X_transformed, y_mapped
        else:
            raise RuntimeError(f"Invalid class balancing strategy: {self.class_balancing_strategy}")

