# utils/training/data_sequencer.py

import logging
from typing import Tuple, Any # Added Any for tf_module

import numpy as np
# Do NOT import tensorflow directly here. It will be passed via constructor.

logger = logging.getLogger(__name__)

class DataSequencer:
    """
    Prepares data into sequences suitable for LSTM models.
    """
    def __init__(self, sequence_length: int, tf_module: Any): # Accept tf_module
        """
        Initializes the DataSequencer.

        Args:
            sequence_length (int): The number of time steps in each sequence.
            tf_module (Any): The TensorFlow module itself.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("Sequence length must be a positive integer.")
        self.sequence_length = sequence_length
        self._tf = tf_module # Store TensorFlow module
        if self._tf is None:
            self.logger.warning("TensorFlow module is None in DataSequencer. to_categorical calls might fail.")
        self.logger.info(f"DataSequencer initialized with sequence length: {self.sequence_length}")

    def create_sequences(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares data into sequences for LSTM training/prediction.

        Args:
            X_data (np.ndarray): Feature data (numpy array).
            y_data (np.ndarray): Label data (numpy array, 0, 1, or 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - X_sequences (np.ndarray): Feature sequences for LSTM input.
                - y_sequences_one_hot (np.ndarray): One-hot encoded labels for the end of each sequence.
        """
        if X_data.ndim != 2:
            raise ValueError(f"X_data must be 2-dimensional (samples, features), got {X_data.ndim} dimensions.")
        if y_data.ndim != 1:
            raise ValueError(f"y_data must be 1-dimensional (samples,), got {y_data.ndim} dimensions.")
        if X_data.shape[0] != y_data.shape[0]:
            raise ValueError("X_data and y_data must have the same number of samples.")

        n_samples = X_data.shape[0]
        n_features = X_data.shape[1]

        if n_samples < self.sequence_length:
            self.logger.warning(f"Not enough data points ({n_samples}) to create sequences of length {self.sequence_length}. Returning empty arrays.")
            return np.empty((0, self.sequence_length, n_features)), np.empty((0, 3)) # 3 classes for one-hot

        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length - 1, n_samples):
            X_sequences.append(X_data[i - self.sequence_length + 1 : i + 1])
            y_sequences.append(y_data[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Use the TensorFlow module passed in the constructor
        if self._tf is None:
            raise ImportError("TensorFlow module is not available in DataSequencer for one-hot encoding.")

        y_sequences_one_hot = self._tf.keras.utils.to_categorical(y_sequences, num_classes=3)

        self.logger.info(f"Prepared {len(X_sequences)} LSTM sequences with shape {X_sequences.shape}")
        self.logger.info(f"Prepared {len(y_sequences_one_hot)} LSTM labels with shape {y_sequences_one_hot.shape}")

        return X_sequences, y_sequences_one_hot

