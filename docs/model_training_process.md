# **Model Training Process**

This document details the end-to-end process for training, evaluating, and saving machine learning models within the algorithmic trading bot project. The primary script for this workflow is scripts/train\_model.py, which leverages functionalities from utils/model\_trainer.py and utils/data\_manager.py.

## **1\. Overview of the Training Pipeline**

The model training pipeline is designed to:

1. **Load and Prepare Data**: Fetch processed features and generated labels, align them, and split them into training, validation (optional for LSTM), and test sets.  
2. **Hyperparameter Tuning (Optional)**: Perform randomized search cross-validation to find optimal hyperparameters for tree-based models (RandomForest, XGBoost).  
3. **Model Training**: Train the selected machine learning model (RandomForest, XGBoost, or LSTM) using the prepared training data.  
4. **Model Evaluation**: Assess the trained model's performance on a held-out test set using various metrics.  
5. **Model Saving**: Persist the trained model, its preprocessor, and associated metadata for future use in backtesting or live trading.

## **2\. Data Loading and Splitting (load\_and\_split\_data function)**

The load\_and\_split\_data function within train\_model.py is responsible for preparing the dataset for model training.

### **2.1. Data Loading**

* It uses DataManager.load\_data to load two main components:  
  * **Processed Features**: Data from data/processed/ containing all engineered features. The open\_time column is explicitly dropped if present, as it's not a feature for the model.  
  * **Labeled Data**: Data from data/labeled/ containing the target labels (-1 for Short, 0 for Neutral, 1 for Long).  
* **Index Alignment**: The loaded features and labels DataFrames are merged using an inner join on their DatetimeIndex to ensure perfect alignment of timestamps.

### **2.2. Feature Selection**

* The function supports an optional features\_to\_use argument. If provided, only these specified columns will be used as features for training.  
* If features\_to\_use is None, the script automatically infers all columns in the processed\_data (excluding known non-feature columns like OHLCV data, volume, and the label itself) as features. This ensures flexibility and avoids hardcoding feature lists.

### **2.3. Data Cleaning and Validation**

* **NaN/Inf Removal**: Crucially, rows containing any NaN (Not a Number) or Inf (Infinity) values in the selected features or the 'label' column are removed. This is a critical step to ensure data quality and prevent errors during model training.  
* **Label Validation**: Ensures that the 'label' column exists and its values are converted to integers. It also filters out any labels that are not \-1, 0, or 1\.  
* **Type Checks**: Basic checks are performed to ensure DataFrames have DatetimeIndex for proper time-series handling.

### **2.4. Time-Series Split**

* The cleaned and prepared data is split into training, validation (optional), and test sets using a time-series split approach. This means the data is divided chronologically, preserving the temporal order.  
* The train\_ratio argument (defaulting to 0.8) determines the proportion of data allocated for training. A val\_ratio (defaulting to 0.1) is used to create a validation set, primarily for LSTM models. The remaining data forms the test set.  
* The function returns X\_train, X\_val, X\_test (features) and y\_train, y\_val, y\_test (labels), along with the full cleaned datasets (X\_full\_cleaned, y\_full\_cleaned) which are used for hyperparameter tuning.

## **3\. Hyperparameter Tuning (run\_tuning function)**

The run\_tuning function orchestrates the hyperparameter optimization process for tree-based models (RandomForest, XGBoost). LSTM models currently skip this step, using their default parameters.

### **3.1. Randomized Search with TimeSeriesSplit**

* **Strategy**: It employs RandomizedSearchCV from scikit-learn to efficiently explore a defined hyperparameter space.  
* **Cross-Validation**: TimeSeriesSplit is used for cross-validation. This is vital for time-series data as it ensures that validation folds always come *after* the training folds, preventing data leakage and providing a more realistic evaluation of model generalization. The number of splits (cv\_n\_splits) is configurable.  
* **Scoring Metric**: The tuning\_scoring\_metric (e.g., 'f1\_macro') is used to evaluate model performance during tuning, guiding the search towards optimal parameters.

### **3.2. Preprocessing and Sampling in Tuning Pipeline**

* A ColumnTransformer (for preprocessing) and optional imblearn samplers (RandomUnderSampler or SMOTE) are integrated into an imblearn.pipeline.Pipeline. This ensures that preprocessing and class balancing are consistently applied within each cross-validation fold during tuning, preventing data leakage from the validation sets.  
* The class\_balancing strategy (undersampling or oversampling) is taken from MODEL\_CONFIG.

### **3.3. Output**

* Upon completion, run\_tuning returns the best\_params\_ found by RandomizedSearchCV, which are then used to update the model\_specific\_config for the actual model training.

## **4\. Model Training (ModelTrainer class and train method)**

The ModelTrainer class (utils/model\_trainer.py) encapsulates the logic for building, training, evaluating, and saving different model types.

### **4.1. Model Initialization and Building**

* The ModelTrainer is initialized with a config dictionary specific to the chosen model type (RandomForest, XGBoost, or LSTM).  
* **Feature Columns Tracking**: It stores feature\_columns\_original (the names of features before any preprocessing) and feature\_columns\_processed (names after preprocessing) for consistent data handling during prediction and analysis.  
* **Preprocessor Creation**: The \_create\_preprocessor method generates a ColumnTransformer that applies StandardScaler to all numeric features. This preprocessor is fitted on the training data.  
* **LSTM Model Architecture (\_build\_lstm\_model)**: If model\_type is 'lstm', a Sequential Keras model is built with configurable LSTM layers, dropout, and a softmax output layer for ternary classification. The number of input features is determined by the output of the preprocessor, and the sequence\_length (from model\_params) defines the lookback window for the LSTM.

### **4.2. Training Process (train method)**

* **Data Transformation**: For all model types, the input features (X\_train) are passed through the fitted preprocessor. For LSTM models, the preprocessed data is further transformed into sequences using \_prepare\_lstm\_sequences.  
* **Class Imbalance Handling**:  
  * For scikit-learn compatible models (RandomForest, XGBoost), imblearn samplers (e.g., RandomUnderSampler, SMOTE) are integrated into the Pipeline to address class imbalance in the training data, if configured via class\_balancing in model\_params.  
  * For LSTM models, class\_weight can be calculated and applied during training to give more importance to minority classes.  
* **Model Fitting**:  
  * **Scikit-learn Models**: The imblearn.pipeline.Pipeline (containing the preprocessor, optional sampler, and the model) is fitted directly on X\_train and y\_train (mapped to integers 0, 1, 2).  
  * **LSTM Models**: The Keras model is trained using model.fit with the prepared sequences. Early stopping and learning rate reduction callbacks are optionally configured based on model\_params.

## **5\. Model Evaluation (evaluate method)**

The evaluate method in ModelTrainer assesses the performance of the trained model on the test set.

* **Prediction**: The trained model (or pipeline) makes predictions on the X\_test data. For LSTM, this involves preparing sequences from the scaled test data.  
* **Metrics**: It calculates standard classification metrics:  
  * overall\_accuracy  
  * balanced\_accuracy\_score (important for imbalanced datasets)  
  * classification\_report (providing precision, recall, f1-score for each class)  
  * confusion\_matrix  
* **NaN Handling**: It explicitly handles NaN values in predictions, removing corresponding samples from the test set before calculating metrics to ensure robust evaluation.  
* **Output**: Returns a dictionary containing all calculated metrics.

## **6\. Model Saving (save method)**

The save method in ModelTrainer uses DataManager to persist the trained model and its associated components.

* **Artifacts Saved**:  
  * **Metadata**: A dictionary containing crucial information about the model, including its type, original and processed feature column names, label mappings, model parameters, sequence length (for LSTM), and the optional feature subset used. This metadata is essential for correctly loading and using the model later.  
  * **Model/Pipeline**:  
    * For scikit-learn models (RandomForest, XGBoost), the entire imblearn.pipeline.Pipeline object is saved using joblib via DataManager.save\_model\_artifact with artifact\_type='pipeline'. The preprocessor is implicitly saved as part of this pipeline.  
    * For LSTM models, the Keras model (tf.keras.Model) is saved in the .keras format, and the ColumnTransformer preprocessor is saved separately using joblib with artifact\_type='preprocessor'.  
* **Path Management**: DataManager handles the construction of appropriate file paths within the models/trained\_models/ directory, organizing models by symbol, interval, and model\_key.

## **7\. Command-Line Usage (scripts/train\_model.py)**

The scripts/train\_model.py script provides a command-line interface to execute the training pipeline.

### **7.1. Arguments**

* \--symbol: Trading pair symbol (e.g., ADAUSDT).  
* \--interval: Time interval for candles (e.g., 5m, 1h).  
* \--model: Type of model to train (random\_forest, xgboost, lstm). Defaults to xgboost.  
* \--train\_ratio: Fraction of data to use for training (0.0 to 1.0, exclusive). Defaults to 0.8.  
* \--skip\_tuning: A flag to skip hyperparameter tuning and use default parameters from config.params.MODEL\_CONFIG.  
* \--features: Optional list of specific feature names to use for training. If not provided, all available features are used.

### **7.2. Examples**

\# Train the default XGBoost model for ternary classification:  
python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h

\# Train the RandomForest model for ternary classification:  
python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest

\# Train the LSTM model (requires TensorFlow):  
python scripts/train\_model.py \--symbol ETHUSDT \--interval 15m \--model lstm \--train\_ratio 0.7

\# Train with tuning (default for non-LSTM):  
python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest  
python \-m scripts.train\_model \--symbol ADAUSDT \--interval 5m \--model xgboost

\# Train skipping tuning:  
python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest \--skip\_tuning

\# Train using a specific subset of features:  
python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h \--features ema\_10 rsi\_14 macd

### **7.3. Prerequisites**

Before running train\_model.py, ensure the following:

* **Processed and Labeled Data**: You must have generated processed data (with features) and labeled data (containing labels \-1, 0, 1\) in your data/processed/ and data/labeled/ directories, respectively. This typically involves running scripts/fetch\_data.py, scripts/generate\_features.py, and scripts/create\_labels.py.  
* **Configuration**: config/params.py (with MODEL\_CONFIG, GENERAL\_CONFIG, LABELING\_CONFIG) and config/paths.py must be correctly configured. This includes defining trained\_models\_dir and trained\_model\_pattern in paths.py.  
* **ATR Column for Triple Barrier**: If using the triple\_barrier labeling strategy with volatility adjustment, the feature generation script must produce an ATR column named atr\_{lookback} (e.g., atr\_14) matching the vol\_adj\_lookback parameter in LABELING\_CONFIG, and this column must be present in the processed data file.  
* **TensorFlow for LSTM**: If you intend to train LSTM models, ensure TensorFlow is installed in your environment.

## **8\. Model Loading for Analysis and Prediction (ModelTrainer.load method)**

The ModelTrainer.load method is crucial for retrieving a previously trained model and its components for tasks like analysis (scripts/analyze\_model.py) or live trading (trading\_bot.py).

### **8.1. Loading Process**

* It uses DataManager.load\_model\_artifact to load the metadata, the model/pipeline, and the preprocessor (if separate, like for LSTM).  
* **Metadata First**: The metadata is loaded first, as it contains essential information (like feature\_columns\_original, feature\_columns\_processed, label\_map, model\_params, sequence\_length) that is used to correctly initialize the ModelTrainer instance and interpret the loaded model artifacts.  
* **Conditional Loading**:  
  * For LSTM models, it loads the Keras model (.keras file) and the ColumnTransformer preprocessor (.joblib file) separately.  
  * For scikit-learn models (RandomForest, XGBoost), it loads the entire imblearn.pipeline.Pipeline (.joblib file), from which the preprocessor and the final model can be extracted.  
* **Error Handling**: Robust error handling is in place to manage cases where files are not found or loading fails, providing informative messages.

### **8.2. Usage in analyze\_model.py**

The scripts/analyze\_model.py script demonstrates how to load a trained model for post-training analysis:

* It calls load\_trained\_model\_and\_preprocessor which in turn uses ModelTrainer.load.  
* The loaded ModelTrainer instance is then used to make predictions (trainer.predict, trainer.predict\_proba) on the test set and perform various analyses (confusion matrix, feature importance, probability distributions, ROC AUC, Precision-Recall, Calibration Curve).

## **9\. ModelTrainer Prediction Methods (predict and predict\_proba)**

The ModelTrainer class provides predict and predict\_proba methods for making inferences on new, unseen data.

### **9.1. predict(X: pd.DataFrame)**

* Takes a DataFrame X of new features as input. This DataFrame should contain all original features that the model was trained on.  
* **Internal Preprocessing**: The method internally uses the *fitted* preprocessor to transform the input X before making predictions. This ensures consistency with the data transformation applied during training.  
* **LSTM Specifics**: For LSTM models, the preprocessed data is further converted into sequences of the defined sequence\_length. Predictions are then aligned to the original timestamp of the *end* of each sequence.  
* **Output**: Returns a pd.Series of predicted labels (-1, 0, or 1), aligned to the original index of the input X.

### **9.2. predict\_proba(X: pd.DataFrame)**

* Takes a DataFrame X of new features as input, similar to predict.  
* **Internal Preprocessing**: Also uses the *fitted* preprocessor to transform the input X.  
* **LSTM Specifics**: For LSTM, prepares sequences and then calls the Keras model's predict method to get raw probabilities. These probabilities are then aligned to the original timestamp of the *end* of each sequence.  
* **Output**: Returns an Optional\[pd.DataFrame\] containing the predicted probabilities for each class (-1, 0, 1), aligned to the original index of the input X. Returns None if the model does not support probability prediction or if an error occurs.

These methods are designed to be robust, handling data transformations and model-specific prediction logic internally, simplifying their use in other parts of the system (e.g., trading\_bot.py for live predictions).