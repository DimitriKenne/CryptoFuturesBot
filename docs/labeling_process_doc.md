This document provides a detailed overview of the labeling process within the project, explaining its purpose, key components, workflow, configuration, and output.

## **1\. Overview of the Labeling Process**

The labeling process is a critical step in preparing financial time-series data for machine learning models. Its primary goal is to transform raw price movements into discrete, actionable signals (labels) that a model can learn to predict. These labels typically represent desired outcomes, such as "buy," "sell," or "hold."

**Why is Labeling Important?**

* **Supervised Learning:** Machine learning models for trading often rely on supervised learning, which requires labeled data (input features paired with a known output label).  
* **Defining "Success":** Labeling allows you to define what constitutes a "successful" or "unsuccessful" trade opportunity based on your specific strategy goals (e.g., a certain percentage profit, avoiding a stop-loss).  
* **Handling Noise:** Financial data is inherently noisy. Labeling strategies help to filter out minor fluctuations and focus on significant price movements or patterns.

**General Steps:**

1. **Load Processed Data:** The labeling process begins with clean, processed OHLCV (Open, High, Low, Close, Volume) data, which often includes engineered technical features.  
2. **Define Labeling Strategy:** A specific strategy (e.g., directional, triple-barrier) is chosen based on the desired trade outcome definition.  
3. **Calculate Raw Labels:** The chosen strategy applies its logic to the data to generate initial, "raw" labels.  
4. **Apply Label Propagation/Smoothing:** To account for trade holding periods or to smooth out rapid label changes, a propagation mechanism is applied.  
5. **Save Labeled Data:** The final labels are saved to disk, typically as a DataFrame containing only the label column and the original timestamp index.  
6. **Analyze Label Distribution and Performance:** After generation, the labels are analyzed to understand their distribution, potential profitability, and characteristics, which helps in refining the labeling strategy and informing model training.

## **2\. Key Components**

The labeling process is orchestrated by several interconnected Python files:

### **2.1. Configuration Files (config/)**

* **config/params.py**:  
  * **LABELING\_CONFIG**: This dictionary holds general and strategy-specific parameters for the labeling process. Examples include min\_holding\_period (for label propagation) and parameters unique to each labeling strategy (e.g., forward\_window\_bars for directional ternary, max\_holding\_bars for triple barrier).  
  * **STRATEGY\_CONFIG**: While primarily for overall trading strategy, it can contain parameters that influence labeling, such as volatility\_regime\_filter\_enabled or analysis\_future\_horizons for the analysis part.  
* **config/paths.py**:  
  * Defines the directory structure for all project artifacts.  
  * Crucially, it specifies where raw, processed, and labeled data are stored.  
  * It defines patterns for output files, including:  
    * LABELED\_DATA\_PATTERN: For saving the generated labeled data files (e.g., {symbol}\_{interval}\_labeled.parquet). **Note: This pattern no longer includes the label\_strategy in the filename, as per your preference.**  
    * LABELING\_ANALYSIS\_BASE\_DIR: The root directory for all labeling analysis results.  
    * LABELING\_STRATEGY\_ANALYSIS\_DIR\_PATTERN\_STR: A string pattern (e.g., .../results/analysis/labeling/{label\_strategy}) used to create strategy-specific subfolders for analysis outputs.  
    * LABELING\_ANALYSIS\_PLOT\_PATTERN and LABELING\_ANALYSIS\_TABLE\_PATTERN: Patterns for the filenames of plots and tables within these strategy-specific folders.

### **2.2. Labeling Strategy Implementations (utils/labeling\_strategies/)**

All concrete labeling strategies inherit from BaseLabelingStrategy, ensuring a consistent interface.

* **utils/labeling\_strategies/base\_strategy.py**:  
  * An Abstract Base Class (ABC) that defines the calculate\_raw\_labels abstract method.  
  * Provides common utility methods like \_validate\_input\_df to ensure input DataFrame integrity.  
  * Establishes a shared logger and FLOAT\_EPSILON for numerical stability.  
* **utils/labeling\_strategies/directional\_ternary.py**:  
  * **Purpose:** Labels data based on a fixed future price movement.  
  * **Logic:** Looks forward\_window\_bars into the future and assigns a label (1 for long, \-1 for short, 0 for neutral) if the price change exceeds a price\_threshold\_pct.  
  * **Output:** 1 (long), \-1 (short), or 0 (neutral).  
* **utils/labeling\_strategies/triple\_barrier.py**:  
  * **Purpose:** Labels data based on which of three barriers (Take Profit, Stop Loss, or Time) is hit first.  
  * **Logic:** For each bar, it defines a potential trade exit window (max\_holding\_bars) and price barriers (Take Profit and Stop Loss). These barriers can be fixed percentages or dynamically adjusted based on volatility (e.g., ATR). The label is determined by the first barrier encountered.  
  * **Output:** 1 (long TP hit), \-1 (short TP hit), or 0 (SL hit or timeout).  
* **utils/labeling\_strategies/max\_return\_quantile.py**:  
  * **Purpose:** Labels data based on extreme (maximum favorable or maximum adverse) returns within a lookahead window, compared against quantile-derived thresholds.  
  * **Logic:** Calculates the maximum positive and negative returns within a quantile\_forward\_window\_bars window. If these extreme returns exceed a quantile\_threshold\_pct (e.g., the 90th percentile of all absolute returns), a label is assigned.  
  * **Output:** 1 (significant positive return), \-1 (significant negative return), or 0 (otherwise).  
* **utils/labeling\_strategies/ema\_return\_percentile.py**:  
  * **Purpose:** Labels data based on future returns relative to a backward Exponential Moving Average (EMA), with thresholds derived from percentiles of Open-Close changes.  
  * **Logic:** Computes a "future return" based on the f\_window future close price and a b\_window backward EMA. It then uses percentiles of historical Open-Close changes to define dynamic alpha and beta thresholds. Labels are assigned if the future return falls within specific ranges defined by these thresholds, considering transaction fees.  
  * **Output:** 1 (buy signal), \-1 (sell signal), or 0 (neutral).

### **2.3. Orchestration and Analysis Utilities (utils/)**

* **utils/label\_generator.py**:  
  * **Role:** The central orchestrator for label generation.  
  * **Functionality:**  
    * Initializes the chosen BaseLabelingStrategy based on LABELING\_CONFIG\['label\_type'\].  
    * **calculate\_labels(df)**: This is the core method. It takes the processed DataFrame (df), calls the selected strategy's calculate\_raw\_labels method, and then applies **label propagation (smoothing)**.  
    * **Label Propagation:** Uses min\_holding\_period from LABELING\_CONFIG to propagate a non-zero label forward. If a signal (1 or \-1) appears, it maintains that label for at least min\_holding\_period bars, unless a conflicting signal appears earlier. This helps in creating more stable and realistic trade entry/exit points.  
    * **Index Handling:** Crucially, it includes logic to ensure that the DataFrame returned by calculate\_labels always has a correct DatetimeIndex that aligns with the input df.  
* **utils/label\_analyzer.py**:  
  * **Role:** A dedicated utility class for performing various analyses on labeled data.  
  * **Functionality:**  
    * Encapsulates all analysis functions: analyze\_label\_streaks, analyze\_mfe\_mae, analyze\_future\_returns, analyze\_regime\_profitability.  
    * **perform\_all\_analyses(df\_combined, symbol, interval, label\_strategy, future\_horizons)**: This method orchestrates the execution of all individual analysis functions.  
    * **Dynamic Directory Creation:** It uses the label\_strategy and the LABELING\_STRATEGY\_ANALYSIS\_DIR\_PATTERN\_STR from paths.py to create a unique subfolder for each labeling strategy's analysis results.  
    * Handles saving plots (PNG) and summary tables (CSV) to these specific folders using the defined patterns.

### **2.4. Main Scripts (scripts/)**

* **scripts/create\_labels.py**:  
  * **Purpose:** The primary script to run the entire label generation and initial analysis pipeline.  
  * **Workflow:**  
    1. Loads configuration from params.py and paths.py.  
    2. Initializes DataManager and LabelGenerator.  
    3. Loads processed data using DataManager.load\_data(data\_type='processed').  
    4. Calls LabelGenerator.calculate\_labels() to get the final labels (as a DataFrame with only the 'label' column).  
    5. **Crucially, it merges the original df\_input (with all OHLCV and features) with the generated label column.** This combined DataFrame (df\_combined\_for\_analysis) is essential for the LabelAnalyzer to perform analyses that require more than just the label (e.g., MFE/MAE needs high/low, regime analysis needs volatility\_regime).  
    6. Saves the labeled\_data\_to\_save (which is just the label column with the correct index) using DataManager.save\_data(data\_type='labeled'). **The name\_suffix is intentionally omitted here, so the file is saved with a generic name like {symbol}\_{interval}\_labeled.parquet.**  
    7. Initializes LabelAnalyzer and calls its perform\_all\_analyses method, passing the df\_combined\_for\_analysis and the label\_strategy so analysis results are saved to the correct, dedicated *strategy-specific folder*.  
* **scripts/analyze\_labels.py**:  
  * **Purpose:** A standalone script to perform comprehensive analyses on *already generated* labeled data.  
  * **Workflow:**  
    1. Parses command-line arguments, including \--label-strategy (which is used *only* for organizing the analysis output, not for loading the labeled file).  
    2. Loads both the processed data and the generic labeled data file (e.g., ADAUSDT\_5m\_labeled.parquet) via DataManager.load\_data(data\_type='labeled'). **The name\_suffix is intentionally omitted here.**  
    3. Merges the loaded processed and labeled DataFrames into a df\_combined DataFrame.  
    4. Initializes LabelAnalyzer and calls its perform\_all\_analyses method, passing the df\_combined and the label\_strategy to ensure analysis results are saved correctly to the strategy-specific folder.  
  * **Benefit:** Allows for re-analysis of existing labels without re-generating them, and for running specific analyses with different parameters (e.g., future\_horizons).  
* **scripts/train\_model.py**:  
  * **Purpose:** The script responsible for training machine learning models.  
  * **Workflow:**  
    1. Parses command-line arguments (no longer requires \--label-strategy as an input for loading labels).  
    2. Loads the processed data and the generic labeled data file (e.g., ADAUSDT\_5m\_labeled.parquet) via DataManager.load\_data(data\_type='labeled'). **The name\_suffix is intentionally omitted here.**  
    3. Splits the data into training, validation, and test sets.  
    4. (Optionally) performs hyperparameter tuning.  
    5. Initializes and trains the model using ModelTrainer.  
    6. Evaluates the trained model.  
    7. Saves the trained model and its metadata using DataManager.  
  * **Note:** This script assumes that a single, desired labeled data file has already been generated by create\_labels.py and is available in the data/labeled/ directory under its generic name.

## **3\. Workflow Description**

The typical workflow for labeling involves:

1. **Data Fetching:** (e.g., scripts/fetch\_data.py) \- Obtains raw OHLCV data.  
2. **Feature Engineering:** (e.g., scripts/generate\_features.py) \- Transforms raw data into processed data with technical indicators and other features (including volatility\_regime if used by a strategy). This processed data is saved to data/processed/.  
3. **Label Generation and Analysis:** (using scripts/create\_labels.py)  
   * create\_labels.py loads the processed data.  
   * It uses LabelGenerator to apply the chosen labeling strategy (e.g., ema\_return\_percentile) to create raw labels.  
   * LabelGenerator then applies label propagation based on min\_holding\_period.  
   * The final label column is saved as a separate .parquet file in data/labeled/ (e.g., ADAUSDT\_5m\_labeled.parquet).  
   * Crucially, create\_labels.py then combines the original processed data with these new labels and passes this comprehensive DataFrame to LabelAnalyzer.  
   * LabelAnalyzer performs various analyses (streaks, MFE/MAE, future returns, regime profitability) and saves the output (plots and tables) to a dedicated subfolder like results/analysis/labeling/ema\_return\_percentile/.  
4. **Independent Label Analysis:** (using scripts/analyze\_labels.py)  
   * If you want to re-analyze existing labels or run specific analyses without regenerating them, you can use analyze\_labels.py.  
   * This script loads the processed data and the generic labeled data file (e.g., ADAUSDT\_5m\_labeled.parquet).  
   * It combines them and passes the combined DataFrame to LabelAnalyzer to perform the desired analyses, saving results to the strategy-specific folder.  
5. **Model Training:** (using scripts/train\_model.py)  
   * train\_model.py loads the processed data and the generic labeled data file.  
   * It then proceeds with model training, evaluation, and saving.

## **4\. Configuration Details**

The primary configuration for labeling resides in config/params.py:

\# Example snippet from config/params.py  
LABELING\_CONFIG \= {  
    'label\_type': 'ema\_return\_percentile', \# Default strategy to use  
    'min\_holding\_period': 1, \# Bars to propagate a label forward

    \# \--- Directional Ternary Strategy Parameters \---  
    'forward\_window\_bars': 20, \# Lookahead window for price change  
    'price\_threshold\_pct': 1.5, \# Percentage change required for a signal

    \# \--- Triple Barrier Strategy Parameters \---  
    'max\_holding\_bars': 100, \# Max duration for a trade  
    'fixed\_take\_profit\_pct': 3, \# Fixed TP percentage  
    'fixed\_stop\_loss\_pct': 1, \# Fixed SL percentage  
    'use\_volatility\_adjustment': True, \# Use ATR for dynamic barriers  
    'vol\_adj\_lookback': 20, \# ATR lookback period  
    'alpha\_take\_profit': 9, \# ATR multiplier for TP  
    'alpha\_stop\_loss': 3, \# ATR multiplier for SL

    \# \--- Max Return Quantile Strategy Parameters \---  
    'quantile\_forward\_window\_bars': 60, \# Lookahead window for max returns  
    'quantile\_threshold\_pct': 60.0, \# Percentile threshold for significant returns

    \# \--- EMA Return Percentile Strategy Parameters \---  
    'f\_window': 2, \# Forward lookahead window for future close  
    'b\_window': 25, \# Backward EMA window for reference price  
    'fee': 0.0005, \# Transaction fee  
    'beta\_increment': 0.1, \# Increment factor for beta per forward window  
    'lower\_percentile': 85, \# Percentile for alpha  
    'upper\_percentile': 99.9, \# Percentile for beta  
}

STRATEGY\_CONFIG \= {  
    \# ... other strategy configs ...  
    'volatility\_regime\_filter\_enabled': True, \# If True, regime parameters are merged into LABELING\_CONFIG  
    'volatility\_regime\_max\_holding\_bars': {0: 5, 1: 21, 2: 11}, \# Example regime-specific holding periods  
    'allow\_trading\_in\_volatility\_regime': {0: False, 1: True, 2: False}, \# Example regime-specific trading allowance

    'analysis\_future\_horizons': \[5, 10, 20, 50, 100\], \# Horizons for future returns analysis  
}

* **label\_type**: Specifies which labeling strategy to use. This must match a key in STRATEGY\_MAP in label\_generator.py.  
* **min\_holding\_period**: An integer indicating how many bars a non-zero label should be propagated forward. This smooths signals and ensures a minimum trade duration.  
* **Strategy-Specific Parameters**: Each strategy has its own set of parameters (e.g., forward\_window\_bars, price\_threshold\_pct for directional ternary; max\_holding\_bars, fixed\_take\_profit\_pct, use\_volatility\_adjustment for triple barrier, etc.). These are read by the respective strategy classes.  
* **analysis\_future\_horizons**: (from STRATEGY\_CONFIG) A list of integer bar counts for which future returns analysis will be performed.

## **5\. Output and Storage**

The labeling process generates two main types of output:

1. **Labeled Data Files:**  
   * **Location:** data/labeled/  
   * **Filename Pattern:** {symbol}\_{interval}\_labeled.parquet (e.g., ADAUSDT\_5m\_labeled.parquet)  
   * **Content:** A Parquet file containing a pandas DataFrame with the original DatetimeIndex (named 'timestamp') and a single column named 'label' (with values 1, \-1, or 0).  
   * **Note:** Only one labeled file for a given symbol and interval exists at a time. Running create\_labels.py with a new strategy will *overwrite* the previous labeled data file.  
2. **Analysis Results (Plots and Tables):**  
   * **Location:** results/analysis/labeling/{label\_strategy}/ (e.g., results/analysis/labeling/ema\_return\_percentile/)  
   * **Filenames:**  
     * **Plots:** PNG images (e.g., ADAUSDT\_5m\_label\_streak\_duration\_plot.png, ADAUSDT\_5m\_mfe\_mae\_distribution\_plot.png).  
     * **Tables:** CSV files containing summary statistics (e.g., ADAUSDT\_5m\_label\_streak\_duration.csv, ADAUSDT\_5m\_mfe\_mae\_summary.csv).  
   * **Content:** Visualizations and tabular summaries of label distribution, streak durations, Maximum Favorable Excursion (MFE), Maximum Adverse Excursion (MAE), and future returns by label and volatility regime. These are saved in strategy-specific subfolders to preserve the analysis results for each labeling approach you test.

## **6\. Usage Examples**

### **6.1. Generating Labels and Performing Initial Analysis**

To generate labels and automatically run the initial analysis for a specific symbol, interval, and labeling strategy:

python \-m scripts.create\_labels \--symbol ADAUSDT \--interval 5m \--label-strategy ema\_return\_percentile

This command will:

* Load processed data for ADAUSDT at 5m interval.  
* Apply the ema\_return\_percentile labeling strategy.  
* Save the labeled data to data/labeled/ADAUSDT\_5m\_labeled.parquet (overwriting any existing file).  
* Generate analysis plots and tables, saving them to results/analysis/labeling/ema\_return\_percentile/.

### **6.2. Analyzing Existing Labeled Data**

To perform analysis on an already generated labeled dataset without regenerating the labels:

python \-m scripts.analyze\_labels \--symbol ADAUSDT \--interval 5m \--label-strategy ema\_return\_percentile \--future-horizons 10 30 60

This command will:

* Load processed data for ADAUSDT at 5m interval.  
* Load the *existing* labeled data from data/labeled/ADAUSDT\_5m\_labeled.parquet.  
* Perform all defined analyses (streaks, MFE/MAE, future returns for horizons 10, 30, 60, and volatility regime profitability).  
* Save the analysis results to results/analysis/labeling/ema\_return\_percentile/.

### **6.3. Training a Model with the Labeled Data**

To train a model using the currently saved labeled data:

python \-m scripts.train\_model \--symbol BTCUSDT \--interval 1h \--model xgboost \--train\_ratio 0.8

This command will:

* Load processed data for BTCUSDT at 1h interval.  
* Load the *existing* labeled data from data/labeled/BTCUSDT\_1h\_labeled.parquet.  
* Train an XGBoost model, evaluate it, and save the model artifacts.

This comprehensive labeling process provides a robust framework for defining, generating, and evaluating trading signals for your machine learning models.