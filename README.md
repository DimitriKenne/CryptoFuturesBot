# **Algorithmic Futures Trading Bot: Machine Learning-Driven Trading Strategy**

This project presents a sophisticated algorithmic trading bot designed for futures markets, leveraging advanced machine learning techniques for predictive signal generation and robust risk management. Built with a modular and extensible architecture, this bot facilitates end-to-end automation of trading strategies, from data acquisition and feature engineering to model training, backtesting, and live execution.

## **Motivation & Background**

This project was initiated in **April 2025**, directly inspired by the research presented in the paper "A profitable trading algorithm for cryptocurrencies using a Neural Network model". Building upon a foundational understanding of non-systematic trading strategies, which I had been exploring since **September 2022**, this systematic framework was rapidly developed to its current state by **early May 2025**. The iterative development process involved numerous versions and was significantly accelerated by leveraging modern AI tools like Gemini and ChatGPT for efficient ideation and implementation. The primary objective of this ongoing work is to build a robust and adaptable framework for exploring and identifying profitable quantitative trading strategies.

## **Features**

* **Modular and Scalable Architecture**: Designed for clarity and extensibility, separating core functionalities into distinct, manageable modules (data management, feature engineering, model training, strategy execution, notifications, exchange integration). This modularity allows for easy adaptation to new exchanges, models, or strategy components.  
* **Multi-Exchange Compatibility**: Features an adaptable design with dedicated adapters for seamless integration with various cryptocurrency futures exchanges (currently implemented for Binance Futures), enabling flexible deployment across different liquidity venues.  
* **Advanced Feature Engineering Pipeline**: Transforms raw OHLCV data into a rich set of predictive features, encompassing:  
  * **Comprehensive Technical Indicators**: Calculation of a wide array of indicators (SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, MACD) across multiple configurable periods.  
  * **Candlestick Pattern Recognition**: Utilizes TA-Lib for automated detection of significant candlestick patterns.  
  * **Custom Statistical & Price Action Features**: Includes Fair Value Gaps (FVG), Z-scores, Average Daily Range (ADR), and trend strength metrics.  
  * **Temporal Safety Validation**: Crucial for preventing lookahead bias, ensuring all features are derived exclusively from past data, making the backtesting and live trading results statistically sound.  
* **Machine Learning-Driven Signal Generation**: Integrates state-of-the-art machine learning models for market direction prediction:  
  * **Ternary Classification**: Models are trained to classify future price movements into three categories: Long (1), Short (-1), or Neutral (0).  
  * **Supported Models**: Includes robust implementations for Long Short-Term Memory (LSTM) networks, XGBoost, and RandomForest classifiers.  
  * **Hyperparameter Tuning**: Supports advanced hyperparameter optimization using RandomizedSearchCV with TimeSeriesSplit for robust model selection.  
* **Configurable and Dynamic Strategy Logic**: Implements a highly customizable trading strategy layer with intelligent filtering mechanisms:  
  * **Confidence-Based Entry Filtering**: Filters trade entries based on the model's prediction probability (confidence score), allowing for higher conviction trades.  
  * **Adaptive Volatility Regime Filtering**: Dynamically adjusts trade entry allowances and maximum position holding periods based on the prevailing market volatility regime (low, medium, high), enhancing adaptability to changing market conditions.  
  * **Trend Alignment Filter**: Incorporates an EMA-based trend filter to ensure trades are aligned with the dominant market trend, reducing counter-trend exposure.  
  * **Directional Control**: Provides explicit control to enable or disable long and short trade entries independently.  
  * **Neutral Signal Management**: Configurable behavior for exiting existing positions upon a neutral signal from the model.  
  * **Dynamic Take Profit/Stop Loss (TP/SL)**: Calculates TP/SL levels dynamically based on market volatility (Average True Range \- ATR) or fixed percentages, adapting risk parameters to current market conditions.  
* **Robust Backtesting Engine**: A comprehensive simulation environment for rigorous strategy evaluation:  
  * **Realistic Simulation**: Accurately accounts for trading fees, slippage, and liquidation mechanics, ensuring precise performance assessment.  
  * **Detailed Performance Metrics**: Tracks and calculates a wide array of performance metrics including total return, Compound Annual Growth Rate (CAGR), maximum drawdown, win rate, profit factor, and average PnL per trade.  
  * **Trade Management Simulation**: Accurately simulates position opening, closing, and immediate reversal logic with proper fee and margin handling.  
  * **Persistent Results**: Saves detailed trade logs (Parquet format) and summary performance metrics (JSON) for post-analysis.  
* **Live Trading Capabilities**: Enables real-time, automated trading in live market environments:  
  * **Real-time Data Integration**: Connects directly to exchange data feeds for up-to-the-minute market information.  
  * **Automated Trade Execution**: Executes market orders, manages open positions, and places/cancels associated Stop Loss and Take Profit orders.  
  * **Resilient State Management**: Implements periodic state saving for current capital and open positions, allowing for graceful restarts and recovery from interruptions.  
  * **Critical Event Notifications**: Integrates with notification services (e.g., Telegram) to send alerts for trade executions, critical errors, and important bot status updates.  
* **Data Management & Persistence**: Handles the complete data lifecycle:  
  * **Data Fetching**: Scripts for fetching historical OHLCV data.  
  * **Data Processing & Storage**: Efficiently processes and stores raw, processed (with features), and labeled data in optimized Parquet format.  
* **Flexible Label Generation**: Supports multiple sophisticated labeling strategies for creating high-quality training datasets:  
  * **Triple Barrier Method (triple\_barrier)**: A robust technique that defines profit, loss, and time barriers. Labels are assigned based on which barrier is hit first within a forward-looking window. This strategy supports dynamic barrier levels adjusted by market volatility (e.g., Average True Range \- ATR) for adaptive risk management.  
  * **Directional Ternary Strategy (directional\_ternary)**: A simpler approach that labels based on a predefined percentage price change within a forward window. A positive change exceeding the threshold results in a 'Long' label, a negative change below the threshold results in a 'Short' label, and anything in between is 'Neutral'.  
  * **Max Return Quantile Strategy (max\_return\_quantile)**: This strategy identifies "significant" price movements by analyzing the maximum favorable excursion within a forward window. It assigns 'Long' or 'Short' labels if the maximum future return (or inverse for shorts) falls within a specified quantile of historical returns, indicating a strong potential move.  
  * **EMA Return Percentile Strategy (ema\_return\_percentile)**: A custom strategy that combines Exponential Moving Average (EMA) based trend analysis with percentile thresholds for future returns. It assigns labels based on the price's position relative to the EMA and whether the future return exceeds a certain percentile, aiming to capture trend-aligned, significant movements.  
  * **Label Propagation Smoothing**: All strategies incorporate a configurable min\_holding\_period to smooth raw labels. This ensures that a generated signal persists for a minimum number of bars, which helps to filter out high-frequency noise and create more stable, actionable training targets for the models.  
* **Comprehensive Logging**: Detailed, rotating logging for all stages of bot operation, backtesting, and data processing, facilitating debugging and performance monitoring.

## **Project Structure**

.  
├── .env \# Environment variables (API keys, secrets)  
├── .gitignore \# Git ignore file  
├── README.md \# Project README  
├── requirements.txt \# Python dependencies  
├── ta\_lib-0.6.3-cp312-cp312-win\_amd64.whl \# TA-Lib wheel (Windows specific)  
├── trade\_ohlcv\_visualization.htm \# HTML for visualizing trades  
├── trading\_bot.py \# Main live trading bot script  
├── adapters/ \# Exchange API adapters  
│ ├── init.py  
│ └── binance\_futures\_adapter.py  
├── config/ \# Configuration files  
│ ├── init.py  
│ ├── params.py \# Centralized parameters for all modules  
│ └── paths.py \# Defines project paths  
├── data/ \# Data storage  
│ ├── labeled/ \# Labeled data for model training  
│ ├── processed/ \# Processed data with engineered features  
│ └── raw/ \# Raw OHLCV data  
├── docs/ \# Project documentation  
│ ├── label\_analysis\_results.md \# Results of label analysis  
│ ├── labeling\_analysis.md \# Documentation on how to analyze labels  
│ ├── labeling\_strategy.md \# Detailed explanation of labeling strategies  
│ ├── model\_analysis.md \# Documentation on model performance analysis  
│ └── model\_training\_doc.md \# Documentation on the model training process  
├── logs/ \# Application logs  
├── models/ \# Trained machine learning models  
│ └── trained\_models/  
│ ├── lstm/  
│ ├── random\_forest/  
│ └── xgboost/  
├── results/ \# Backtesting and live trading results  
│ ├── analysis/ \# Analysis outputs (plots, summary metrics)  
│ ├── backtesting/ \# Raw backtesting trade logs and equity curves  
│ └── live\_trading/ \# Live trading state and history  
├── scripts/ \# Utility scripts for workflow automation  
│ ├── analyze\_labels.py  
│ ├── analyze\_model.py  
│ ├── backtest.py \# Script to run backtests  
│ ├── convert\_trades\_to\_json.py  
│ ├── create\_labels.py  
│ ├── fetch\_data.py  
│ ├── generate\_features.py  
│ └── train\_model.py  
└── utils/ \# Core utility modules  
├── init.py  
├── backtester.py \# Backtesting engine  
├── data\_manager.py  
├── exceptions.py  
├── exchange\_interface.py  
├── features\_engineer.py \# Feature engineering module  
├── label\_generator.py \# Label generation module  
├── labeling\_strategies/ \# Different labeling strategy implementations  
│ ├── base\_strategy.py  
│ ├── directional\_ternary.py  
│ ├── ema\_return\_percentile.py  
│ ├── max\_return\_quantile.py  
│ └── triple\_barrier.py  
├── logger\_config.py  
├── model\_trainer.py  
├── notification\_manager.py  
└── results\_analyzer.py

## **Key Components Explained**

### **trading\_bot.py**

This is the heart of the live trading system. It orchestrates the entire trading process:

* **Initialization**: Loads configuration, sets up logging, initializes exchange adapters, feature engineers, and loads the trained machine learning model. It also loads previous bot state (capital and open positions) for seamless restarts.  
* **Main Loop (run method)**: Continuously fetches new candle data, processes it through the feature engineering pipeline, obtains a signal from the loaded model, and executes trade logic.  
* **Signal Processing**: Calls \_get\_signal to get predictions and probabilities from the model, and then \_apply\_entry\_filters to validate the signal against various strategy rules (trend, confidence, volatility regime).  
* **Trade Execution**: Manages opening, closing, and reversing positions based on filtered signals and current market conditions. It handles position sizing, setting dynamic Take Profit (TP) and Stop Loss (SL) levels, and tracking trade details.  
* **State Management**: Periodically saves the bot's internal capital and current open position to a JSON file, and appends closed trades to a Parquet file for persistent storage and later analysis.

### **config/params.py**

This file serves as the central hub for all configurable parameters across the entire project. It defines:

* **General Configuration**: Random seeds, parallel processing settings, and hyperparameter tuning defaults.  
* **Exchange Configuration (EXCHANGE\_CONFIG)**: API keys (loaded from environment variables for security), connection settings (testnet, TLD, timeout), and default market specifics (precision, min quantity/notional).  
* **Feature Engineering Configuration (FEATURE\_CONFIG)**: Defines periods for various technical indicators (SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, etc.), parameters for candlestick pattern detection, Fair Value Gap (FVG) lookbacks, Z-score periods, Average Daily Range (ADR), and trend strength. It also includes settings for temporal safety validation.  
* **Labeling Configuration (LABELING\_CONFIG)**: Specifies the chosen labeling strategy (e.g., triple\_barrier, ema\_return\_percentile) and its specific parameters, along with a min\_holding\_period for label smoothing.  
* **Model Training Configuration (MODEL\_CONFIG)**: Contains hyperparameters and data handling settings for different machine learning models (Random Forest, XGBoost, LSTM), including their specific tuning parameter distributions.  
* **Strategy Configuration (STRATEGY\_CONFIG)**: Crucial for defining the bot's trading behavior, including:  
  * initial\_capital, risk\_per\_trade\_pct, leverage, trading\_fee\_rate.  
  * exit\_on\_neutral\_signal: Controls whether the bot exits a position when the model predicts a neutral signal.  
  * allow\_long\_trades, allow\_short\_trades: Boolean flags to enable or disable long/short entries.  
  * confidence\_filter\_enabled, confidence\_threshold\_long\_pct, confidence\_threshold\_short\_pct: Parameters for filtering trades based on model prediction confidence.  
  * volatility\_regime\_filter\_enabled, volatility\_regime\_max\_holding\_bars, allow\_trading\_in\_volatility\_regime: Parameters for the dynamic volatility regime filter.  
  * trend\_filter\_enabled, trend\_filter\_ema\_period: Settings for the EMA-based trend filter.  
  * sequence\_length\_bars: Important for LSTM models, defining the input sequence length.

### **utils/backtester.py**

This module provides a robust and configurable backtesting engine for simulating trading strategies.

* **Initialization**: Takes historical OHLCV data, model predictions, and model probabilities as input. It now robustly merges various configuration dictionaries (STRATEGY\_CONFIG, BACKTESTER\_CONFIG, EXCHANGE\_CONFIG, FEATURE\_CONFIG) to create a comprehensive set of parameters for the simulation, ensuring all necessary attributes (including maintenance\_margin\_rate, liquidation\_fee\_rate, confidence thresholds, and volatility regime settings) are properly initialized.  
* **Data Preparation**: Validates input data, ensures correct data types, and can calculate missing technical indicators (like ATR or EMA) on the fly if they are required by the strategy but not present in the input data. It also aligns model signals and probabilities with the data index and handles NaNs.  
* **Simulation Loop**: Iterates through historical bars, applying the trading strategy logic bar by bar.  
* **Trade Logic**:  
  * **Entry Filters**: Implements \_apply\_entry\_filters which checks various conditions (EMA trend, confidence thresholds, volatility regime, allowed trade directions) before a position can be opened. These filters now correctly utilize their configured parameters.  
  * **Position Sizing**: Calculates optimal position size based on risk per trade, initial capital, and stop-loss levels.  
  * **Dynamic TP/SL**: Computes take-profit and stop-loss prices, dynamically adjusting them based on Average True Range \- ATR if enabled.  
  * **Exit Conditions**: Monitors for multiple exit conditions including Stop Loss (SL), Take Profit (TP), Liquidation (with accurate liquidation\_price estimation using maintenance\_margin\_rate), Maximum Holding Period (dynamically set by volatility regime), and neutral signals (if configured).  
  * **Reversal Logic**: Handles scenarios where a new signal dictates reversing an existing position (closing the current one and opening an opposite one), ensuring proper fee calculation using current\_position\_entry\_fees.  
* **Results & Metrics**: Tracks all executed trades, calculates unrealized and realized PnL, maintains an equity curve, and generates comprehensive performance metrics (e.g., total return, CAGR, max drawdown, win rate, profit factor). Includes a PnL consistency check to verify calculations.  
* **Saving Results**: Saves detailed trade logs (Parquet) and summary metrics (JSON) to specified output directories.

### **utils/features\_engineer.py**

This module is responsible for transforming raw OHLCV (Open, High, Low, Close, Volume) data into a rich set of technical and statistical features suitable for machine learning models.

* **Comprehensive Indicator Calculation**: Computes a wide range of popular technical indicators for multiple periods, including:  
  * Moving Averages (SMA, EMA)  
  * Momentum Oscillators (RSI, Stochastic, Awesome Oscillator, CCI, MFI)  
  * Volatility Indicators (Bollinger Bands, Average True Range \- ATR)  
  * Volume Indicators (On-Balance Volume \- OBV, Chaikin Money Flow \- CMF)  
  * MACD (Moving Average Convergence Divergence)  
* **Pattern Recognition**: Detects various candlestick patterns using TA-Lib.  
* **Custom Features**: Includes calculations for Fair Value Gaps (FVG), Z-scores, Average Daily Range (ADR), and trend strength indicators.  
* **Temporal Safety**: Crucially, all features are calculated in a time-safe manner, using only past data to prevent lookahead bias, which is essential for realistic backtesting and live trading. It includes built-in validation checks to detect potential lookahead bias during development.  
* **Configurable Periods**: Allows users to specify multiple lookback periods for indicators via params.py, generating a diverse set of features.  
* **Volatility Regime Calculation**: Determines the market's volatility regime (e.g., low, medium, high) based on ATR, which is then used by the strategy for dynamic adjustments.

### **utils/label\_generator.py**

This module is designed to create the target labels used for training the machine learning models. It abstracts the complexity of defining what constitutes a "buy," "sell," or "neutral" signal.

* **Strategy-Based Labeling**: Supports various popular labeling strategies, each with its own logic for defining trade outcomes:  
  * **Triple Barrier Method (triple\_barrier)**: A robust technique that defines profit, loss, and time barriers. Labels are assigned based on which barrier is hit first within a forward-looking window. It supports dynamic barrier levels adjusted by market volatility (e.g., Average True Range \- ATR) for adaptive risk management.  
  * **Directional Ternary Strategy (directional\_ternary)**: A simpler approach that labels based on a predefined percentage price change within a forward window. A positive change exceeding the threshold results in a 'Long' label, a negative change below the threshold results in a 'Short' label, and anything in between is 'Neutral'.  
  * **Max Return Quantile Strategy (max\_return\_quantile)**: This strategy identifies "significant" price movements by analyzing the maximum favorable excursion within a forward window. It assigns 'Long' or 'Short' labels if the maximum future return (or inverse for shorts) falls within a specified quantile of historical returns, indicating a strong potential move.  
  * **EMA Return Percentile Strategy (ema\_return\_percentile)**: A custom strategy that combines Exponential Moving Average (EMA) based trend analysis with percentile thresholds for future returns. It assigns labels based on the price's position relative to the EMA and whether the future return exceeds a certain percentile, aiming to capture trend-aligned, significant movements.  
* **Label Propagation Smoothing**: All strategies incorporate a configurable min\_holding\_period to smooth raw labels. This ensures that a generated signal persists for a minimum number of bars, which helps to filter out high-frequency noise and create more stable, actionable training targets for the models.  
* **Input Validation**: Performs checks on the input DataFrame to ensure it contains the necessary OHLCV data and features before label calculation.  
* **Extensible Design**: Designed to easily integrate new labeling strategies by extending a BaseLabelingStrategy class.

## **Installation**

1. Clone the repository:  
   git clone https://github.com/DimitriKenne/CryptoFutureBot.git  
   cd CryptoFutureBot  
2. Create a virtual environment (recommended):  
   python \-m venv .venv  
   source .venv/bin/activate \# On Windows: .venv\\Scripts\\activate

3. # **Install dependencies:**    **pip install \-r requirements.txt**    **Note: If you encounter issues with TA-Lib, refer to the ta\_lib-0.6.3-cp312-cp312-win\_amd64.whl file for Windows installation or consult TA-Lib documentation for other OS. 4\. Set up environment variables:**    **Create a .env file in the project root and add your exchange API keys and any notification tokens:**    **BINANCE\_API\_KEY="YOUR\_BINANCE\_API\_KEY"**    **BINANCE\_API\_SECRET="YOUR\_BINANCE\_API\_SECRET"**    **TELEGRAM\_ENABLED=True**    **TELEGRAM\_BOT\_TOKEN="YOUR\_TELEGRAM\_BOT\_TOKEN"**    **TELEGRAM\_CHAT\_ID="YOUR\_TELEGRAM\_CHAT\_ID"**    **Add other exchange or notification credentials as needed**

## **Configuration**

All core parameters are managed in config/params.py. Before running any scripts or the bot, review and adjust these settings according to your preferences and risk tolerance. Key sections include:

* GENERAL\_CONFIG  
* EXCHANGE\_CONFIG  
* FEATURE\_CONFIG  
* LABELING\_CONFIG  
* MODEL\_CONFIG  
* STRATEGY\_CONFIG

## **Usage**

The scripts/ directory contains various utility scripts to manage the workflow. Here are detailed instructions for each:

### **trading\_bot.py**

Execute the live trading bot.

python trading\_bot.py

### **scripts/fetch\_data.py**

Download historical OHLCV data from exchanges.

1. Create a .env file in the project root with BINANCE\_API\_KEY and BINANCE\_API\_SECRET.  
2. Ensure config/params.py contains relevant configuration in STRATEGY\_CONFIG\['exchange\_adapter\_params'\].  
3. Ensure config/paths.py contains 'raw\_data\_dir' and 'raw\_data\_pattern' keys in the PATHS dictionary.  
4. Run the script from the project root:  
   python scripts/fetch\_data.py \--symbol ADAUSDT \--interval 5m \--start\_date 2024-01-01 \--end\_date 2024-03-01  
   To fetch data up to the current time:  
   python scripts/fetch\_data.py \--symbol ADAUSDT \--interval 5m \--start\_date 2024-01-01

### **scripts/generate\_features.py**

Apply feature engineering to raw data.

python scripts/generate\_features.py \--symbol BTCUSDT \--interval 1h

Or using the module syntax:

python \-m scripts.generate\_features \--symbol ADAUSDT \--interval 5m

Ensure you have run the fetch\_data script first to obtain the raw data:

python \-m scripts.fetch\_data \--symbol ADAUSDT \--interval 5m \--start\_date YYYY-MM-DD

Ensure config/params.py (with FEATURE\_CONFIG and STRATEGY\_CONFIG) and config/paths.py are correctly configured.

### **scripts/create\_labels.py**

Generate labels for model training using a chosen strategy.

* Generate labels using the simple directional strategy:  
  python scripts/create\_labels.py \--symbol BTCUSDT \--interval 1h \--label-strategy directional\_ternary  
* Generate labels using the triple barrier strategy:  
  python scripts/create\_labels.py \--symbol ADAUSDT \--interval 5m \--label-strategy triple\_barrier  
* Generate labels using the max return quantile strategy:  
  python scripts/create\_labels.py \--symbol ADAUSDT \--interval 15m \--label-strategy max\_return\_quantile

Ensure you have processed data files (including necessary features like ATR columns if using triple\_barrier) in your data/processed directory, and that config/params.py and config/paths.py are correct. The feature generation script must produce an ATR column named atr\_{lookback} (e.g., atr\_14) matching the vol\_adj\_lookback parameter in LABELING\_CONFIG if using the triple\_barrier strategy, and save it to the processed data file.

### **scripts/train\_model.py**

Train and save machine learning models.

* Train the default XGBoost model for ternary classification:  
  python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h  
* Train the RandomForest model for ternary classification:  
  python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest  
* Train the LSTM model (requires TensorFlow):  
  python scripts/train\_model.py \--symbol ETHUSDT \--interval 15m \--model lstm \--train\_ratio 0.7  
* Train with tuning (default for non-LSTM):  
  python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest  
  python \-m scripts.train\_model \--symbol ADAUSDT \--interval 5m \--model xgboost  
* Train skipping tuning:  
  python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest \--skip\_tuning  
* Train using a specific subset of features:  
  python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h \--features ema\_10 rsi\_14 macd

Ensure you have processed and labeled data files (including label 0\) in your data/ directory and config/params.py (with MODEL\_CONFIG, GENERAL\_CONFIG) and config/paths.py are correctly configured (including 'trained\_models\_dir' path and 'trained\_model\_pattern'). The feature generation script must produce an ATR column named atr\_{lookback} (e.g., atr\_14) matching the vol\_adj\_lookback parameter in LABELING\_CONFIG if using the triple\_barrier strategy, and save it to the processed data file.

### **scripts/analyze\_model.py**

Analyze a trained machine learning model.

* Analyze the trained RandomForest model for ADAUSDT 5m data:  
  python scripts/analyze\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest

Ensure you have run the train\_model.py script successfully for the specified symbol, interval, and model before running this analysis script. The script will load data from data/processed/ and data/labeled/ and the trained model and metadata from models/trained\_models/, all managed by DataManager. Analysis results (metrics and plots) will be saved to results/analysis/model\_analysis/.

### **scripts/backtest.py**

Run backtests using trained models and strategy configurations.

* Run backtest on the test set for the trained XGBoost model for BTCUSDT 1h data:  
  python scripts/backtest.py \--symbol BTCUSDT \--interval 1h \--model xgboost  
* Run backtest on the test set for the trained RandomForest model for ADAUSDT 5m data:  
  python scripts/backtest.py \--symbol ADAUSDT \--interval 5m \--model random\_forest  
* Run backtest on the full dataset for the trained LSTM model for ADAUSDT 5m data:  
  python scripts/backtest.py \--symbol ADAUSDT \--interval 5m \--model lstm \--backtest\_mode full \--train\_ratio 0.7

Ensure you have processed data files in your data/processed directory, labeled data files (containing labels) in your data/labeled directory, and trained model files in your models/trained\_models directory. config/params.py (with MODEL\_CONFIG, GENERAL\_CONFIG, STRATEGY\_CONFIG, BACKTESTER\_CONFIG) and config/paths.py are correctly configured.

### **scripts/analyze\_results.py**

Analyze trading results (backtest or live).

* Analyze backtest results for BTCUSDT 1h data using the xgboost model:  
  python scripts/analyze\_results.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_type backtest  
* Analyze live trading results for ADAUSDT 5m data using the random\_forest model:  
  python scripts/analyze\_results.py \--symbol ADAUSDT \--interval 5m \--model\_type random\_forest \--results\_type live

You can optionally specify custom directories for results or analysis output:

python scripts/analyze\_results.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_dir /path/to/my/backtest\_results \--analysis\_dir /path/to/my/analysis\_output

Ensure you have results files (trades.parquet and equity.parquet) in the specified or configured results directory, and that config/paths.py is correctly configured.

### **scripts/analyze\_labels.py**

Perform key analyses on generated trading labels and processed data to inform strategy configuration (stop-loss, take-profit, max\_holding\_period). Results, including plots and summary statistics, will be saved to the directory specified by PATHS\['labeling\_analysis\_dir'\] (default: results/analysis/labeling\_analysis/) and documented in docs/label\_analysis\_results.md.

* Run all analyses for ADAUSDT 5m data with default future horizons:  
  python scripts/analyze\_labels.py \--symbol ADAUSDT \--interval 5m  
* Run analyses for specific horizons (10, 30, 60 bars):  
  python scripts/analyze\_labels.py \--symbol BTCUSDT \--interval 1h \--future-horizons 10 30 60

Ensure you have run the data processing and labeling steps first to generate the necessary 'processed' (including volatility\_regime feature) and 'labeled' data files in your data directory.

### **scripts/convert\_trades\_to\_json.py**

Convert trade history and OHLCV data to JSON for visualization.

python scripts/convert\_trades\_to\_json.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_type backtest

You can optionally specify a custom output file path:

python scripts/convert\_trades\_to\_json.py \--symbol ADAUSDT \--interval 5m \--model\_type random\_forest \--output\_file /path/to/my/output.json

## **Contributing**

Contributions are welcome\! Please follow standard GitHub flow: fork the repository, create a feature branch, commit your changes, and open a pull request.

## **License**

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE).