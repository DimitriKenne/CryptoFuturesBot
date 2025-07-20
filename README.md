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

    * **Dynamic Take Profit/Stop Loss (TP/SL)**: Calculates TP/SL levels dynamically based on market volatility (Average True Range - ATR) or fixed percentages, adapting risk parameters to current market conditions.

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

    * **Triple Barrier Method (triple_barrier)**: A robust technique that defines profit, loss, and time barriers. Labels are assigned based on which barrier is hit first within a forward-looking window. This strategy supports dynamic barrier levels adjusted by market volatility (e.g., Average True Range - ATR) for adaptive risk management.

* **Net Forward Return Quantile Strategy (net_forward_return_quantile)**: This strategy labels based on the *net* percentage return (accounting for fees and slippage) over a forward window. It assigns 'Long' or 'Short' labels if the net future return exceeds a specific quantile threshold for positive or negative movements, ensuring signals target genuinely profitable moves.

    * **Future Range Dominance Strategy (future_range_dominance)**: This strategy identifies labels based on the ratio of the highest potential net profit in one direction (long or short) to the highest potential net profit in the *opposite* direction within a forward window. It assigns 'Long' or 'Short' labels if one direction's potential net profit significantly dominates the other, and both meet a minimum profitability threshold, aiming for high-conviction, directional moves.

    * **Label Propagation Smoothing**: All strategies incorporate a configurable **`min_holding_period`** to smooth raw labels. This ensures that a generated signal persists for a minimum number of bars, which helps to filter out high-frequency noise and create more stable, actionable training targets for the models.

* **Comprehensive Logging**: Detailed, rotating logging for all stages of bot operation, backtesting, and data processing, facilitating debugging and performance monitoring.

## **Project Structure**
```
.
├── .env # Environment variables (API keys, secrets)
├── .gitignore # Git ignore file
├── README.md # Project README
├── requirements.txt # Python dependencies
├── ta_lib-0.6.3-cp312-cp312-win_amd64.whl # TA-Lib wheel (Windows specific)
├── trade_ohlcv_visualization.htm # HTML for visualizing trades
├── trading_bot.py # Main live trading bot script
├── adapters/ # Exchange API adapters
│ ├── init.py
│ └── binance_futures_adapter.py
├── config/ # Configuration files
│ ├── init.py
│ ├── params.py # Centralized parameters for all modules
│ └── paths.py # Defines project paths
├── data/ # Data storage
│ ├── labeled/ # Labeled data for model training
│ ├── processed/ # Processed data with engineered features
│ └── raw/ # Raw OHLCV data
├── docs/ # Project documentation
│ ├── label_analysis_results.md # Results of label analysis
│ ├── labeling_analysis.md # Documentation on how to analyze labels
│ ├── labeling_strategy.md # Detailed explanation of labeling strategies
│ ├── model_analysis.md # Documentation on model performance analysis
│ └── model_training_doc.md # Documentation on the model training process
├── logs/ # Application logs
├── models/ # Trained machine learning models
│ └── trained_models/
│ ├── lstm/
│ ├── random_forest/
│ └── xgboost/
├── results/ # Backtesting and live trading results
│ ├── analysis/ # Analysis outputs (plots, summary metrics)
│ ├── backtesting/ # Raw backtesting trade logs and equity curves
│ └── live_trading/ # Live trading state and history
├── scripts/ # Utility scripts for workflow automation
│ ├── analyze_labels.py
│ ├── analyze_model.py
│ ├── backtest.py # Script to run backtests
│ ├── convert_trades_to_json.py
│ ├── create_labels.py
│ ├── fetch_data.py
│ ├── generate_features.py
│ ├── monte_carlo_backtest.py # NEW: Script to run Monte Carlo backtests
│ └── train_model.py
└── utils/ # Core utility modules
├── init.py
├── backtester.py # Backtesting engine
├── data_manager.py
├── exceptions.py
├── exchange_interface.py
├── features_engineer.py # Feature engineering module
├── label_generator.py # Label generation module
├── labeling_strategies/ # Different labeling strategy implementations
│ ├── base_strategy.py
│ ├── directional_ternary.py
│ ├── ema_return_percentile.py
│ ├── max_return_quantile.py
│ └── triple_barrier.py
├── logger_config.py
├── model_trainer.py
├── notification_manager.py
└── results_analyzer.py
```

## **Key Components Explained**

### **trading_bot.py**

This is the heart of the live trading system. It orchestrates the entire trading process:

* **Initialization**: Loads configuration, sets up logging, initializes exchange adapters, feature engineers, and loads the trained machine learning model. It also loads previous bot state (capital and open positions) for seamless restarts.

* **Main Loop (run method)**: Continuously fetches new candle data, processes it through the feature engineering pipeline, obtains a signal from the loaded model, and executes trade logic.

* **Signal Processing**: Calls \_get_signal to get predictions and probabilities from the model, and then \_apply_entry_filters to validate the signal against various strategy rules (trend, confidence, volatility regime).

* **Trade Execution**: Manages opening, closing, and reversing positions based on filtered signals and current market conditions. It handles position sizing, setting dynamic Take Profit (TP) and Stop Loss (SL) levels, and tracking trade details.

* **State Management**: Periodically saves the bot's internal capital and current open position to a JSON file, and appends closed trades to a Parquet file for persistent storage and later analysis.

### **config/params.py**

This file serves as the central hub for all configurable parameters across the entire project. It defines:

* **General Configuration**: Random seeds, parallel processing settings, and hyperparameter tuning defaults.

* **Exchange Configuration (EXCHANGE_CONFIG)**: API keys (loaded from environment variables for security), connection settings (testnet, TLD, timeout), and default market specifics (precision, min quantity/notional).

* **Feature Engineering Configuration (FEATURE_CONFIG)**: Defines periods for various technical indicators (SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, etc.), parameters for candlestick pattern detection, Fair Value Gap (FVG) lookbacks, Z-score periods, Average Daily Range (ADR), and trend strength. It also includes settings for temporal safety validation.

* **Labeling Configuration (LABELING_CONFIG)**: Specifies the chosen labeling strategy (e.g., triple_barrier, ema_return_percentile) and its specific parameters, along with a min_holding_period for label smoothing.

* **Model Training Configuration (MODEL_CONFIG)**: Contains hyperparameters and data handling settings for different machine learning models (Random Forest, XGBoost, LSTM), including their specific tuning parameter distributions.

* **Strategy Configuration (STRATEGY_CONFIG)**: Crucial for defining the bot's trading behavior, including:

    * initial_capital, risk_per_trade_pct, leverage, trading_fee_rate.

    * exit_on_neutral_signal: Controls whether the bot exits a position when the model predicts a neutral signal.

    * allow_long_trades, allow_short_trades: Boolean flags to enable or disable long/short entries.

    * confidence_filter_enabled, confidence_threshold_long_pct, confidence_threshold_short_pct: Parameters for filtering trades based on model prediction confidence.

    * volatility_regime_filter_enabled, volatility_regime_max_holding_bars, allow_trading_in_volatility_regime: Parameters for the dynamic volatility regime filter.

    * trend_filter_enabled, trend_filter_ema_period: Settings for the EMA-based trend filter.

    * sequence_length_bars: Important for LSTM models, defining the input sequence length.

### **utils/backtester.py**

This module provides a robust and configurable backtesting engine for simulating trading strategies.

* **Initialization**: Takes historical OHLCV data, model predictions, and model probabilities as input. It now robustly merges various configuration dictionaries (STRATEGY_CONFIG, BACKTESTER_CONFIG, EXCHANGE_CONFIG, FEATURE_CONFIG) to create a comprehensive set of parameters for the simulation, ensuring all necessary attributes (including maintenance_margin_rate, liquidation_fee_rate, confidence thresholds, and volatility regime settings) are properly initialized.

* **Data Preparation**: Validates input data, ensures correct data types, and can calculate missing technical indicators (like ATR or EMA) on the fly if they are required by the strategy but not present in the input data. It also aligns model signals and probabilities with the data index and handles NaNs.

* **Simulation Loop**: Iterates through historical bars, applying the trading strategy logic bar by bar.

* **Trade Logic**:

    * **Entry Filters**: Implements \_apply_entry_filters which checks various conditions (EMA trend, confidence thresholds, volatility regime, allowed trade directions) before a position can be opened. These filters now correctly utilize their configured parameters.

    * **Position Sizing**: Calculates optimal position size based on risk per trade, initial capital, and stop-loss levels.

    * **Dynamic TP/SL**: Computes take-profit and stop-loss prices, dynamically adjusting them based on Average True Range - ATR if enabled.

    * **Exit Conditions**: Monitors for multiple exit conditions including Stop Loss (SL), Take Profit (TP), Liquidation (with accurate liquidation_price estimation using maintenance_margin_rate), Maximum Holding Period (dynamically set by volatility regime), and neutral signals (if configured).

    * **Reversal Logic**: Handles scenarios where a new signal dictates reversing an existing position (closing the current one and opening an opposite one), ensuring proper fee calculation using current_position_entry_fees.

* **Results & Metrics**: Tracks all executed trades, calculates unrealized and realized PnL, maintains an equity curve, and generates comprehensive performance metrics (e.g., total return, CAGR, max drawdown, win rate, profit factor). Includes a PnL consistency check to verify calculations.

* **Saving Results**: Saves detailed trade logs (Parquet) and summary metrics (JSON) to specified output directories.

### **utils/features_engineer.py**

This module is responsible for transforming raw OHLCV (Open, High, Low, Close, Volume) data into a rich set of technical and statistical features suitable for machine learning models.

* **Comprehensive Indicator Calculation**: Computes a wide range of popular technical indicators for multiple periods, including:

    * Moving Averages (SMA, EMA)

    * Momentum Oscillators (RSI, Stochastic, Awesome Oscillator, CCI, MFI)

    * Volatility Indicators (Bollinger Bands, Average True Range - ATR)

    * Volume Indicators (On-Balance Volume - OBV, Chaikin Money Flow - CMF)

    * MACD (Moving Average Convergence Divergence)

* **Pattern Recognition**: Detects various candlestick patterns using TA-Lib.

* **Custom Features**: Includes calculations for Fair Value Gaps (FVG), Z-scores, Average Daily Range (ADR), and trend strength indicators.

* **Temporal Safety**: Crucially, all features are calculated in a time-safe manner, using only past data to prevent lookahead bias, which is essential for realistic backtesting and live trading. It includes built-in validation checks to detect potential lookahead bias during development.

* **Configurable Periods**: Allows users to specify multiple lookback periods for indicators via params.py, generating a diverse set of features.

* **Volatility Regime Calculation**: Determines the market's volatility regime (e.g., low, medium, high) based on ATR, which is then used by the strategy for dynamic adjustments.

### **utils/label_generator.py**

This module is designed to create the target labels used for training the machine learning models. It abstracts the complexity of defining what constitutes a "buy," "sell," or "neutral" signal.

* **Strategy-Based Labeling**: Supports various popular labeling strategies, each with its own logic for defining trade outcomes:

    * **Triple Barrier Method (triple_barrier)**: A robust technique that defines profit, loss, and time barriers. Labels are assigned based on which barrier is hit first within a forward-looking window. It supports dynamic barrier levels adjusted by market volatility (e.g., Average True Range - ATR) for adaptive risk management.

    * **Directional Ternary Strategy (directional_ternary)**: A simpler approach that labels based on a predefined percentage price change within a forward window. A positive change exceeding the threshold results in a 'Long' label, a negative change below the threshold results in a 'Short' label, and anything in between is 'Neutral'.

    * **Max Return Quantile Strategy (max_return_quantile)**: This strategy identifies "significant" price movements by analyzing the maximum favorable excursion within a forward window. It assigns 'Long' or 'Short' labels if the maximum future return (or inverse for shorts) falls within a specified quantile of historical returns, indicating a strong potential move.

    * **EMA Return Percentile Strategy (ema_return_percentile)**: A custom strategy that combines Exponential Moving Average (EMA) based trend analysis with percentile thresholds for future returns. It assigns labels based on the price's position relative to the EMA and whether the future return exceeds a certain percentile, aiming to capture trend-aligned, significant movements.

    * **Label Propagation Smoothing**: All strategies incorporate a configurable min_holding_period to smooth raw labels. This ensures that a generated signal persists for a minimum number of bars, which helps to filter out high-frequency noise and create more stable, actionable training targets for the models.

* **Input Validation**: Performs checks on the input DataFrame to ensure it contains the necessary OHLCV data and features before label calculation.

* **Extensible Design**: Designed to easily integrate new labeling strategies by extending a BaseLabelingStrategy class.

## **Installation**

1.  Clone the repository:
    `git clone https://github.com/DimitriKenne/CryptoFutureBot.git`
    `cd CryptoFutureBot`

2.  Create a virtual environment (recommended):
    `python -m venv .venv`
    `source .venv/bin/activate` # On Windows: `.venv\Scripts\activate`

3.  **Install dependencies:**
    `pip install -r requirements.txt`
    **Note:** If you encounter issues with TA-Lib, refer to the `ta_lib-0.6.3-cp312-cp312-win_amd64.whl` file for Windows installation or consult TA-Lib documentation for other OS.

4.  **Set up environment variables:**
    **Create a `.env` file in the project root and add your exchange API keys and any notification tokens:**
    `BINANCE_API_KEY="YOUR_BINANCE_API_KEY"`
    `BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"`
    `TELEGRAM_ENABLED=True`
    `TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"`
    `TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"`
    **Add other exchange or notification credentials as needed**

## **Configuration**

All core parameters are managed in `config/params.py`. Before running any scripts or the bot, review and adjust these settings according to your preferences and risk tolerance. Key sections include:

* `GENERAL_CONFIG`

* `EXCHANGE_CONFIG`

* `FEATURE_CONFIG`

* `LABELING_CONFIG`

* `MODEL_CONFIG`

* `STRATEGY_CONFIG`

## **Usage**

The `scripts/` directory contains various utility scripts to manage the workflow. Here are detailed instructions for each:

### **trading_bot.py**

Execute the live trading bot.

`python trading_bot.py`

### **scripts/fetch_data.py**

Download historical OHLCV data from exchanges.

1.  Create a `.env` file in the project root with `BINANCE_API_KEY` and `BINANCE_API_SECRET`.

2.  Ensure `config/params.py` contains relevant configuration in `STRATEGY_CONFIG['exchange_adapter_params']`.

3.  Ensure `config/paths.py` contains 'raw_data_dir' and 'raw_data_pattern' keys in the `PATHS` dictionary.

4.  Run the script from the project root:
    `python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01 --end_date 2024-03-01`
    To fetch data up to the current time:
    `python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01`

### **scripts/generate_features.py**

Apply feature engineering to raw data.

`python scripts/generate_features.py --symbol BTCUSDT --interval 1h`

Or using the module syntax:

`python -m scripts.generate_features --symbol ADAUSDT --interval 5m`

Ensure you have run the `fetch_data` script first to obtain the raw data:

`python -m scripts.fetch_data --symbol ADAUSDT --interval 5m --start_date YYYY-MM-DD`

Ensure `config/params.py` (with `FEATURE_CONFIG` and `STRATEGY_CONFIG`) and `config/paths.py` are correctly configured.

### **scripts/create_labels.py**

Generate labels for model training using a chosen strategy.

* Generate labels using the net forward return quantile strategy:
    `python scripts/create_labels.py --symbol BTCUSDT --interval 1h --label-strategy net_forward_return_quantile`

* Generate labels using the future range dominance strategy:
    `python scripts/create_labels.py --symbol ADAUSDT --interval 5m --label-strategy future_range_dominance`

* Generate labels using the triple barrier strategy:
    `python scripts/create_labels.py --symbol ADAUSDT --interval 15m --label-strategy triple_barrier`

Ensure you have processed data files (including necessary features like ATR columns if using triple_barrier) in your `data/processed` directory, and that `config/params.py` and `config/paths.py` are correct. The feature generation script must produce an ATR column named `atr_{lookback}` (e.g., `atr_14`) matching the `vol_adj_lookback` parameter in `LABELING_CONFIG` if using the triple_barrier strategy, and save it to the processed data file.

### **scripts/train_model.py**

Train and save machine learning models.

* Train the default XGBoost model for ternary classification:
    `python scripts/train_model.py --symbol BTCUSDT --interval 1h`

* Train the RandomForest model for ternary classification:
    `python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest`

* Train the LSTM model (requires TensorFlow):
    `python scripts/train_model.py --symbol ETHUSDT --interval 15m --model lstm --train_ratio 0.7`

* Train with tuning (default for non-LSTM):
    `python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest`
    `python -m scripts.train_model --symbol ADAUSDT --interval 5m --model xgboost`

* Train skipping tuning:
    `python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest --skip_tuning`

* Train using a specific subset of features:
    `python scripts/train_model.py --symbol BTCUSDT --interval 1h --features ema_10 rsi_14 macd`

Ensure you have processed and labeled data files (including label 0) in your `data/` directory and `config/params.py` (with `MODEL_CONFIG`, `GENERAL_CONFIG`) and `config/paths.py` are correctly configured (including 'trained_models_dir' path and 'trained_model_pattern'). The feature generation script must produce an ATR column named `atr_{lookback}` (e.g., `atr_14`) matching the `vol_adj_lookback` parameter in `LABELING_CONFIG` if using the triple_barrier strategy, and save it to the processed data file.

### **scripts/analyze_model.py**

Analyze a trained machine learning model.

* Analyze the trained RandomForest model for ADAUSDT 5m data:
    `python scripts/analyze_model.py --symbol ADAUSDT --interval 5m --model random_forest`

Ensure you have run the `train_model.py` script successfully for the specified symbol, interval, and model before running this analysis script. The script will load data from `data/processed/` and `data/labeled/` and the trained model and metadata from `models/trained_models/`, all managed by `DataManager`. Analysis results (metrics and plots) will be saved to `results/analysis/model_analysis/`.

### **scripts/backtest.py**

Run deterministic backtests using trained models and strategy configurations.

* Run backtest on the test set for the trained XGBoost model for BTCUSDT 1h data:
    `python scripts/backtest.py --symbol BTCUSDT --interval 1h --model xgboost`

* Run backtest on the test set for the trained RandomForest model for ADAUSDT 5m data:
    `python scripts/backtest.py --symbol ADAUSDT --interval 5m --model random_forest`

* Run backtest on the full dataset for the trained LSTM model for ADAUSDT 5m data:
    `python scripts/backtest.py --symbol ADAUSDT --interval 5m --model lstm --backtest_mode full --train_ratio 0.7`

Ensure you have processed data files in your `data/processed` directory, labeled data files (containing labels) in your `data/labeled` directory, and trained model files in your `models/trained_models` directory. `config/params.py` (with `MODEL_CONFIG`, `GENERAL_CONFIG`, `STRATEGY_CONFIG`, `BACKTESTER_CONFIG`) and `config/paths.py` are correctly configured.

### **scripts/monte_carlo_backtest.py**

Orchestrates advanced Monte Carlo backtests using (GARCH+jump)-simulated price paths. This script allows you to evaluate your ML-based strategy's performance across many alternative market scenarios, providing a more robust assessment of risk and potential returns.

* Run Monte Carlo simulations with GARCH(1,1)+jump price path generation for the LSTM model on ADAUSDT 5m data:
    `python scripts/monte_carlo_backtest.py --symbol ADAUSDT --interval 5m --model lstm --num_simulations 100`

Ensure you have processed data files in your `data/processed` directory and trained model files in your `models/trained_models` directory. `config/params.py` (with `MODEL_CONFIG`, `GENERAL_CONFIG`, `STRATEGY_CONFIG`, `BACKTESTER_CONFIG`, `FEATURE_CONFIG`) and `config/paths.py` are correctly configured.

### **scripts/analyze_results.py**

Analyze trading results (backtest or live).

* Analyze backtest results for BTCUSDT 1h data using the xgboost model:
    `python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest`

* Analyze live trading results for ADAUSDT 5m data using the random_forest model:
    `python scripts/analyze_results.py --symbol ADAUSDT --interval 5m --model_type random_forest --results_type live`

You can optionally specify custom directories for results or analysis output:

`python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_dir /path/to/my/backtest_results --analysis_dir /path/to/my/analysis_output`

Ensure you have results files (`trades.parquet` and `equity.parquet`) in the specified or configured results directory, and that `config/paths.py` is correctly configured.

### **scripts/analyze_labels.py**

Perform key analyses on generated trading labels and processed data to inform strategy configuration (stop-loss, take-profit, max_holding_period). Results, including plots and summary statistics, will be saved to the directory specified by `PATHS['labeling_analysis_dir']` (default: `results/analysis/labeling_analysis/`) and documented in `docs/label_analysis_results.md`.

* Run all analyses for ADAUSDT 5m data with default future horizons:
    `python scripts/analyze_labels.py --symbol ADAUSDT --interval 5m`

* Run analyses for specific horizons (10, 30, 60 bars):
    `python scripts/analyze_labels.py --symbol BTCUSDT --interval 1h --future-horizons 10 30 60`

Ensure you have run the data processing and labeling steps first to generate the necessary 'processed' (including volatility_regime feature) and 'labeled' data files in your data directory.

### **scripts/convert_trades_to_json.py**

Convert trade history and OHLCV data to JSON for visualization.

`python scripts/convert_trades_to_json.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest`

You can optionally specify a custom output file path:

`python scripts/convert_trades_to_json.py --symbol ADAUSDT --interval 5m --model_type random_forest --output_file /path/to/my/output.json`

## **Contributing**

Contributions are welcome! Please follow standard GitHub flow: fork the repository, create a feature branch, commit your changes, and open a pull request.

## **License**

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE).
