# **Algorithmic Futures Trading Bot: Machine Learning-Driven Trading Strategy**

This project presents a sophisticated algorithmic trading bot designed for futures markets, leveraging advanced machine learning techniques for predictive signal generation and robust risk management. Built with a modular and extensible architecture, this bot facilitates end-to-end automation of trading strategies, from data acquisition and feature engineering to model training, backtesting, and live execution.

---

## **Motivation & Background**

This project was initiated in **April 2025**, directly inspired by the research presented in the paper "A profitable trading algorithm for cryptocurrencies using a Neural Network model". Building upon a foundational understanding of non-systematic trading strategies, which I had been exploring since **September 2022**, this systematic framework was rapidly developed to its current state by **early May 2025**. The iterative development process involved numerous versions and was significantly accelerated by leveraging modern AI tools like Gemini and ChatGPT for efficient ideation and implementation. The primary objective of this ongoing work is to build a robust and adaptable framework for exploring and identifying profitable quantitative trading strategies.

---

## **Features**

* **Modular and Scalable Architecture**
  - Clear separation of core functionalities into distinct modules (data management, feature engineering, model training, strategy execution, notifications, exchange integration).
  - Easily adaptable to new exchanges, models, or strategy components.
* **Multi-Exchange Compatibility**
  - Dedicated adapters for seamless integration with various cryptocurrency futures exchanges (currently implemented for Binance Futures).
* **Advanced Feature Engineering Pipeline**
  - Transforms raw OHLCV data into a rich set of predictive features:
    - **Comprehensive Technical Indicators:** SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, MACD (multi-period).
    - **Candlestick Pattern Recognition:** Automated detection using TA-Lib.
    - **Custom Statistical & Price Action Features:** Fair Value Gaps (FVG), Z-scores, Average Daily Range (ADR), trend strength metrics, lagged prices, differenced prices.
    - **Temporal Safety Validation:** Prevents lookahead bias by ensuring all features are derived exclusively from past data.
    - **Volatility Regime Calculation:** Market regime (low/medium/high) based on Bollinger Bands width or ATR.
* **Machine Learning-Driven Signal Generation**
  - **Ternary Classification:** Models classify future price movements as Long (1), Short (-1), or Neutral (0).
  - **Supported Models:** LSTM, XGBoost, RandomForest.
  - **Hyperparameter Tuning:** RandomizedSearchCV with TimeSeriesSplit for robust model selection.
* **Configurable and Dynamic Strategy Logic**
  - **Confidence-Based Entry Filtering:** Filters trades based on model prediction probability (confidence score).
  - **Adaptive Volatility Regime Filtering:** Dynamically adjusts trade entry and max holding periods based on market volatility regime.
  - **Trend Alignment Filter:** EMA-based filter to ensure trades are aligned with the dominant market trend.
  - **Directional Control:** Enable/disable long and short entries independently.
  - **Neutral Signal Management:** Configurable behavior for exiting positions on neutral signals.
  - **Dynamic Take Profit/Stop Loss (TP/SL):** TP/SL levels calculated dynamically based on ATR or fixed percentages.
* **Robust Backtesting Engine**
  - **Realistic Simulation:** Accounts for trading fees, slippage, and liquidation mechanics.
  - **Detailed Performance Metrics:** Total return, CAGR, max drawdown, win rate, profit factor, average PnL per trade.
  - **Trade Management Simulation:** Position opening, closing, reversal logic with fee and margin handling.
  - **Persistent Results:** Trade logs (Parquet) and summary metrics (JSON) for post-analysis.
* **Live Trading Capabilities**
  - **Real-time Data Integration:** Direct connection to exchange data feeds.
  - **Automated Trade Execution:** Market orders, open position management, TP/SL order placement/cancellation.
  - **Resilient State Management:** Periodic state saving for capital and open positions.
  - **Critical Event Notifications:** Telegram alerts for trade executions, errors, and bot status.
* **Data Management & Persistence**
  - **Data Fetching:** Scripts for historical OHLCV data.
  - **Data Processing & Storage:** Raw, processed (features), and labeled data in Parquet format.
* **Flexible Label Generation**
  - **Strategy 1 (Triple Barrier):** Labels assigned based on which barrier (profit, loss, time) is hit first; supports dynamic ATR-based barriers.
  - **Strategy 2 (Net Forward Return Quantile):** Labels based on net percentage return over a forward window, targeting genuinely profitable moves.
  - **Strategy 3 (Future Range Dominance):** Labels based on dominance of net profit in one direction over the other, with minimum profitability threshold.
  - **Strategy 4 (Clustering-Based Labeling):** Labels based on market regimes identified by clustering of technical features and mapping to historical profitability.
  - **Label Propagation Smoothing:** Configurable min_holding_period to smooth raw labels and filter out noise.
* **Comprehensive Logging**
  - Rotating logs for all stages of bot operation, backtesting, and data processing.

---

## **Project Structure**

```
.
├── .env                        # Environment variables (API keys, secrets)
├── .gitignore                  # Git ignore file
├── README.md                   # Project README
├── requirements.txt            # Python dependencies
├── ta_lib-0.6.3-cp312-cp312-win_amd64.whl # TA-Lib wheel (Windows specific)
├── trade_ohlcv_visualization.htm # HTML for visualizing trades
├── trading_bot.py              # Main live trading bot script
├── adapters/                   # Exchange API adapters
│   ├── __init__.py
│   └── binance_futures_adapter.py
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── params.py               # Central import hub for configuration schemas
│   ├── general_config_schema.py # Schema and default values for General settings
│   ├── exchange_config_schema.py # Schema and default values for Exchange settings
│   ├── feature_config_schema.py # Schema and default values for Feature Engineering
│   ├── label_config_schema.py   # Schema and default values for Labeling
│   ├── model_config_schema.py   # Schema and default values for Model Training
│   ├── strategy_config_schema.py # Schema and default values for Strategy Logic
│   ├── notifier_config_schema.py # Schema and default values for Notifier settings
│   └── paths.py                # Defines project paths
├── data/                       # Data storage
│   ├── labeled/                # Labeled data for model training
│   ├── processed/              # Processed data with engineered features
│   └── raw/                    # Raw OHLCV data
├── docs/                       # Project documentation
│   ├── label_analysis_results.md
│   ├── labeling_analysis.md
│   ├── labeling_strategy.md
│   ├── model_analysis.md
│   └── model_training_doc.md
├── logs/                       # Application logs
├── models/                     # Trained machine learning models
│   └── trained_models/
│       ├── lstm/
│       ├── random_forest/
│       └── xgboost/
├── results/                    # Backtesting and live trading results
│   ├── analysis/
│   ├── backtesting/
│   └── live_trading/
├── scripts/                    # Utility scripts for workflow automation
│   ├── analyze_labels.py
│   ├── analyze_model.py
│   ├── backtest.py
│   ├── convert_trades_to_json.py
│   ├── create_labels.py
│   ├── fetch_data.py
│   ├── generate_features.py
│   ├── monte_carlo_backtest.py
│   ├── setup_environment.py    # NEW: Potential script for environment setup (e.g., creating dirs)
│   └── train_model.py
└── utils/                      # Core utility modules
    ├── __init__.py
    ├── backtester.py
    ├── data_manager.py
    ├── exceptions.py
    ├── exchange_interface.py
    ├── logger_config.py
    ├── model_trainer.py
    ├── notification_manager.py
    ├── results_analyzer.py
    ├── feature_engineering/
    │   ├── __init__.py
    │   ├── feature_engineer.py
    │   ├── feature_name_generator.py
    │   └── technical_indicator_calculator.py
    └── labeling/
        ├── __init__.py
        ├── label_analyzer.py
        ├── label_generator.py
        └── strategies/
            ├── __init__.py
            ├── base_strategy.py
            ├── strategy1.py
            ├── strategy2.py
            ├── strategy3.py
            └── strategy4.py
```

---

## **Key Components Explained**

### **trading_bot.py**

Orchestrates the live trading process:

* **Initialization:** Loads configuration, logging, exchange adapters, feature engineers, trained ML model, and previous bot state.
* **Main Loop:** Fetches new candle data, processes features, gets model signal, and executes trade logic.
* **Signal Processing:** Uses _get_signal and _apply_entry_filters for model predictions and strategy rules.
* **Trade Execution:** Manages position opening, closing, reversal, sizing, dynamic TP/SL, and trade tracking.
* **State Management:** Periodically saves capital and open positions to JSON, appends closed trades to Parquet.

---

### **config/params.py**

This file now serves as a **central import hub** for all granular configuration schemas. It allows other modules to easily access a comprehensive set of parameters defined and validated in their respective dedicated files. The actual parameters are defined in:

* config/general_config_schema.py
* config/exchange_config_schema.py
* config/feature_config_schema.py
* config/label_config_schema.py
* config/model_config_schema.py
* config/strategy_config_schema.py
* config/notifier_config_schema.py

---

### **utils/feature_engineering/feature_engineer.py**

Transforms raw OHLCV data into technical/statistical features:

* **Indicator Calculation:** SMA, EMA, RSI, Stochastic, CCI, MFI, Bollinger Bands, ATR, OBV, CMF, MACD (multi-period).
* **Pattern Recognition:** Candlestick patterns via TA-Lib.
* **Custom Features:** FVG, Z-scores, ADR, trend strength, lagged/differenced prices, support/resistance.
* **Temporal Safety:** All features calculated using only past data; validation checks for lookahead bias.
* **Configurable Periods:** Multiple lookback periods via config/feature_config_schema.py.
* **Volatility Regime Calculation:** Market regime (low/medium/high) based on Bollinger Bands width or ATR.

---

### **utils/backtester.py**

Robust backtesting engine:

* **Initialization:** Merges configs for simulation (strategy, backtester, exchange, feature).
* **Data Preparation:** Validates input, calculates missing indicators, aligns signals/probabilities.
* **Simulation Loop:** Applies strategy logic bar by bar.
* **Trade Logic:** Entry filters, position sizing, dynamic TP/SL, exit conditions (SL, TP, liquidation, max holding, neutral), reversal logic.
* **Results & Metrics:** Tracks trades, PnL, equity curve, performance metrics, PnL consistency check.
* **Saving Results:** Trade logs (Parquet) and metrics (JSON).

---

### **utils/labeling/label_generator.py**

Creates target labels for ML models:

* **Strategy-Based Labeling:** Multiple strategies selectable via command-line argument:
  * **strategy_1 (Triple Barrier):** Labels assigned based on which barrier (profit, loss, time) is hit first; supports dynamic ATR-based barriers.
  * **strategy_2 (Net Forward Return Quantile):** Labels based on net percentage return over a forward window, targeting genuinely profitable moves.
  * **strategy_3 (Future Range Dominance):** Labels based on dominance of net profit in one direction over the other, with minimum profitability threshold.
  * **strategy_4 (Clustering-Based Labeling):** Labels based on market regimes identified by clustering of technical features and mapping to historical profitability.
* **Label Propagation Smoothing:** Configurable min_holding_period to smooth raw labels and filter out noise.
* **Input Validation:** Checks for required OHLCV/features.
* **Extensible Design:** Easily add new strategies via BaseLabelingStrategy.

---

## **Installation**

1. **Clone the repository:**
   ```sh
   git clone https://github.com/DimitriKenne/CryptoFutureBot.git
   cd CryptoFutureBot
   ```

2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   *Note: For TA-Lib issues, use the provided wheel for Windows or consult TA-Lib docs for other OS.*

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
   BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"
   TELEGRAM_ENABLED=True
   TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
   TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
   ```
   Add other credentials as needed.

---

## **Configuration**

All core parameters are now defined and managed within dedicated configuration schema files in the config/ directory. These dataclass-based schemas provide structure, default values, and validation for different aspects of the project. Review and adjust settings in these files as needed before running scripts or the bot:

* config/params.py (Central import hub for all schemas)
* config/general_config_schema.py
* config/exchange_config_schema.py
* config/feature_config_schema.py
* config/label_config_schema.py
* config/model_config_schema.py
* config/strategy_config_schema.py
* config/notifier_config_schema.py

---

## **Usage**

The scripts/ directory contains utility scripts for workflow automation. Example usage:

### **trading_bot.py**

Run the live trading bot:
```sh
python trading_bot.py
```

---

### **scripts/fetch_data.py**

Download historical OHLCV data:
```sh
python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01 --end_date 2024-03-01
python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01
```

---

### **scripts/generate_features.py**

Apply feature engineering:
```sh
python scripts/generate_features.py --symbol BTCUSDT --interval 1h
python -m scripts.generate_features --symbol ADAUSDT --interval 5m
```

---

### **scripts/create_labels.py**

Generate labels for model training:
```sh
python scripts/create_labels.py --symbol BTCUSDT --interval 1h --label-strategy strategy_2
python scripts/create_labels.py --symbol ADAUSDT --interval 5m --label-strategy strategy_3
python scripts/create_labels.py --symbol ADAUSDT --interval 15m --label-strategy strategy_1
python scripts/create_labels.py --symbol ETHUSDT --interval 1h --label-strategy strategy_4
```

---

### **scripts/train_model.py**

Train and save ML models:
```sh
python scripts/train_model.py --symbol BTCUSDT --interval 1h
python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest
python scripts/train_model.py --symbol ETHUSDT --interval 15m --model lstm --train_ratio 0.7
python scripts/train_model.py --symbol ADAUSDT --interval 5m --model random_forest --skip_tuning
python scripts/train_model.py --symbol BTCUSDT --interval 1h --features ema_10 rsi_14 macd
```

---

### **scripts/analyze_model.py**

Analyze a trained model:
```sh
python scripts/analyze_model.py --symbol ADAUSDT --interval 5m --model random_forest
```

---

### **scripts/backtest.py**

Run deterministic backtests:
```sh
python scripts/backtest.py --symbol BTCUSDT --interval 1h --model xgboost
python scripts/backtest.py --symbol ADAUSDT --interval 5m --model random_forest
python scripts/backtest.py --symbol ADAUSDT --interval 5m --model lstm --backtest_mode full --train_ratio 0.7
```

---

### **scripts/monte_carlo_backtest.py**

Run Monte Carlo backtests:
```sh
python scripts/monte_carlo_backtest.py --symbol ADAUSDT --interval 5m --model lstm --num_simulations 100
```

---

### **scripts/analyze_results.py**

Analyze trading results:
```sh
python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest
python scripts/analyze_results.py --symbol ADAUSDT --interval 5m --model_type random_forest --results_type live
python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_dir /path/to/my/backtest_results --analysis_dir /path/to/my/analysis_output
```

---

### **scripts/analyze_labels.py**

Analyze generated trading labels:
```sh
python scripts/analyze_labels.py --symbol ADAUSDT --interval 5m --label-strategy strategy_2
python scripts/analyze_labels.py --symbol BTCUSDT --interval 1h --label-strategy strategy_1 --future-horizons 10 30 60
```

---

### **scripts/convert_trades_to_json.py**

Convert trade history and OHLCV data to JSON:
```sh
python scripts/convert_trades_to_json.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest
python scripts/convert_trades_to_json.py --symbol ADAUSDT --interval 5m --model_type random_forest --output_file /path/to/my/output.json
```

---

## **Contributing**

Contributions are welcome! Please follow standard GitHub flow: fork the repository, create a feature branch, commit your changes, and open a pull request.

---

## **License**

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE)
