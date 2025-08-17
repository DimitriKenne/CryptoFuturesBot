# **Algorithmic Futures Trading Bot: Machine Learning-Driven Trading Strategy**

A modular, extensible trading bot for cryptocurrency futures, powered by machine learning.

Automates the full workflow: data acquisition, feature engineering, model training, backtesting, and live trading.

---

## **Motivation**

* **Started:** April 2025, inspired by "A profitable trading algorithm for cryptocurrencies using a Neural Network model".
* **Background:** Built on non-systematic trading research since September 2022.
* **Development:** Rapid, iterative, and AI-assisted (Gemini, ChatGPT).
* **Goal:** Build a robust, adaptable framework for profitable quantitative strategies.

---

## **Features**

* **Modular Architecture:**  
  - Separate modules for data, features, models, strategy, notifications, exchange integration.
  - Easily extendable to new exchanges, models, or strategies.
* **Multi-Exchange Support:**  
  - Adapters for Binance Futures (others possible).
* **Advanced Feature Engineering:**  
  - Technical indicators: SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, MACD.
  - Candlestick pattern recognition (TA-Lib).
  - Custom features: Fair Value Gaps, Z-scores, ADR, trend metrics, lagged/differenced prices.
  - Volatility regime detection (Bollinger Bands width, ATR).
  - **Lookahead bias prevention:** Features are engineered dynamically on raw data streams within the MarketDataHandler for backtesting and live trading, ensuring temporal safety.
* **ML Signal Generation:**  
  - Ternary classification: Long (1), Short (-1), Neutral (0).
  - Models: LSTM, XGBoost, RandomForest.
  - Hyperparameter tuning with TimeSeriesSplit.
* **Configurable Strategy Logic:**  
  - Confidence-based entry filtering.
  - Volatility regime filtering.
  - Trend alignment filter (EMA).
  - Enable/disable long/short entries.
  - Neutral signal exit options.
  - Dynamic TP/SL (ATR or fixed %).
  - Precise position sizing (min notional, exchange precision).
* **Backtesting & Simulation:**  
  - Deterministic backtester: realistic simulation with fees, slippage, liquidation. **Now processes raw data internally via MarketDataHandler.**
  - Monte Carlo backtester: GARCH + jumps for synthetic price paths, probabilistic risk analysis. **Also feeds raw synthetic data to MarketDataHandler for internal feature engineering.**
* **Live Trading:**  
  - Real-time data integration.
  - Automated trade execution (market orders, TP/SL).
  - State management and Telegram notifications.
* **Data Management:**  
  - Historical raw data fetching.
  - **Feature-engineered and labeled data can be optionally saved/loaded (Parquet) for model training, but core backtesting now performs feature engineering dynamically.**
* **Flexible Label Generation:**  
  - Multiple strategies (triple barrier, net forward return, range dominance, clustering).
  - Label smoothing, input validation, extensible design.
* **Comprehensive Logging:**  
  - Rotating logs for all stages.

---

## **Project Structure**

```
CryptoFuturesBot/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── ta_lib-0.6.3-cp312-cp312-win_amd64.whl
├── trade_ohlcv_visualization.htm
├── trading_bot.py
├── config/
│   ├── params.py
│   ├── validator.py
│   ├── general.py
│   ├── exchange.py
│   ├── feature.py
│   ├── label.py
│   ├── model.py
│   ├── trading.py
│   ├── paths.py
│   └── notifier.py
├── data/
│   ├── labeled/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── label_analysis_results.md
│   ├── labeling_analysis.md
│   ├── labeling_strategy.md
│   ├── model_analysis.md
│   └── model_training_doc.md
├── logs/
├── models/
│   └── trained_models/
│       ├── lstm/
│       ├── random_forest/
│       └── xgboost/
├── results/
│   ├── analysis/
│   ├── backtesting/
│   └── live_trading/
├── scripts/
│   ├── analyze_labels.py
│   ├── analyze_model.py
│   ├── analyze_results.py
│   ├── backtest.py
│   ├── convert_trades_to_json.py
│   ├── create_labels.py
│   ├── fetch_data.py
│   ├── generate_features.py
│   ├── monte_carlo_backtest.py
│   └── train_model.py
└── utils/
    ├── exceptions.py
    ├── logger_config.py
    ├── notification_manager.py
    ├── adapters/
    │   ├── exchange_interface.py
    │   └── binance_futures_adapter.py
    ├── analysis/
    │   ├── performance_analyzer.py
    │   └── monte_carlo_analyzer.py
    ├── data_management/
    │   ├── data_manager.py
    │   └── market_data_handler.py
    ├── feature_engineering/
    │   ├── feature_engineer.py
    │   ├── feature_name_generator.py
    │   ├── indicator_feature_processor.py
    │   ├── price_action_feature_processor.py
    │   └── technical_indicator_calculator.py
    ├── labeling/
    │   ├── label_analyzer.py
    │   ├── label_generator.py
    │   └── strategies/
    │       ├── base_strategy.py
    │       ├── strategy1.py
    │       ├── strategy2.py
    │       ├── strategy3.py
    │       └── strategy4.py
    ├── simulation/
    │   └── price_path_simulator.py
    ├── strategy_execution/
    │   ├── backtester.py
    │   ├── trade_calculation_helpers.py
    │   ├── trade_execution_engine.py
    │   └── trading_session_manager.py
    └── training/
        ├── data_sequencer.py
        ├── model_builder.py
        ├── model_trainer.py
        └── preprocessor_builder.py
```

---

## **Key Components**

* **trading_bot.py:** Main live trading script. Orchestrates data, signals, execution, state, notifications.
* **config/params.py:** Central config import hub. Ensures consistency across schemas.
* **utils/data_management/market_data_handler.py:** Loads/fetches **raw** market data, applies features **dynamically**, generates signals.
* **utils/strategy_execution/backtester.py:** Deterministic backtesting engine. Simulates trades, tracks metrics. **Its MarketDataHandler now performs on-the-fly feature engineering.**
* **utils/simulation/price_path_simulator.py:** Generates synthetic OHLCV paths for Monte Carlo analysis.
* **utils/analysis/monte_carlo_analyzer.py:** Aggregates and visualizes Monte Carlo results.

---

## **Installation**

```sh
git clone https://github.com/DimitriKenne/CryptoFutureBot.git
cd CryptoFutureBot

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

* For TA-Lib issues, use the provided wheel for Windows or consult TA-Lib docs for other OS.

**Set up environment variables:** Create a .env file in the project root:

```
BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"
TELEGRAM_ENABLED=True
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
```

---

## **Configuration**

All parameters are managed in config/ as dataclass schemas.

Edit these files to adjust settings before running scripts or the bot.

---

## **Usage Examples**

### **Live Trading**

```sh
python trading_bot.py --symbol BTCUSDT --interval 1h --model xgboost
```

### **Data Fetching**

```sh
python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01 --end_date 2024-03-01
```

### **Feature Engineering (for Model Training or External Analysis)**

# This script processes raw data and saves features to data/processed.
# While not strictly required for backtesting/live-trading due to dynamic FE in MarketDataHandler,
# it can be used to pre-generate data for model training or standalone analysis.
```sh
python scripts/generate_features.py --symbol BTCUSDT --interval 1h
```

### **Label Generation**

```sh
python scripts/create_labels.py --symbol BTCUSDT --interval 1h --label-strategy strategy_2
```

### **Model Training**

```sh
python scripts/train_model.py --symbol BTCUSDT --interval 1h
```

### **Model Analysis**

```sh
python scripts/analyze_model.py --symbol ADAUSDT --interval 5m --model random_forest
```

### **Backtesting**

# The backtester now loads raw data and performs feature engineering internally.
# No need to run generate_features.py beforehand unless you want to pre-process.
```sh
python scripts/backtest.py --symbol BTCUSDT --interval 1h --model xgboost
```

### **Monte Carlo Backtesting**

# The Monte Carlo backtester generates raw synthetic data, which is then
# feature-engineered internally by MarketDataHandler during each simulation.
```sh
python scripts/monte_carlo_backtest.py --symbol ADAUSDT --interval 5m --model lstm --num_simulations 100
```

### **Results Analysis**

```sh
python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest
```

### **Label Analysis**

```sh
python scripts/analyze_labels.py --symbol ADAUSDT --interval 5m --label-strategy strategy_2
```

### **Convert Trades to JSON**

```sh
python scripts/convert_trades_to_json.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest
```

---

## **Contributing**

Contributions are welcome!

Fork the repo, create a feature branch, commit changes, and open a pull request.

---

## **License**

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE)
