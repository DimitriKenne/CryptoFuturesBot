# **Algorithmic Futures Trading Bot: Machine Learning-Driven Trading Strategy**

A modular, extensible trading bot for cryptocurrency futures, powered by machine learning.

Automates the full workflow: data acquisition, feature engineering, model training, backtesting, and live trading.

## **Motivation**

* **Started:** April 2025, inspired by "A profitable trading algorithm for cryptocurrencies using a Neural Network model".  
* **Background:** Built on non-systematic trading research since September 2022\.  
* **Development:** Rapid, iterative, and AI-assisted (Gemini, ChatGPT).  
* **Goal:** Build a robust, adaptable framework for profitable quantitative strategies.

## **Features**

* **Modular Architecture:** \- Separate modules for data, features, models, strategy, notifications, exchange integration.  
  * Easily extendable to new exchanges, models, or strategies.  
* **Multi-Exchange Support:** \- Adapters for Binance Futures (others possible).  
* **Advanced Feature Engineering:** \- Technical indicators: SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, MACD.  
  * Candlestick pattern recognition (TA-Lib).  
  * Custom features: Fair Value Gaps, Z-scores, ADR, trend metrics, lagged/differenced prices.  
  * Volatility regime detection (Bollinger Bands width, ATR).  
  * **Lookahead bias prevention: Features are engineered dynamically on raw data streams within the MarketDataHandler for backtesting and live trading, ensuring temporal safety.**  
* **ML Signal Generation:** \- Ternary classification: Long (1), Short (-1), Neutral (0).  
  * Models: LSTM, XGBoost, RandomForest.  
  * Hyperparameter tuning with TimeSeriesSplit.  
* **Configurable Strategy Logic:** \- Confidence-based entry filtering.  
  * Volatility regime filtering.  
  * Trend alignment filter (EMA).  
  * Enable/disable long/short entries.  
  * Neutral signal exit options.  
  * Dynamic TP/SL (ATR or fixed %).  
  * Precise position sizing (min notional, exchange precision).  
* **Backtesting & Simulation:** \- Deterministic backtester: realistic simulation with fees, slippage, liquidation. **Now processes raw data internally via MarketDataHandler.**  
  * Monte Carlo backtester: GARCH \+ jumps for synthetic price paths, probabilistic risk analysis. **Also feeds raw synthetic data to MarketDataHandler for internal feature engineering.**  
* **Live Trading:** \- Real-time data integration.  
  * Automated trade execution (market orders, TP/SL).  
  * State management and Telegram notifications.  
* **Data Management:** \- Historical raw data fetching.  
  * **Feature-engineered and labeled data can be optionally saved/loaded (Parquet) for model training, but core backtesting now performs feature engineering dynamically.**  
* **Flexible Label Generation:** \- Multiple strategies (triple barrier, net forward return, range dominance, clustering).  
  * Label smoothing, input validation, extensible design.  
* **Comprehensive Logging:** \- Rotating logs for all stages.

## **Project Structure**

CryptoFuturesBot/  
├── .env  
├── .gitignore  
├── README.md  
├── requirements.txt  
├── ta\_lib-0.6.3-cp312-cp312-win\_amd64.whl  
├── trade\_ohlcv\_visualization.htm  
├── trading\_bot.py  
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
│   ├── label\_analysis\_results.md  
│   ├── labeling\_analysis.md  
│   ├── labeling\_strategy.md  
│   ├── model\_analysis.md  
│   └── model\_training\_doc.md  
├── logs/  
├── models/  
│   └── trained\_models/  
│       ├── lstm/  
│       ├── random\_forest/  
│       └── xgboost/  
├── results/  
│   ├── analysis/  
│   ├── backtesting/  
│   └── live\_trading/  
├── scripts/  
│   ├── analyze\_labels.py  
│   ├── analyze\_model.py  
│   ├── analyze\_results.py  
│   ├── backtest.py  
│   ├── convert\_trades\_to\_json.py  
│   ├── create\_labels.py  
│   ├── fetch\_data.py  
│   ├── generate\_features.py  
│   ├── monte\_carlo\_backtest.py  
│   └── train\_model.py  
└── utils/  
    ├── exceptions.py  
    ├── logger\_config.py  
    ├── notification\_manager.py  
    ├── adapters/  
    │   ├── exchange\_interface.py  
    │   └── binance\_futures\_adapter.py  
    ├── analysis/  
    │   ├── performance\_analyzer.py  
    │   └── monte\_carlo\_analyzer.py  
    ├── data\_management/  
    │   ├── data\_manager.py  
    │   └── market\_data\_handler.py  
    ├── feature\_engineering/  
    │   ├── feature\_engineer.py  
    │   ├── feature\_name\_generator.py  
    │   ├── indicator\_feature\_processor.py  \# Added for clarity  
    │   ├── price\_action\_feature\_processor.py \# Added for clarity  
    │   └── technical\_indicator\_calculator.py  
    ├── labeling/  
    │   ├── label\_analyzer.py  
    │   ├── label\_generator.py  
    │   └── strategies/  
    │       ├── base\_strategy.py  
    │       ├── strategy1.py  
    │       ├── strategy2.py  
    │       ├── strategy3.py  
    │       └── strategy4.py  
    ├── simulation/  
    │   └── price\_path\_simulator.py  
    ├── strategy\_execution/  
    │   ├── backtester.py  
    │   ├── trade\_calculation\_helpers.py  
    │   ├── trade\_execution\_engine.py  
    │   └── trading\_session\_manager.py  
    └── training/  
        ├── data\_sequencer.py  
        ├── model\_builder.py  
        ├── model\_trainer.py  
        └── preprocessor\_builder.py \# Corrected typo in file name from \`processor\_builder.py\`

## **Key Components**

* **trading\_bot.py:** Main live trading script. Orchestrates data, signals, execution, state, notifications.  
* **config/params.py:** Central config import hub. Ensures consistency across schemas.  
* **utils/data\_management/market\_data\_handler.py:** Loads/fetches **raw** market data, applies features **dynamically**, generates signals.  
* **utils/strategy\_execution/backtester.py:** Deterministic backtesting engine. Simulates trades, tracks metrics. **Its MarketDataHandler now performs on-the-fly feature engineering.**  
* **utils/simulation/price\_path\_simulator.py:** Generates synthetic OHLCV paths for Monte Carlo analysis.  
* **utils/analysis/monte\_carlo\_analyzer.py:** Aggregates and visualizes Monte Carlo results.

## **Installation**

git clone https://github.com/DimitriKenne/CryptoFutureBot.git  
cd CryptoFutureBot

python \-m venv .venv  
\# Windows:  
.venv\\Scripts\\activate  
\# Linux/Mac:  
source .venv/bin/activate

pip install \-r requirements.txt

* For TA-Lib issues, use the provided wheel for Windows or consult TA-Lib docs for other OS.

**Set up environment variables:** Create a .env file in the project root:

BINANCE\_API\_KEY="YOUR\_BINANCE\_API\_KEY"  
BINANCE\_API\_SECRET="YOUR\_BINANCE\_API\_SECRET"  
TELEGRAM\_ENABLED=True  
TELEGRAM\_BOT\_TOKEN="YOUR\_TELEGRAM\_BOT\_TOKEN"  
TELEGRAM\_CHAT\_ID="YOUR\_TELEGRAM\_CHAT\_ID"

## **Configuration**

All parameters are managed in config/ as dataclass schemas.

Edit these files to adjust settings before running scripts or the bot.

## **Usage Examples**

### **Live Trading**

python trading\_bot.py \--symbol BTCUSDT \--interval 1h \--model xgboost

### **Data Fetching**

python scripts/fetch\_data.py \--symbol ADAUSDT \--interval 5m \--start\_date 2024-01-01 \--end\_date 2024-03-01

### **Feature Engineering (for Model Training or External Analysis)**

\# This script processes raw data and saves features to data/processed.  
\# While not strictly required for backtesting/live-trading due to dynamic FE in MarketDataHandler,  
\# it can be used to pre-generate data for model training or standalone analysis.  
python scripts/generate\_features.py \--symbol BTCUSDT \--interval 1h

### **Label Generation**

python scripts/create\_labels.py \--symbol BTCUSDT \--interval 1h \--label-strategy strategy\_2

### **Model Training**

python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h

### **Model Analysis**

python scripts/analyze\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest

### **Backtesting**

\# The backtester now loads raw data and performs feature engineering internally.  
\# No need to run generate\_features.py beforehand unless you want to pre-process.  
python scripts/backtest.py \--symbol BTCUSDT \--interval 1h \--model xgboost

### **Monte Carlo Backtesting**

\# The Monte Carlo backtester generates raw synthetic data, which is then  
\# feature-engineered internally by MarketDataHandler during each simulation.  
python scripts/monte\_carlo\_backtest.py \--symbol ADAUSDT \--interval 5m \--model lstm \--num\_simulations 100

### **Results Analysis**

python scripts/analyze\_results.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_type backtest

### **Label Analysis**

python scripts/analyze\_labels.py \--symbol ADAUSDT \--interval 5m \--label-strategy strategy\_2

### **Convert Trades to JSON**

python scripts/convert\_trades\_to\_json.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_type backtest

## **Contributing**

Contributions are welcome\!

Fork the repo, create a feature branch, commit changes, and open a pull request.

## **License**

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE)