# Algorithmic Futures Trading Bot: Machine Learning-Driven Trading Strategy

This project presents a sophisticated algorithmic trading bot designed for futures markets, leveraging advanced machine learning techniques for predictive signal generation and robust risk management. Built with a modular and extensible architecture, this bot facilitates end-to-end automation of trading strategies, from data acquisition and feature engineering to model training, backtesting, and live execution.

## Motivation & Background

This project was initiated in **April 2025**, directly inspired by the research presented in the paper "A profitable trading algorithm for cryptocurrencies using a Neural Network model". Building upon a foundational understanding of non-systematic trading strategies, which I had been exploring since **September 2022**, this systematic framework was rapidly developed to its current state by **early May 2025**. The iterative development process involved numerous versions and was significantly accelerated by leveraging modern AI tools like Gemini and ChatGPT for efficient ideation and implementation. The primary objective of this ongoing work is to build a robust and adaptable framework for exploring and identifying profitable quantitative trading strategies.

## Features

* **Modular and Scalable Architecture**  
  * Clear separation of core functionalities into distinct modules (data management, feature engineering, model training, strategy execution, notifications, exchange integration).  
  * Easily adaptable to new exchanges, models, or strategy components.  
* **Multi-Exchange Compatibility**  
  * Dedicated adapters for seamless integration with various cryptocurrency futures exchanges (currently implemented for Binance Futures).  
* **Advanced Feature Engineering Pipeline**  
  * Transforms raw OHLCV data into a rich set of predictive features:  
    * **Comprehensive Technical Indicators:** SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, MACD (multi-period).  
    * **Candlestick Pattern Recognition:** Automated detection using TA-Lib.  
    * **Custom Statistical & Price Action Features:** Fair Value Gaps (FVG), Z-scores, Average Daily Range (ADR), trend strength metrics, lagged prices, differenced prices.  
    * **Temporal Safety Validation:** Prevents lookahead bias by ensuring all features are derived exclusively from past data.  
    * **Volatility Regime Calculation:** Market regime (low/medium/high) based on Bollinger Bands width or ATR.  
* **Machine Learning-Driven Signal Generation**  
  * **Ternary Classification:** Models classify future price movements as Long (1), Short (-1), or Neutral (0).  
  * **Supported Models:** LSTM, XGBoost, RandomForest.  
  * **Hyperparameter Tuning:** RandomizedSearchCV with TimeSeriesSplit for robust model selection.  
* **Configurable and Dynamic Strategy Logic**  
  * **Confidence-Based Entry Filtering:** Filters trades based on model prediction probability (confidence score), with robust handling for missing probabilities.  
  * **Adaptive Volatility Regime Filtering:** Dynamically adjusts trade entry and max holding periods based on market volatility regime.  
  * **Trend Alignment Filter:** EMA-based filter to ensure trades are aligned with the dominant market trend.  
  * **Directional Control:** Enable/disable long and short entries independently.  
  * **Neutral Signal Management:** Configurable behavior for exiting positions on neutral signals.  
  * **Dynamic Take Profit/Stop Loss (TP/SL):** TP/SL levels calculated dynamically based on ATR or fixed percentages, with minimum distance from liquidation.  
  * **Precise Position Sizing**: Advanced logic to calculate trade quantity, ensuring adherence to minimum notional values and exchange precision.  
* **Robust Backtesting Engine**  
  * **Realistic Simulation:** Accounts for trading fees, slippage, and liquidation mechanics.  
  * **Detailed Performance Metrics:** Total return, CAGR, max drawdown, win rate, profit factor, average PnL per trade, with proper annualization via bars\_per\_year.  
  * **Trade Management Simulation:** Position opening, closing, reversal logic with accurate fee and margin handling, and precise capital tracking.  
  * **Persistent Results:** Trade logs (Parquet), **equity curve (Parquet)**, and summary metrics (JSON) for post-analysis.  
* **Live Trading Capabilities**  
  * **Real-time Data Integration:** Direct connection to exchange data feeds.  
  * **Automated Trade Execution:** Market orders, open position management, TP/SL order placement/cancellation.  
  * **Resilient State Management:** Periodic state saving for capital and open positions.  
  * **Critical Event Notifications:** Telegram alerts for trade executions, errors, and bot status.  
* **Data Management & Persistence**  
  * **Data Fetching:** Scripts for historical OHLCV data.  
  * **Data Processing & Storage:** Raw, processed (features), and labeled data in Parquet format.  
* **Flexible Label Generation**  
  * **Strategy 1 (Triple Barrier):** Labels assigned based on which barrier (profit, loss, time) is hit first; supports dynamic ATR-based barriers.  
  * **Strategy 2 (Net Forward Return Quantile):** Labels based on net percentage return over a forward window, targeting genuinely profitable moves.  
  * **Strategy 3 (Future Range Dominance):** Labels based on dominance of net profit in one direction over the other, with minimum profitability threshold.  
  * **Strategy 4 (Clustering-Based Labeling):** Labels based on market regimes identified by clustering of technical features and mapping to historical profitability.  
  * **Label Propagation Smoothing:** Configurable min\_holding\_period to smooth raw labels and filter out noise.  
* **Comprehensive Logging**  
  * Rotating logs for all stages of bot operation, backtesting, and data processing.

## Project Structure

.  
├── .env                        \# Environment variables (API keys, secrets)  
├── .gitignore                  \# Git ignore file  
├── README.md                   \# Project README  
├── requirements.txt            \# Python dependencies  
├── ta\_lib-0.6.3-cp312-cp312-win\_amd64.whl \# TA-Lib wheel (Windows specific)  
├── trade\_ohlcv\_visualization.htm \# HTML for visualizing trades  
├── trading\_bot.py              \# Main live trading bot script  
├── adapters/                   \# Exchange API adapters  
│   ├── \_\_init\_\_.py  
│   └── binance\_futures\_adapter.py  
├── config/                     \# Configuration files  
│   ├── \_\_init\_\_.py  
│   ├── params.py               \# Central import hub for configuration schemas  
│   ├── general\_config\_schema.py \# Schema and default values for General settings  
│   ├── exchange\_config\_schema.py \# Schema and default values for Exchange settings  
│   ├── feature\_config\_schema.py \# Schema and default values for Feature Engineering  
│   ├── label\_config\_schema.py   \# Schema and default values for Labeling  
│   ├── model\_config\_schema.py   \# Schema and default values for Model Training  
│   ├── strategy\_config\_schema.py \# Schema and default values for Strategy Logic  
│   ├── notifier\_config\_schema.py \# Schema and default values for Notifier settings  
│   ├── backtest\_config\_schema.py \# Schema and default values for Backtesting settings  
│   └── paths.py                \# Defines project paths  
├── data/                       \# Data storage  
│   ├── labeled/                \# Labeled data for model training  
│   ├── processed/              \# Processed data with engineered features  
│   └── raw/                    \# Raw OHLCV data  
├── docs/                       \# Project documentation  
│   ├── label\_analysis\_results.md  
│   ├── labeling\_analysis.md  
│   ├── labeling\_strategy.md  
│   ├── model\_analysis.md  
│   └── model\_training\_doc.md  
├── logs/                       \# Application logs  
├── models/                     \# Trained machine learning models  
│   └── trained\_models/  
│       ├── lstm/  
│       ├── random\_forest/  
│       └── xgboost/  
├── results/                    \# Backtesting and live trading results  
│   ├── analysis/  
│   ├── backtesting/  
│   └── live\_trading/  
├── scripts/                    \# Utility scripts for workflow automation  
│   ├── analyze\_labels.py  
│   ├── analyze\_model.py  
│   ├── backtest.py  
│   ├── convert\_trades\_to\_json.py  
│   ├── create\_labels.py  
│   ├── fetch\_data.py  
│   ├── generate\_features.py  
│   ├── monte\_carlo\_backtest.py  
│   ├── setup\_environment.py  
│   └── train\_model.py  
└── utils/                      \# Core utility modules  
    ├── \_\_init\_\_.py  
    ├── backtest\_engine/        \# Renamed from 'backtester.py' for modularity  
    │   ├── \_\_init\_\_.py  
    │   └── simulator.py  
    ├── bot\_state/              \# New directory for state management  
    │   ├── \_\_init\_\_.py  
    │   └── handler.py  
    ├── data\_processing/        \# Data handling, processing, and time synchronization
    │   ├── \_\_init\_\_.py  
    │   ├── data\_manager.py  
    │   └── ohlcv\_processor.py  
    ├── exceptions.py  
    ├── exchange\_interface.py  
    ├── logger\_config.py  
    ├── notification\_manager.py  
    ├── results\_analyzer.py  
    ├── feature\_engineering/  
    │   ├── \_\_init\_\_.py  
    │   ├── feature\_engineer.py  
    │   ├── feature\_name\_generator.py  
    │   └── technical\_indicator\_calculator.py  
    ├── labeling/  
    │   ├── \_\_init\_\_.py  
    │   ├── label\_analyzer.py  
    │   ├── label\_generator.py  
    │   └── strategies/  
    │       ├── \_\_init\_\_.py  
    │       ├── base\_strategy.py  
    │       ├── strategy1.py  
    │       ├── strategy2.py  
    │       ├── strategy3.py  
    │       └── strategy4.py  
    ├── strategy\_execution/     \# New directory for strategy components  
    │   ├── \_\_init\_\_.py  
    │   ├── entry\_filters.py  
    │   ├── exit\_conditions.py  
    │   └── trade\_manager.py  
    ├── trade\_core/             \# New directory for fundamental trade logic  
    │   ├── \_\_init\_\_.py  
    │   ├── financial\_math.py  
    │   ├── liquidation.py  
    │   ├── order\_precision.py  
    │   └── position\_sizing.py  
    └── training/  
        ├── \_\_init\_\_.py  
        ├── model\_trainer.py  
        ├── model\_builder.py  
        ├── data\_sequencer.py  
        └── processor\_builder.py

## Key Components Explained

### trading\_bot.py

Orchestrates the live trading process:

* **Initialization:** Loads configuration, logging, exchange adapters, feature engineers, trained ML model, and previous bot state.  
* **Main Loop:** Fetches new candle data, processes features, gets model signal, and executes trade logic.  
* **Signal Processing:** Uses \_get\_signal and \_apply\_entry\_filters for model predictions and strategy rules.  
* **Trade Execution:** Manages position opening, closing, reversal, sizing, dynamic TP/SL, and trade tracking.  
* **State Management:** Periodically saves capital and open positions to JSON, appends closed trades to Parquet.

### config/params.py

This file now serves as a **central import hub** for all granular configuration schemas. It allows other modules to easily access a comprehensive set of parameters defined and validated in their respective dedicated files. It also contains **cross-schema consistency checks** to ensure related parameters across different configurations are aligned (e.g., ensuring LSTM input timesteps match feature sequence lengths). The actual parameters are defined in:

* config/general\_config\_schema.py  
* config/exchange\_config\_schema.py  
* config/feature\_config\_schema.py  
* config/label\_config\_schema.py  
* config/model\_config\_schema.py  
* config/strategy\_config\_schema.py  
* config/notifier\_config\_schema.py  
* config/backtest\_config\_schema.py

### config/backtest\_config\_schema.py

This schema defines configuration parameters specifically for the **backtesting engine**, including maintenance margin rates, liquidation fees, maximum concurrent trades, and flags for saving various backtest results (trades, equity curve, metrics). It also allows for overriding specific strategy parameters solely for backtesting purposes.

### utils/data\_processing/ohlcv\_processor.py

Centralizes OHLCV data processing tasks, including validation, numeric type conversion, **calculation of missing indicators (as fallback if not pre-generated)**, and **robust alignment of model predictions and probability scores**. It ensures data is consistently prepared for both backtesting and live trading, properly renaming probability columns (e.g., from numeric \-1 to string 'proba\_-1').

### utils/trade\_core/order\_precision.py

Manages **exchange-specific requirements for price and quantity precision**, including rounding methods. It now includes round\_quantity\_up to ensure calculated quantities meet minimum thresholds by rounding upwards to the nearest valid precision step, preventing "notional below minimum" errors. It also validates minimum/maximum order quantities and notional values.

### utils/trade\_core/position\_sizing.py

Calculates the appropriate trade quantity based on risk management parameters (capital, risk per trade, stop-loss distance) and leverage. This module now incorporates a **smarter approach to meeting minimum notional values**, potentially adjusting the calculated quantity upwards to ensure the order is valid for the exchange while balancing risk.

### utils/trade\_core/financial\_math.py

Provides core financial calculation utilities for trading, including accurate **PnL calculation (gross and net)**, fee computation (entry, exit, total), margin calculation, and capital adjustments. It handles both long and short positions precisely.

### utils/trade\_core/liquidation.py

Handles calculations for **estimating liquidation prices** for futures positions, considering entry price, leverage, and maintenance margin. It also provides a utility to **check if a stop-loss price is sufficiently safe from liquidation**, preventing trades with dangerously tight stops.

### utils/strategy\_execution/trade\_manager.py

Orchestrates the lifecycle of a single trade. It calls upon lower-level trade\_core utilities (FinancialMath, PositionSizing, Liquidation, OrderPrecision) to **calculate comprehensive trade parameters** (entry, SL/TP, quantity, margin, liquidation price, max holding bars) and to **process trade closures**. It ensures all configuration parameters are correctly passed to these underlying utilities.

### utils/strategy\_execution/entry\_filters.py

Applies various **entry conditions (filters)** to a potential trade signal, ensuring a trade only proceeds if all conditions are met. This includes confidence thresholds, volatility regime filters, and trend alignment. It now correctly processes model probability values by **expecting and checking for consistent probability column names** (e.g., 'proba\_1').

### utils/strategy\_execution/exit\_conditions.py

Evaluates various conditions to determine if an open trade should be exited. This includes checking for Stop Loss (SL) and Take Profit (TP) hits, adherence to maximum holding periods, neutral model signals, and **dynamically adjusting SL/TP levels if they are too close to the estimated liquidation price** to maintain a safety buffer.

### utils/training/model\_trainer.py

Manages the end-to-end process of **training, evaluating, and loading machine learning models** for the trading bot. This includes data splitting, preprocessing, model fitting (with optional hyperparameter tuning), performance evaluation, and persistence of the trained model pipeline (model \+ preprocessor) and its metadata.

### utils/training/model\_builder.py

Responsible for **constructing and configuring the specific machine learning models** (e.g., XGBoost, RandomForest, LSTM) based on the parameters defined in model\_config\_schema.py. It abstracts the model instantiation details away from the core ModelTrainer.

### utils/training/data\_sequencer.py

Handles the **preparation and sequencing of time-series data** for models that require sequential inputs (like LSTMs). It ensures data is correctly formatted into sequences, potentially with padding or rolling window techniques, before being fed into the model's preprocessing pipeline.

### utils/training/processor\_builder.py

Focuses on **building and configuring the data preprocessing pipeline** (e.g., scaling, PCA, handling categorical features) using scikit-learn's ColumnTransformer or similar tools. It ensures that features are transformed consistently before model training and prediction.

### utils/backtest\_engine/simulator.py

The core **backtesting engine**, now located at utils/backtest\_engine/simulator.py. It simulates trading operations bar by bar, integrating all strategy components. It features **robust data preparation**, precise **capital and margin tracking**, accurate **PnL and fee calculations**, and ensures **correct saving of trade history and the equity curve** (now saved as a DataFrame for compatibility). It also calculates and saves comprehensive performance metrics, using bars\_per\_year from strategy config for proper annualization.

### utils/labeling/label\_generator.py

Creates target labels for ML models:

* **Strategy-Based Labeling:** Multiple strategies selectable via command-line argument:  
  * **strategy\_1 (Triple Barrier):** Labels assigned based on which barrier (profit, loss, time) is hit first; supports dynamic ATR-based barriers.  
  * **strategy\_2 (Net Forward Return Quantile):** Labels based on net percentage return over a forward window, targeting genuinely profitable moves.  
  * **strategy\_3 (Future Range Dominance):** Labels based on dominance of net profit in one direction over the other, with minimum profitability threshold.  
  * **strategy\_4 (Clustering-Based Labeling):** Labels based on market regimes identified by clustering of technical features and mapping to historical profitability.  
* **Label Propagation Smoothing:** Configurable min\_holding\_period to smooth raw labels and filter out noise.  
* **Input Validation:** Checks for required OHLCV/features.  
* **Extensible Design:** Easily add new strategies via BaseLabelingStrategy.

## Installation

1. **Clone the repository:**  
   git clone https://github.com/DimitriKenne/CryptoFutureBot.git  
   cd CryptoFutureBot

2. **Create a virtual environment (recommended):**  
   python \-m venv .venv  
   \# On Windows:  
   .venv\\Scripts\\activate  
   \# On Linux/Mac:  
   source .venv/bin/activate

3. **Install dependencies:**  
   pip install \-r requirements.txt

   *Note: For TA-Lib issues, use the provided wheel for Windows or consult TA-Lib docs for other OS.*  
4. Set up environment variables:  
   Create a .env file in the project root:  
   BINANCE\_API\_KEY="YOUR\_BINANCE\_API\_KEY"  
   BINANCE\_API\_SECRET="YOUR\_BINANCE\_API\_SECRET"  
   TELEGRAM\_ENABLED=True  
   TELEGRAM\_BOT\_TOKEN="YOUR\_TELEGRAM\_BOT\_TOKEN"  
   TELEGRAM\_CHAT\_ID="YOUR\_TELEGRAM\_CHAT\_ID"

   Add other credentials as needed.

## Configuration

All core parameters are now defined and managed within dedicated configuration schema files in the config/ directory. These dataclass-based schemas provide structure, default values, and validation for different aspects of the project. Review and adjust settings in these files as needed before running scripts or the bot:

* config/params.py (Central import hub for all schemas)  
* config/general\_config\_schema.py  
* config/exchange\_config\_schema.py  
* config/feature\_config\_schema.py  
* config/label\_config\_schema.py  
* config/model\_config\_schema.py  
* config/strategy\_config\_schema.py  
* config/notifier\_config\_schema.py  
* config/backtest\_config\_schema.py

## Usage

The scripts/ directory contains utility scripts for workflow automation. Example usage:

### trading\_bot.py

Run the live trading bot:

python trading\_bot.py

### scripts/fetch\_data.py

Download historical OHLCV data:

python scripts/fetch\_data.py \--symbol ADAUSDT \--interval 5m \--start\_date 2024-01-01 \--end\_date 2024-03-01  
python scripts/fetch\_data.py \--symbol ADAUSDT \--interval 5m \--start\_date 2024-01-01

### scripts/generate\_features.py

Apply feature engineering:

python scripts/generate\_features.py \--symbol BTCUSDT \--interval 1h  
python \-m scripts.generate\_features \--symbol ADAUSDT \--interval 5m

### scripts/create\_labels.py

Generate labels for model training:

python scripts/create\_labels.py \--symbol BTCUSDT \--interval 1h \--label-strategy strategy\_2  
python scripts/create\_labels.py \--symbol ADAUSDT \--interval 5m \--label-strategy strategy\_3  
python scripts/create\_labels.py \--symbol ADAUSDT \--interval 15m \--label-strategy strategy\_1  
python scripts/create\_labels.py \--symbol ETHUSDT \--interval 1h \--label-strategy strategy\_4

### scripts/train\_model.py

Train and save ML models:

python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h  
python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest  
python scripts/train\_model.py \--symbol ETHUSDT \--interval 15m \--model lstm \--train\_ratio 0.7  
python scripts/train\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest \--skip\_tuning  
python scripts/train\_model.py \--symbol BTCUSDT \--interval 1h \--features ema\_10 rsi\_14 macd

### scripts/analyze\_model.py

Analyze a trained model:

python scripts/analyze\_model.py \--symbol ADAUSDT \--interval 5m \--model random\_forest

### scripts/backtest.py

Run deterministic backtests:

python scripts/backtest.py \--symbol BTCUSDT \--interval 1h \--model xgboost  
python scripts/backtest.py \--symbol ADAUSDT \--interval 5m \--model random\_forest  
python scripts/backtest.py \--symbol ADAUSDT \--interval 5m \--model lstm \--backtest\_mode full \--train\_ratio 0.7

### scripts/monte\_carlo\_backtest.py

Run Monte Carlo backtests:

python scripts/monte\_carlo\_backtest.py \--symbol ADAUSDT \--interval 5m \--model lstm \--num\_simulations 100

### scripts/analyze\_results.py

Analyze trading results:

python scripts/analyze\_results.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_type backtest  
python scripts/analyze\_results.py \--symbol ADAUSDT \--interval 5m \--model\_type random\_forest \--results\_type live  
python scripts/analyze\_results.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_dir /path/to/my/backtest\_results \--analysis\_dir /path/to/my/analysis\_output

### scripts/analyze\_labels.py

Analyze generated trading labels:

python scripts/analyze\_labels.py \--symbol ADAUSDT \--interval 5m \--label-strategy strategy\_2  
python scripts/analyze\_labels.py \--symbol BTCUSDT \--interval 1h \--label-strategy strategy\_1 \--future-horizons 10 30 60

### scripts/convert\_trades\_to\_json.py

Convert trade history and OHLCV data to JSON:

python scripts/convert\_trades\_to\_json.py \--symbol BTCUSDT \--interval 1h \--model\_type xgboost \--results\_type backtest  
python scripts/convert\_trades\_to\_json.py \--symbol ADAUSDT \--interval 5m \--model\_type random\_forest \--output\_file /path/to/my/output.json

## Contributing

Contributions are welcome\! Please follow standard GitHub flow: fork the repository, create a feature branch, commit your changes, and open a pull request.

## License

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE)