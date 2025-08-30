# Algorithmic Futures Trading Bot: Machine Learning-Driven Trading Strategy

**Author:** DimitriKenne  
**Last Updated:** 2025-08-28

This project presents a sophisticated algorithmic trading bot designed for futures markets, leveraging advanced machine learning techniques for predictive signal generation and robust risk management. Built with a modular and extensible architecture, this bot facilitates end-to-end automation of trading strategies, from data acquisition and feature engineering to model training, backtesting, and live execution.

---

## Motivation & Background

This project was initiated in April 2025, directly inspired by the research presented in the paper "A profitable trading algorithm for cryptocurrencies using a Neural Network model". Building upon a foundational understanding of non-systematic trading strategies, which I had been exploring since September 2022, this systematic framework was rapidly developed to its current state by early May 2025. The iterative development process involved numerous versions and was significantly accelerated by leveraging modern AI tools like Gemini and ChatGPT for efficient ideation and implementation. The primary objective of this ongoing work is to build a robust and adaptable framework for exploring and identifying profitable quantitative trading strategies.

---

## Features

### Modular and Scalable Architecture
- Clear separation of core functionalities into distinct modules (data management, feature engineering, model training, strategy execution, notifications, exchange integration).
- Easily adaptable to new exchanges, models, or strategy components.

### Multi-Exchange Compatibility
- Dedicated adapters for seamless integration with various cryptocurrency futures exchanges (currently implemented for Binance Futures).

### Advanced Feature Engineering Pipeline
- Transforms raw OHLCV data into a rich set of predictive features:
  - **Comprehensive Technical Indicators:** SMA, EMA, RSI, Bollinger Bands, ATR, Stochastic, CCI, MFI, MACD (multi-period).
  - **Candlestick Pattern Recognition:** Automated detection using TA-Lib.
  - **Custom Statistical & Price Action Features:** Fair Value Gaps (FVG), Z-scores, Average Daily Range (ADR), trend strength metrics, lagged prices, differenced prices.
  - **Temporal Safety Validation:** Prevents lookahead bias by ensuring all features are derived exclusively from past data.
  - **Volatility Regime Calculation:** Market regime (low/medium/high) based on Bollinger Bands width or ATR.

### Machine Learning-Driven Signal Generation
- **Ternary Classification:** Models classify future price movements as Long (1), Short (-1), or Neutral (0).
- **Supported Models:** LSTM, XGBoost, RandomForest.
- **Hyperparameter Tuning:** RandomizedSearchCV with TimeSeriesSplit for robust model selection.

### Configurable and Dynamic Strategy Logic
- **Confidence-Based Entry Filtering:** Filters trades based on model prediction probability (confidence score), with robust handling for missing probabilities.
- **Adaptive Volatility Regime Filtering:** Dynamically adjusts trade entry and max holding periods based on market volatility regime.
- **Trend Alignment Filter:** EMA-based filter to ensure trades are aligned with the dominant market trend.
- **Directional Control:** Enable/disable long and short entries independently.
- **Neutral Signal Management:** Configurable behavior for exiting positions on neutral signals.
- **Dynamic Take Profit/Stop Loss (TP/SL):** TP/SL levels calculated dynamically based on ATR or fixed percentages, with minimum distance from liquidation.
- **Precise Position Sizing:** Advanced logic to calculate trade quantity, ensuring adherence to minimum notional values and exchange precision.

### Robust Backtesting Engine
- **Realistic Simulation:** Accounts for trading fees, slippage, and liquidation mechanics.
- **Detailed Performance Metrics:** Total return, CAGR, max drawdown, win rate, profit factor, average PnL per trade, with proper annualization via bars_per_year.
- **Trade Management Simulation:** Position opening, closing, reversal logic with accurate fee and margin handling, and precise capital tracking.
- **Persistent Results:** Trade logs (Parquet), equity curve (Parquet), and summary metrics (JSON) for post-analysis.

### Live Trading Capabilities
- **Real-time Data Integration:** Direct connection to exchange data feeds.
- **Automated Trade Execution:** Market orders, open position management, TP/SL order placement/cancellation.
- **Resilient State Management:** Periodic state saving for capital and open positions.
- **Critical Event Notifications:** Telegram alerts for trade executions, errors, and bot status.

### Data Management & Persistence
- **Data Fetching:** Scripts for historical OHLCV data.
- **Dynamic Feature Engineering:** Features are engineered dynamically on raw data streams within the MarketDataHandler for backtesting and live trading, ensuring temporal safety. Processed and labeled data can still be optionally saved/loaded for model training.

### Flexible Label Generation
- **Strategy-Based Labeling:** Multiple strategies selectable via command-line argument (`--labeling-strategy`): Triple Barrier, Net Forward Return Quantile, Future Range Dominance, Clustering-Based Labeling.
- **Label Propagation Smoothing:** Configurable min_holding_period to smooth raw labels and filter out noise.
- **Input Validation:** Checks for required OHLCV/features.
- **Extensible Design:** Easily add new strategies via BaseLabelingStrategy.

### Comprehensive Logging
- Rotating logs for all stages of bot operation, backtesting, and data processing.

---

## Project Structure

```
├── .env                          # Environment variables (API keys, secrets)
├── .gitignore                    # Git ignore file
├── README.md                     # Project README
├── requirements.txt              # Python dependencies
├── ta_lib-0.6.3-cp312-cp312-win_amd64.whl # TA-Lib wheel (Windows specific)
├── trade_ohlcv_visualization.htm # HTML for visualizing trades
├── trading_bot.py                # Main live trading bot script
├── config/                       # Centralized configuration files
│   ├── params.py                 # Central import hub for configuration schemas
│   ├── paths.py                  # Defines all project-specific file paths
│   ├── validator.py              # Validation logic for config schemas
│   ├── general.py                # General application settings
│   ├── exchange.py               # Exchange connection and order settings
│   ├── feature.py                # Feature engineering configurations
│   ├── label.py                  # Labeling strategy configurations
│   ├── model.py                  # Model training and inference configurations
│   ├── trading.py                # Trading strategy, risk, and backtesting configurations
│   └── notifier.py               # Notification service settings
├── data/                         # Data storage
│   ├── labeled/                  # Labeled data for model training
│   ├── processed/                # Processed data with engineered features
│   └── raw/                      # Raw OHLCV data
├── docs/                         # Project documentation
├── logs/                         # Application logs
├── models/                       # Trained machine learning models, organized by type (e.g., models/xgboost/, models/lstm/)
├── results/                      # Backtesting and live trading results
│   ├── labeling/                 # Labeling analysis results
│   ├── model_analysis/           # Model analysis results
│   ├── backtesting/              # Backtesting results
│   ├── monte_carlo/              # Monte Carlo backtesting results
│   └── live_trading/             # Live trading results
├── scripts/                      # Utility scripts for workflow automation
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
└── utils/                        # Core utility modules
    ├── bot_management/           # Bot management modules (trade_cycle_processor.py, lifecycle_manager.py)
    ├── exceptions.py             # Custom exception classes
    ├── logger_config.py          # Logging configuration
    ├── notification_manager.py   # Handles sending notifications
    ├── strategy_evaluation/      # Modules for evaluating strategy performance, metrics, and simulation (replaces analysis/)
    │   ├── performance_analyzer.py
    │   ├── monte_carlo_analyzer.py
    │   ├── metrics_calculator.py
    │   ├── plotting_utils.py
    │   └── path_simulator.py
    ├── data_management/          # Utilities for data handling and live data processing
    │   ├── data_manager.py
    │   └── market_data_handler.py
    ├── exchange_adapters/        # Exchange API adapters and related utilities
    │   ├── exchange_interface.py # Abstract base class for exchange adapters
    │   └── binance/              # Binance-specific adapters and helpers
    │       ├── futures_adapter.py
    │       ├── client_manager.py
    │       ├── decorators.py
    │       ├── account_configurator.py
    │       └── exchange_info_helper.py
    ├── feature_engineering/      # Utilities for feature generation
    │   ├── feature_engineer.py
    │   ├── feature_name_generator.py
    │   ├── indicator_feature_processor.py
    │   ├── price_action_feature_processor.py
    │   ├── price_feature_calculator.py
    │   └── technical_indicator_calculator.py
    ├── labeling/                 # Utilities for generating labels for ML models
    │   ├── label_generator.py
    │   ├── label_analyzer.py
    │   ├── analysis_plotter.py
    │   └── analysis_calculator.py
    ├── strategy_execution/       # Trade execution and session management modules
    │   ├── backtester.py         # Main backtesting logic
    │   ├── entry_filters.py
    │   ├── trade_calculation_helpers.py
    │   ├── trade_execution_engine.py
    │   ├── trading_session_manager.py
    │   └── live_trading_session_manager.py
    └── training/                 # Utilities used during the model training phase
        ├── model_trainer.py
        ├── model_builder.py
        ├── data_sequencer.py
        └── preprocessor_builder.py
```

---

## Key Components Explained

### trading_bot.py
The main script orchestrating the live trading process. It handles initialization, continuously fetches and processes market data, applies trading logic, manages state, and sends notifications.

### config/ Directory
This directory serves as the central hub for all configuration parameters, organized into dedicated schema files for clarity and validation.

- **params.py:** Central import hub, aggregating all config dataclasses.
- **paths.py:** Defines all project-specific file and directory paths.
- **validator.py:** Contains validation logic for all configuration schemas.
- **general.py:** General application settings (e.g., random seed, processor count, minimum lookback for data).
- **exchange.py:** Exchange connection settings (API keys, testnet, rate limits, precision).
- **feature.py:** Defines configurations for feature engineering (indicator periods, pattern enablement, temporal safety).
- **label.py:** Configures parameters for different labeling strategies.
- **model.py:** Model training and inference configurations (model type, hyperparameters, features to use).
- **trading.py:** Aggregates risk management, trade execution, entry/exit filters, and backtesting parameters.
- **notifier.py:** Settings for notification services (e.g., Telegram token and chat ID).

### utils/bot_management/
Contains bot management modules such as `trade_cycle_processor.py` and `lifecycle_manager.py`, responsible for supervising trading cycles, user interaction in hybrid mode, and safe lifecycle management.

### utils/strategy_evaluation/
Contains modules for evaluating strategy performance, metrics, and simulations (replaces the old `utils/analysis/` folder). This now includes `path_simulator.py` (previously in utils/simulator/).

### utils/data_management/
Handles market data loading, storage, and live data stream processing.

- **data_manager.py:** Manages loading and saving of various data types (raw, processed, labeled, models) from/to persistent storage.
- **market_data_handler.py:** Responsible for fetching (historical/recent), processing (feature engineering), and providing data with integrated model signals for both backtesting and live trading. It dynamically applies feature engineering on raw data streams.

### utils/exchange_adapters/
Manages interactions with different cryptocurrency exchanges.

- **exchange_interface.py:** An Abstract Base Class (ABC) defining the common interface for all exchange adapters, ensuring consistency.
- **binance/:** A subfolder containing Binance-specific implementations:
  - **futures_adapter.py:** Concrete implementation of ExchangeInterface for Binance Futures, orchestrating API calls and delegating to specialized helpers.
  - **client_manager.py:** Manages the lifecycle and singleton instance of the Binance AsyncClient.
  - **decorators.py:** Contains decorators like async_retry_api_call for robust API interaction.
  - **account_configurator.py:** Handles Binance account-level settings such as leverage and margin mode.
  - **exchange_info_helper.py:** Fetches and caches exchange-specific information (e.g., precision, minimums) and provides utility methods for data adjustment.

### utils/feature_engineering/
Contains modules for transforming raw OHLCV data into a rich set of predictive features.

- **feature_engineer.py:** Orchestrates the overall feature engineering process, combining various processors.
- **feature_name_generator.py:** Generates consistent naming conventions for features.
- **indicator_feature_processor.py:** Processes standard technical indicators.
- **price_action_feature_processor.py:** Calculates features based on price action patterns and pivot points.
- **price_feature_calculator.py:** Core calculations for basic price-derived features.
- **technical_indicator_calculator.py:** Implements the calculation logic for various technical indicators (SMA, EMA, RSI, etc.).

### utils/labeling/
Provides utilities for generating and analyzing labels for machine learning models.

- **label_generator.py:** Creates target labels for ML models based on various strategies (e.g., Triple Barrier, Net Forward Return Quantile).
- **label_analyzer.py:** Analyzes the distribution and characteristics of generated labels.
- **analysis_plotter.py:** Provides plotting utilities specifically for label analysis.
- **analysis_calculator.py:** Calculates metrics related to label analysis.

### utils/strategy_execution/
Houses the core components for trade strategy execution, encompassing backtesting and live trading logic.

- **backtester.py:** The core backtesting engine, simulating trading operations bar by bar. It integrates with MarketDataHandler for dynamic feature engineering.
- **entry_filters.py:** Applies various conditions (filters) to potential trade signals before an entry is made.
- **trade_calculation_helpers.py:** Provides fundamental calculations required for trade management (e.g., PnL, fees, liquidation price).
- **trade_execution_engine.py:** Central engine for handling all trade-related calculations and strategy logic, including position sizing, SL/TP, and PnL.
- **trading_session_manager.py:** Manages the financial state, open positions, and trade history for a trading session, including persistence for live trading.
- **live_trading_session_manager.py:** Manages live trading session state, open positions, and capital tracking.

### utils/training/
Contains utilities essential for the machine learning model training phase.

- **model_trainer.py:** Manages the end-to-end process of training, evaluating, and loading ML models.
- **model_builder.py:** Responsible for constructing and configuring specific ML models (e.g., XGBoost, RandomForest, LSTM).
- **data_sequencer.py:** Handles the preparation and sequencing of time-series data for models requiring sequential inputs (e.g., LSTMs).
- **preprocessor_builder.py:** Builds and configures the data preprocessing pipeline (scaling, PCA) for features.

---

## Installation

Clone the repository:
```sh
git clone https://github.com/DimitriKenne/CryptoFutureBot.git
cd CryptoFutureBot
```

Create a virtual environment (recommended):
```sh
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

Install dependencies:
```sh
pip install -r requirements.txt
```

*Note: For TA-Lib installation issues, please refer to the provided wheel for Windows (`ta_lib-0.6.3-cp312-cp312-win_amd64.whl`) or consult TA-Lib documentation for specific operating system installation instructions.*

Set up environment variables:  
Create a `.env` file in the project root with your API keys and other sensitive information:
```
BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"
TELEGRAM_ENABLED=True
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
```
Add other credentials as needed based on your chosen exchange and notification services.

---

## Configuration

All core parameters for the bot are now defined and managed within dedicated, dataclass-based configuration schema files located in the `config/` directory. These schemas provide structure, default values, and validation for different aspects of the project (general settings, exchange, features, labeling, models, trading strategy, and notifications). Review and adjust settings in these files as needed before running scripts or the bot.

---

## Usage

The `scripts/` directory contains utility scripts for automating various parts of the trading workflow. Example usage:

### Live Trading
```sh
python trading_bot.py --symbol ADAUSDT --interval 5m --model_type lstm
```

### Data Fetching
```sh
python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01 --end_date 2024-03-01
python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01
```

### Feature Engineering (for Model Training or External Analysis)
This script processes raw data and saves features to `data/processed`. While not strictly required for backtesting/live-trading due to dynamic feature engineering in MarketDataHandler, it can be used to pre-generate data for model training or standalone analysis.
```sh
python scripts/generate_features.py --symbol BTCUSDT --interval 1h
python -m scripts.generate_features --symbol ADAUSDT --interval 5m
```

### Label Generation
```sh
python scripts/create_labels.py --symbol BTCUSDT --interval 1h --labeling-strategy strategy_2
python scripts/create_labels.py --symbol ADAUSDT --interval 5m --labeling-strategy strategy_3
python scripts/create_labels.py --symbol ADAUSDT --interval 15m --labeling-strategy strategy_1
python scripts/create_labels.py --symbol ETHUSDT --interval 1h --labeling-strategy strategy_4
```

### Model Training
```sh
python scripts/train_model.py --symbol BTCUSDT --interval 1h --model_type xgboost
python scripts/train_model.py --symbol ADAUSDT --interval 5m --model_type random_forest
python scripts/train_model.py --symbol ETHUSDT --interval 15m --model_type lstm --train_ratio 0.7
python scripts/train_model.py --symbol ADAUSDT --interval 5m --model_type random_forest --skip_tuning
python scripts/train_model.py --symbol BTCUSDT --interval 1h --features ema_10 rsi_14 macd
```

### Model Analysis
```sh
python scripts/analyze_model.py --symbol ADAUSDT --interval 5m --model_type random_forest
```

### Backtesting
The backtester now loads raw data and performs feature engineering internally via MarketDataHandler. No need to run `generate_features.py` beforehand unless you want to pre-process.
```sh
python scripts/backtest.py --symbol BTCUSDT --interval 1h --model_type xgboost
python scripts/backtest.py --symbol ADAUSDT --interval 5m --model_type random_forest
python scripts/backtest.py --symbol ADAUSDT --interval 5m --model_type lstm --backtest_mode full --train_ratio 0.7
```

### Monte Carlo Backtesting
The Monte Carlo backtester generates raw synthetic data, which is then feature-engineered internally by MarketDataHandler during each simulation.
```sh
python scripts/monte_carlo_backtest.py --symbol ADAUSDT --interval 5m --model_type lstm --num_simulations 100
```

### Results Analysis
```sh
python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest
python scripts/analyze_results.py --symbol ADAUSDT --interval 5m --model_type random_forest --results_type live
python scripts/analyze_results.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_dir /path/to/my/backtest_results --analysis_dir /path/to/my/analysis_output
```

### Label Analysis
```sh
python scripts/analyze_labels.py --symbol ADAUSDT --interval 5m --labeling-strategy strategy_2
python scripts/analyze_labels.py --symbol BTCUSDT --interval 1h --labeling-strategy strategy_1 --future-horizons 10 30 60
```

### Convert Trades to JSON
```sh
python scripts/convert_trades_to_json.py --symbol BTCUSDT --interval 1h --model_type xgboost --results_type backtest
python scripts/convert_trades_to_json.py --symbol ADAUSDT --interval 5m --model_type random_forest --output_file /path/to/my/output.json
```

---

## Contributing

Contributions are welcome! Please follow standard GitHub flow: fork the repository, create a feature branch, commit your changes, and open a pull request.

---

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).