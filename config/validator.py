"""
Config validation for schema dataclasses.
Call validate_config(config_instance) for any config object.
"""

from config.feature import FeatureConfig, TemporalValidationConfig

def validate_general_config(config):
    if not isinstance(config.random_seed, int):
        raise TypeError("random_seed must be int.")
    if config.n_processors < -1:
        raise ValueError("n_processors must be >= -1.")
    if config.hyperparameter_tuning_n_iter <= 0:
        raise ValueError("hyperparameter_tuning_n_iter must be > 0.")
    if config.hyperparameter_tuning_cv_folds <= 0:
        raise ValueError("hyperparameter_tuning_cv_folds must be > 0.")
    
    if not isinstance(config.data_granularity_minutes, (int, float)) or config.data_granularity_minutes <= 0:
        raise ValueError("data_granularity_minutes must be a positive number.")
    if not isinstance(config.min_trade_loop_interval_seconds, (int, float)) or config.min_trade_loop_interval_seconds <= 0:
        raise ValueError("min_trade_loop_interval_seconds must be a positive number.")
    if not isinstance(config.polling_frequency_factor, float) or not (0 < config.polling_frequency_factor < 1):
        raise ValueError("polling_frequency_factor must be a float between 0 and 1 (exclusive).")


def validate_feature_config(config: FeatureConfig):
    """
    Validates the fields of a FeatureConfig instance.
    This function contains the comprehensive field-level validation logic.
    """
    period_lists = ['sma_periods', 'ema_periods', 'rsi_periods', 'bollinger_periods', 'atr_periods',
                    'stochastic_periods', 'ao_periods', 'cci_periods', 'mfi_periods', 'volume_periods',
                    'support_resistance_periods', 'z_score_periods', 'adr_periods', 'trend_strength_periods']
    for attr in period_lists:
        val = getattr(config, attr)
        if not isinstance(val, list) or not all(isinstance(x, int) and x > 0 for x in val):
            raise ValueError(f"'{attr}' must be a list of positive integers.")

    if config.pivot_point_calculation_period not in ['daily', 'weekly', 'monthly']:
        raise ValueError("pivot_point_calculation_period must be 'daily', 'weekly', or 'monthly'.")
    if config.pivot_point_method not in ['standard']:
        raise ValueError("pivot_point_method must be 'standard'.")
    if not isinstance(config.candlestick_patterns, list) or not all(isinstance(x, str) for x in config.candlestick_patterns):
        raise ValueError("'candlestick_patterns' must be a list of strings.")
    if not isinstance(config.fvg_lookback_bars, int) or config.fvg_lookback_bars <= 0:
        raise ValueError("'fvg_lookback_bars' must be a positive integer.")
    
    if not isinstance(config.swing_pivot_left_bars, int) or config.swing_pivot_left_bars <= 0:
        raise ValueError("swing_pivot_left_bars must be a positive integer.")
    if not isinstance(config.swing_pivot_right_bars, int) or config.swing_pivot_right_bars <= 0:
        raise ValueError("swing_pivot_right_bars must be a positive integer.")
    if not isinstance(config.volume_oscillator_short_ema, int) or config.volume_oscillator_short_ema <= 0:
        raise ValueError("volume_oscillator_short_ema must be a positive integer.")
    if not isinstance(config.volume_oscillator_long_ema, int) or config.volume_oscillator_long_ema <= 0:
        raise ValueError("volume_oscillator_long_ema must be a positive integer.")
    if config.volume_oscillator_short_ema >= config.volume_oscillator_long_ema:
        raise ValueError("volume_oscillator_short_ema must be less than volume_oscillator_long_ema.")
    if not isinstance(config.volume_threshold, (int, float)) or config.volume_threshold < 0:
        raise ValueError("volume_threshold must be a non-negative number.")

    if not isinstance(config.remove_nan_rows, bool):
        raise TypeError("'remove_nan_rows' must be a boolean.")

    for col, lags in config.lagged_features.items():
        if not isinstance(col, str) or not col:
            raise ValueError(f"Lagged feature column name must be a non-empty string, got '{col}'.")
        if not (isinstance(lags, list) and all(isinstance(l, int) and l > 0 for l in lags)):
            raise ValueError(f"Lags for '{col}' must be a list of positive integers, got {lags}.")

    for col, orders in config.differenced_features.items():
        if not isinstance(col, str) or not col:
            raise ValueError(f"Differenced feature column name must be a non-empty string, got '{col}'.")
        if not (isinstance(orders, list) and all(isinstance(o, int) and o > 0 for o in orders)):
            raise ValueError(f"Differencing orders for '{col}' must be a list of positive integers, got {orders}.")
    
    tv = config.temporal_validation
    if not isinstance(tv.enabled, bool):
        raise TypeError("TemporalValidationConfig: enabled must be bool.")
    if not isinstance(tv.warning_correlation_threshold, float) or not (0 <= tv.warning_correlation_threshold <= 1):
        raise ValueError("TemporalValidationConfig: warning_correlation_threshold must be float between 0 and 1.")
    if not isinstance(tv.error_correlation_threshold, float) or not (0 <= tv.error_correlation_threshold <= 1):
        raise ValueError("TemporalValidationConfig: error_correlation_threshold must be float between 0 and 1.")
    if tv.warning_correlation_threshold > tv.error_correlation_threshold:
        raise ValueError("TemporalValidationConfig: warning_correlation_threshold cannot be greater than error_correlation_threshold.")
    
    if not isinstance(config.talib_available, bool):
        raise TypeError("talib_available must be a boolean.")
    if not isinstance(config.ta_lib_available, bool):
        raise TypeError("ta_lib_available must be a boolean.")

def validate_trading_config(config):
    validate_risk_config(config.risk)
    validate_trade_execution_config(config.trade_execution)
    validate_entry_filter_config(config.entry_filter)
    validate_volatility_regime_config(config.volatility_regime)
    validate_sltp_config(config.sltp)
    validate_backtest_config(config.backtest)
    if config.bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive.")

def validate_exchange_config(config):
    if config.exchange != 'binance':
        raise ValueError("exchange must be 'binance'.")
    if config.default_type not in ['future', 'spot']:
        raise ValueError("default_type must be 'future' or 'spot'.")
    if config.rateLimit <= 0:
        raise ValueError("rateLimit must be positive.")
    if config.timeout <= 0:
        raise ValueError("timeout must be positive.")
    if config.tld not in ['com', 'us']:
        raise ValueError("tld must be 'com' or 'us'.")
    if config.price_precision < 0:
        raise ValueError("price_precision must be >= 0.")
    if config.quantity_precision < 0:
        raise ValueError("quantity_precision must be >= 0.")
    if config.min_quantity < 0:
        raise ValueError("min_quantity must be >= 0.")
    if config.min_notional < 0:
        raise ValueError("min_notional must be >= 0.")

def validate_notifier_config(config):
    tg = config.telegram
    if not isinstance(tg.enabled, bool):
        raise TypeError("TelegramConfig: enabled must be bool.")
    if tg.enabled:
        if not isinstance(tg.token, str) or not tg.token:
            raise ValueError("TelegramConfig: token required if enabled.")
        if not isinstance(tg.chat_id, str) or not tg.chat_id:
            raise ValueError("TelegramConfig: chat_id required if enabled.")

def validate_xgboost_params(params):
    if not isinstance(params.n_estimators, int) or params.n_estimators <= 0:
        raise ValueError("XGBoostParams: n_estimators must be positive int.")
    if not isinstance(params.learning_rate, float) or not (0 < params.learning_rate <= 1):
        raise ValueError("XGBoostParams: learning_rate must be float in (0, 1].")
    if not isinstance(params.max_depth, int) or params.max_depth <= 0:
        raise ValueError("XGBoostParams: max_depth must be positive int.")
    if not isinstance(params.subsample, float) or not (0 < params.subsample <= 1):
        raise ValueError("XGBoostParams: subsample must be float in (0, 1].")
    if not isinstance(params.colsample_bytree, float) or not (0 < params.colsample_bytree <= 1):
        raise ValueError("XGBoostParams: colsample_bytree must be float in (0, 1].")
    if not isinstance(params.n_jobs, int):
        raise TypeError("XGBoostParams: n_jobs must be int.")

def validate_random_forest_params(params):
    if not isinstance(params.n_estimators, int) or params.n_estimators <= 0:
        raise ValueError("RandomForestParams: n_estimators must be positive int.")
    if not isinstance(params.max_depth, int) or params.max_depth <= 0:
        raise ValueError("RandomForestParams: max_depth must be positive int.")
    if not isinstance(params.min_samples_leaf, int) or params.min_samples_leaf <= 0:
        raise ValueError("RandomForestParams: min_samples_leaf must be positive int.")
    if not isinstance(params.min_samples_split, int) or params.min_samples_split <= 0:
        raise ValueError("RandomForestParams: min_samples_split must be positive int.")
    if not isinstance(params.n_jobs, int):
        raise TypeError("RandomForestParams: n_jobs must be int.")

def validate_lstm_params(params):
    if not isinstance(params.sequence_length_bars, int) or params.sequence_length_bars <= 0:
        raise ValueError("LSTMParams: sequence_length_bars must be positive int.")
    if params.n_features is not None and (not isinstance(params.n_features, int) or params.n_features <= 0):
        raise ValueError("LSTMParams: n_features must be None or positive int.")
    if not isinstance(params.units_per_layer, int) or params.units_per_layer <= 0:
        raise ValueError("LSTMParams: units_per_layer must be positive int.")
    if not isinstance(params.n_layers, int) or params.n_layers <= 0:
        raise ValueError("LSTMParams: n_layers must be positive int.")
    if not isinstance(params.epochs, int) or params.epochs <= 0:
        raise ValueError("LSTMParams: epochs must be positive int.")
    if not isinstance(params.batch_size, int) or params.batch_size <= 0:
        raise ValueError("LSTMParams: batch_size must be positive int.")
    if not isinstance(params.validation_split, float) or not (0 < params.validation_split < 1):
        raise ValueError("LSTMParams: validation_split must be float in (0, 1).")
    if not isinstance(params.dropout_rate, float) or not (0 <= params.dropout_rate < 1):
        raise ValueError("LSTMParams: dropout_rate must be float in [0, 1).")
    if not isinstance(params.learning_rate, float) or params.learning_rate <= 0:
        raise ValueError("LSTMParams: learning_rate must be positive float.")

def validate_model_config(config):
    if config.model_type not in ['xgboost', 'random_forest', 'lstm']:
        raise ValueError("model_type must be 'xgboost', 'random_forest', or 'lstm'.")
    if config.features_to_use is not None and not isinstance(config.features_to_use, list):
        raise TypeError("features_to_use must be a list or None.")
    if not isinstance(config.label_column, str) or not config.label_column:
        raise ValueError("label_column must be a non-empty string.")
    if not isinstance(config.train_test_split_ratio, float) or not (0 < config.train_test_split_ratio < 1):
        raise ValueError("train_test_split_ratio must be float between 0 and 1.")
    if config.scaler_type not in ['standard', 'minmax', None]:
        raise ValueError("scaler_type must be 'standard', 'minmax', or None.")
    if not isinstance(config.pca_enabled, bool):
        raise TypeError("pca_enabled must be bool.")
    if not isinstance(config.pca_n_components, (int, float)) or \
       (isinstance(config.pca_n_components, float) and not (0 < config.pca_n_components <= 1)) or \
       (isinstance(config.pca_n_components, int) and config.pca_n_components <= 0):
        raise ValueError("pca_n_components must be an int > 0 or a float in (0, 1].")

    # --- ADDED: Validate each model type config ---
    validate_xgboost_params(config.xgboost_params)
    validate_random_forest_params(config.random_forest_params)
    validate_lstm_params(config.lstm_params)

def validate_label_config(config):
    if config.labeling_strategy_type not in [
        'labeling_strategy_1',
        'labeling_strategy_2',
        'labeling_strategy_3',
        'labeling_strategy_4'
    ]:
        raise ValueError("labeling_strategy_type must be one of the supported strategies.")
    if config.min_holding_period < 1:
        raise ValueError("min_holding_period must be at least 1.")
    if not (0 <= config.trading_fee_pct <= 100):
        raise ValueError("trading_fee_pct must be 0-100.")
    if not (0 <= config.slippage_tolerance_pct <= 100):
        raise ValueError("slippage_tolerance_pct must be 0-100.")
    if not isinstance(config.analysis_future_horizons, list) or not all(isinstance(x, int) and x > 0 for x in config.analysis_future_horizons):
        raise ValueError("analysis_future_horizons must be a list of positive integers.")

    s1 = config.labeling_strategy_1
    if not (0 <= s1.profit_multiplier_pct <= 1000):
        raise ValueError("LabelingStrategy1Config: profit_multiplier_pct must be 0-1000.")
    if not (0 <= s1.stop_loss_multiplier_pct <= 1000):
        raise ValueError("LabelingStrategy1Config: stop_loss_multiplier_pct must be 0-1000.")

    s2 = config.labeling_strategy_2
    if not (0 <= s2.quantile_threshold_long_pct <= 100):
        raise ValueError("LabelingStrategy2Config: quantile_threshold_long_pct must be 0-100.")
    if not (0 <= s2.quantile_threshold_short_pct <= 100):
        raise ValueError("LabelingStrategy2Config: quantile_threshold_short_pct must be 0-100.")

    s3 = config.labeling_strategy_3
    if not (0 <= s3.long_ratio_quantile_pct <= 100):
        raise ValueError("LabelingStrategy3Config: long_ratio_quantile_pct must be 0-100.")
    if not (0 <= s3.short_ratio_quantile_pct <= 100):
        raise ValueError("LabelingStrategy3Config: short_ratio_quantile_pct must be 0-100.")
    if not (0 <= s3.min_profit_threshold_pct <= 100):
        raise ValueError("LabelingStrategy3Config: min_profit_threshold_pct must be 0-100.")

    s4 = config.labeling_strategy_4
    if not (0 < s4.pca_n_components_pct <= 100):
        raise ValueError("LabelingStrategy4Config: pca_n_components_pct must be 0-100.")

def validate_risk_config(config):
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if not (0 < config.risk_per_trade_pct <= 100):
        raise ValueError("risk_per_trade_pct must be 0-100.")
    if config.leverage <= 0:
        raise ValueError("leverage must be positive.")

def validate_trade_execution_config(config):
    if not (0 <= config.trading_fee_pct <= 100):
        raise ValueError("trading_fee_pct must be 0-100.")
    if not (0 <= config.slippage_tolerance_pct <= 100):
        raise ValueError("slippage_tolerance_pct must be 0-100.")
    if not (0 <= config.min_liq_distance_pct <= 100):
        raise ValueError("min_liq_distance_pct must be 0-100.")

def validate_entry_filter_config(config):
    if not (0 <= config.confidence_threshold_long_pct <= 100):
        raise ValueError("confidence_threshold_long_pct must be 0-100.")
    if not (0 <= config.confidence_threshold_short_pct <= 100):
        raise ValueError("confidence_threshold_short_pct must be 0-100.")
    if config.trend_filter_ema_period <= 0:
        raise ValueError("trend_filter_ema_period must be positive.")
    if not isinstance(config.allow_long_trades, bool):
        raise TypeError("allow_long_trades must be bool.")
    if not isinstance(config.allow_short_trades, bool):
        raise TypeError("allow_short_trades must be bool.")
    if hasattr(config, "min_long_proba") and not (0 <= config.min_long_proba <= 1):
        raise ValueError("min_long_proba must be between 0 and 1.")
    if hasattr(config, "min_short_proba") and not (0 <= config.min_short_proba <= 1):
        raise ValueError("min_short_proba must be between 0 and 1.")
    if hasattr(config, "allowed_volatility_regimes") and config.allowed_volatility_regimes is not None:
        if not isinstance(config.allowed_volatility_regimes, (list, set, tuple)) or \
           not all(isinstance(x, int) for x in config.allowed_volatility_regimes):
            raise ValueError("allowed_volatility_regimes must be a list/set/tuple of integers.")

def validate_volatility_regime_config(config):
    if set(config.max_holding_bars.keys()) != {0, 1, 2}:
        raise ValueError("max_holding_bars keys must be 0, 1, 2.")
    if set(config.allow_trading.keys()) != {0, 1, 2}:
        raise ValueError("allow_trading keys must be 0, 1, 2.")

def validate_sltp_config(config):
    if not (0 < config.fixed_take_profit_pct <= 100):
        raise ValueError("fixed_take_profit_pct must be 0-100.")
    if not (0 < config.fixed_stop_loss_pct <= 100):
        raise ValueError("fixed_stop_loss_pct must be 0-100.")
    if not (0 <= config.min_sl_tp_pct <= 100):
        raise ValueError("min_sl_tp_pct must be 0-100.")
    if config.volatility_window_bars <= 0:
        raise ValueError("volatility_window_bars must be positive.")
    if config.alpha_take_profit <= 0:
        raise ValueError("alpha_take_profit must be positive.")
    if config.alpha_stop_loss <= 0:
        raise ValueError("alpha_stop_loss must be positive.")

def validate_backtest_config(config):
    if not (0 <= config.maintenance_margin_pct <= 100):
        raise ValueError("maintenance_margin_pct must be 0-100.")
    if not (0 <= config.liquidation_fee_pct <= 100):
        raise ValueError("liquidation_fee_pct must be 0-100.")
    if config.max_concurrent_trades <= 0:
        raise ValueError("max_concurrent_trades must be positive.")
    if config.backtest_mode not in ['full', 'train', 'test']:
        raise ValueError("Invalid backtest_mode.")
    if not isinstance(config.save_trades, bool):
        raise TypeError("save_trades must be bool.")
    if not isinstance(config.save_equity_curve, bool):
        raise TypeError("save_equity_curve must be bool.")
    if not isinstance(config.save_metrics, bool):
        raise TypeError("save_metrics must be bool.")
    if not isinstance(config.override_strategy_params, dict):
        raise TypeError("override_strategy_params must be dict.")

def validate_config(config):
    name = type(config).__name__
    if name == "GeneralConfig":
        validate_general_config(config)
    elif name == "FeatureConfig":
        validate_feature_config(config)
    elif name == "TradingConfig":
        validate_trading_config(config)
    elif name == "ExchangeConfig":
        validate_exchange_config(config)
    elif name == "NotifierConfig":
        validate_notifier_config(config)
    elif name == "ModelConfig":
        validate_model_config(config)
    elif name == "LabelConfig":
        validate_label_config(config)
    elif name == "AppConfig":
        validate_general_config(config.general)
        validate_feature_config(config.features)
        validate_trading_config(config.trading)
        validate_exchange_config(config.exchange)
        validate_notifier_config(config.notifier)
        validate_model_config(config.model)
        validate_label_config(config.labeling)