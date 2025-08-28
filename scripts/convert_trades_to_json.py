# scripts/convert_trades_to_json.py

import pandas as pd
import json
import sys
import logging
from pathlib import Path
import argparse
import numpy as np
import math
from typing import Optional 

# Add the project root to the system path to allow importing modules
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import the DataManager utility and logger setup
try:
    from utils.data_management.data_manager import DataManager
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}", file=sys.stderr)
    sys.exit(1)

# Set up logging for this script
logger = setup_rotating_logging(
    log_filename_base=Path(__file__).stem,
    log_level=logging.INFO
)

def handle_value(value):
    """
    Helper function to handle different data types for JSON serialization,
    converting Timestamps to milliseconds since epoch and handling NaN/Infinity values.
    """
    if pd.isna(value):
        return None 
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
             value = value.tz_localize('UTC')
        return int(value.timestamp() * 1000)
    if isinstance(value, (int, float, np.integer, np.floating)):
        python_value = value.item() if isinstance(value, (np.integer, np.floating)) else value
        if math.isinf(python_value) or math.isnan(python_value):
            return None
        return python_value
    if isinstance(value, bool):
        return bool(value)
    return value

def convert_trade_history_and_ohlcv_to_json(
    ohlcv_df: pd.DataFrame,
    trade_history_df: pd.DataFrame,
    output_file_path: Path
):
    """
    Uses loaded OHLCV and trade history data, combines them, and saves as a JSON file
    structured for financial chart visualization.
    """
    required_ohlcv_cols = ['open', 'high', 'low', 'close']
    if not all(col in ohlcv_df.columns for col in required_ohlcv_cols + ['open_time']):
         missing = [col for col in required_ohlcv_cols + ['open_time'] if col not in ohlcv_df.columns]
         logger.error(f"Loaded OHLCV data is missing required columns: {missing}. Aborting.")
         sys.exit(1)

    ohlcv_data = ohlcv_df[required_ohlcv_cols].astype(float).copy()
    ohlcv_data.columns = ['open', 'high', 'low', 'close'] 

    if not pd.api.types.is_datetime64_any_dtype(ohlcv_df['open_time']):
        logger.warning("ohlcv_df['open_time'] is not datetime type. Attempting conversion.")
        ohlcv_df['open_time'] = pd.to_datetime(ohlcv_df['open_time'], errors='coerce', utc=True)
        ohlcv_df.dropna(subset=['open_time'], inplace=True)

    ohlcv_data['time'] = (ohlcv_df['open_time'].astype(np.int64) // 10**6)

    trade_markers = []
    if not trade_history_df.empty:
        for col in ['entry_time', 'exit_time']:
            if col in trade_history_df.columns:
                 trade_history_df[col] = pd.to_datetime(trade_history_df[col], errors='coerce', utc=True)
        
        trade_history_df.dropna(subset=['entry_time', 'exit_time'], inplace=True)

        if not trade_history_df.empty:
            trade_history_df = trade_history_df.sort_values(by='entry_time').reset_index(drop=True)
            for index, trade in trade_history_df.iterrows():
                entry_time_ms = handle_value(trade.get('entry_time'))
                exit_time_ms = handle_value(trade.get('exit_time'))
                direction = trade.get('direction_int')

                if entry_time_ms is None or exit_time_ms is None or direction is None:
                     logger.warning(f"Skipping trade {index} due to missing time or direction.")
                     continue
                try:
                    direction_numeric = float(direction)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping trade {index} due to invalid direction value: {direction}")
                    continue

                net_pnl = trade.get('net_pnl')
                exit_reason = trade.get('exit_reason')
                entry_price = trade.get('entry_price')
                exit_price = trade.get('exit_price')

                trade_markers.append({
                    'time': entry_time_ms,
                    'position': 'belowBar' if direction_numeric > 0 else 'aboveBar',
                    'color': '#26A69A' if direction_numeric > 0 else '#EF5350',
                    'shape': 'arrowUp' if direction_numeric > 0 else 'arrowDown',
                    'text': f'Entry ({"Long" if direction_numeric > 0 else "Short"}): {entry_price:.6f}' if entry_price is not None else f'Entry ({"Long" if direction_numeric > 0 else "Short"})',
                    'size': 1.5,
                    'tradeDetails': {k: handle_value(v) for k, v in trade.items()}
                })
                if exit_time_ms is not None:
                    trade_markers.append({
                        'time': exit_time_ms,
                        'position': 'aboveBar' if direction_numeric > 0 else 'belowBar',
                        'color': '#26A69A' if (net_pnl is not None and net_pnl >= 0) else '#EF5350',
                        'shape': 'circle',
                        'text': f'Exit ({exit_reason}): {exit_price:.6f} PnL: {net_pnl:.2f}' if exit_price is not None and net_pnl is not None else f'Exit ({exit_reason})',
                        'size': 1.5,
                        'tradeDetails': {k: handle_value(v) for k, v in trade.items()}
                    })
    else:
        logger.warning("Trade history is empty. No trade markers will be generated.")

    output_data = {
        'ohlcv': ohlcv_data.to_dict(orient='records'),
        'tradeMarkers': trade_markers
    }

    logger.info(f"Saving combined data to JSON file: {output_file_path}...")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Combined data successfully converted and saved to {output_file_path}")

def main():
    """
    Main function to parse arguments and run the conversion, using DataManager for all path logic.
    """
    parser = argparse.ArgumentParser(description="Convert trade history and OHLCV data to JSON for visualization.")
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', required=True, help='Data interval (e.g., 1h, 5m)')
    parser.add_argument('--model_type', required=True, help='Model type (e.g., xgboost, lstm)')
    parser.add_argument('--results_type', default='backtest', choices=['backtest', 'live'],
                        help='Type of results to visualize (backtest or live).')
    parser.add_argument('--output_file', type=Path,
                        help='Optional: Full path for the output JSON file. Overrides default location.')
    args = parser.parse_args()

    dm = DataManager()

    # --- Load OHLCV data ---
    ohlcv_df = dm.load_dataframe(symbol=args.symbol, interval=args.interval, data_type='raw')
    if ohlcv_df is None or ohlcv_df.empty:
        logger.error("Failed to load OHLCV data. Aborting.")
        sys.exit(1)
    if ohlcv_df.index.name == 'timestamp':
        ohlcv_df = ohlcv_df.reset_index().rename(columns={'timestamp': 'open_time'})

    # --- Load Trade History using DataManager logic ---
    run_dir = None
    trades_file_pattern_key = None
    if args.results_type == 'backtest':
        run_dir = dm.get_backtesting_dir(model_type=args.model_type, symbol=args.symbol, interval=args.interval)
        trades_file_pattern_key = 'backtest_trades'
    elif args.results_type == 'live':
        run_dir = dm.get_live_trading_dir(model_type=args.model_type, symbol=args.symbol, interval=args.interval)
        trades_file_pattern_key = 'live_trades'

    trade_history_df = pd.DataFrame()
    if run_dir and trades_file_pattern_key:
        trades_filename = dm.path_config['patterns'][trades_file_pattern_key]
        trade_history_file_path = run_dir / trades_filename
        
        if trade_history_file_path.exists():
            try:
                trade_history_df = pd.read_parquet(trade_history_file_path)
                logger.info(f"Trade history loaded successfully from {trade_history_file_path}.")
            except Exception as e:
                logger.error(f"Error loading trade history from {trade_history_file_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Trade history file not found: {trade_history_file_path}. Proceeding without trade markers.")
    
    # --- Determine Output Path ---
    if args.output_file:
        output_file_path = args.output_file
    else:
        if run_dir:
            output_filename = dm.path_config['patterns']['trade_visualization_json']
            output_file_path = run_dir / output_filename
        else:
            logger.error(f"Could not determine run directory for '{args.results_type}'. Cannot generate default output path.")
            sys.exit(1)

    # --- Run Conversion ---
    convert_trade_history_and_ohlcv_to_json(
        ohlcv_df=ohlcv_df,
        trade_history_df=trade_history_df,
        output_file_path=output_file_path
    )

if __name__ == "__main__":
    main()