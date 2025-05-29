# utils/logger_config.py

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler # Import RotatingFileHandler
# Removed datetime as timestamping is now handled by calling script or not used in filename
# Removed io, collections.Counter as they were not directly used in core logging setup

# Add project root to Python path (Consider packaging the project instead for better practice)
# This might already be done in calling scripts, but good practice for standalone use
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import paths configuration
try:
    from config.paths import PATHS
except ImportError:
    # Define a default log path if config.paths is not available
    DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
    PATHS = {'logs_dir': DEFAULT_LOG_DIR}
    # Ensure the default log directory exists
    try:
        PATHS['logs_dir'].mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # If even default log dir creation fails, use current directory
        PATHS['logs_dir'] = Path(".")
        print(f"Warning: Could not create default log directory {DEFAULT_LOG_DIR}: {e}. Logging to current directory.", file=sys.stderr)

    print(f"Warning: config.paths not found. Using default log directory: {PATHS['logs_dir']}", file=sys.stderr)


# Get a logger instance for internal messages within logger_config itself
_internal_logger = logging.getLogger(__name__)
# Prevent internal logger messages from propagating to root before setup
_internal_logger.propagate = False
# Add a basic handler to internal logger to see its messages even before main setup
_internal_logger.addHandler(logging.StreamHandler(sys.stderr))
_internal_logger.setLevel(logging.DEBUG)


def setup_rotating_logging(log_filename_base: str, log_level=logging.INFO, max_bytes=5*1024*1024, backup_count=5):
    """
    Sets up the root logger with a console handler and a rotating file handler.
    Logs are saved to a file named based on log_filename_base in the directory
    specified by PATHS['logs_dir']. The file handler rotates logs to prevent
    them from becoming too large.

    Args:
        log_filename_base (str): The base name for the log file (e.g., 'create_labels', 'backtester').
                                 The actual file will be like 'create_labels.log'.
        log_level (int): The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
                         This level applies to both console and file handlers by default.
        max_bytes (int): The maximum size of the log file before rotation occurs (in bytes).
                         Defaults to 5MB (5 * 1024 * 1024).
        backup_count (int): The number of backup log files to keep.
                            Defaults to 5.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level) # Set the root logger level

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Remove existing handlers to prevent duplicates on multiple calls ---
    # This is important if setup_rotating_logging might be called more than once
    # or if other modules have already added handlers to the root logger.
    if root_logger.hasHandlers():
        _internal_logger.debug("Removing existing handlers from root logger.")
        # Iterate over a copy of the handlers list because removing modifies the list
        for handler in root_logger.handlers[:]:
            try:
                # Attempt to close handlers that have a close method
                if hasattr(handler, 'close'):
                    handler.close()
                root_logger.removeHandler(handler)
                _internal_logger.debug(f"Removed handler: {handler}")
            except Exception as e:
                _internal_logger.warning(f"Error removing handler {handler}: {e}")


    # --- Console Handler (output to stderr) ---
    # Use a standard StreamHandler for maximum compatibility
    try:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level) # Console level matches root level
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _internal_logger.debug("Added standard StreamHandler to sys.stderr.")
    except Exception as e:
        _internal_logger.error(f"Failed to create standard StreamHandler for sys.stderr: {e}.", exc_info=True)


    # --- Rotating File Handler ---
    log_dir = PATHS.get('logs_dir', PROJECT_ROOT / "logs") # Use default if not in PATHS
    # Ensure log directory exists
    log_dir_path = Path(log_dir)
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        _internal_logger.debug(f"Ensured log directory exists: {log_dir_path}")
    except OSError as e:
        _internal_logger.error(f"Failed to ensure log directory exists at {log_dir_path}: {e}", exc_info=True)
        # Do NOT return here. Continue to return root_logger even if file logging fails.


    # Construct the log file path
    log_filename = f"{log_filename_base}.log"
    log_filepath = log_dir_path / log_filename

    try:
        # Use RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8' # Ensure UTF-8 encoding
        )
        file_handler.setLevel(log_level) # File level matches root level
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        _internal_logger.info(f"Rotating logs will be saved to: {log_filepath}")
        _internal_logger.info(f"Log rotation configured: maxBytes={max_bytes}, backupCount={backup_count}")

    except Exception as e:
        _internal_logger.error(f"Failed to create rotating file logger at {log_filepath}: {e}", exc_info=True)

    # Always return the root_logger, even if file handler creation failed.
    return root_logger


# Example usage (optional, for testing logger_config.py directly)
if __name__ == "__main__":
    # Example: Setup logging for a dummy script named 'test_script'
    setup_rotating_logging('test_script', logging.DEBUG, max_bytes=10*1024, backup_count=3) # Smaller size for testing
    logger = logging.getLogger(__name__) # Get a logger for this module

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Simulate writing enough data to trigger rotation (requires writing > max_bytes)
    # This is just a conceptual example; actual rotation happens when the handler writes.
    # You would need to write log messages repeatedly to see rotation in action.
    # For demonstration, let's just log a few more lines.
    for i in range(5):
        logger.info(f"Writing test message {i+1} to trigger potential rotation.")

    logger.info("Logger setup test complete.")
