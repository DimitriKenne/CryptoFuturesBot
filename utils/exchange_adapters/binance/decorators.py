# utils/exchange_adapters/binance/binance_decorators.py

"""
Provides decorators for Binance API calls, including retry logic with exponential backoff.
Separated to prevent circular import dependencies.
"""

import asyncio
import logging
from binance.exceptions import BinanceAPIException, BinanceRequestException

logger = logging.getLogger(__name__)

# --- Constants for Binance API Error Codes ---
ORDER_NOT_FOUND_CODE = -2011
INSUFFICIENT_FUNDS_CODES = [-2019, -4003, -4007, -4014]
RATE_LIMIT_CODES = [-1003, -1015, -1120]
INVALID_FILTER_CODES = [-1013, -2010]
REDUCE_ONLY_REJECTED_CODE = -2022
INVALID_API_KEY_CODE = -2008


def async_retry_api_call(max_retries: int = 3, initial_delay: float = 1.0, max_delay: float = 10.0):
    """
    Decorator for retrying asynchronous Binance API calls with exponential backoff.
    Specific to Binance API exceptions and common network errors.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay between retries in seconds.
        max_delay (float): Maximum delay between retries in seconds.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries + 1): # +1 to include the initial try
                try:
                    await asyncio.sleep(0.05) # Small fixed pre-call delay
                    return await func(*args, **kwargs)
                except (BinanceAPIException, BinanceRequestException) as e:
                    last_exception = e
                    # Non-retryable errors
                    if e.code in [ORDER_NOT_FOUND_CODE, REDUCE_ONLY_REJECTED_CODE, INVALID_API_KEY_CODE] or \
                       e.code in INSUFFICIENT_FUNDS_CODES or \
                       e.code in INVALID_FILTER_CODES or \
                       (e.status_code >= 400 and e.status_code < 500 and e.status_code not in [429]):
                        logger.error(f"Non-retryable Binance API error on {func.__name__} (Code: {e.code}, Status: {getattr(e, 'status_code', 'N/A')}): {e}", exc_info=False)
                        raise
                    # Retryable errors (rate limits, server errors)
                    elif e.code in RATE_LIMIT_CODES or getattr(e, 'status_code', None) in [429, 500, 502, 503, 504]:
                        if attempt < max_retries:
                            logger.warning(f"Retryable API error on {func.__name__} (Code: {e.code}, Status: {getattr(e, 'status_code', 'N/A')}). Retry {attempt+1}/{max_retries}. Sleeping {delay:.2f}s.")
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, max_delay)
                        else:
                            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__} after retryable error (Code: {e.code}, Status: {getattr(e, 'status_code', 'N/A')}).")
                            raise ConnectionError(f"Max retries exceeded for {func.__name__} after API error {e.code}") from e
                    else:
                        logger.error(f"Unexpected non-retryable Binance API error on {func.__name__} (Code: {e.code}, Status: {getattr(e, 'status_code', 'N/A')}): {e}", exc_info=True)
                        raise
                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Request timeout on {func.__name__}. Retry {attempt+1}/{max_retries}. Sleeping {delay:.2f}s.")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__} after timeout.")
                        raise ConnectionError(f"Max retries exceeded for {func.__name__} after timeout") from e
                except ConnectionError as e:
                     last_exception = e
                     if attempt < max_retries:
                          logger.warning(f"Network connection error on {func.__name__}: {e}. Retry {attempt+1}/{max_retries}. Sleeping {delay:.2f}s.")
                          await asyncio.sleep(delay)
                          delay = min(delay * 2, max_delay)
                     else:
                          logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__} after connection error.")
                          raise ConnectionError(f"Max retries exceeded for {func.__name__} after connection error") from e
                except Exception as e:
                    logger.error(f"Unexpected error during API call {func.__name__}: {e}", exc_info=True)
                    raise
            if last_exception:
                raise last_exception
            else:
                # This path should ideally not be reached if an exception was always captured
                raise ConnectionError(f"API call {func.__name__} failed after {max_retries} retries without a specific exception.")
        return wrapper
    return decorator

