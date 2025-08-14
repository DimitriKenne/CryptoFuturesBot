# utils/data_processing/time_synchronizer.py

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd # Used for pd.Timestamp type hinting

logger = logging.getLogger(__name__)

class TimeSynchronizer:
    """
    Manages time-related operations for synchronizing with market candles,
    parsing intervals, and waiting for the next candle close.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_next_candle_time(self, last_candle_time: Optional[pd.Timestamp], interval: str) -> Optional[datetime]:
        """
        Calculates the expected UTC close time of the next candle based on the last known candle.

        Args:
            last_candle_time (Optional[pd.Timestamp]): The timestamp of the last observed candle.
            interval (str): The OHLCV interval string (e.g., '1m', '1h', '1d').

        Returns:
            Optional[datetime]: The expected UTC datetime of the next candle's close, or None if calculation fails.
        """
        if last_candle_time is None:
            self.logger.error("Cannot determine next candle time: last_candle_time is not set.")
            return None
        if not isinstance(last_candle_time, pd.Timestamp):
            self.logger.error(f"Invalid last_candle_time type: {type(last_candle_time)}. Expected pandas Timestamp.")
            return None

        try:
            interval_timedelta = self.parse_interval_to_timedelta(interval)
            if interval_timedelta is None:
                raise ValueError(f"Could not parse interval string: {interval}")

            # Ensure last_candle_time is timezone-aware UTC
            last_time_utc = last_candle_time.tz_convert(timezone.utc) if last_candle_time.tz else last_candle_time.tz_localize(timezone.utc)

            # Next candle's open time is last candle's close time.
            # Next candle's close time is last candle's close time + interval_timedelta
            next_candle_time_utc = last_time_utc + interval_timedelta
            return next_candle_time_utc

        except Exception as e:
            self.logger.error(f"Error calculating next candle time: {e}", exc_info=True)
            return None

    def parse_interval_to_timedelta(self, interval_str: str) -> Optional[timedelta]:
        """
        Parses a common exchange interval string (e.g., '1m', '1h', '1d') into a timedelta object.

        Args:
            interval_str (str): The interval string.

        Returns:
            Optional[timedelta]: The corresponding timedelta object, or None if the interval is unsupported/invalid.
        """
        try:
            if not isinstance(interval_str, str) or len(interval_str) < 2:
                self.logger.warning(f"Invalid interval string format: '{interval_str}'.")
                return None

            unit = interval_str[-1].lower()
            value = int(interval_str[:-1])

            if unit == 'm': return timedelta(minutes=value)
            if unit == 'h': return timedelta(hours=value)
            if unit == 'd': return timedelta(days=value)
            if unit == 'w': return timedelta(weeks=value)
            # Add 'M' for months if needed, but it's more complex with timedelta
            # Consider using dateutil.relativedelta for more complex calendar intervals (e.g., '1M' for one month)
            # if unit == 'M': return timedelta(days=value * 30) # Approximation, safer to use a date offset library

            self.logger.warning(f"Unsupported interval unit '{unit}' for precise timedelta calculation.")
            return None
        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to parse interval string '{interval_str}': {e}", exc_info=True)
            return None

    async def wait_until_next_candle(self, next_candle_time_utc: datetime, buffer_sec: int = 10):
        """
        Asynchronously waits until the specified UTC time, plus a small buffer, to ensure
        the next candle has fully closed and is available on the exchange.

        Args:
            next_candle_time_utc (datetime): The target UTC datetime to wait until.
            buffer_sec (int): A small buffer in seconds to wait past the candle close time.
        """
        wait_until_utc = next_candle_time_utc + timedelta(seconds=buffer_sec)
        now_utc = datetime.now(timezone.utc)
        time_to_wait_sec = (wait_until_utc - now_utc).total_seconds()

        if time_to_wait_sec > 0:
            self.logger.info(f"Waiting {time_to_wait_sec:.2f}s for next candle close ({next_candle_time_utc.isoformat()})...")
            await asyncio.sleep(time_to_wait_sec)
            self.logger.debug("Wait complete.")
        else:
            lag_seconds = abs(time_to_wait_sec)
            self.logger.warning(f"Next candle time ({next_candle_time_utc.isoformat()}) is in the past. Lagging by {lag_seconds:.2f}s. Processing immediately.")

