# utils/exchange_adapters/binance/client_manager.py

import asyncio
import logging
from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException
from typing import Optional, Any
import numpy as np # Used for np.nan

# Custom exceptions
from utils.exceptions import ExchangeConnectionError, ConfigurationError

# Import AppConfig to access exchange configuration directly
from config.params import AppConfig
from config.exchange import ExchangeConfig # For type hinting

# Decorator for retry logic
from utils.exchange_adapters.binance.decorators import async_retry_api_call

logger = logging.getLogger(__name__)

class BinanceAPIClientManager:
    """
    Manages the lifecycle of the Binance AsyncClient.
    Ensures a single, robust client instance is used across the application.
    Handles connection, disconnection, and client retrieval.
    """
    _instance: Optional[AsyncClient] = None
    _is_connected: bool = False
    _lock = asyncio.Lock() # Class-level lock for async operations
    _client_config: Optional[ExchangeConfig] = None
    _logger: Optional[logging.Logger] = None

    def __init__(self, api_key: str, api_secret: str, exchange_config: ExchangeConfig, logger: logging.Logger):
        if not BinanceAPIClientManager._instance:
            BinanceAPIClientManager._client_config = exchange_config
            BinanceAPIClientManager._logger = logger
            self.api_key = api_key
            self.api_secret = api_secret
            self.exchange_config = exchange_config # Keep a reference to the config
            self.logger = logger # Keep a reference to the logger

    @property
    def get_client(self) -> AsyncClient:
        """
        Provides the singleton AsyncClient instance.
        Raises ExchangeConnectionError if the client is not yet connected.
        """
        if not BinanceAPIClientManager._is_connected or BinanceAPIClientManager._instance is None:
            self.logger.critical("Binance AsyncClient is not initialized. Call connect() first.")
            raise ExchangeConnectionError("Binance AsyncClient is not initialized. Call connect() first.")
        return BinanceAPIClientManager._instance
    
    def set_client(self, client: Any):
        """
        Sets the AsyncClient instance directly. This is useful for passing
        an already initialized client (e.g., from a test environment).
        """
        if not isinstance(client, AsyncClient):
            raise TypeError("Provided client must be an instance of binance.AsyncClient")
        BinanceAPIClientManager._instance = client
        BinanceAPIClientManager._is_connected = True
        self.logger.debug("Binance AsyncClient instance set manually.")


    @async_retry_api_call(max_retries=3, initial_delay=1)
    async def connect(self):
        """
        Establishes an asynchronous connection to the Binance API.
        This method is idempotent; it will only create a new client if one doesn't exist.
        """
        async with BinanceAPIClientManager._lock:
            if not BinanceAPIClientManager._is_connected:
                self.logger.info("Attempting to connect to Binance AsyncClient...")
                try:
                    # Retrieve config from the class-level storage
                    config = BinanceAPIClientManager._client_config
                    if not config:
                        raise ConfigurationError("BinanceAPIClientManager configuration not set during connect.")

                    # Construct options dictionary for the client
                    client_options = config.options.copy() # Start with default options

                    # Add verbose and enableRateLimit if specified in config
                    client_options['verbose'] = config.verbose
                    client_options['enableRateLimit'] = config.enableRateLimit

                    # Use testnet base URL if testnet is enabled
                    if config.testnet:
                        client_options['base_url'] = 'https://testnet.binancefuture.com' # Futures testnet URL
                        self.logger.info("Connecting to Binance Futures TESTNET.")
                    else:
                        self.logger.info("Connecting to Binance Futures MAINNET.")

                    # Create the client instance using the collected options
                    # Pass the client_options dictionary directly as the 'options' argument
                    BinanceAPIClientManager._instance = await AsyncClient.create(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        tld=config.tld,
                        testnet=config.testnet,
                    )
                    
                    # Synchronize time with Binance server
                    await BinanceAPIClientManager._instance.futures_time()
                    
                    BinanceAPIClientManager._is_connected = True
                    self.logger.info("Successfully connected to Binance AsyncClient.")
                except Exception as e:
                    self.logger.critical(f"Failed to create Binance client: {e}", exc_info=True)
                    BinanceAPIClientManager._instance = None
                    BinanceAPIClientManager._is_connected = False
                    raise ExchangeConnectionError(f"Failed to create Binance client: {e}") from e
            else:
                self.logger.info("Binance AsyncClient already connected.")

    async def close_connection(self):
        """
        Closes the Binance AsyncClient connection.
        """
        async with BinanceAPIClientManager._lock:
            if BinanceAPIClientManager._is_connected and BinanceAPIClientManager._instance:
                self.logger.info("Closing Binance AsyncClient connection...")
                try:
                    await BinanceAPIClientManager._instance.close_connection()
                    self.logger.info("Binance AsyncClient connection closed.")
                except Exception as e:
                    self.logger.error(f"Error closing Binance AsyncClient connection: {e}", exc_info=True)
                finally:
                    BinanceAPIClientManager._instance = None
                    BinanceAPIClientManager._is_connected = False
            else:
                self.logger.info("Binance AsyncClient is not active or already closed.")

    @classmethod
    def get_status(cls) -> bool:
        """
        Returns the connection status of the Binance AsyncClient.
        """
        return cls._is_connected
    
    async def periodic_time_sync(self, interval_seconds: int = 600):
        """
        Periodically synchronizes time with Binance server to prevent timestamp errors.
        Should be launched as a background task after connecting.
        """
        while BinanceAPIClientManager._is_connected and BinanceAPIClientManager._instance:
            try:
                await BinanceAPIClientManager._instance.futures_time()
                self.logger.debug("Binance API time sync successful.")
            except Exception as e:
                self.logger.warning(f"Failed Binance time sync: {e}")
            await asyncio.sleep(interval_seconds)
