# utils/simulation/price_path_simulator.py

import logging
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import sys

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Import AppConfig and FLOAT_EPSILON directly
    from config.params import AppConfig, FLOAT_EPSILON 
    from arch import arch_model # For GARCH modeling
except ImportError as e:
    logging.critical(f"Failed to import necessary modules for PricePathSimulator: {e}. Ensure 'arch' library is installed and config is accessible.", exc_info=True)
    sys.exit(1)

logger = logging.getLogger(__name__)

class PricePathSimulator:
    """
    Fits a GARCH model for the diffusion component and generates synthetic OHLCV data paths
    with an added jump component. Parameters for both are estimated from historical data.
    """
    def __init__(self, historical_data: pd.DataFrame, app_config: AppConfig):
        """
        Initializes the PricePathSimulator.

        Args:
            historical_data (pd.DataFrame): DataFrame containing historical OHLCV data,
                                            used to fit models and learn patterns.
            app_config (AppConfig): The global application configuration object.
        """
        if not all(col in historical_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Historical data must contain OHLCV columns: 'open', 'high', 'low', 'close', 'volume'.")
        
        self.hist_data = historical_data
        self.hist_returns = historical_data['close'].pct_change().dropna()
        self.fitted_garch_model = None
        self.mean_drift = self.hist_returns.mean() # Mean drift from historical returns
        self.app_config = app_config # Store app_config to access other config attributes
        
        # Scaling factor for GARCH model fitting. Recommended by 'arch' for numerical stability.
        self.garch_scale_factor = 1000.0 

        # Initialize jump parameters to defaults; they will be estimated later from residuals.
        self.jump_intensity_lambda = 0.001 # Default small value (frequency of jumps)
        self.jump_mean = 0.0               # Default jump size mean
        self.jump_std = 0.001              # Default jump size standard deviation

        self._learn_historical_patterns() # Learn OHLC ratios and volume distribution from historical data
        self.fit_garch_model()            # Fit GARCH model for diffusion
        self._estimate_jump_parameters()  # Estimate jump parameters from GARCH residuals

    def _learn_historical_patterns(self):
        """
        Analyzes historical data to learn empirical distributions for OHLC ratios and Volume.
        These ratios are used to reconstruct realistic candle shapes from simulated close prices.
        """
        # Ratios relative to previous close for 'open'
        self.hist_open_to_prev_close_ratio = (self.hist_data['open'] / self.hist_data['close'].shift(1)).dropna()
        # Ratios relative to current close for 'high' and 'low'
        self.hist_high_to_close_ratio = (self.hist_data['high'] / self.hist_data['close']).dropna()
        self.hist_low_to_close_ratio = (self.hist_data['low'] / self.hist_data['close']).dropna()
        # Historical volume distribution
        self.hist_volume = self.hist_data['volume'].dropna()

    def fit_garch_model(self, p: int = 1, q: int = 1, dist: str = 't'):
        """
        Fits a GARCH(p, q) model to the historical returns to capture the diffusion component (volatility clustering).

        Args:
            p (int): The order of the AR component in the GARCH model (lagged squared residuals).
            q (int): The order of the MA component in the GARCH model (lagged conditional variances).
            dist (str): The distribution assumption for the innovations ('t' for Student's t, 'normal' for normal).
        """
        logger.info(f"Fitting GARCH(p={p}, q={q}) model with '{dist}' distribution for diffusion component...")
        if self.hist_returns.empty:
            logger.warning("Historical returns series is empty. Cannot fit GARCH model.")
            self.fitted_garch_model = None
            return
        
        # Check for constant returns, which can cause issues with GARCH fitting
        # Access FLOAT_EPSILON directly as a global constant
        if self.hist_returns.std() < FLOAT_EPSILON: 
            logger.warning("Historical returns have zero variance. GARCH model cannot be fitted. Simulating with constant returns.")
            self.fitted_garch_model = None # Indicate no GARCH model was fitted
            return

        # Rescale returns before passing to arch_model to improve numerical stability.
        # This is a common practice when working with very small return values.
        scaled_returns = self.hist_returns * self.garch_scale_factor
        
        # 'rescale=False' is used to suppress a warning, as we are manually handling the scaling.
        garch_model = arch_model(scaled_returns, p=p, q=q, vol='Garch', dist=dist, rescale=False)
        try:
            self.fitted_garch_model = garch_model.fit(disp='off') # 'disp=off' suppresses verbose output during fitting
            logger.info("GARCH model fitting complete.")
        except Exception as e:
            logger.warning(f"GARCH model fitting failed: {e}. Simulating returns with historical mean and std dev as fallback.", exc_info=True)
            self.fitted_garch_model = None # Indicate fitting failed

    def _estimate_jump_parameters(self, jump_threshold_std_dev: float = 3.0):
        """
        Estimates jump intensity (lambda), mean, and standard deviation from GARCH standardized residuals.
        Jumps are identified as outliers in the residuals beyond a certain standard deviation threshold.

        Args:
            jump_threshold_std_dev (float): Number of standard deviations from the mean to consider a residual a "jump".
        """
        if self.fitted_garch_model is None:
            logger.warning("GARCH model not fitted. Cannot estimate jump parameters from residuals. Using default jump parameters.")
            return

        # Standardized residuals are innovations divided by conditional volatility; they should ideally be i.i.d.
        standardized_residuals = self.fitted_garch_model.resid / self.fitted_garch_model.conditional_volatility
        
        # Identify potential jumps by looking for residuals that are significant outliers
        jumps_filter = np.abs(standardized_residuals) > jump_threshold_std_dev
        jumps = standardized_residuals[jumps_filter]

        if not jumps.empty:
            # Estimate jump intensity (number of jumps per period)
            self.jump_intensity_lambda = len(jumps) / len(self.hist_returns) # Relative to original historical returns length
            
            # Estimate jump mean and std from the actual return values corresponding to these jump events
            actual_jumps_returns = self.hist_returns.loc[jumps.index]
            
            self.jump_mean = actual_jumps_returns.mean()
            self.jump_std = actual_jumps_returns.std()
            
            logger.info(f"Estimated Jump Parameters: Lambda={self.jump_intensity_lambda:.4f}, Mean={self.jump_mean:.4f}, Std={self.jump_std:.4f}")
        else:
            logger.info("No significant jumps detected in GARCH residuals. Using default (or very small) jump parameters.")
            # Keep initialized small default values if no jumps are found

    def simulate_one_path(self, num_periods: int, start_date: pd.Timestamp, freq) -> Optional[pd.DataFrame]:
        """
        Generates a single, full synthetic OHLCV data path by combining:
        1. A GARCH-modeled diffusion process for volatility.
        2. A Poisson-driven jump process for sudden, large price movements.
        3. Reconstruction of OHLCV from simulated close prices based on historical patterns.

        Args:
            num_periods (int): The number of periods (bars) to simulate.
            start_date (pd.Timestamp): The starting timestamp for the simulated data.
            freq: The frequency of the data (e.g., '1H', '5T').

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the simulated 'open', 'high', 'low', 'close', 'volume'
                                    data, or None if simulation fails.
        """
        # Fallback if GARCH model failed to fit (e.g., due to constant returns)
        if self.fitted_garch_model is None:
            logger.warning("GARCH model not fitted. Simulating returns with historical mean and std dev without GARCH dynamics.")
            sim_returns_base = np.random.normal(self.mean_drift, self.hist_returns.std(), num_periods)
        else:
            # Get the last conditional variance from the historical data for simulation start.
            # Remember to unscale the variance if returns were scaled during fitting.
            last_variance_scaled = self.fitted_garch_model.conditional_volatility.iloc[-1]**2
            last_variance = last_variance_scaled / (self.garch_scale_factor**2)
            
            params = self.fitted_garch_model.params
            omega_scaled = params['omega']
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            nu = params.get('nu', np.inf) # Degrees of freedom for Students-t distribution

            # Unscale omega for direct use in simulation
            omega = omega_scaled / (self.garch_scale_factor**2)

            sim_returns_base = np.zeros(num_periods)
            current_variance = last_variance # Start with the last known variance

            for t in range(num_periods):
                # 1. Diffusion Component (GARCH process)
                if self.fitted_garch_model.model.distribution.name == 'StudentsT':
                    # Draw from a Student's t-distribution and scale to ensure unit variance.
                    random_shock_diffusion = np.random.standard_t(df=nu) * np.sqrt((nu - 2) / nu)
                else: # Assume Normal distribution for innovations if not Students-t
                    random_shock_diffusion = np.random.normal()
                
                # The diffusion_return incorporates the drift and is scaled by current conditional volatility
                diffusion_return = self.mean_drift + random_shock_diffusion * np.sqrt(current_variance)
                sim_returns_base[t] = diffusion_return
                
                # Update conditional variance for the next period using the GARCH equation
                current_variance = omega + alpha * (diffusion_return**2) + beta * current_variance
        
        # 2. Jump Component (Poisson Process) - these are added to the base returns
        num_jumps_per_period = np.random.poisson(self.jump_intensity_lambda, num_periods)
        jump_returns = np.zeros(num_periods)
        for t in range(num_periods):
            if num_jumps_per_period[t] > 0:
                # If jumps occur, draw jump sizes from a normal distribution
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, num_jumps_per_period[t])
                jump_returns[t] = np.sum(jump_sizes) # Sum up multiple jumps if they occur in one period
        
        # Total return combines the diffusion component with any jumps
        sim_returns = sim_returns_base + jump_returns

        # Calculate simulated close prices by compounding returns from the last historical close price
        sim_close_prices = self.hist_data['close'].iloc[-1] * (1 + sim_returns).cumprod()

        # Create a DataFrame for the synthetic path with a DatetimeIndex
        synthetic_df = pd.DataFrame(index=pd.date_range(start=start_date, periods=num_periods, freq=freq))
        synthetic_df['close'] = sim_close_prices
        
        # Initialize all OHLCV columns with NaN to ensure they exist before assignment
        synthetic_df['open'] = np.nan
        synthetic_df['high'] = np.nan
        synthetic_df['low'] = np.nan
        synthetic_df['volume'] = np.nan
        
        # Crucial check: ensure historical ratios are not empty before attempting to use them
        if self.hist_open_to_prev_close_ratio.empty or \
           self.hist_high_to_close_ratio.empty or \
           self.hist_low_to_close_ratio.empty:
            logger.warning("Historical OHLC ratios are empty. Cannot accurately simulate OHLC. Returning None.")
            return None

        # Reconstruct Open, High, Low based on simulated Close and learned historical ratios.
        # The first synthetic 'open' is based on the last historical 'close'.
        if not self.hist_open_to_prev_close_ratio.empty:
            synthetic_df.loc[synthetic_df.index[0], 'open'] = self.hist_data['close'].iloc[-1] * np.random.choice(self.hist_open_to_prev_close_ratio)
        else:
            synthetic_df.loc[synthetic_df.index[0], 'open'] = synthetic_df['close'].iloc[0] # Fallback

        # For subsequent bars, 'open' is based on the *previous synthetic close*.
        for i in range(1, num_periods):
            if not self.hist_open_to_prev_close_ratio.empty:
                synthetic_df.loc[synthetic_df.index[i], 'open'] = synthetic_df['close'].iloc[i-1] * np.random.choice(self.hist_open_to_prev_close_ratio)
            else:
                synthetic_df.loc[synthetic_df.index[i], 'open'] = synthetic_df['close'].iloc[i-1] # Fallback

        # High and Low are relative to their own bar's close.
        if not self.hist_high_to_close_ratio.empty:
            synthetic_df['high'] = synthetic_df['close'] * np.random.choice(self.hist_high_to_close_ratio, size=num_periods)
        else:
            synthetic_df['high'] = synthetic_df['close'] * 1.001 # Fallback, small percentage above close
        
        if not self.hist_low_to_close_ratio.empty:
            synthetic_df['low'] = synthetic_df['close'] * np.random.choice(self.hist_low_to_close_ratio, size=num_periods)
        else:
            synthetic_df['low'] = synthetic_df['close'] * 0.999 # Fallback, small percentage below close

        # Final adjustment to ensure OHLC consistency (high >= open, close; low <= open, close)
        synthetic_df['high'] = synthetic_df[['high', 'open', 'close']].max(axis=1)
        synthetic_df['low'] = synthetic_df[['low', 'open', 'close']].min(axis=1)
        
        # Simulate 'volume' based on historical volumes' distribution
        if not self.hist_volume.empty:
            synthetic_df['volume'] = np.random.choice(self.hist_volume, size=num_periods)
        else:
            synthetic_df['volume'] = np.mean(self.hist_data['volume'].dropna()) if not self.hist_data['volume'].dropna().empty else 1000 # Fallback volume

        return synthetic_df[['open', 'high', 'low', 'close', 'volume']]

