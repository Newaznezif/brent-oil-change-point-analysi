"""
Data Preprocessing Module for Brent Oil Price Analysis

This module handles all data preparation tasks including:
- Date parsing and cleaning
- Stationarity testing and transformations
- Log returns calculation
- Outlier detection and handling
- Feature engineering for time series analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, Union
import warnings
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class BrentDataPreprocessor:
    """
    Comprehensive data preprocessing for Brent oil price analysis
    """
    
    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        data_path : str, optional
            Path to CSV file containing Brent oil price data
        df : pd.DataFrame, optional
            Already loaded DataFrame
        """
        if data_path:
            self.df = self.load_data(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either data_path or df must be provided")
            
        self.original_shape = self.df.shape
        self.processed_data = None
        self.stationarity_results = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load Brent oil price data from CSV
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def parse_dates(self, date_column: str = 'Date', date_format: Optional[str] = None) -> pd.DataFrame:
        """
        Parse date column with robust handling of different formats
        
        Parameters:
        -----------
        date_column : str
            Name of the date column
        date_format : str, optional
            Specific date format to use. If None, tries multiple formats.
            
        Returns:
        --------
        pd.DataFrame : Data with parsed dates
        """
        print(f"\nParsing dates in column '{date_column}'...")
        
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found in data")
        
        original_dates = self.df[date_column].copy()
        
        # Try multiple date formats
        date_formats = [
            '%d-%b-%y',    # 20-May-87
            '%Y-%m-%d',    # 1987-05-20
            '%d/%m/%Y',    # 20/05/1987
            '%m/%d/%Y',    # 05/20/1987
            '%d-%m-%Y',    # 20-05-1987
            '%b %d, %Y',   # May 20, 1987
        ]
        
        if date_format:
            date_formats = [date_format] + date_formats
        
        parsed_successfully = False
        for fmt in date_formats:
            try:
                self.df[date_column] = pd.to_datetime(self.df[date_column], format=fmt)
                print(f"  ✓ Successfully parsed with format: {fmt}")
                parsed_successfully = True
                break
            except:
                continue
        
        if not parsed_successfully:
            # Try pandas inference as last resort
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            print("  ✓ Parsed using pandas inference")
        
        # Fix year 2000 issues (e.g., '87' -> 1987, '22' -> 2022)
        def fix_year(date):
            try:
                if date.year > 2023:  # If year appears as 2087 instead of 1987
                    return date.replace(year=date.year - 100)
                return date
            except:
                return date
        
        self.df[date_column] = self.df[date_column].apply(fix_year)
        
        # Sort by date
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
        
        # Set date as index for time series analysis
        self.df.set_index(date_column, inplace=True)
        
        print(f"Date range after parsing: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Number of unique dates: {self.df.index.nunique()}")
        
        return self.df
    
    def clean_price_data(self, price_column: str = 'Price') -> pd.DataFrame:
        """
        Clean and validate price data
        
        Parameters:
        -----------
        price_column : str
            Name of the price column
            
        Returns:
        --------
        pd.DataFrame : Cleaned data
        """
        print(f"\nCleaning price data in column '{price_column}'...")
        
        if price_column not in self.df.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        # Check for missing values
        missing_before = self.df[price_column].isnull().sum()
        if missing_before > 0:
            print(f"  Found {missing_before} missing values")
            
            # Interpolate missing values
            self.df[price_column] = self.df[price_column].interpolate(method='linear')
            
            # Forward/backward fill any remaining NaNs at edges
            self.df[price_column] = self.df[price_column].fillna(method='ffill').fillna(method='bfill')
            
            missing_after = self.df[price_column].isnull().sum()
            print(f"  Missing values after cleaning: {missing_after}")
        
        # Check for zeros or negative prices (data quality issue)
        zero_or_negative = (self.df[price_column] <= 0).sum()
        if zero_or_negative > 0:
            print(f"  WARNING: Found {zero_or_negative} prices ≤ 0")
            # Replace zeros/negatives with NaN and interpolate
            self.df.loc[self.df[price_column] <= 0, price_column] = np.nan
            self.df[price_column] = self.df[price_column].interpolate(method='linear')
        
        # Check for outliers using IQR method
        Q1 = self.df[price_column].quantile(0.25)
        Q3 = self.df[price_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((self.df[price_column] < lower_bound) | (self.df[price_column] > upper_bound)).sum()
        if outliers > 0:
            print(f"  Found {outliers} potential outliers (outside 3*IQR)")
            # Optionally cap outliers (commented out as oil prices can legitimately spike)
            # self.df[price_column] = np.clip(self.df[price_column], lower_bound, upper_bound)
        
        # Basic statistics
        print(f"  Price statistics after cleaning:")
        print(f"    • Min: ${self.df[price_column].min():.2f}")
        print(f"    • Max: ${self.df[price_column].max():.2f}")
        print(f"    • Mean: ${self.df[price_column].mean():.2f}")
        print(f"    • Std: ${self.df[price_column].std():.2f}")
        
        return self.df
    
    def calculate_log_returns(self, price_column: str = 'Price') -> pd.DataFrame:
        """
        Calculate log returns for stationarity
        
        Parameters:
        -----------
        price_column : str
            Name of the price column
            
        Returns:
        --------
        pd.DataFrame : Data with log returns added
        """
        print(f"\nCalculating log returns from '{price_column}'...")
        
        if price_column not in self.df.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        # Calculate log price (for visualization)
        self.df['log_price'] = np.log(self.df[price_column])
        
        # Calculate log returns (percentage)
        self.df['log_returns'] = self.df['log_price'].diff() * 100
        
        # Calculate simple returns (for comparison)
        self.df['simple_returns'] = self.df[price_column].pct_change() * 100
        
        # Calculate absolute returns (for volatility analysis)
        self.df['abs_returns'] = self.df['log_returns'].abs()
        
        # Drop the first row (NaN from differencing)
        self.df = self.df.dropna(subset=['log_returns', 'simple_returns'])
        
        print(f"  Log returns calculated successfully")
        print(f"  Log returns statistics:")
        print(f"    • Mean: {self.df['log_returns'].mean():.6f}%")
        print(f"    • Std: {self.df['log_returns'].std():.6f}%")
        print(f"    • Min: {self.df['log_returns'].min():.6f}%")
        print(f"    • Max: {self.df['log_returns'].max():.6f}%")
        
        return self.df
    
    def test_stationarity(self, series: pd.Series, series_name: str = 'series') -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        series_name : str
            Name of the series for reporting
            
        Returns:
        --------
        Dict : Stationarity test results
        """
        print(f"\nTesting stationarity of {series_name}...")
        
        result = adfuller(series.dropna())
        
        test_results = {
            'series': series_name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'n_obs': len(series.dropna()),
            'is_stationary': result[1] < 0.05
        }
        
        print(f"  ADF Statistic: {result[0]:.4f}")
        print(f"  p-value: {result[1]:.4f}")
        print(f"  Critical Values:")
        for key, value in result[4].items():
            print(f"    {key}: {value:.4f}")
        
        if test_results['is_stationary']:
            print(f"  ✓ {series_name} is STATIONARY (reject null hypothesis)")
        else:
            print(f"  ✗ {series_name} is NON-STATIONARY (fail to reject null hypothesis)")
        
        self.stationarity_results[series_name] = test_results
        return test_results
    
    def analyze_all_series_stationarity(self) -> pd.DataFrame:
        """
        Test stationarity of all relevant series
        
        Returns:
        --------
        pd.DataFrame : Stationarity test results for all series
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE STATIONARITY ANALYSIS")
        print("="*60)
        
        series_to_test = {
            'price': self.df.get('Price'),
            'log_price': self.df.get('log_price'),
            'log_returns': self.df.get('log_returns'),
            'simple_returns': self.df.get('simple_returns')
        }
        
        all_results = []
        for name, series in series_to_test.items():
            if series is not None:
                result = self.test_stationarity(series, name)
                all_results.append(result)
        
        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)
        
        print("\n" + "="*60)
        print("STATIONARITY SUMMARY")
        print("="*60)
        for _, row in results_df.iterrows():
            status = "STATIONARY" if row['is_stationary'] else "NON-STATIONARY"
            print(f"{row['series']:15} : {status:15} (p = {row['p_value']:.6f})")
        
        return results_df
    
    def decompose_time_series(self, series_column: str = 'Price', 
                             model: str = 'additive', 
                             period: Optional[int] = None) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Parameters:
        -----------
        series_column : str
            Name of the series to decompose
        model : str
            'additive' or 'multiplicative'
        period : int, optional
            Period for seasonal decomposition. If None, tries to auto-detect.
            
        Returns:
        --------
        Dict : Decomposition results
        """
        print(f"\nDecomposing {series_column} time series...")
        
        if series_column not in self.df.columns:
            raise ValueError(f"Column '{series_column}' not found in data")
        
        series = self.df[series_column].dropna()
        
        # Auto-detect period if not provided
        if period is None:
            # For daily financial data, common periods to check
            possible_periods = [5, 20, 21, 22, 30, 60, 90, 252]  # Trading days patterns
            best_period = None
            
            # Simple auto-correlation based period detection
            if len(series) > 252:  # Need enough data
                autocorr = pd.Series(series).autocorr(lag=5)
                if abs(autocorr) > 0.3:
                    best_period = 5  # Weekly pattern
                else:
                    best_period = 252  # Annual pattern
            
            period = best_period or 252  # Default to annual
        
        try:
            decomposition = seasonal_decompose(series, model=model, period=period)
            
            # Store decomposition results
            decomp_results = {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': model
            }
            
            print(f"  Successfully decomposed with {model} model (period={period})")
            
            return decomp_results
            
        except Exception as e:
            print(f"  Warning: Could not decompose series: {e}")
            return {}
    
    def calculate_rolling_statistics(self, window_sizes: list = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for volatility analysis
        
        Parameters:
        -----------
        window_sizes : list, optional
            List of window sizes in days. Default: [21, 63, 252] (1mo, 3mo, 1yr)
            
        Returns:
        --------
        pd.DataFrame : Data with rolling statistics added
        """
        if window_sizes is None:
            window_sizes = [21, 63, 252]  # 1 month, 3 months, 1 year
        
        print(f"\nCalculating rolling statistics with windows: {window_sizes}")
        
        if 'log_returns' not in self.df.columns:
            raise ValueError("Log returns not calculated. Run calculate_log_returns() first.")
        
        returns = self.df['log_returns']
        
        for window in window_sizes:
            # Rolling mean
            self.df[f'rolling_mean_{window}'] = returns.rolling(window=window).mean()
            
            # Rolling standard deviation (volatility)
            self.df[f'rolling_std_{window}'] = returns.rolling(window=window).std()
            
            # Rolling maximum drawdown
            rolling_cumulative = (1 + returns/100).rolling(window=window).apply(
                lambda x: (x.prod() - 1) * 100, raw=False
            )
            self.df[f'rolling_max_{window}'] = rolling_cumulative.rolling(window=window).max()
            self.df[f'rolling_min_{window}'] = rolling_cumulative.rolling(window=window).min()
            self.df[f'drawdown_{window}'] = (
                (self.df[f'rolling_cumulative_{window}'] - self.df[f'rolling_max_{window}']) / 
                self.df[f'rolling_max_{window}'].abs()
            ) * 100
            
            print(f"  Added statistics for {window}-day window")
        
        return self.df
    
    def detect_volatility_clusters(self, window: int = 21, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect periods of high volatility (volatility clustering)
        
        Parameters:
        -----------
        window : int
            Rolling window for volatility calculation
        threshold : float
            Number of standard deviations above mean to consider high volatility
            
        Returns:
        --------
        pd.DataFrame : Data with volatility cluster indicators
        """
        print(f"\nDetecting volatility clusters...")
        
        if 'log_returns' not in self.df.columns:
            raise ValueError("Log returns not calculated. Run calculate_log_returns() first.")
        
        # Calculate rolling volatility
        returns = self.df['log_returns']
        rolling_vol = returns.rolling(window=window).std()
        
        # Identify high volatility periods
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        high_vol_threshold = vol_mean + threshold * vol_std
        
        self.df['high_volatility'] = rolling_vol > high_vol_threshold
        self.df['volatility_regime'] = self.df['high_volatility'].map({True: 'High', False: 'Normal'})
        
        # Calculate cluster statistics
        high_vol_periods = self.df['high_volatility'].sum()
        high_vol_pct = (high_vol_periods / len(self.df)) * 100
        
        print(f"  High volatility threshold: {high_vol_threshold:.4f}%")
        print(f"  High volatility periods: {high_vol_periods} ({high_vol_pct:.1f}% of data)")
        print(f"  Average duration of high volatility clusters: {self._calculate_cluster_duration():.1f} days")
        
        return self.df
    
    def _calculate_cluster_duration(self) -> float:
        """Calculate average duration of high volatility clusters"""
        if 'high_volatility' not in self.df.columns:
            return 0
        
        # Find clusters of consecutive True values
        clusters = []
        in_cluster = False
        cluster_start = None
        
        for i, is_high in enumerate(self.df['high_volatility']):
            if is_high and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif not is_high and in_cluster:
                in_cluster = False
                clusters.append(i - cluster_start)
        
        if in_cluster and cluster_start is not None:
            clusters.append(len(self.df) - cluster_start)
        
        return np.mean(clusters) if clusters else 0
    
    def prepare_event_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create event-based features for modeling
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            DataFrame containing event information with 'date' column
            
        Returns:
        --------
        pd.DataFrame : Data with event features added
        """
        print(f"\nPreparing event features...")
        
        # Ensure events have datetime index
        events_df = events_df.copy()
        events_df['date'] = pd.to_datetime(events_df['date'])
        events_df.set_index('date', inplace=True)
        
        # Create event flags
        for event_type in events_df['event_type'].unique():
            event_dates = events_df[events_df['event_type'] == event_type].index
            self.df[f'event_{event_type.lower().replace(" ", "_")}'] = self.df.index.isin(event_dates)
            print(f"  Added flag for {event_type} events")
        
        # Create impact level features
        for impact_level in events_df['impact_level'].unique():
            event_dates = events_df[events_df['impact_level'] == impact_level].index
            self.df[f'impact_{impact_level.lower()}'] = self.df.index.isin(event_dates)
            print(f"  Added flag for {impact_level} impact events")
        
        # Create rolling event counts
        for window in [5, 21, 63]:  # 1 week, 1 month, 3 months
            event_count = pd.Series(index=self.df.index, data=0)
            for date in events_df.index:
                if date in self.df.index:
                    idx = self.df.index.get_loc(date)
                    start_idx = max(0, idx - window + 1)
                    event_count.iloc[start_idx:idx + 1] += 1
            
            self.df[f'events_last_{window}d'] = event_count
            print(f"  Added {window}-day event count")
        
        return self.df
    
    def create_lagged_features(self, n_lags: int = 5) -> pd.DataFrame:
        """
        Create lagged features for time series modeling
        
        Parameters:
        -----------
        n_lags : int
            Number of lagged features to create
            
        Returns:
        --------
        pd.DataFrame : Data with lagged features
        """
        print(f"\nCreating {n_lags} lagged features...")
        
        if 'log_returns' not in self.df.columns:
            raise ValueError("Log returns not calculated. Run calculate_log_returns() first.")
        
        returns = self.df['log_returns']
        
        for lag in range(1, n_lags + 1):
            self.df[f'returns_lag_{lag}'] = returns.shift(lag)
            print(f"  Added lag {lag}")
        
        # Create rolling window statistics from lagged returns
        self.df['returns_mean_5'] = returns.rolling(window=5).mean()
        self.df['returns_std_5'] = returns.rolling(window=5).std()
        self.df['returns_skew_5'] = returns.rolling(window=5).skew()
        self.df['returns_kurt_5'] = returns.rolling(window=5).apply(
            lambda x: stats.kurtosis(x, nan_policy='omit')
        )
        
        print(f"  Added rolling window statistics")
        
        return self.df
    
    def create_technical_indicators(self) -> pd.DataFrame:
        """
        Create common technical indicators for financial time series
        
        Returns:
        --------
        pd.DataFrame : Data with technical indicators
        """
        print(f"\nCreating technical indicators...")
        
        if 'Price' not in self.df.columns:
            raise ValueError("Price data not available")
        
        price = self.df['Price']
        
        # Simple Moving Averages
        self.df['SMA_20'] = price.rolling(window=20).mean()
        self.df['SMA_50'] = price.rolling(window=50).mean()
        self.df['SMA_200'] = price.rolling(window=200).mean()
        
        # Exponential Moving Averages
        self.df['EMA_12'] = price.ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = price.ewm(span=26, adjust=False).mean()
        
        # MACD
        self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_hist'] = self.df['MACD'] - self.df['MACD_signal']
        
        # RSI
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.df['BB_middle'] = price.rolling(window=20).mean()
        bb_std = price.rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (bb_std * 2)
        self.df['BB_lower'] = self.df['BB_middle'] - (bb_std * 2)
        self.df['BB_width'] = self.df['BB_upper'] - self.df['BB_lower']
        
        print(f"  Added 12 technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)")
        
        return self.df
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the fully processed DataFrame
        
        Returns:
        --------
        pd.DataFrame : Processed data ready for analysis
        """
        self.processed_data = self.df.copy()
        return self.processed_data
    
    def save_processed_data(self, output_path: str):
        """
        Save processed data to CSV
        
        Parameters:
        -----------
        output_path : str
            Path to save the processed data
        """
        if self.processed_data is None:
            self.processed_data = self.df.copy()
        
        self.processed_data.to_csv(output_path)
        print(f"\nProcessed data saved to: {output_path}")
        print(f"Shape: {self.processed_data.shape}")
        print(f"Columns: {list(self.processed_data.columns)}")
    
    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report of data preprocessing
        
        Returns:
        --------
        Dict : Summary statistics and metrics
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING SUMMARY REPORT")
        print("="*60)
        
        summary = {
            'original_shape': self.original_shape,
            'processed_shape': self.df.shape,
            'date_range': {
                'start': self.df.index.min().strftime('%Y-%m-%d'),
                'end': self.df.index.max().strftime('%Y-%m-%d'),
                'days': (self.df.index.max() - self.df.index.min()).days
            },
            'price_statistics': {},
            'returns_statistics': {},
            'stationarity': self.stationarity_results,
            'missing_values': {},
            'data_quality': {}
        }
        
        # Price statistics
        if 'Price' in self.df.columns:
            price = self.df['Price']
            summary['price_statistics'] = {
                'min': float(price.min()),
                'max': float(price.max()),
                'mean': float(price.mean()),
                'std': float(price.std()),
                'skewness': float(price.skew()),
                'kurtosis': float(price.kurtosis())
            }
        
        # Returns statistics
        if 'log_returns' in self.df.columns:
            returns = self.df['log_returns']
            summary['returns_statistics'] = {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'min': float(returns.min()),
                'max': float(returns.max()),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            }
        
        # Missing values
        summary['missing_values'] = self.df.isnull().sum().to_dict()
        
        # Data quality metrics
        total_obs = len(self.df)
        summary['data_quality'] = {
            'completeness_rate': 1 - (self.df.isnull().sum().sum() / (total_obs * len(self.df.columns))),
            'zero_prices': int((self.df.get('Price', pd.Series([])) <= 0).sum()) if 'Price' in self.df.columns else 0,
            'duplicate_dates': self.df.index.duplicated().sum()
        }
        
        # Print summary
        print(f"\n1. DATA OVERVIEW:")
        print(f"   • Original shape: {summary['original_shape']}")
        print(f"   • Processed shape: {summary['processed_shape']}")
        print(f"   • Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   • Total days: {summary['date_range']['days']}")
        
        print(f"\n2. PRICE STATISTICS:")
        if summary['price_statistics']:
            stats = summary['price_statistics']
            print(f"   • Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
            print(f"   • Mean: ${stats['mean']:.2f} (±${stats['std']:.2f})")
            print(f"   • Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
        
        print(f"\n3. RETURNS STATISTICS:")
        if summary['returns_statistics']:
            stats = summary['returns_statistics']
            print(f"   • Daily mean: {stats['mean']:.4f}%")
            print(f"   • Daily volatility: {stats['std']:.4f}%")
            print(f"   • Annualized Sharpe: {stats['sharpe_ratio']:.3f}")
            print(f"   • Range: {stats['min']:.2f}% to {stats['max']:.2f}%")
        
        print(f"\n4. STATIONARITY SUMMARY:")
        for series_name, result in summary['stationarity'].items():
            status = "✓ STATIONARY" if result['is_stationary'] else "✗ NON-STATIONARY"
            print(f"   • {series_name:15}: {status:20} (p = {result['p_value']:.6f})")
        
        print(f"\n5. DATA QUALITY:")
        quality = summary['data_quality']
        print(f"   • Completeness rate: {quality['completeness_rate']:.1%}")
        print(f"   • Zero/negative prices: {quality['zero_prices']}")
        print(f"   • Duplicate dates: {quality['duplicate_dates']}")
        
        print(f"\n6. FEATURES CREATED:")
        feature_counts = {
            'price_features': sum(1 for col in self.df.columns if 'Price' in col or 'price' in col),
            'returns_features': sum(1 for col in self.df.columns if 'return' in col.lower()),
            'technical_features': sum(1 for col in self.df.columns if any(x in col for x in ['SMA', 'EMA', 'MACD', 'RSI', 'BB'])),
            'event_features': sum(1 for col in self.df.columns if 'event' in col.lower() or 'impact' in col.lower()),
            'lagged_features': sum(1 for col in self.df.columns if 'lag' in col.lower())
        }
        
        for feature_type, count in feature_counts.items():
            if count > 0:
                print(f"   • {feature_type.replace('_', ' ').title()}: {count}")
        
        print("\n" + "="*60)
        
        return summary


# Utility functions for standalone use
def load_and_preprocess_data(data_path: str, 
                           events_path: Optional[str] = None,
                           save_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete data preprocessing pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to Brent oil price data
    events_path : str, optional
        Path to events data
    save_path : str, optional
        Path to save processed data
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]: Processed data and summary report
    """
    print("="*60)
    print("COMPLETE DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = BrentDataPreprocessor(data_path=data_path)
    
    # Execute preprocessing steps
    preprocessor.parse_dates(date_column='Date')
    preprocessor.clean_price_data(price_column='Price')
    preprocessor.calculate_log_returns(price_column='Price')
    
    # Test stationarity
    preprocessor.analyze_all_series_stationarity()
    
    # Add rolling statistics
    preprocessor.calculate_rolling_statistics()
    
    # Detect volatility clusters
    preprocessor.detect_volatility_clusters()
    
    # Create lagged features
    preprocessor.create_lagged_features(n_lags=5)
    
    # Create technical indicators
    preprocessor.create_technical_indicators()
    
    # Add event features if events data provided
    if events_path:
        events_df = pd.read_csv(events_path)
        preprocessor.prepare_event_features(events_df)
    
    # Generate summary report
    summary = preprocessor.generate_summary_report()
    
    # Save processed data if path provided
    if save_path:
        preprocessor.save_processed_data(save_path)
    
    # Get processed data
    processed_data = preprocessor.get_processed_data()
    
    print("\n✓ Preprocessing pipeline completed successfully!")
    print("="*60)
    
    return processed_data, summary


def create_analysis_ready_dataset(data_path: str, 
                                 target_column: str = 'log_returns',
                                 test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split for modeling
    
    Parameters:
    -----------
    data_path : str
        Path to processed data
    target_column : str
        Column to use as target variable
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # Drop rows with NaN in target column
    df = df.dropna(subset=[target_column])
    
    # Split chronologically (time series split)
    split_idx = int(len(df) * (1 - test_size))
    
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    print(f"Train data: {train_data.shape} ({train_data.index.min()} to {train_data.index.max()})")
    print(f"Test data: {test_data.shape} ({test_data.index.min()} to {test_data.index.max()})")
    print(f"Test size: {test_size:.1%}")
    
    return train_data, test_data


if __name__ == '__main__':
    # Example usage
    data_path = "../data/raw/BrentOilPrices.csv"
    events_path = "../data/events/geopolitical_events.csv"
    output_path = "../data/processed/brent_processed.csv"
    
    try:
        processed_data, summary = load_and_preprocess_data(
            data_path=data_path,
            events_path=events_path,
            save_path=output_path
        )
        print(f"\nProcessed data shape: {processed_data.shape}")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise
