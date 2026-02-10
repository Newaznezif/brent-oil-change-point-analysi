"""
Data models and processing helpers for Brent Oil Analysis Dashboard
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """Manages all data operations for the dashboard"""
    
    def __init__(self, data_dir='../../data'):
        self.data_dir = os.path.abspath(data_dir)
        # Initialize attributes to prevent AttributeErrors
        self.processed_data = pd.DataFrame()
        self.change_points = pd.DataFrame()
        self.events = pd.DataFrame()
        self.analysis_results = {}
        self._load_data()
    
    def _load_data(self):
        """Load all data files"""
        try:
            # Load processed data
            processed_path = os.path.join(self.data_dir, 'processed', 'brent_processed_task2.csv')
            if os.path.exists(processed_path):
                self.processed_data = pd.read_csv(processed_path, parse_dates=['Date'])
                self.processed_data.set_index('Date', inplace=True)
                print(f"✓ Loaded processed data: {len(self.processed_data)} rows")
            else:
                print(f"✗ Processed data not found: {processed_path}")
                self.processed_data = pd.DataFrame()
            
            # Load change point results
            cp_path = os.path.join(self.data_dir, 'processed', 'change_points_results.csv')
            if os.path.exists(cp_path):
                self.change_points = pd.read_csv(cp_path, parse_dates=['change_date'])
                print(f"✓ Loaded change points: {len(self.change_points)} points")
            else:
                print(f"✗ Change points not found: {cp_path}")
                self.change_points = pd.DataFrame()
            
            # Load events
            events_path = os.path.join(self.data_dir, 'events', 'geopolitical_events.csv')
            if os.path.exists(events_path):
                self.events = pd.read_csv(events_path, parse_dates=['date'])
                print(f"✓ Loaded events: {len(self.events)} events")
            else:
                print(f"✗ Events not found: {events_path}")
                self.events = pd.DataFrame()
            
            # Load analysis results
            analysis_path = os.path.join(self.data_dir, 'processed', 'comprehensive_analysis_report.json')
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    self.analysis_results = json.load(f)
                print("✓ Loaded analysis results")
            else:
                print(f"✗ Analysis results not found: {analysis_path}")
                self.analysis_results = {}
            
            # If critical data is missing, create sample data
            if self.processed_data.empty or self.change_points.empty:
                print("Critical data missing, triggering sample data creation...")
                self._create_sample_data()
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data if real data is not available"""
        print("Creating sample data for demonstration...")
        
        # Create sample prices
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        # Generate realistic price series with trends
        np.random.seed(42)
        base_price = 70
        trend = np.cumsum(np.random.randn(n)) * 0.5
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
        noise = np.random.randn(n) * 2
        
        prices = base_price + trend + seasonal + noise
        prices = np.maximum(prices, 10)  # Ensure positive prices
        
        self.processed_data = pd.DataFrame({
            'Price': prices,
        }, index=dates)
        self.processed_data['Log_Returns'] = np.log(self.processed_data['Price']).diff().fillna(0) * 100
        
        # Create sample change points
        change_dates = ['2014-11-01', '2016-01-15', '2020-04-01', '2022-03-01']
        self.change_points = pd.DataFrame({
            'change_point_id': range(1, 5),
            'change_date': pd.to_datetime(change_dates),
            'mean_before': [0.08, -0.05, 0.12, -0.15],
            'mean_after': [-0.15, 0.10, -0.25, 0.08],
            'mean_difference': [-0.23, 0.15, -0.37, 0.23],
            'std_before': [1.8, 2.1, 2.5, 3.0],
            'std_after': [3.2, 1.5, 4.8, 2.2],
            'probability_mu2_gt_mu1': [0.12, 0.85, 0.05, 0.92],
            'effect_size': [-0.8, 0.6, -1.2, 0.9],
            'impact_magnitude': ['Medium', 'Medium', 'High', 'High']
        })
        
        # Create sample events
        sample_events = [
            ['OPEC Production Cut', '2014-11-27', 'OPEC Decision', 'High'],
            ['Brexit Vote', '2016-06-23', 'Political Event', 'High'],
            ['COVID-19 Pandemic', '2020-03-11', 'Pandemic', 'High'],
            ['Russia-Ukraine War', '2022-02-24', 'Geopolitical Conflict', 'High'],
            ['Federal Reserve Rate Hike', '2022-03-16', 'Monetary Policy', 'Medium']
        ]
        
        self.events = pd.DataFrame(sample_events, 
                                 columns=['event_name', 'date', 'event_type', 'impact_level'])
        self.events['date'] = pd.to_datetime(self.events['date'])
        
        print("✓ Sample data created successfully")
    
    # Data access methods
    
    def get_prices(self, start_date=None, end_date=None):
        """Get price data with optional filtering"""
        df = self.processed_data.copy()
        
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Prepare for JSON serialization
        result = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': df['Price'].tolist() if 'Price' in df.columns else [],
            'count': len(df)
        }
        
        if not df.empty:
            result.update({
                'min_price': float(df['Price'].min()),
                'max_price': float(df['Price'].max()),
                'avg_price': float(df['Price'].mean())
            })
        
        return result
    
    def get_returns(self):
        """Get returns data"""
        if 'Log_Returns' in self.processed_data.columns:
            returns = self.processed_data['Log_Returns'].dropna()
            return {
                'dates': returns.index.strftime('%Y-%m-%d').tolist(),
                'returns': returns.tolist(),
                'count': len(returns),
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std())
            }
        return {'dates': [], 'returns': [], 'count': 0}
    
    def get_change_points(self):
        """Get change points data"""
        if not self.change_points.empty:
            result = self.change_points.to_dict('records')
            # Convert dates to string for JSON serialization
            for item in result:
                if 'change_date' in item and hasattr(item['change_date'], 'strftime'):
                    item['change_date'] = item['change_date'].strftime('%Y-%m-%d')
            return result
        return []
    
    def get_events(self):
        """Get events data"""
        if not self.events.empty:
            result = self.events.to_dict('records')
            # Convert dates to string
            for item in result:
                if 'date' in item and hasattr(item['date'], 'strftime'):
                    item['date'] = item['date'].strftime('%Y-%m-%d')
            return result
        return []
    
    def get_events_near_change_points(self, max_days=30):
        """Get events near change points"""
        if self.change_points.empty or self.events.empty:
            return []
        
        associations = []
        for _, cp in self.change_points.iterrows():
            cp_date = cp['change_date']
            
            # Find nearby events
            nearby_events = []
            for _, event in self.events.iterrows():
                days_diff = abs((cp_date - event['date']).days)
                if days_diff <= max_days:
                    nearby_events.append({
                        'event_name': event['event_name'],
                        'event_date': event['date'].strftime('%Y-%m-%d'),
                        'days_difference': days_diff,
                        'event_type': event['event_type'],
                        'impact_level': event['impact_level']
                    })
            
            if nearby_events:
                associations.append({
                    'change_point_id': int(cp['change_point_id']),
                    'change_date': cp_date.strftime('%Y-%m-%d'),
                    'mean_difference': float(cp['mean_difference']),
                    'nearby_events': sorted(nearby_events, key=lambda x: x['days_difference'])[:3]
                })
        
        return associations
    
    def get_regimes(self):
        """Get market regimes defined by change points"""
        if self.change_points.empty:
            return []
        
        regimes = []
        dates = self.processed_data.index
        
        # Sort change points by date
        cps = self.change_points.sort_values('change_date')
        cps_dates = cps['change_date'].tolist()
        
        # Add start and end boundaries
        boundaries = [dates.min()] + cps_dates + [dates.max()]
        
        for i in range(len(boundaries) - 1):
            start_date = boundaries[i]
            end_date = boundaries[i + 1]
            
            # Get data for this regime
            mask = (dates >= start_date) & (dates <= end_date)
            regime_data = self.processed_data.loc[mask]
            
            if len(regime_data) > 0 and 'Log_Returns' in regime_data.columns:
                returns = regime_data['Log_Returns'].dropna()
                
                regimes.append({
                    'regime_id': i + 1,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': (end_date - start_date).days,
                    'mean_return': float(returns.mean()),
                    'volatility': float(returns.std()),
                    'observations': len(returns),
                    'is_change_point': i > 0  # All but first regime start with change point
                })
        
        return regimes
    
    def get_statistics(self):
        """Get overall statistics"""
        stats = {
            'total_days': len(self.processed_data),
            'date_range': {
                'start': self.processed_data.index.min().strftime('%Y-%m-%d'),
                'end': self.processed_data.index.max().strftime('%Y-%m-%d')
            }
        }
        
        if 'Price' in self.processed_data.columns:
            prices = self.processed_data['Price']
            stats['price_stats'] = {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'mean': float(prices.mean()),
                'median': float(prices.median())
            }
        
        if 'Log_Returns' in self.processed_data.columns:
            returns = self.processed_data['Log_Returns'].dropna()
            stats['return_stats'] = {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'min': float(returns.min()),
                'max': float(returns.max()),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis())
            }
        
        stats['change_points_count'] = len(self.change_points)
        stats['events_count'] = len(self.events)
        
        return stats
    
    def get_dashboard_summary(self):
        """Get summary for dashboard"""
        return {
            'title': 'Brent Oil Analysis Dashboard',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_status': {
                'prices_loaded': not self.processed_data.empty,
                'change_points_loaded': not self.change_points.empty,
                'events_loaded': not self.events.empty
            },
            'key_metrics': self.get_statistics(),
            'recent_change_points': self.get_change_points()[-3:] if self.get_change_points() else []
        }
    
    def get_single_change_point_analysis(self):
        """Get single change point analysis"""
        if self.change_points.empty:
            return {}
        
        # Get the most significant change point
        cp = self.change_points.iloc[self.change_points['effect_size'].abs().idxmax()]
        
        return {
            'change_point': {
                'date': cp['change_date'].strftime('%Y-%m-%d'),
                'mean_before': float(cp['mean_before']),
                'mean_after': float(cp['mean_after']),
                'difference': float(cp['mean_difference']),
                'effect_size': float(cp['effect_size']),
                'probability': float(cp['probability_mu2_gt_mu1'])
            },
            'interpretation': self._get_interpretation(cp)
        }
    
    def get_multiple_change_points_analysis(self):
        """Get multiple change points analysis"""
        if self.change_points.empty:
            return {}
        
        analysis = {
            'total_change_points': len(self.change_points),
            'change_points_by_impact': self.change_points.sort_values('effect_size', key=abs, ascending=False).to_dict('records'),
            'summary': {
                'average_effect_size': float(self.change_points['effect_size'].abs().mean()),
                'bullish_changes': int((self.change_points['mean_difference'] > 0).sum()),
                'bearish_changes': int((self.change_points['mean_difference'] < 0).sum()),
                'high_impact_changes': int((self.change_points['effect_size'].abs() > 0.8).sum())
            }
        }
        
        return analysis
    
    def get_impact_analysis(self):
        """Get impact analysis"""
        if self.change_points.empty:
            return {}
        
        impact = {
            'economic_impact': {
                'total_annualized_impact': float(self.change_points['mean_difference'].sum() * 252),
                'largest_positive_impact': float(self.change_points['mean_difference'].max()),
                'largest_negative_impact': float(self.change_points['mean_difference'].min())
            },
            'volatility_impact': {
                'average_volatility_change': float(((self.change_points['std_after'] - self.change_points['std_before']) / self.change_points['std_before'] * 100).mean()),
                'regimes_with_increased_volatility': int((self.change_points['std_after'] > self.change_points['std_before']).sum())
            }
        }
        
        return impact
    
    def get_recommendations(self):
        """Get recommendations based on analysis"""
        recommendations = {
            'investors': [
                'Monitor change points for regime shifts',
                'Adjust portfolio allocation based on detected regimes',
                'Use change points as entry/exit signals for trading strategies'
            ],
            'risk_managers': [
                'Update risk models when change points are detected',
                'Adjust position sizing based on regime volatility',
                'Implement dynamic hedging strategies around change points'
            ],
            'analysts': [
                'Correlate change points with fundamental events',
                'Use Bayesian methods for ongoing monitoring',
                'Combine with other indicators for confirmation'
            ]
        }
        
        # Add specific recommendations based on data
        if not self.change_points.empty:
            latest_cp = self.change_points.iloc[-1]
            if latest_cp['mean_difference'] > 0:
                recommendations['investors'].append('Consider increasing exposure - recent bullish regime detected')
            else:
                recommendations['investors'].append('Consider defensive positioning - recent bearish regime detected')
        
        return recommendations
    
    def _get_interpretation(self, change_point):
        """Get interpretation of a change point"""
        effect_size = abs(change_point['effect_size'])
        prob = change_point['probability_mu2_gt_mu1']
        
        interpretation = {
            'magnitude': 'Large' if effect_size > 0.8 else ('Medium' if effect_size > 0.5 else 'Small'),
            'confidence': 'High' if prob > 0.95 else ('Medium' if prob > 0.8 else 'Low'),
            'direction': 'Bullish' if change_point['mean_difference'] > 0 else 'Bearish',
            'volatility_change': 'Increased' if change_point['std_after'] > change_point['std_before'] else 'Decreased'
        }
        
        return interpretation
