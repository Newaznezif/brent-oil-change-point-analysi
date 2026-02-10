"""
Bayesian Change Point Models for Brent Oil Price Analysis

This module implements Bayesian change point detection models using PyMC.
Includes single and multiple change point models with MCMC sampling.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union, Any
from dataclasses import dataclass
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


@dataclass
class ChangePointResult:
    """Container for change point analysis results"""
    model_name: str
    trace: az.InferenceData
    summary: pd.DataFrame
    change_points: List[int]
    change_dates: List[pd.Timestamp]
    parameters: Dict[str, np.ndarray]
    convergence: Dict[str, Any]
    model_fit: Dict[str, Any]
    

class BayesianChangePointModel:
    """
    Base class for Bayesian change point models using PyMC
    """
    
    def __init__(self, data: np.ndarray, dates: pd.DatetimeIndex, 
                 model_name: str = "Bayesian_Change_Point"):
        """
        Initialize change point model
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data (should be stationary, e.g., log returns)
        dates : pd.DatetimeIndex
            Corresponding dates for the data
        model_name : str
            Name for the model
        """
        self.data = data
        self.dates = dates
        self.n_obs = len(data)
        self.model_name = model_name
        self.model = None
        self.trace = None
        self.result = None
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data"""
        if len(self.data) != len(self.dates):
            raise ValueError("Data and dates must have same length")
        
        if self.n_obs < 100:
            warnings.warn(f"Small dataset ({self.n_obs} observations). Results may be unstable.")
        
        if np.any(np.isnan(self.data)):
            raise ValueError("Data contains NaN values. Clean data before modeling.")
    
    def build_model(self) -> pm.Model:
        """
        Build the PyMC model. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def sample(self, draws: int = 2000, tune: int = 1000, 
               chains: int = 2, cores: int = 1, 
               random_seed: int = 42, progressbar: bool = True) -> az.InferenceData:
        """
        Run MCMC sampling
        
        Parameters:
        -----------
        draws : int
            Number of posterior samples per chain
        tune : int
            Number of tuning samples per chain
        chains : int
            Number of MCMC chains
        cores : int
            Number of CPU cores to use
        random_seed : int
            Random seed for reproducibility
        progressbar : bool
            Whether to show progress bar
            
        Returns:
        --------
        az.InferenceData : MCMC trace
        """
        if self.model is None:
            self.model = self.build_model()
        
        print(f"Sampling {self.model_name} model...")
        print(f"  Draws: {draws}, Tune: {tune}, Chains: {chains}")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
                target_accept=0.9  # Higher acceptance rate for better mixing
            )
        
        print(f"✓ Sampling completed")
        return self.trace
    
    def diagnose_convergence(self, trace: az.InferenceData) -> Dict:
        """
        Diagnose MCMC convergence
        
        Parameters:
        -----------
        trace : az.InferenceData
            MCMC trace
            
        Returns:
        --------
        Dict : Convergence diagnostics
        """
        print(f"\nConvergence diagnostics for {self.model_name}:")
        
        diagnostics = {}
        
        # 1. R-hat statistics (should be < 1.01)
        summary = az.summary(trace)
        rhat_stats = summary['r_hat']
        
        diagnostics['rhat'] = {
            'values': rhat_stats.to_dict(),
            'all_converged': (rhat_stats < 1.01).all(),
            'max': float(rhat_stats.max()),
            'min': float(rhat_stats.min()),
            'mean': float(rhat_stats.mean())
        }
        
        print(f"  R-hat statistics:")
        print(f"    • All < 1.01: {diagnostics['rhat']['all_converged']}")
        print(f"    • Range: [{diagnostics['rhat']['min']:.3f}, {diagnostics['rhat']['max']:.3f}]")
        
        # 2. Effective sample size (should be > 400)
        ess_stats = summary['ess_bulk']
        
        diagnostics['ess'] = {
            'values': ess_stats.to_dict(),
            'all_sufficient': (ess_stats > 400).all(),
            'min': float(ess_stats.min()),
            'mean': float(ess_stats.mean())
        }
        
        print(f"  Effective Sample Size (ESS):")
        print(f"    • All > 400: {diagnostics['ess']['all_sufficient']}")
        print(f"    • Min ESS: {diagnostics['ess']['min']:.0f}")
        
        # 3. Trace plots data (for visualization)
        diagnostics['trace_data'] = {
            var: trace.posterior[var].values for var in trace.posterior.data_vars
        }
        
        # 4. Divergences check
        try:
            n_divergences = trace.sample_stats.divergences.sum().item()
            diagnostics['divergences'] = {
                'count': n_divergences,
                'problematic': n_divergences > 0
            }
            print(f"  Divergences: {n_divergences} (problematic: {n_divergences > 0})")
        except:
            diagnostics['divergences'] = {'count': 0, 'problematic': False}
        
        return diagnostics
    
    def extract_results(self, trace: az.InferenceData) -> ChangePointResult:
        """
        Extract meaningful results from trace
        
        Parameters:
        -----------
        trace : az.InferenceData
            MCMC trace
            
        Returns:
        --------
        ChangePointResult : Analysis results
        """
        # Get summary statistics
        summary = az.summary(trace)
        
        # Extract change points (implementation depends on model)
        change_points, change_dates = self._extract_change_points(trace)
        
        # Extract parameter estimates
        parameters = self._extract_parameters(trace)
        
        # Check convergence
        convergence = self.diagnose_convergence(trace)
        
        # Assess model fit
        model_fit = self._assess_model_fit(trace)
        
        # Create result object
        self.result = ChangePointResult(
            model_name=self.model_name,
            trace=trace,
            summary=summary,
            change_points=change_points,
            change_dates=change_dates,
            parameters=parameters,
            convergence=convergence,
            model_fit=model_fit
        )
        
        return self.result
    
    def _extract_change_points(self, trace: az.InferenceData) -> Tuple[List[int], List[pd.Timestamp]]:
        """
        Extract change point locations. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _extract_change_points()")
    
    def _extract_parameters(self, trace: az.InferenceData) -> Dict:
        """
        Extract model parameters. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _extract_parameters()")
    
    def _assess_model_fit(self, trace: az.InferenceData) -> Dict:
        """
        Assess model fit using posterior predictive checks
        """
        print(f"\nAssessing model fit for {self.model_name}...")
        
        fit_metrics = {}
        
        # 1. Posterior predictive checks
        with self.model:
            ppc = pm.sample_posterior_predictive(trace, random_seed=42)
        
        # Compare observed vs predicted
        observed = self.data
        predicted = ppc.posterior_predictive['y'].values.flatten()
        
        # Calculate fit metrics
        fit_metrics['ppc_mean_diff'] = float(np.mean(predicted) - np.mean(observed))
        fit_metrics['ppc_std_diff'] = float(np.std(predicted) - np.std(observed))
        
        # 2. Predictive accuracy (WAIC, LOO)
        try:
            comparison = az.compare({'model': trace})
            fit_metrics['waic'] = float(comparison.loc['model', 'waic'])
            fit_metrics['loo'] = float(comparison.loc['model', 'loo'])
            fit_metrics['p_waic'] = float(comparison.loc['model', 'p_waic'])
        except:
            fit_metrics['waic'] = None
            fit_metrics['loo'] = None
            fit_metrics['p_waic'] = None
        
        # 3. R-squared like metric
        ss_res = np.sum((observed - predicted.mean())**2)
        ss_tot = np.sum((observed - observed.mean())**2)
        fit_metrics['r_squared'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
        
        print(f"  R²: {fit_metrics['r_squared']:.3f}")
        if fit_metrics['waic']:
            print(f"  WAIC: {fit_metrics['waic']:.1f}")
        
        return fit_metrics
    
    def plot_posteriors(self, var_names: List[str] = None, 
                       hdi_prob: float = 0.95) -> plt.Figure:
        """
        Plot posterior distributions
        
        Parameters:
        -----------
        var_names : List[str], optional
            Variables to plot. If None, plots all.
        hdi_prob : float
            Highest density interval probability
            
        Returns:
        --------
        matplotlib.figure.Figure : Plot figure
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample() first.")
        
        if var_names is None:
            var_names = list(self.trace.posterior.data_vars)
        
        n_vars = len(var_names)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'Posterior Distributions - {self.model_name}', fontsize=14, fontweight='bold')
        
        if n_vars == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, var_name in enumerate(var_names):
            if var_name in self.trace.posterior:
                az.plot_posterior(self.trace, var_names=[var_name], 
                                hdi_prob=hdi_prob, ax=axes[i])
                axes[i].set_title(f'{var_name}', fontweight='bold')
            else:
                axes[i].text(0.5, 0.5, f'Variable {var_name} not found',
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Missing: {var_name}', fontweight='bold')
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_trace(self, var_names: List[str] = None) -> plt.Figure:
        """
        Plot trace plots for convergence assessment
        
        Parameters:
        -----------
        var_names : List[str], optional
            Variables to plot. If None, plots all.
            
        Returns:
        --------
        matplotlib.figure.Figure : Plot figure
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample() first.")
        
        if var_names is None:
            var_names = list(self.trace.posterior.data_vars)[:4]  # Limit to 4 for readability
        
        n_vars = len(var_names)
        fig, axes = plt.subplots(n_vars, 2, figsize=(12, 3*n_vars))
        fig.suptitle(f'Trace Plots - {self.model_name}', fontsize=14, fontweight='bold')
        
        if n_vars == 1:
            axes = axes.reshape(1, 2)
        
        for i, var_name in enumerate(var_names):
            az.plot_trace(self.trace, var_names=[var_name], axes=axes[i])
            axes[i, 0].set_ylabel(var_name, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_dir: str = '../data/processed'):
        """
        Save model results to files
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.result is None:
            raise ValueError("No results available. Run extract_results() first.")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, f'{self.model_name}_summary.csv')
        self.result.summary.to_csv(summary_path)
        
        # Save change points
        cp_data = {
            'model_name': self.model_name,
            'change_points': self.result.change_points,
            'change_dates': [str(d) for d in self.result.change_dates],
            'parameters': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in self.result.parameters.items()},
            'convergence': self.result.convergence,
            'model_fit': self.result.model_fit
        }
        
        cp_path = os.path.join(output_dir, f'{self.model_name}_results.json')
        with open(cp_path, 'w') as f:
            json.dump(cp_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to {output_dir}/")
        print(f"  • Summary: {self.model_name}_summary.csv")
        print(f"  • Results: {self.model_name}_results.json")


class SingleChangePointModel(BayesianChangePointModel):
    """
    Bayesian model with single change point (mean shift)
    
    Model: y_t ~ N(μ₁, σ) for t < τ, N(μ₂, σ) for t ≥ τ
    where τ is the change point location
    """
    
    def __init__(self, data: np.ndarray, dates: pd.DatetimeIndex):
        super().__init__(data, dates, model_name="Single_Change_Point")
    
    def build_model(self) -> pm.Model:
        """
        Build single change point model
        
        Model structure:
        τ ~ DiscreteUniform(1, n_obs-1)  # Change point location
        μ₁ ~ Normal(0, 1)                # Mean before change
        μ₂ ~ Normal(0, 1)                # Mean after change
        σ ~ HalfNormal(1)                # Standard deviation (constant)
        y ~ Normal(switch(τ > t, μ₁, μ₂), σ)  # Likelihood
        """
        print(f"Building single change point model...")
        
        with pm.Model() as model:
            # ===== PRIORS =====
            
            # Prior for change point location (discrete uniform over all possible points)
            τ = pm.DiscreteUniform("τ", lower=1, upper=self.n_obs-1)
            
            # Priors for means before and after change point
            μ1 = pm.Normal("μ1", mu=0, sigma=1)
            μ2 = pm.Normal("μ2", mu=0, sigma=1)
            
            # Prior for standard deviation (assumed constant for simplicity)
            # Using HalfNormal with scale based on data std
            data_std = np.std(self.data)
            σ = pm.HalfNormal("σ", sigma=data_std/2)
            
            # ===== LIKELIHOOD =====
            
            # Create time index array
            t = np.arange(self.n_obs)
            
            # Switch between means based on change point
            # μ = μ1 if t < τ, μ2 if t ≥ τ
            μ = pm.math.switch(τ > t, μ1, μ2)
            
            # Likelihood: Normal distribution with switching mean
            y = pm.Normal("y", mu=μ, sigma=σ, observed=self.data)
            
            # Store additional information
            pm.Deterministic("μ_diff", μ2 - μ1)
            pm.Deterministic("μ_ratio", μ2 / μ1)
            
            print(f"  Model built with {self.n_obs} observations")
            print(f"  Parameters: τ (change point), μ1, μ2, σ")
            
        self.model = model
        return model
    
    def _extract_change_points(self, trace: az.InferenceData) -> Tuple[List[int], List[pd.Timestamp]]:
        """
        Extract the single change point location
        """
        # Get posterior samples of τ
        τ_samples = trace.posterior['τ'].values.flatten()
        
        # Calculate mean and HDI
        τ_mean = int(np.mean(τ_samples))
        τ_hdi = az.hdi(τ_samples, hdi_prob=0.95)
        
        # Convert to dates
        τ_date = self.dates[τ_mean]
        τ_hdi_dates = [self.dates[int(τ_hdi[0])], self.dates[int(τ_hdi[1])]]
        
        # Return as lists (single change point)
        change_points = [τ_mean]
        change_dates = [τ_date]
        
        # Store additional information
        self.change_point_stats = {
            'mean_index': τ_mean,
            'mean_date': τ_date,
            'hdi_95': τ_hdi.tolist(),
            'hdi_dates': τ_hdi_dates,
            'posterior_samples': τ_samples
        }
        
        return change_points, change_dates
    
    def _extract_parameters(self, trace: az.InferenceData) -> Dict:
        """
        Extract parameter estimates
        """
        parameters = {}
        
        # Extract posterior samples
        for var in ['μ1', 'μ2', 'σ', 'μ_diff', 'μ_ratio']:
            if var in trace.posterior:
                samples = trace.posterior[var].values.flatten()
                parameters[var] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'hdi_95': az.hdi(samples, hdi_prob=0.95).tolist(),
                    'samples': samples
                }
        
        # Calculate probability that μ2 > μ1
        if 'μ_diff' in parameters:
            μ_diff_samples = parameters['μ_diff']['samples']
            prob_μ2_gt_μ1 = np.mean(μ_diff_samples > 0)
            parameters['prob_μ2_gt_μ1'] = float(prob_μ2_gt_μ1)
        
        return parameters
    
    def get_detailed_results(self) -> Dict:
        """
        Get detailed results including impact quantification
        """
        if self.result is None:
            raise ValueError("No results available. Run extract_results() first.")
        
        detailed = {
            'model': self.model_name,
            'change_point': {
                'index': self.change_point_stats['mean_index'],
                'date': str(self.change_point_stats['mean_date']),
                'hdi_95_indices': self.change_point_stats['hdi_95'],
                'hdi_95_dates': [str(d) for d in self.change_point_stats['hdi_dates']],
                'certainty': np.std(self.change_point_stats['posterior_samples']) / self.n_obs
            },
            'parameters': self.result.parameters,
            'impact': self._quantify_impact(),
            'convergence': self.result.convergence['rhat']['all_converged']
        }
        
        return detailed
    
    def _quantify_impact(self) -> Dict:
        """
        Quantify the impact of the change point
        """
        impact = {}
        
        # Get parameter estimates
        μ1_mean = self.result.parameters['μ1']['mean']
        μ2_mean = self.result.parameters['μ2']['mean']
        σ_mean = self.result.parameters['σ']['mean']
        
        # Calculate impact metrics
        mean_change = μ2_mean - μ1_mean
        mean_change_pct = (mean_change / abs(μ1_mean)) * 100 if μ1_mean != 0 else 0
        
        # Effect size (Cohen's d)
        effect_size = mean_change / σ_mean if σ_mean > 0 else 0
        
        impact['mean_before'] = float(μ1_mean)
        impact['mean_after'] = float(μ2_mean)
        impact['mean_change'] = float(mean_change)
        impact['mean_change_pct'] = float(mean_change_pct)
        impact['effect_size'] = float(effect_size)
        impact['volatility'] = float(σ_mean)
        impact['probability_increase'] = self.result.parameters.get('prob_μ2_gt_μ1', 0.5)
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            impact['magnitude'] = 'Small'
        elif abs(effect_size) < 0.5:
            impact['magnitude'] = 'Medium'
        else:
            impact['magnitude'] = 'Large'
        
        return impact


class MeanVarianceChangePointModel(BayesianChangePointModel):
    """
    Bayesian model with single change point in both mean and variance
    
    Model: y_t ~ N(μ₁, σ₁) for t < τ, N(μ₂, σ₂) for t ≥ τ
    """
    
    def __init__(self, data: np.ndarray, dates: pd.DatetimeIndex):
        super().__init__(data, dates, model_name="Mean_Variance_Change_Point")
    
    def build_model(self) -> pm.Model:
        """
        Build change point model with mean and variance shifts
        """
        print(f"Building mean-variance change point model...")
        
        with pm.Model() as model:
            # ===== PRIORS =====
            
            # Prior for change point location
            τ = pm.DiscreteUniform("τ", lower=1, upper=self.n_obs-1)
            
            # Priors for means
            μ1 = pm.Normal("μ1", mu=0, sigma=1)
            μ2 = pm.Normal("μ2", mu=0, sigma=1)
            
            # Priors for standard deviations (HalfNormal for positive values)
            data_std = np.std(self.data)
            σ1 = pm.HalfNormal("σ1", sigma=data_std)
            σ2 = pm.HalfNormal("σ2", sigma=data_std)
            
            # ===== LIKELIHOOD =====
            
            # Create time index array
            t = np.arange(self.n_obs)
            
            # Switch parameters based on change point
            μ = pm.math.switch(τ > t, μ1, μ2)
            σ = pm.math.switch(τ > t, σ1, σ2)
            
            # Likelihood
            y = pm.Normal("y", mu=μ, sigma=σ, observed=self.data)
            
            # Store additional information
            pm.Deterministic("μ_diff", μ2 - μ1)
            pm.Deterministic("σ_ratio", σ2 / σ1)
            pm.Deterministic("σ_diff", σ2 - σ1)
            
            print(f"  Model built with {self.n_obs} observations")
            print(f"  Parameters: τ, μ1, μ2, σ1, σ2")
            
        self.model = model
        return model
    
    def _extract_change_points(self, trace: az.InferenceData) -> Tuple[List[int], List[pd.Timestamp]]:
        """
        Extract the change point location
        """
        τ_samples = trace.posterior['τ'].values.flatten()
        τ_mean = int(np.mean(τ_samples))
        τ_date = self.dates[τ_mean]
        
        return [τ_mean], [τ_date]
    
    def _extract_parameters(self, trace: az.InferenceData) -> Dict:
        """
        Extract parameter estimates
        """
        parameters = {}
        
        for var in ['μ1', 'μ2', 'σ1', 'σ2', 'μ_diff', 'σ_ratio', 'σ_diff']:
            if var in trace.posterior:
                samples = trace.posterior[var].values.flatten()
                parameters[var] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'hdi_95': az.hdi(samples, hdi_prob=0.95).tolist(),
                    'samples': samples
                }
        
        return parameters


class MultipleChangePointModel(BayesianChangePointModel):
    """
    Bayesian model with multiple change points using a Poisson process prior
    """
    
    def __init__(self, data: np.ndarray, dates: pd.DatetimeIndex, 
                 max_changepoints: int = 5):
        """
        Initialize multiple change point model
        
        Parameters:
        -----------
        max_changepoints : int
            Maximum number of change points to consider
        """
        super().__init__(data, dates, model_name="Multiple_Change_Points")
        self.max_changepoints = max_changepoints
    
    def build_model(self) -> pm.Model:
        """
        Build multiple change point model using a hierarchical approach
        
        Note: This model is computationally intensive and may require
        careful tuning for convergence.
        """
        print(f"Building multiple change point model (max {self.max_changepoints} changes)...")
        print("Note: This model is computationally intensive.")
        
        with pm.Model() as model:
            # ===== PRIORS FOR NUMBER OF CHANGE POINTS =====
            
            # Prior for number of change points (Poisson distribution)
            # λ = expected number of change points
            λ = pm.Gamma("λ", alpha=2, beta=1)  # Prior mean ~2 change points
            n_changepoints = pm.Poisson("n_changepoints", mu=λ)
            
            # Constrain to reasonable range
            n_changepoints = pt.clip(n_changepoints, 0, self.max_changepoints)
            
            # ===== PRIORS FOR CHANGE POINT LOCATIONS =====
            
            # Use a Dirichlet Process prior for change point locations
            α = pm.Gamma("α", alpha=1, beta=1)  # Concentration parameter
            
            # Stick-breaking construction
            v = pm.Beta("v", alpha=1, beta=α, shape=self.max_changepoints)
            
            # Calculate weights
            stick_breaking_weights = v * pt.concatenate([
                pt.ones(1),
                pt.extra_ops.cumprod(1 - v)[:-1]
            ])
            
            # Ensure weights sum to 1
            weights = stick_breaking_weights / pt.sum(stick_breaking_weights)
            
            # Categorical distribution for change point locations
            tau = pm.Categorical("tau", p=weights, shape=self.n_obs-1)
            
            # ===== PRIORS FOR REGIME PARAMETERS =====
            
            # Means for each regime (including regimes between change points)
            μ = pm.Normal("μ", mu=0, sigma=1, shape=self.max_changepoints+1)
            
            # Volatilities for each regime
            σ = pm.HalfNormal("σ", sigma=1, shape=self.max_changepoints+1)
            
            # ===== LIKELIHOOD =====
            
            # Create regime assignments
            regime = pt.zeros(self.n_obs, dtype='int32')
            for i in range(self.n_obs-1):
                regime = pt.set_subtensor(regime[i+1:], regime[i+1:] + (tau[i] == 1))
            
            # Select parameters based on regime
            μ_selected = μ[regime]
            σ_selected = σ[regime]
            
            # Likelihood
            y = pm.Normal("y", mu=μ_selected, sigma=σ_selected, observed=self.data)
            
            print(f"  Model built with hierarchical structure")
            print(f"  Parameters: n_changepoints, τ locations, regime means and volatilities")
            print(f"  Warning: May require extensive tuning for convergence")
            
        self.model = model
        return model
    
    def _extract_change_points(self, trace: az.InferenceData) -> Tuple[List[int], List[pd.Timestamp]]:
        """
        Extract multiple change point locations
        """
        # This is complex for the hierarchical model
        # For simplicity, we'll use a threshold on the tau probabilities
        
        if 'tau' not in trace.posterior:
            return [], []
        
        # Get posterior samples of tau
        tau_samples = trace.posterior['tau'].values
        
        # Calculate probability of change at each time point
        change_probs = np.mean(tau_samples, axis=(0, 1))
        
        # Threshold for change point detection
        threshold = 0.3
        
        # Find change points above threshold
        change_points = np.where(change_probs > threshold)[0].tolist()
        
        # Filter: require minimum distance between change points
        min_distance = self.n_obs // 20  # At least 5% of data between changes
        filtered_points = []
        
        for cp in change_points:
            if not filtered_points or (cp - filtered_points[-1]) >= min_distance:
                filtered_points.append(cp)
        
        # Convert to dates
        change_dates = [self.dates[cp] for cp in filtered_points if cp < len(self.dates)]
        
        return filtered_points, change_dates
    
    def _extract_parameters(self, trace: az.InferenceData) -> Dict:
        """
        Extract parameter estimates
        """
        parameters = {}
        
        # Extract number of change points
        if 'n_changepoints' in trace.posterior:
            n_cp_samples = trace.posterior['n_changepoints'].values.flatten()
            parameters['n_changepoints'] = {
                'mean': float(np.mean(n_cp_samples)),
                'std': float(np.std(n_cp_samples)),
                'hdi_95': az.hdi(n_cp_samples, hdi_prob=0.95).tolist(),
                'distribution': np.bincount(n_cp_samples.astype(int))
            }
        
        return parameters


class BayesianOnlineChangePointDetector:
    """
    Bayesian online change point detection (Adams & MacKay, 2007)
    
    This implements a simplified version for demonstration.
    For production use, consider specialized implementations.
    """
    
    def __init__(self, hazard_rate: float = 1/250):
        """
        Initialize online detector
        
        Parameters:
        -----------
        hazard_rate : float
            Prior probability of change at each time point
            1/250 ≈ yearly change probability for daily data
        """
        self.hazard_rate = hazard_rate
        self.run_length_posterior = None
        self.change_point_probs = None
        
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run online change point detection
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            - Run length posterior probabilities
            - Change point probabilities
        """
        n = len(data)
        
        # Initialize
        self.run_length_posterior = np.zeros((n, n))
        self.run_length_posterior[0, 0] = 1
        
        self.change_point_probs = np.zeros(n)
        
        # Prior parameters for Normal-Gamma conjugate prior
        μ0 = 0      # Prior mean
        κ0 = 1      # Prior precision scale
        α0 = 1      # Prior shape
        β0 = 1      # Prior rate
        
        for t in range(1, n):
            # Initialize predictive probabilities
            predictive_probs = np.zeros(t + 1)
            
            for r in range(t + 1):  # Possible run lengths
                if r == 0:  # Change point
                    # Reset to prior
                    μ = μ0
                    κ = κ0
                    α = α0
                    β = β0
                else:
                    # Update sufficient statistics
                    x_window = data[t - r:t]
                    n_r = len(x_window)
                    
                    # Conjugate prior updates
                    x̄ = np.mean(x_window)
                    μ = (κ0 * μ0 + n_r * x̄) / (κ0 + n_r)
                    κ = κ0 + n_r
                    α = α0 + n_r / 2
                    β = β0 + 0.5 * np.sum((x_window - x̄)**2) + \
                        (κ0 * n_r * (x̄ - μ0)**2) / (2 * (κ0 + n_r))
                
                # Student's t predictive distribution
                # Simplified: use Gaussian approximation
                predictive_mean = μ
                predictive_var = β * (κ + 1) / (α * κ)
                
                # Calculate probability
                prob = stats.norm.pdf(data[t], predictive_mean, np.sqrt(predictive_var))
                predictive_probs[r] = prob
            
            # Calculate growth probabilities (no change)
            growth_probs = predictive_probs * self.run_length_posterior[t-1, :t+1] * (1 - self.hazard_rate)
            
            # Calculate change point probability
            cp_prob = predictive_probs[0] * np.sum(self.run_length_posterior[t-1, :t+1] * self.hazard_rate)
            
            # Update run length posterior
            self.run_length_posterior[t, 0] = cp_prob
            self.run_length_posterior[t, 1:t+1] = growth_probs[:t]
            
            # Normalize
            total = np.sum(self.run_length_posterior[t, :t+1])
            if total > 0:
                self.run_length_posterior[t, :t+1] /= total
            
            # Store change point probability
            self.change_point_probs[t] = self.run_length_posterior[t, 0]
        
        return self.run_length_posterior, self.change_point_probs
    
    def get_change_points(self, threshold: float = 0.1) -> List[int]:
        """
        Get detected change points based on probability threshold
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for change point detection
            
        Returns:
        --------
        List[int] : Indices of detected change points
        """
        if self.change_point_probs is None:
            raise ValueError("Run detect() first")
        
        change_points = np.where(self.change_point_probs > threshold)[0].tolist()
        
        # Filter: minimum distance between changes
        if len(change_points) > 1:
            min_distance = len(self.change_point_probs) // 50  # 2% of data
            filtered = [change_points[0]]
            
            for cp in change_points[1:]:
                if cp - filtered[-1] >= min_distance:
                    filtered.append(cp)
            
            return filtered
        
        return change_points


# Utility functions for model comparison and selection
def compare_models(model_results: List[ChangePointResult]) -> pd.DataFrame:
    """
    Compare multiple change point models
    
    Parameters:
    -----------
    model_results : List[ChangePointResult]
        Results from different models
        
    Returns:
    --------
    pd.DataFrame : Comparison metrics
    """
    comparison_data = []
    
    for result in model_results:
        # Extract key metrics
        metrics = {
            'Model': result.model_name,
            'Change Points': len(result.change_points),
            'WAIC': result.model_fit.get('waic', np.nan),
            'LOO': result.model_fit.get('loo', np.nan),
            'R²': result.model_fit.get('r_squared', np.nan),
            'Converged': result.convergence['rhat']['all_converged'],
            'Min ESS': result.convergence['ess']['min'],
            'Parameters': len(result.summary)
        }
        
        # Add change point dates if available
        if result.change_dates:
            metrics['First Change'] = result.change_dates[0].strftime('%Y-%m-%d')
            metrics['Last Change'] = result.change_dates[-1].strftime('%Y-%m-%d')
        
        comparison_data.append(metrics)
    
    return pd.DataFrame(comparison_data)


def select_best_model(model_results: List[ChangePointResult], 
                     criterion: str = 'waic') -> ChangePointResult:
    """
    Select the best model based on information criterion
    
    Parameters:
    -----------
    model_results : List[ChangePointResult]
        Results from different models
    criterion : str
        Criterion to use: 'waic' or 'loo'
        
    Returns:
    --------
    ChangePointResult : Best model according to criterion
    """
    valid_models = []
    
    for result in model_results:
        # Check if model has required criterion and converged
        if (criterion in result.model_fit and 
            result.model_fit[criterion] is not None and
            result.convergence['rhat']['all_converged']):
            valid_models.append(result)
    
    if not valid_models:
        raise ValueError("No valid models with convergence and criterion available")
    
    # Select model with lowest criterion value (lower is better)
    best_model = min(valid_models, 
                     key=lambda x: x.model_fit[criterion])
    
    print(f"Best model selected: {best_model.model_name}")
    print(f"  Criterion ({criterion}): {best_model.model_fit[criterion]:.1f}")
    print(f"  Change points: {len(best_model.change_points)}")
    
    return best_model


def run_complete_analysis(data: np.ndarray, dates: pd.DatetimeIndex,
                         draws: int = 2000, tune: int = 1000) -> Dict:
    """
    Run complete change point analysis with multiple models
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    dates : pd.DatetimeIndex
        Corresponding dates
    draws : int
        Number of MCMC draws
    tune : int
        Number of tuning samples
        
    Returns:
    --------
    Dict : Complete analysis results
    """
    print("="*60)
    print("COMPLETE CHANGE POINT ANALYSIS")
    print("="*60)
    
    results = {}
    
    # 1. Single change point model
    print("\n1. Running single change point model...")
    single_model = SingleChangePointModel(data, dates)
    single_model.build_model()
    single_trace = single_model.sample(draws=draws//2, tune=tune//2)  # Fewer samples for speed
    single_result = single_model.extract_results(single_trace)
    results['single'] = single_result
    
    # 2. Mean-variance change point model
    print("\n2. Running mean-variance change point model...")
    mv_model = MeanVarianceChangePointModel(data, dates)
    mv_model.build_model()
    mv_trace = mv_model.sample(draws=draws//2, tune=tune//2)
    mv_result = mv_model.extract_results(mv_trace)
    results['mean_variance'] = mv_result
    
    # 3. Bayesian online detection (for comparison)
    print("\n3. Running Bayesian online detection...")
    online_detector = BayesianOnlineChangePointDetector(hazard_rate=1/500)
    run_length_post, cp_probs = online_detector.detect(data[:2000])  # Subset for speed
    cp_indices = online_detector.get_change_points(threshold=0.1)
    cp_dates = [dates[i] for i in cp_indices if i < len(dates)]
    
    results['online'] = {
        'model_name': 'Bayesian_Online',
        'change_points': cp_indices,
        'change_dates': cp_dates,
        'change_probs': cp_probs,
        'run_length_post': run_length_post
    }
    
    # 4. Model comparison
    print("\n4. Comparing models...")
    model_results_list = [single_result, mv_result]
    comparison_df = compare_models(model_results_list)
    
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # 5. Select best model
    try:
        best_model = select_best_model(model_results_list, criterion='waic')
        results['best_model'] = best_model
        print(f"\n✓ Selected best model: {best_model.model_name}")
    except ValueError as e:
        print(f"\n⚠ Could not select best model: {e}")
        results['best_model'] = None
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    
    return results


# Example usage
if __name__ == '__main__':
    # Example data generation for testing
    np.random.seed(42)
    n_points = 1000
    
    # Create synthetic data with a change point
    data1 = np.random.normal(0, 1, 500)
    data2 = np.random.normal(2, 1.5, 500)
    synthetic_data = np.concatenate([data1, data2])
    
    # Create dates
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    print("Testing Bayesian Change Point Models...")
    
    try:
        # Test single change point model
        model = SingleChangePointModel(synthetic_data, dates)
        model.build_model()
        
        # Sample (with reduced samples for quick testing)
        trace = model.sample(draws=500, tune=250, chains=2)
        
        # Extract results
        result = model.extract_results(trace)
        
        print(f"\nChange point detected at index: {result.change_points[0]}")
        print(f"Change point date: {result.change_dates[0]}")
        print(f"Mean before: {result.parameters['μ1']['mean']:.3f}")
        print(f"Mean after: {result.parameters['μ2']['mean']:.3f}")
        print(f"Probability μ2 > μ1: {result.parameters.get('prob_μ2_gt_μ1', 0):.2%}")
        
        # Save results
        model.save_results()
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()
