"""
Visualization Utilities for Bayesian Change Point Analysis

This module provides comprehensive plotting functions for:
- Posterior distributions and trace plots
- Change point visualization on time series
- Model diagnostics and convergence checking
- Event association and impact visualization
- Publication-ready figure generation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import arviz as az
from typing import List, Dict, Tuple, Optional, Union, Any
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set global plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'

# Color schemes
COLOR_SCHEMES = {
    # Primary colors
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Raspberry
    'accent': '#F18F01',       # Orange
    'highlight': '#C73E1D',    # Brick red
    'neutral': '#6C757D',      # Gray
    
    # Regime colors
    'regime1': '#4ECDC4',      # Turquoise
    'regime2': '#FF6B6B',      #Coral
    'regime3': '#FFD166',      # Yellow
    'regime4': '#06D6A0',      # Mint
    'regime5': '#118AB2',      # Blue
    'regime6': '#EF476F',      # Pink
    
    # Event type colors
    'conflict': '#DC3545',     # Red
    'policy': '#28A745',       # Green
    'economic': '#FFC107',     # Yellow
    'disaster': '#FD7E14',     # Orange
    'health': '#6F42C1',       # Purple
    
    # Impact level colors
    'high': '#DC3545',         # Red
    'medium': '#FFC107',       # Yellow
    'low': '#28A745',          # Green
    
    # Statistical colors
    'mean': '#2E86AB',
    'median': '#A23B72',
    'hdi': 'lightgray',
    'ci': 'lightblue'
}

# Event type markers
EVENT_MARKERS = {
    'Conflict': 'x',
    'Policy': '^',
    'Economic': 's',
    'Disaster': 'v',
    'Health': 'd'
}


def setup_plotting_context(context='notebook', style='darkgrid'):
    """
    Set up plotting context and style
    
    Parameters:
    -----------
    context : str
        Plotting context ('paper', 'notebook', 'talk', 'poster')
    style : str
        Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
    """
    if context == 'paper':
        plt.rcParams['figure.figsize'] = (7, 5)
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['axes.labelsize'] = 10
    elif context == 'talk':
        plt.rcParams['figure.figsize'] = (10, 7)
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
    elif context == 'poster':
        plt.rcParams['figure.figsize'] = (12, 9)
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 16
    
    plt.style.use(f'seaborn-v0_8-{style}')
    sns.set_palette("husl")
    
    return plt.rcParams.copy()


def plot_time_series_with_change_points(data: np.ndarray,
                                       dates: pd.DatetimeIndex,
                                       change_points: List[int],
                                       change_point_names: Optional[List[str]] = None,
                                       change_point_probs: Optional[List[float]] = None,
                                       title: str = "Time Series with Change Points",
                                       ylabel: str = "Value",
                                       xlabel: str = "Date",
                                       highlight_major: bool = True,
                                       regime_colors: bool = True,
                                       show_hdi: bool = False,
                                       hdi_data: Optional[Dict] = None,
                                       event_markers: Optional[Dict] = None,
                                       figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot time series with vertical lines at change points
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    dates : pd.DatetimeIndex
        Date index
    change_points : List[int]
        Indices of change points
    change_point_names : List[str], optional
        Names for each change point
    change_point_probs : List[float], optional
        Probabilities for each change point
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    highlight_major : bool
        Whether to highlight major change points
    regime_colors : bool
        Whether to color code regimes
    show_hdi : bool
        Whether to show HDI intervals for change points
    hdi_data : Dict, optional
        HDI intervals for change points
    event_markers : Dict, optional
        Event markers to overlay {date: event_info}
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot time series
    ax.plot(dates, data, linewidth=1.2, alpha=0.9,
            color=COLOR_SCHEMES['primary'], label='Time Series', zorder=1)
    
    # Color regimes if requested
    if regime_colors and change_points:
        regime_boundaries = [0] + change_points + [len(data)]
        regime_colors_list = [
            COLOR_SCHEMES[f'regime{i+1}'] for i in range(len(regime_boundaries) - 1)
        ]
        
        for i in range(len(regime_boundaries) - 1):
            start_idx = regime_boundaries[i]
            end_idx = min(regime_boundaries[i + 1], len(dates))
            
            if start_idx < len(dates):
                start_date = dates[start_idx]
                end_date = dates[min(end_idx, len(dates) - 1)]
                
                ax.axvspan(start_date, end_date, alpha=0.15,
                          color=regime_colors_list[i % len(regime_colors_list)],
                          label=f'Regime {i+1}' if i < 4 else None)
    
    # Plot change points
    legend_elements = []
    
    for i, cp_idx in enumerate(change_points):
        if cp_idx < len(dates):
            cp_date = dates[cp_idx]
            
            # Determine line properties
            if highlight_major and change_point_probs:
                prob = change_point_probs[i]
                if prob > 0.8:
                    color = COLOR_SCHEMES['highlight']
                    linewidth = 3
                    linestyle = '-'
                    label = 'Major Change Point' if i == 0 else None
                elif prob > 0.5:
                    color = COLOR_SCHEMES['accent']
                    linewidth = 2
                    linestyle = '--'
                    label = 'Moderate Change Point' if i == 0 else None
                else:
                    color = COLOR_SCHEMES['secondary']
                    linewidth = 1.5
                    linestyle = ':'
                    label = 'Minor Change Point' if i == 0 else None
            else:
                color = COLOR_SCHEMES['secondary']
                linewidth = 2
                linestyle = '--'
                label = 'Change Point' if i == 0 else None
            
            # Plot change point line
            line = ax.axvline(x=cp_date, color=color, linestyle=linestyle,
                             linewidth=linewidth, alpha=0.8, label=label, zorder=2)
            
            # Add HDI interval if available
            if show_hdi and hdi_data and i in hdi_data:
                hdi_start = hdi_data[i]['lower']
                hdi_end = hdi_data[i]['upper']
                hdi_start_date = dates[min(max(0, hdi_start), len(dates)-1)]
                hdi_end_date = dates[min(max(0, hdi_end), len(dates)-1)]
                
                ax.axvspan(hdi_start_date, hdi_end_date, alpha=0.2,
                          color=color, label=f'95% HDI' if i == 0 else None)
            
            # Add annotation
            annotation_text = f'CP{i+1}'
            if change_point_names and i < len(change_point_names):
                annotation_text = f'{change_point_names[i]}'
            if change_point_probs and i < len(change_point_probs):
                annotation_text += f'\nP={change_point_probs[i]:.2f}'
            
            ax.annotate(annotation_text,
                       xy=(cp_date, ax.get_ylim()[1] * 0.95),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor=color,
                                alpha=0.8),
                       zorder=3)
            
            if i == 0:
                legend_elements.append(
                    Line2D([0], [0], color=color, linestyle=linestyle,
                          linewidth=linewidth, label=label)
                )
    
    # Add event markers if provided
    if event_markers:
        event_legend_added = False
        for date_str, event_info in event_markers.items():
            event_date = pd.to_datetime(date_str)
            if event_date >= dates[0] and event_date <= dates[-1]:
                # Get marker and color based on event type
                event_type = event_info.get('type', 'Unknown')
                marker = EVENT_MARKERS.get(event_type, 'o')
                color = COLOR_SCHEMES.get(event_type.lower(), COLOR_SCHEMES['neutral'])
                
                # Find closest data point
                idx = np.argmin(np.abs(dates - event_date))
                y_value = data[idx]
                
                ax.scatter(event_date, y_value, marker=marker, s=100,
                          color=color, edgecolor='black', linewidth=1.5,
                          zorder=4, alpha=0.9,
                          label='Event' if not event_legend_added else None)
                event_legend_added = True
                
                # Add event label
                ax.annotate(event_info.get('name', 'Event'),
                           xy=(event_date, y_value),
                           xytext=(0, 15),
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=8, fontstyle='italic',
                           bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='white',
                                    edgecolor=color,
                                    alpha=0.7))
    
    # Formatting
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    return fig


def plot_posterior_distributions(trace: az.InferenceData,
                                var_names: Optional[List[str]] = None,
                                hdi_prob: float = 0.95,
                                show_kde: bool = True,
                                show_mean: bool = True,
                                show_median: bool = False,
                                figsize: Tuple[int, int] = (12, 8),
                                title_suffix: str = "") -> plt.Figure:
    """
    Plot posterior distributions from MCMC trace
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC trace data
    var_names : List[str], optional
        Variables to plot. If None, plots all.
    hdi_prob : float
        Highest density interval probability
    show_kde : bool
        Whether to show KDE overlay
    show_mean : bool
        Whether to mark the mean
    show_median : bool
        Whether to mark the median
    figsize : Tuple[int, int]
        Figure size
    title_suffix : str
        Suffix to add to title
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    if var_names is None:
        var_names = list(trace.posterior.data_vars)
    
    n_vars = len(var_names)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'Posterior Distributions {title_suffix}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    if n_vars == 1:
        axes = np.array([axes])
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    
    for i, var_name in enumerate(var_names):
        if var_name in trace.posterior:
            # Extract samples
            samples = trace.posterior[var_name].values.flatten()
            
            # Plot histogram
            ax = axes[i] if n_vars > 1 else axes
            n, bins, patches = ax.hist(samples, bins=50, density=True,
                                      alpha=0.7, color=COLOR_SCHEMES['primary'],
                                      edgecolor='white', linewidth=0.5)
            
            # Plot KDE if requested
            if show_kde:
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(samples)
                    x_range = np.linspace(samples.min(), samples.max(), 1000)
                    ax.plot(x_range, kde(x_range), color=COLOR_SCHEMES['secondary'],
                           linewidth=2, label='KDE')
                except:
                    pass
            
            # Add mean line
            if show_mean:
                mean_val = np.mean(samples)
                ax.axvline(x=mean_val, color=COLOR_SCHEMES['highlight'],
                          linestyle='-', linewidth=2,
                          label=f'Mean: {mean_val:.3f}')
            
            # Add median line
            if show_median:
                median_val = np.median(samples)
                ax.axvline(x=median_val, color=COLOR_SCHEMES['accent'],
                          linestyle='--', linewidth=1.5,
                          label=f'Median: {median_val:.3f}')
            
            # Add HDI interval
            hdi = az.hdi(samples, hdi_prob=hdi_prob)
            ax.axvspan(hdi[0], hdi[1], alpha=0.2,
                      color=COLOR_SCHEMES['hdi'],
                      label=f'{int(hdi_prob*100)}% HDI')
            
            # Add statistics text
            stats_text = (f'Mean: {np.mean(samples):.3f}\n'
                         f'Std: {np.std(samples):.3f}\n'
                         f'95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]')
            
            ax.text(0.95, 0.95, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(var_name, fontweight='bold', fontsize=12)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':')
            
        else:
            ax = axes[i] if n_vars > 1 else axes
            ax.text(0.5, 0.5, f'Variable {var_name} not found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Missing: {var_name}', fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_vars, n_rows * n_cols):
        if n_rows * n_cols > 1:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_trace_diagnostics(trace: az.InferenceData,
                          var_names: Optional[List[str]] = None,
                          max_vars: int = 6,
                          figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot comprehensive trace diagnostics including trace plots and autocorrelation
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC trace data
    var_names : List[str], optional
        Variables to plot. If None, selects first max_vars.
    max_vars : int
        Maximum number of variables to plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    if var_names is None:
        var_names = list(trace.posterior.data_vars)[:max_vars]
    else:
        var_names = var_names[:max_vars]
    
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 3, figsize=figsize)
    fig.suptitle('MCMC Trace Diagnostics', fontsize=16, fontweight='bold', y=1.02)
    
    if n_vars == 1:
        axes = axes.reshape(1, 3)
    
    for i, var_name in enumerate(var_names):
        if var_name in trace.posterior:
            samples = trace.posterior[var_name].values
            
            # Reshape for trace plot (chains × samples)
            n_chains, n_samples = samples.shape[:2]
            trace_data = samples.reshape(n_chains, n_samples)
            
            # Column 1: Trace plot
            ax_trace = axes[i, 0]
            for chain in range(n_chains):
                ax_trace.plot(trace_data[chain], alpha=0.7, linewidth=0.8,
                             label=f'Chain {chain+1}')
            ax_trace.set_title(f'{var_name} - Trace Plot', fontsize=11, fontweight='bold')
            ax_trace.set_xlabel('Sample')
            ax_trace.set_ylabel('Value')
            ax_trace.legend(fontsize=8)
            ax_trace.grid(True, alpha=0.3)
            
            # Column 2: Autocorrelation plot
            ax_acf = axes[i, 1]
            max_lag = min(100, n_samples // 2)
            
            # Calculate autocorrelation for each chain
            for chain in range(n_chains):
                chain_data = trace_data[chain]
                acf = np.correlate(chain_data - np.mean(chain_data),
                                  chain_data - np.mean(chain_data), mode='full')
                acf = acf[len(acf)//2:len(acf)//2 + max_lag]
                acf = acf / acf[0]  # Normalize
                
                ax_acf.plot(range(max_lag), acf, alpha=0.7, linewidth=1)
            
            ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_acf.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax_acf.axhline(y=-0.1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax_acf.set_title(f'{var_name} - Autocorrelation', fontsize=11, fontweight='bold')
            ax_acf.set_xlabel('Lag')
            ax_acf.set_ylabel('Autocorrelation')
            ax_acf.grid(True, alpha=0.3)
            
            # Column 3: Distribution + convergence statistics
            ax_dist = axes[i, 2]
            
            # Plot histogram of all chains combined
            all_samples = trace_data.flatten()
            ax_dist.hist(all_samples, bins=40, density=True,
                        alpha=0.7, color=COLOR_SCHEMES['primary'],
                        edgecolor='white')
            
            # Add KDE
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(all_samples)
                x_range = np.linspace(all_samples.min(), all_samples.max(), 1000)
                ax_dist.plot(x_range, kde(x_range), color=COLOR_SCHEMES['secondary'],
                           linewidth=2)
            except:
                pass
            
            # Add convergence statistics
            # R-hat (from ArviZ if available, otherwise approximate)
            try:
                summary = az.summary(trace, var_names=[var_name])
                r_hat = summary['r_hat'].iloc[0]
                ess = summary['ess_bulk'].iloc[0]
            except:
                # Simple R-hat approximation
                between_chain_var = np.var(np.mean(trace_data, axis=1))
                within_chain_var = np.mean(np.var(trace_data, axis=1))
                r_hat = np.sqrt((between_chain_var/within_chain_var + n_samples - 1)/n_samples)
                ess = n_chains * n_samples
            
            stats_text = (f'R-hat: {r_hat:.3f}\n'
                         f'ESS: {ess:.0f}\n'
                         f'Mean: {np.mean(all_samples):.3f}\n'
                         f'Std: {np.std(all_samples):.3f}')
            
            ax_dist.text(0.95, 0.95, stats_text,
                        transform=ax_dist.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_dist.set_title(f'{var_name} - Distribution', fontsize=11, fontweight='bold')
            ax_dist.set_xlabel('Value')
            ax_dist.set_ylabel('Density')
            ax_dist.grid(True, alpha=0.3)
            
        else:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'Variable {var_name} not found',
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'Missing: {var_name}', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_change_point_posterior(tau_samples: np.ndarray,
                               dates: pd.DatetimeIndex,
                               true_change_points: Optional[List[int]] = None,
                               title: str = "Change Point Posterior Distribution",
                               show_hdi: bool = True,
                               hdi_prob: float = 0.95,
                               show_mean: bool = True,
                               show_mode: bool = True,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot posterior distribution of change point locations
    
    Parameters:
    -----------
    tau_samples : np.ndarray
        Samples of change point locations (indices)
    dates : pd.DatetimeIndex
        Date index for converting indices to dates
    true_change_points : List[int], optional
        True change point indices (for validation)
    title : str
        Plot title
    show_hdi : bool
        Whether to show HDI interval
    hdi_prob : float
        HDI probability level
    show_mean : bool
        Whether to show mean location
    show_mode : bool
        Whether to show mode location
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Convert indices to dates for x-axis
    tau_dates = [dates[min(max(0, int(tau)), len(dates)-1)] for tau in tau_samples]
    tau_dates = pd.to_datetime(tau_dates)
    
    # Plot 1: Histogram of change point dates
    ax1.hist(tau_dates, bins=50, density=True, alpha=0.7,
            color=COLOR_SCHEMES['primary'], edgecolor='white')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    try:
        # Convert dates to numeric for KDE
        tau_numeric = mdates.date2num(tau_dates)
        kde = gaussian_kde(tau_numeric)
        x_range_numeric = np.linspace(tau_numeric.min(), tau_numeric.max(), 1000)
        x_range_dates = mdates.num2date(x_range_numeric)
        ax1.plot(x_range_dates, kde(x_range_numeric),
                color=COLOR_SCHEMES['secondary'], linewidth=2, label='KDE')
    except:
        pass
    
    # Add mean change point
    if show_mean:
        mean_tau = int(np.mean(tau_samples))
        mean_date = dates[min(max(0, mean_tau), len(dates)-1)]
        ax1.axvline(x=mean_date, color=COLOR_SCHEMES['highlight'],
                   linestyle='-', linewidth=2.5,
                   label=f'Mean: {mean_date.strftime("%Y-%m-%d")}')
    
    # Add mode change point
    if show_mode:
        from scipy.stats import mode
        mode_result = mode(tau_samples)
        mode_tau = int(mode_result.mode[0]) if len(mode_result.mode) > 0 else mean_tau
        mode_date = dates[min(max(0, mode_tau), len(dates)-1)]
        ax1.axvline(x=mode_date, color=COLOR_SCHEMES['accent'],
                   linestyle='--', linewidth=2,
                   label=f'Mode: {mode_date.strftime("%Y-%m-%d")}')
    
    # Add HDI interval
    if show_hdi:
        hdi = az.hdi(tau_samples, hdi_prob=hdi_prob)
        hdi_start = dates[min(max(0, int(hdi[0])), len(dates)-1)]
        hdi_end = dates[min(max(0, int(hdi[1])), len(dates)-1)]
        ax1.axvspan(hdi_start, hdi_end, alpha=0.2,
                   color=COLOR_SCHEMES['hdi'],
                   label=f'{int(hdi_prob*100)}% HDI')
    
    # Add true change points if provided
    if true_change_points:
        for i, true_cp in enumerate(true_change_points):
            if true_cp < len(dates):
                true_date = dates[true_cp]
                ax1.axvline(x=true_date, color='black',
                           linestyle=':', linewidth=1.5,
                           label=f'True CP{i+1}' if i == 0 else None)
    
    ax1.set_title(title, fontweight='bold', pad=10)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # Plot 2: Cumulative probability
    ax2.hist(tau_dates, bins=50, density=True, cumulative=True,
            alpha=0.7, color=COLOR_SCHEMES['neutral'],
            edgecolor='white', histtype='step', linewidth=2)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Probability')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    return fig


def plot_regime_comparison(regime_stats: pd.DataFrame,
                          metrics: List[str] = ['mean_return', 'std_return', 'sharpe_ratio'],
                          title: str = "Regime Comparison",
                          figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot comparison of different regimes
    
    Parameters:
    -----------
    regime_stats : pd.DataFrame
        DataFrame with regime statistics
    metrics : List[str]
        Metrics to compare
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    
    if n_metrics == 1:
        axes = [axes]
    
    # Get regime colors
    n_regimes = len(regime_stats)
    regime_colors = [COLOR_SCHEMES[f'regime{i+1}'] for i in range(n_regimes)]
    
    for i, metric in enumerate(metrics):
        if metric in regime_stats.columns:
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(range(n_regimes), regime_stats[metric],
                         color=regime_colors, alpha=0.8,
                         edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.3f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
            
            # Customize axis
            ax.set_xlabel('Regime')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xticks(range(n_regimes))
            ax.set_xticklabels([f'R{r+1}' for r in range(n_regimes)])
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add regime duration as secondary information
            if 'duration_days' in regime_stats.columns:
                for idx, (_, row) in enumerate(regime_stats.iterrows()):
                    duration = row['duration_days'] / 365.25  # Convert to years
                    ax.text(idx, ax.get_ylim()[0] * 1.02,
                           f'{duration:.1f}y', ha='center', va='bottom',
                           fontsize=8, style='italic')
    
    plt.tight_layout()
    return fig


def plot_event_impact_association(associations: pd.DataFrame,
                                 impact_metric: str = 'mean_change',
                                 size_metric: str = 'vol_change',
                                 title: str = "Event Impact Association",
                                 color_by: str = 'event_type',
                                 show_stats: bool = True,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot event-change point associations with impact visualization
    
    Parameters:
    -----------
    associations : pd.DataFrame
        DataFrame with event-change point associations
    impact_metric : str
        Metric to use for impact (y-axis)
    size_metric : str
        Metric to use for bubble size
    title : str
        Plot title
    color_by : str
        Column to use for coloring ('event_type', 'impact_level', 'region')
    show_stats : bool
        Whether to show summary statistics
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    if associations.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No associations to plot',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Scatter plot of impact vs timing
    ax1 = axes[0, 0]
    
    # Prepare data
    x = associations['days_difference']
    y = associations[impact_metric]
    
    # Determine colors
    if color_by in associations.columns:
        unique_categories = associations[color_by].unique()
        color_map = {}
        for j, category in enumerate(unique_categories):
            color_map[category] = COLOR_SCHEMES.get(category.lower(), 
                                                   f'C{j}')
    else:
        color_map = {'All': COLOR_SCHEMES['primary']}
        associations['_color'] = 'All'
        color_by = '_color'
    
    # Determine bubble sizes
    if size_metric in associations.columns:
        sizes = 50 + 200 * (associations[size_metric] - associations[size_metric].min()) / \
               (associations[size_metric].max() - associations[size_metric].min())
    else:
        sizes = 100
    
    # Create scatter plot
    for category in associations[color_by].unique():
        mask = associations[color_by] == category
        ax1.scatter(x[mask], y[mask], s=sizes[mask] if hasattr(sizes, '__getitem__') else sizes,
                   color=color_map[category], alpha=0.7, edgecolor='black', linewidth=0.5,
                   label=category)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Days from Event to Change Point')
    ax1.set_ylabel(impact_metric.replace('_', ' ').title())
    ax1.set_title('Impact vs. Timing', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Impact by category
    ax2 = axes[0, 1]
    
    if color_by != '_color' and color_by in associations.columns:
        impact_by_category = associations.groupby(color_by)[impact_metric].agg(['mean', 'std', 'count'])
        
        if not impact_by_category.empty:
            categories = impact_by_category.index.tolist()
            x_pos = range(len(categories))
            colors = [color_map.get(cat, COLOR_SCHEMES['neutral']) for cat in categories]
            
            bars = ax2.bar(x_pos, impact_by_category['mean'],
                          yerr=impact_by_category['std'],
                          capsize=5, color=colors, alpha=0.8,
                          edgecolor='black')
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(categories, rotation=45, ha='right')
            ax2.set_ylabel(f'Mean {impact_metric.replace("_", " ")}')
            ax2.set_title(f'Impact by {color_by.replace("_", " ").title()}', fontweight='bold')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add count labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = impact_by_category.iloc[i]['count']
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'n={int(count)}', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
    
    # Plot 3: Distribution of impacts
    ax3 = axes[1, 0]
    
    ax3.hist(y, bins=20, density=True, alpha=0.7,
            color=COLOR_SCHEMES['primary'], edgecolor='white')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(y.dropna())
        x_range = np.linspace(y.min(), y.max(), 1000)
        ax3.plot(x_range, kde(x_range), color=COLOR_SCHEMES['secondary'],
                linewidth=2, label='KDE')
    except:
        pass
    
    # Add statistics
    stats_text = (f'Mean: {y.mean():.3f}\n'
                 f'Std: {y.std():.3f}\n'
                 f'Min: {y.min():.3f}\n'
                 f'Max: {y.max():.3f}\n'
                 f'N: {len(y)}')
    
    ax3.text(0.95, 0.95, stats_text,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel(impact_metric.replace('_', ' ').title())
    ax3.set_ylabel('Density')
    ax3.set_title('Impact Distribution', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timeline of events and impacts
    ax4 = axes[1, 1]
    
    if 'change_point_date' in associations.columns and 'event_date' in associations.columns:
        # Convert to datetime if needed
        cp_dates = pd.to_datetime(associations['change_point_date'])
        event_dates = pd.to_datetime(associations['event_date'])
        
        # Sort by change point date
        sort_idx = np.argsort(cp_dates)
        cp_dates = cp_dates.iloc[sort_idx]
        event_dates = event_dates.iloc[sort_idx]
        impacts = y.iloc[sort_idx]
        
        # Create timeline
        for i, (cp_date, event_date, impact) in enumerate(zip(cp_dates, event_dates, impacts)):
            # Color based on impact direction
            color = COLOR_SCHEMES['high'] if impact > 0 else COLOR_SCHEMES['accent']
            
            # Plot connection line
            ax4.plot([event_date, cp_date], [i, i],
                    color=color, alpha=0.5, linewidth=1)
            
            # Plot event marker
            ax4.scatter(event_date, i, color=COLOR_SCHEMES['primary'],
                       s=100, marker='o', edgecolor='black', linewidth=1,
                       label='Event' if i == 0 else None)
            
            # Plot change point marker
            ax4.scatter(cp_date, i, color=color,
                       s=150, marker='s', edgecolor='black', linewidth=1.5,
                       label='Change Point' if i == 0 else None)
            
            # Add impact value
            ax4.text(cp_date, i + 0.1, f'{impact:.2f}',
                    ha='left', va='bottom', fontsize=8)
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Association Index')
        ax4.set_title('Event-Change Point Timeline', fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame,
                         metrics: List[str] = ['WAIC', 'LOO', 'R²'],
                         title: str = "Model Comparison",
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot comparison of multiple models
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison metrics
    metrics : List[str]
        Metrics to include in comparison
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Radar chart for model comparison
    ax1 = axes[0, 0]
    
    # Prepare data for radar chart
    models = comparison_df['Model'].tolist()
    n_models = len(models)
    
    # Normalize metrics for radar chart
    radar_metrics = []
    for metric in metrics:
        if metric in comparison_df.columns:
            values = comparison_df[metric].values
            # Handle missing values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                # Normalize to 0-1 (lower is better for WAIC/LOO, higher for R²)
                if metric in ['WAIC', 'LOO']:
                    norm_values = 1 - (values - valid_values.min()) / (valid_values.max() - valid_values.min())
                else:
                    norm_values = (values - valid_values.min()) / (valid_values.max() - valid_values.min())
                radar_metrics.append((metric, norm_values))
    
    if radar_metrics:
        n_metrics = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        for i, model in enumerate(models):
            values = [radar_metrics[j][1][i] for j in range(n_metrics)]
            values += values[:1]  # Close the circle
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=model)
            ax1.fill(angles, values, alpha=0.1)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([m[0] for m in radar_metrics])
        ax1.set_title('Model Performance Radar', fontweight='bold')
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax1.grid(True)
    
    # Plot 2: Bar chart for key metrics
    ax2 = axes[0, 1]
    
    if 'Change Points' in comparison_df.columns:
        x_pos = range(n_models)
        bars = ax2.bar(x_pos, comparison_df['Change Points'],
                      color=COLOR_SCHEMES['primary'], alpha=0.8,
                      edgecolor='black')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Number of Change Points')
        ax2.set_title('Change Point Count', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Convergence metrics
    ax3 = axes[1, 0]
    
    convergence_data = []
    model_names = []
    for _, row in comparison_df.iterrows():
        if 'Converged' in row and 'Min ESS' in row:
            convergence_data.append([1 if row['Converged'] else 0, row['Min ESS']])
            model_names.append(row['Model'])
    
    if convergence_data:
        convergence_data = np.array(convergence_data)
        x_pos = range(len(model_names))
        
        width = 0.35
        ax3.bar(x_pos, convergence_data[:, 0], width,
               color=COLOR_SCHEMES['accent'], label='Converged (1=Yes)')
        ax3.bar([p + width for p in x_pos], convergence_data[:, 1], width,
               color=COLOR_SCHEMES['secondary'], label='Min ESS')
        
        ax3.set_xticks([p + width/2 for p in x_pos])
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.set_title('Convergence Metrics', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Information criteria
    ax4 = axes[1, 1]
    
    if 'WAIC' in comparison_df.columns:
        metrics_to_plot = ['WAIC', 'LOO'] if 'LOO' in comparison_df.columns else ['WAIC']
        x_pos = range(n_models)
        width = 0.35
        
        for i, metric in enumerate(metrics_to_plot):
            offset = i * width
            values = comparison_df[metric].values
            valid_mask = ~np.isnan(values)
            
            if np.any(valid_mask):
                ax4.bar([p + offset for p in x_pos[valid_mask]], values[valid_mask], width,
                       label=metric, alpha=0.8)
        
        ax4.set_xticks([p + width/2 for p in x_pos])
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.set_ylabel('Value')
        ax4.set_title('Information Criteria', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_parameter_recovery(true_params: Dict[str, float],
                          estimated_params: Dict[str, Dict],
                          title: str = "Parameter Recovery Analysis",
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot true vs estimated parameter comparison for model validation
    
    Parameters:
    -----------
    true_params : Dict[str, float]
        True parameter values
    estimated_params : Dict[str, Dict]
        Estimated parameters with uncertainty
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Plot figure
    """
    param_names = list(true_params.keys())
    n_params = len(param_names)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    
    # Plot 1: True vs Estimated comparison
    true_values = []
    estimated_means = []
    estimated_errors = []
    
    for param in param_names:
        if param in true_params and param in estimated_params:
            true_values.append(true_params[param])
            estimated_means.append(estimated_params[param]['mean'])
            
            # Calculate error bar (95% HDI half-width)
            if 'hdi_95' in estimated_params[param]:
                hdi = estimated_params[param]['hdi_95']
                error = (hdi[1] - hdi[0]) / 2
                estimated_errors.append(error)
            else:
                estimated_errors.append(estimated_params[param].get('std', 0))
    
    if true_values:
        x_pos = range(len(true_values))
        
        ax1.errorbar(x_pos, estimated_means, yerr=estimated_errors,
                    fmt='o', color=COLOR_SCHEMES['primary'],
                    ecolor=COLOR_SCHEMES['secondary'], capsize=5,
                    label='Estimated ± 95% HDI')
        
        ax1.scatter(x_pos, true_values, color=COLOR_SCHEMES['highlight'],
                   s=100, marker='s', edgecolor='black', linewidth=2,
                   label='True Value', zorder=5)
        
        # Add perfect recovery line
        ax1.plot([min(true_values + estimated_means), max(true_values + estimated_means)],
                [min(true_values + estimated_means), max(true_values + estimated_means)],
                '--', color='gray', alpha=0.5, label='Perfect Recovery')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.set_xlabel('Parameter')
        ax1.set_ylabel('Value')
        ax1.set_title('True vs Estimated Parameters', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recovery error distribution
    if true_values and estimated_means:
        errors = np.array(estimated_means) - np.array(true_values)
        relative_errors = errors / np.array(true_values)
        
        ax2.hist(errors, bins=15, density=True, alpha=0.7,
                color=COLOR_SCHEMES['primary'], edgecolor='white',
                label=f'Absolute Error\nMean: {np.mean(np.abs(errors)):.3f}')
        
        ax2.hist(relative_errors, bins=15, density=True, alpha=0.7,
                color=COLOR_SCHEMES['accent'], edgecolor='white',
                label=f'Relative Error\nMean: {np.mean(np.abs(relative_errors)):.3f}')
        
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Parameter Recovery Error', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_dashboard(analysis_results: Dict,
                            dates: pd.DatetimeIndex,
                            data: np.ndarray,
                            figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create a comprehensive summary dashboard for change point analysis
    
    Parameters:
    -----------
    analysis_results : Dict
        Dictionary containing analysis results from different models
    dates : pd.DatetimeIndex
        Date index
    data : np.ndarray
        Time series data
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : Dashboard figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Bayesian Change Point Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Create subplot grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Time series with all detected change points
    ax1 = fig.add_subplot(gs[0, :])
    
    # Get change points from all models
    all_change_points = []
    change_point_labels = []
    
    for model_name, results in analysis_results.items():
        if hasattr(results, 'change_points'):
            all_change_points.extend(results.change_points)
            change_point_labels.extend([f'{model_name}-CP{i+1}' 
                                      for i in range(len(results.change_points))])
    
    # Plot time series
    ax1.plot(dates, data, linewidth=1.0, alpha=0.8,
            color=COLOR_SCHEMES['primary'])
    
    # Plot change points
    for cp_idx, label in zip(all_change_points, change_point_labels):
        if cp_idx < len(dates):
            cp_date = dates[cp_idx]
            ax1.axvline(x=cp_date, color=COLOR_SCHEMES['highlight'],
                       linestyle='--', linewidth=1, alpha=0.7)
            ax1.annotate(label, xy=(cp_date, ax1.get_ylim()[1] * 0.95),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax1.set_title('Time Series with Detected Change Points', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    model_names = []
    n_change_points = []
    convergence_status = []
    
    for model_name, results in analysis_results.items():
        if hasattr(results, 'change_points'):
            model_names.append(model_name)
            n_change_points.append(len(results.change_points))
            convergence_status.append(results.convergence['rhat']['all_converged'])
    
    if model_names:
        x_pos = range(len(model_names))
        colors = [COLOR_SCHEMES['primary'] if conv else COLOR_SCHEMES['secondary'] 
                 for conv in convergence_status]
        
        bars = ax2.bar(x_pos, n_change_points, color=colors, alpha=0.8,
                      edgecolor='black')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('Number of Change Points')
        ax2.set_title('Model Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add convergence indicators
        for i, (bar, conv) in enumerate(zip(bars, convergence_status)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    '✓' if conv else '✗', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    # Plot 3: Impact summary
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Collect impact metrics if available
    impact_metrics = []
    impact_labels = []
    
    for model_name, results in analysis_results.items():
        if hasattr(results, 'parameters') and 'μ_diff' in results.parameters:
            impact = results.parameters['μ_diff']['mean']
            impact_metrics.append(impact)
            impact_labels.append(model_name)
    
    if impact_metrics:
        x_pos = range(len(impact_metrics))
        colors = [COLOR_SCHEMES['high'] if imp > 0 else COLOR_SCHEMES['accent'] 
                 for imp in impact_metrics]
        
        bars = ax3.bar(x_pos, impact_metrics, color=colors, alpha=0.8,
                      edgecolor='black')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(impact_labels, rotation=45, ha='right')
        ax3.set_ylabel('Mean Change (μ₂ - μ₁)')
        ax3.set_title('Impact Magnitude', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Uncertainty summary
    ax4 = fig.add_subplot(gs[1, 2])
    
    uncertainty_data = []
    uncertainty_labels = []
    
    for model_name, results in analysis_results.items():
        if hasattr(results, 'parameters'):
            # Calculate average HDI width as uncertainty measure
            hdi_widths = []
            for param_name, param_info in results.parameters.items():
                if 'hdi_95' in param_info:
                    hdi = param_info['hdi_95']
                    hdi_widths.append(hdi[1] - hdi[0])
            
            if hdi_widths:
                uncertainty_data.append(np.mean(hdi_widths))
                uncertainty_labels.append(model_name)
    
    if uncertainty_data:
        x_pos = range(len(uncertainty_data))
        ax4.bar(x_pos, uncertainty_data, color=COLOR_SCHEMES['neutral'],
               alpha=0.8, edgecolor='black')
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(uncertainty_labels, rotation=45, ha='right')
        ax4.set_ylabel('Average HDI Width')
        ax4.set_title('Parameter Uncertainty', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Convergence diagnostics summary
    ax5 = fig.add_subplot(gs[2, 0])
    
    ess_data = []
    rhat_data = []
    model_labels = []
    
    for model_name, results in analysis_results.items():
        if hasattr(results, 'convergence'):
            ess_data.append(results.convergence['ess']['min'])
            rhat_data.append(results.convergence['rhat']['max'])
            model_labels.append(model_name)
    
    if ess_data and rhat_data:
        x_pos = range(len(model_labels))
        width = 0.35
        
        ax5.bar(x_pos, ess_data, width, color=COLOR_SCHEMES['accent'],
               label='Min ESS', alpha=0.8)
        
        ax5_twin = ax5.twinx()
        ax5_twin.bar([p + width for p in x_pos], rhat_data, width,
                    color=COLOR_SCHEMES['secondary'], label='Max R-hat', alpha=0.8)
        
        ax5.axhline(y=400, color='red', linestyle='--', linewidth=1,
                   label='ESS Threshold')
        ax5_twin.axhline(y=1.01, color='green', linestyle='--', linewidth=1,
                        label='R-hat Threshold')
        
        ax5.set_xticks([p + width/2 for p in x_pos])
        ax5.set_xticklabels(model_labels, rotation=45, ha='right')
        ax5.set_ylabel('Min ESS')
        ax5_twin.set_ylabel('Max R-hat')
        ax5.set_title('Convergence Diagnostics', fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 6: Model fit summary
    ax6 = fig.add_subplot(gs[2, 1:])
    
    fit_metrics = []
    fit_labels = []
    
    for model_name, results in analysis_results.items():
        if hasattr(results, 'model_fit') and 'r_squared' in results.model_fit:
            fit_metrics.append({
                'R²': results.model_fit['r_squared'],
                'WAIC': results.model_fit.get('waic', np.nan),
                'LOO': results.model_fit.get('loo', np.nan)
            })
            fit_labels.append(model_name)
    
    if fit_metrics:
        # Create dataframe for easier plotting
        fit_df = pd.DataFrame(fit_metrics, index=fit_labels)
        
        # Plot multiple metrics
        fit_df.plot(kind='bar', ax=ax6, alpha=0.8, edgecolor='black')
        
        ax6.set_xticklabels(fit_labels, rotation=45, ha='right')
        ax6.set_ylabel('Value')
        ax6.set_title('Model Fit Metrics', fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300,
               formats: List[str] = ['png', 'pdf'], 
               bbox_inches: str = 'tight'):
    """
    Save figure to multiple formats
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    dpi : int
        Resolution for raster formats
    formats : List[str]
        Formats to save ('png', 'pdf', 'svg', 'eps')
    bbox_inches : str
        Bounding box inches setting
    """
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', 
               exist_ok=True)
    
    for fmt in formats:
        save_path = f"{filename}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
        print(f"✓ Saved: {save_path}")


def create_publication_figure(fig: plt.Figure, 
                            title: str,
                            width: str = 'single',
                            font_scale: float = 1.0):
    """
    Format figure for publication
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to format
    title : str
        Figure title
    width : str
        Figure width ('single', 'double', 'full')
    font_scale : float
        Font scaling factor
    """
    # Set figure size based on publication requirements
    if width == 'single':
        fig.set_size_inches(3.5, 2.5)  # Single column
    elif width == 'double':
        fig.set_size_inches(7.2, 4.8)  # Double column
    elif width == 'full':
        fig.set_size_inches(10, 6.5)   # Full page
    
    # Update font sizes
    font_props = {
        'font.size': 8 * font_scale,
        'axes.titlesize': 9 * font_scale,
        'axes.labelsize': 8 * font_scale,
        'xtick.labelsize': 7 * font_scale,
        'ytick.labelsize': 7 * font_scale,
        'legend.fontsize': 7 * font_scale,
        'figure.titlesize': 10 * font_scale
    }
    
    plt.rcParams.update(font_props)
    
    # Update title
    if fig._suptitle is not None:
        fig._suptitle.set_text(title)
        fig._suptitle.set_fontsize(10 * font_scale)
    
    # Adjust layout
    plt.tight_layout(pad=1.0)
    
    return fig


# Example usage function
def create_complete_visualization_report(analysis_results: Dict,
                                        dates: pd.DatetimeIndex,
                                        data: np.ndarray,
                                        output_dir: str = '../reports/figures'):
    """
    Create and save a complete set of visualizations
    
    Parameters:
    -----------
    analysis_results : Dict
        Analysis results from change point models
    dates : pd.DatetimeIndex
        Date index
    data : np.ndarray
        Time series data
    output_dir : str
        Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualization report...")
    
    # 1. Summary dashboard
    print("  Creating summary dashboard...")
    fig_dashboard = create_summary_dashboard(analysis_results, dates, data)
    save_figure(fig_dashboard, os.path.join(output_dir, 'summary_dashboard'))
    plt.close(fig_dashboard)
    
    # 2. Time series with change points for each model
    print("  Creating time series plots...")
    for model_name, results in analysis_results.items():
        if hasattr(results, 'change_points'):
            fig_ts = plot_time_series_with_change_points(
                data, dates, results.change_points,
                title=f'{model_name} - Detected Change Points',
                ylabel='Log Returns (%)'
            )
            save_figure(fig_ts, os.path.join(output_dir, f'{model_name}_time_series'))
            plt.close(fig_ts)
    
    # 3. Posterior distributions for each model
    print("  Creating posterior distribution plots...")
    for model_name, results in analysis_results.items():
        if hasattr(results, 'trace'):
            fig_post = plot_posterior_distributions(
                results.trace,
                title_suffix=f' - {model_name}'
            )
            save_figure(fig_post, os.path.join(output_dir, f'{model_name}_posteriors'))
            plt.close(fig_post)
    
    # 4. Trace diagnostics for each model
    print("  Creating trace diagnostics...")
    for model_name, results in analysis_results.items():
        if hasattr(results, 'trace'):
            fig_trace = plot_trace_diagnostics(results.trace)
            save_figure(fig_trace, os.path.join(output_dir, f'{model_name}_trace_diagnostics'))
            plt.close(fig_trace)
    
    print(f"\n✓ Visualization report saved to {output_dir}/")


if __name__ == '__main__':
    # Example usage
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("- plot_time_series_with_change_points()")
    print("- plot_posterior_distributions()")
    print("- plot_trace_diagnostics()")
    print("- plot_change_point_posterior()")
    print("- plot_regime_comparison()")
    print("- plot_event_impact_association()")
    print("- plot_model_comparison()")
    print("- create_summary_dashboard()")
    print("- save_figure()")
