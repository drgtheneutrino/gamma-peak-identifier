"""
Peak fitting algorithms for gamma spectroscopy.

This module provides functions for fitting detected peaks with various
peak shape models (Gaussian, Voigt, etc.) and background models.
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
import warnings

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf, voigt_profile
from scipy.stats import chi2


# Peak shape functions

def gaussian(x: np.ndarray, amplitude: float, centroid: float, sigma: float) -> np.ndarray:
    """
    Gaussian peak function.
    
    Parameters:
        x: Channel numbers
        amplitude: Peak amplitude
        centroid: Peak center position
        sigma: Peak width parameter (std dev)
        
    Returns:
        Gaussian peak values
    """
    return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2)


def gaussian_with_background(x: np.ndarray, 
                            amplitude: float, 
                            centroid: float, 
                            sigma: float,
                            bg_slope: float, 
                            bg_intercept: float) -> np.ndarray:
    """
    Gaussian peak with linear background.
    
    Parameters:
        x: Channel numbers
        amplitude: Peak amplitude
        centroid: Peak center position
        sigma: Peak width parameter
        bg_slope: Background slope
        bg_intercept: Background intercept
        
    Returns:
        Model values (peak + background)
    """
    peak = gaussian(x, amplitude, centroid, sigma)
    background = bg_slope * x + bg_intercept
    return peak + background


def gaussian_with_tail(x: np.ndarray,
                       amplitude: float,
                       centroid: float,
                       sigma: float,
                       tail_amp: float,
                       tail_slope: float) -> np.ndarray:
    """
    Gaussian peak with low-energy tail (for detector effects).
    
    Parameters:
        x: Channel numbers
        amplitude: Peak amplitude
        centroid: Peak center position
        sigma: Peak width parameter
        tail_amp: Tail amplitude
        tail_slope: Tail decay parameter
        
    Returns:
        Peak values with tail
    """
    peak = gaussian(x, amplitude, centroid, sigma)
    
    # Add exponential tail on low-energy side
    tail = np.zeros_like(x, dtype=float)
    mask = x < centroid
    if np.any(mask):
        tail[mask] = tail_amp * np.exp((x[mask] - centroid) / tail_slope)
    
    return peak + tail


def voigt(x: np.ndarray,
          amplitude: float,
          centroid: float,
          sigma: float,
          gamma: float) -> np.ndarray:
    """
    Voigt profile (convolution of Gaussian and Lorentzian).
    
    Parameters:
        x: Channel numbers
        amplitude: Peak amplitude
        centroid: Peak center position
        sigma: Gaussian width parameter
        gamma: Lorentzian width parameter
        
    Returns:
        Voigt profile values
    """
    z = (x - centroid + 1j * gamma) / (sigma * np.sqrt(2))
    profile = voigt_profile(z.imag, z.real)
    return amplitude * profile / np.max(profile)


def double_gaussian(x: np.ndarray,
                   amp1: float, cent1: float, sig1: float,
                   amp2: float, cent2: float, sig2: float,
                   bg_slope: float, bg_intercept: float) -> np.ndarray:
    """
    Double Gaussian for fitting overlapping peaks.
    
    Parameters:
        x: Channel numbers
        amp1, cent1, sig1: First peak parameters
        amp2, cent2, sig2: Second peak parameters
        bg_slope, bg_intercept: Background parameters
        
    Returns:
        Model values
    """
    peak1 = gaussian(x, amp1, cent1, sig1)
    peak2 = gaussian(x, amp2, cent2, sig2)
    background = bg_slope * x + bg_intercept
    return peak1 + peak2 + background


# Background models

def linear_background(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Linear background model."""
    return slope * x + intercept


def quadratic_background(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Quadratic background model."""
    return a * x**2 + b * x + c


def step_background(x: np.ndarray, 
                   height: float, 
                   position: float,
                   width: float,
                   baseline: float) -> np.ndarray:
    """
    Step function background (for Compton edge).
    
    Parameters:
        x: Channel numbers
        height: Step height
        position: Step position
        width: Step width (smoothing)
        baseline: Baseline level
        
    Returns:
        Step function values
    """
    # Use error function for smooth step
    step = 0.5 * (1 + erf((x - position) / width))
    return baseline + height * step


# Main fitting functions

def estimate_initial_params(channels: np.ndarray,
                          counts: np.ndarray,
                          peak_idx: int,
                          peak_model: str = 'gaussian') -> Tuple[tuple, Tuple[int, int]]:
    """
    Estimate initial parameters for peak fitting.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_idx: Index of peak maximum
        peak_model: Peak model type
        
    Returns:
        tuple: (initial_parameters, (fit_start, fit_end))
    """
    # Get peak properties
    peak_channel = channels[peak_idx]
    peak_height = counts[peak_idx]
    
    # Estimate FWHM
    half_max = peak_height / 2
    
    # Search for half-maximum points
    left_idx = peak_idx
    while left_idx > 0 and counts[left_idx] > half_max:
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < len(counts) - 1 and counts[right_idx] > half_max:
        right_idx += 1
    
    # Calculate FWHM and sigma
    fwhm_channels = right_idx - left_idx
    if fwhm_channels < 1:
        fwhm_channels = 3  # Minimum reasonable width
    
    fwhm = channels[min(right_idx, len(channels)-1)] - channels[max(left_idx, 0)]
    sigma_estimate = fwhm / 2.355  # FWHM = 2.355 * sigma for Gaussian
    
    # Define fitting region (default 3x FWHM on each side)
    fit_width = max(int(3 * fwhm_channels), 10)
    fit_start = max(0, peak_idx - fit_width)
    fit_end = min(len(channels), peak_idx + fit_width)
    
    # Estimate background from edges of fit region
    edge_points = 5
    left_bg = counts[fit_start:fit_start + edge_points]
    right_bg = counts[max(fit_end - edge_points, fit_start + edge_points):fit_end]
    
    if len(left_bg) > 0 and len(right_bg) > 0:
        bg_level = (np.mean(left_bg) + np.mean(right_bg)) / 2
        bg_slope = (np.mean(right_bg) - np.mean(left_bg)) / (channels[fit_end-1] - channels[fit_start])
    else:
        bg_level = np.min(counts[fit_start:fit_end])
        bg_slope = 0
    
    # Peak amplitude above background
    amplitude_estimate = peak_height - bg_level
    
    # Build initial parameters based on model
    if peak_model == 'gaussian':
        initial_params = (amplitude_estimate, peak_channel, sigma_estimate,
                         bg_slope, bg_level)
    elif peak_model == 'gaussian_tail':
        initial_params = (amplitude_estimate, peak_channel, sigma_estimate,
                         amplitude_estimate * 0.1, sigma_estimate * 2,
                         bg_slope, bg_level)
    elif peak_model == 'voigt':
        initial_params = (amplitude_estimate, peak_channel, sigma_estimate,
                         sigma_estimate * 0.5, bg_slope, bg_level)
    else:
        initial_params = (amplitude_estimate, peak_channel, sigma_estimate,
                         bg_slope, bg_level)
    
    return initial_params, (fit_start, fit_end)


def fit_single_peak(channels: np.ndarray,
                   counts: np.ndarray,
                   peak_idx: int,
                   peak_model: str = 'gaussian',
                   background_method: str = 'linear',
                   window_scale: float = 3.0,
                   max_iterations: int = 5000) -> Dict[str, Any]:
    """
    Fit a single peak with specified model.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_idx: Index of peak maximum
        peak_model: Peak model ('gaussian', 'gaussian_tail', 'voigt')
        background_method: Background model ('linear', 'quadratic', 'step')
        window_scale: Scale factor for fitting window
        max_iterations: Maximum fitting iterations
        
    Returns:
        Dictionary with fitted parameters
    """
    # Get initial parameters
    initial_params, (fit_start, fit_end) = estimate_initial_params(
        channels, counts, peak_idx, peak_model
    )
    
    # Adjust window scale
    if window_scale != 3.0:
        current_width = fit_end - fit_start
        new_width = int(current_width * window_scale / 3.0)
        center = (fit_start + fit_end) // 2
        fit_start = max(0, center - new_width // 2)
        fit_end = min(len(channels), center + new_width // 2)
    
    # Extract fitting region
    x_fit = channels[fit_start:fit_end]
    y_fit = counts[fit_start:fit_end]
    
    # Calculate weights (Poisson statistics)
    weights = 1.0 / np.sqrt(np.maximum(y_fit, 1))
    
    # Select fitting function
    if peak_model == 'gaussian' and background_method == 'linear':
        fit_func = gaussian_with_background
        param_names = ['amplitude', 'centroid', 'sigma', 'bg_slope', 'bg_intercept']
    else:
        # Build composite function
        fit_func, param_names = build_composite_function(peak_model, background_method)
    
    # Perform fitting
    try:
        # Set bounds
        bounds = get_parameter_bounds(x_fit, y_fit, peak_model, initial_params)
        
        # Fit with scipy.curve_fit
        popt, pcov = curve_fit(
            fit_func,
            x_fit, 
            y_fit,
            p0=initial_params,
            sigma=weights,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=max_iterations
        )
        
        # Calculate uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate fitted values and residuals
        y_fitted = fit_func(x_fit, *popt)
        residuals = y_fit - y_fitted
        
        # Calculate chi-square
        chi_square = np.sum((residuals * weights) ** 2)
        dof = len(y_fit) - len(popt)  # Degrees of freedom
        reduced_chi_square = chi_square / dof if dof > 0 else np.inf
        
        # Extract peak parameters
        if peak_model in ['gaussian', 'gaussian_tail', 'voigt']:
            amplitude = popt[0]
            centroid = popt[1]
            sigma = popt[2]
        else:
            amplitude = popt[0]
            centroid = popt[1] 
            sigma = popt[2]
        
        # Calculate derived quantities
        fwhm = 2.355 * sigma  # For Gaussian
        
        # Calculate peak area (integral of peak without background)
        if peak_model == 'gaussian':
            area = amplitude * sigma * np.sqrt(2 * np.pi)
            area_err = area * np.sqrt((perr[0]/amplitude)**2 + (perr[2]/sigma)**2)
        else:
            # Numerical integration for other models
            peak_only = fit_func(x_fit, *popt)
            if background_method == 'linear':
                peak_only -= (popt[-2] * x_fit + popt[-1])
            area = np.trapz(peak_only, x_fit)
            area_err = np.sqrt(area)  # Rough estimate
        
        # Calculate SNR
        noise = np.std(residuals)
        snr = amplitude / noise if noise > 0 else np.inf
        
        # Calculate resolution
        resolution = fwhm / centroid * 100 if centroid > 0 else 0
        
        # Success flag
        fit_success = True
        fit_message = "Fit converged successfully"
        
    except Exception as e:
        # Fitting failed - return estimates
        warnings.warn(f"Peak fitting failed: {e}")
        
        fit_success = False
        fit_message = str(e)
        
        # Use initial estimates
        centroid = channels[peak_idx]
        amplitude = counts[peak_idx]
        sigma = 2.0
        fwhm = 2.355 * sigma
        area = amplitude * 5  # Rough estimate
        area_err = np.sqrt(area)
        snr = amplitude / np.sqrt(amplitude) if amplitude > 0 else 0
        resolution = fwhm / centroid * 100 if centroid > 0 else 0
        
        popt = None
        perr = None
        reduced_chi_square = np.inf
        residuals = None
    
    # Build result dictionary
    result = {
        'centroid': centroid,
        'centroid_err': perr[1] if perr is not None else 1.0,
        'amplitude': amplitude,
        'amplitude_err': perr[0] if perr is not None else np.sqrt(amplitude),
        'sigma': sigma,
        'sigma_err': perr[2] if perr is not None else 0.5,
        'area': area,
        'area_err': area_err,
        'fwhm': fwhm,
        'fwhm_err': 2.355 * (perr[2] if perr is not None else 0.5),
        'snr': snr,
        'resolution': resolution,
        'chi_square': reduced_chi_square,
        'fit_success': fit_success,
        'fit_message': fit_message,
        'fit_params': popt,
        'fit_errors': perr,
        'fit_region': (fit_start, fit_end),
        'peak_model': peak_model,
        'background_method': background_method
    }
    
    # Add background parameters if available
    if popt is not None and background_method == 'linear':
        result['bg_slope'] = popt[-2]
        result['bg_intercept'] = popt[-1]
    
    return result


def fit_peaks(channels: np.ndarray,
             counts: np.ndarray,
             peak_indices: np.ndarray,
             peak_model: str = 'gaussian',
             background_method: str = 'linear',
             window_scale: float = 3.0,
             parallel: bool = False) -> List[Dict[str, Any]]:
    """
    Fit all detected peaks.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_indices: Indices of detected peaks
        peak_model: Peak model to use
        background_method: Background model
        window_scale: Scale factor for fitting window
        parallel: Whether to use parallel processing
        
    Returns:
        List of fitted peak dictionaries
    """
    fitted_peaks = []
    
    # Check for overlapping peaks
    peak_groups = identify_overlapping_peaks(channels, counts, peak_indices, window_scale)
    
    for group in peak_groups:
        if len(group) == 1:
            # Single peak - standard fitting
            peak_result = fit_single_peak(
                channels, counts, group[0],
                peak_model=peak_model,
                background_method=background_method,
                window_scale=window_scale
            )
            fitted_peaks.append(peak_result)
        else:
            # Multiple overlapping peaks - fit together
            multiplet_results = fit_multiplet(
                channels, counts, group,
                peak_model=peak_model,
                background_method=background_method
            )
            fitted_peaks.extend(multiplet_results)
    
    return fitted_peaks


def fit_multiplet(channels: np.ndarray,
                 counts: np.ndarray,
                 peak_indices: List[int],
                 peak_model: str = 'gaussian',
                 background_method: str = 'linear') -> List[Dict[str, Any]]:
    """
    Fit multiple overlapping peaks simultaneously.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_indices: List of peak indices in multiplet
        peak_model: Peak model to use
        background_method: Background model
        
    Returns:
        List of fitted peak dictionaries
    """
    n_peaks = len(peak_indices)
    
    if n_peaks == 2 and peak_model == 'gaussian' and background_method == 'linear':
        # Use optimized double Gaussian
        return fit_double_gaussian(channels, counts, peak_indices)
    
    # General multiplet fitting
    # Define fitting region
    fit_start = max(0, min(peak_indices) - 20)
    fit_end = min(len(channels), max(peak_indices) + 20)
    
    x_fit = channels[fit_start:fit_end]
    y_fit = counts[fit_start:fit_end]
    weights = 1.0 / np.sqrt(np.maximum(y_fit, 1))
    
    # Build initial parameters
    initial_params = []
    for peak_idx in peak_indices:
        rel_idx = peak_idx - fit_start
        if 0 <= rel_idx < len(y_fit):
            initial_params.extend([
                y_fit[rel_idx],  # amplitude
                x_fit[rel_idx],  # centroid
                2.0  # sigma
            ])
    
    # Add background parameters
    if background_method == 'linear':
        bg_level = np.min(y_fit)
        bg_slope = 0
        initial_params.extend([bg_slope, bg_level])
    
    # Create multiplet function
    def multiplet_function(x, *params):
        result = np.zeros_like(x)
        
        # Add each peak
        for i in range(n_peaks):
            amp = params[3*i]
            cent = params[3*i + 1]
            sig = params[3*i + 2]
            result += gaussian(x, amp, cent, sig)
        
        # Add background
        if background_method == 'linear':
            result += params[-2] * x + params[-1]
        
        return result
    
    # Fit the multiplet
    try:
        popt, pcov = curve_fit(
            multiplet_function,
            x_fit, y_fit,
            p0=initial_params,
            sigma=weights,
            absolute_sigma=True,
            maxfev=5000
        )
        
        perr = np.sqrt(np.diag(pcov))
        fit_success = True
        
    except Exception as e:
        warnings.warn(f"Multiplet fitting failed: {e}")
        fit_success = False
        popt = initial_params
        perr = np.ones_like(initial_params)
    
    # Extract individual peak results
    results = []
    for i in range(n_peaks):
        amp = popt[3*i]
        cent = popt[3*i + 1]
        sig = popt[3*i + 2]
        
        result = {
            'centroid': cent,
            'centroid_err': perr[3*i + 1] if fit_success else 1.0,
            'amplitude': amp,
            'amplitude_err': perr[3*i] if fit_success else np.sqrt(amp),
            'sigma': sig,
            'sigma_err': perr[3*i + 2] if fit_success else 0.5,
            'area': amp * sig * np.sqrt(2 * np.pi),
            'area_err': amp * sig * np.sqrt(2 * np.pi) * 0.1,  # Rough estimate
            'fwhm': 2.355 * sig,
            'fwhm_err': 2.355 * perr[3*i + 2] if fit_success else 1.0,
            'snr': amp / np.std(y_fit - multiplet_function(x_fit, *popt)) if fit_success else 0,
            'resolution': 2.355 * sig / cent * 100 if cent > 0 else 0,
            'fit_success': fit_success,
            'fit_region': (fit_start, fit_end),
            'multiplet': True,
            'multiplet_size': n_peaks
        }
        
        results.append(result)
    
    return results


def fit_double_gaussian(channels: np.ndarray,
                       counts: np.ndarray,
                       peak_indices: List[int]) -> List[Dict[str, Any]]:
    """
    Specialized fitting for double Gaussian peaks.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_indices: List of two peak indices
        
    Returns:
        List of two fitted peak dictionaries
    """
    if len(peak_indices) != 2:
        raise ValueError("fit_double_gaussian requires exactly 2 peaks")
    
    # Define fitting region
    fit_start = max(0, min(peak_indices) - 15)
    fit_end = min(len(channels), max(peak_indices) + 15)
    
    x_fit = channels[fit_start:fit_end]
    y_fit = counts[fit_start:fit_end]
    
    # Initial parameters
    idx1 = peak_indices[0] - fit_start
    idx2 = peak_indices[1] - fit_start
    
    initial_params = [
        y_fit[idx1], x_fit[idx1], 2.0,  # Peak 1
        y_fit[idx2], x_fit[idx2], 2.0,  # Peak 2
        0, np.min(y_fit)  # Background
    ]
    
    # Fit
    try:
        popt, pcov = curve_fit(
            double_gaussian,
            x_fit, y_fit,
            p0=initial_params,
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov))
        fit_success = True
    except:
        fit_success = False
        popt = initial_params
        perr = np.ones_like(initial_params)
    
    # Build results
    results = []
    for i in range(2):
        base_idx = 3 * i
        amp = popt[base_idx]
        cent = popt[base_idx + 1]
        sig = popt[base_idx + 2]
        
        result = {
            'centroid': cent,
            'centroid_err': perr[base_idx + 1] if fit_success else 1.0,
            'amplitude': amp,
            'amplitude_err': perr[base_idx] if fit_success else np.sqrt(amp),
            'sigma': sig,
            'sigma_err': perr[base_idx + 2] if fit_success else 0.5,
            'area': amp * sig * np.sqrt(2 * np.pi),
            'area_err': amp * sig * np.sqrt(2 * np.pi) * 0.1,
            'fwhm': 2.355 * sig,
            'fwhm_err': 2.355 * perr[base_idx + 2] if fit_success else 1.0,
            'snr': amp / np.std(y_fit) if np.std(y_fit) > 0 else 0,
            'resolution': 2.355 * sig / cent * 100 if cent > 0 else 0,
            'fit_success': fit_success,
            'fit_region': (fit_start, fit_end),
            'multiplet': True,
            'multiplet_size': 2,
            'bg_slope': popt[-2],
            'bg_intercept': popt[-1]
        }
        results.append(result)
    
    return results


def identify_overlapping_peaks(channels: np.ndarray,
                              counts: np.ndarray,
                              peak_indices: np.ndarray,
                              window_scale: float = 3.0) -> List[List[int]]:
    """
    Identify groups of overlapping peaks.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_indices: Array of peak indices
        window_scale: Scale factor for determining overlap
        
    Returns:
        List of peak groups (each group contains indices of overlapping peaks)
    """
    if len(peak_indices) == 0:
        return []
    
    # Sort peaks by position
    sorted_indices = np.argsort(peak_indices)
    sorted_peaks = peak_indices[sorted_indices]
    
    # Estimate fitting windows for each peak
    windows = []
    for peak_idx in sorted_peaks:
        # Estimate FWHM
        half_max = counts[peak_idx] / 2
        left = peak_idx
        while left > 0 and counts[left] > half_max:
            left -= 1
        right = peak_idx
        while right < len(counts) - 1 and counts[right] > half_max:
            right += 1
        
        fwhm_channels = right - left
        if fwhm_channels < 1:
            fwhm_channels = 3
        
        window_width = int(window_scale * fwhm_channels)
        windows.append((peak_idx - window_width, peak_idx + window_width))
    
    # Group overlapping peaks
    groups = []
    current_group = [sorted_peaks[0]]
    current_window = windows[0]
    
    for i in range(1, len(sorted_peaks)):
        peak = sorted_peaks[i]
        window = windows[i]
        
        # Check if this peak overlaps with current group
        if window[0] <= current_window[1]:
            # Overlapping - add to current group
            current_group.append(peak)
            # Extend the group window
            current_window = (min(current_window[0], window[0]),
                            max(current_window[1], window[1]))
        else:
            # No overlap - start new group
            groups.append(current_group)
            current_group = [peak]
            current_window = window
    
    # Add the last group
    groups.append(current_group)
    
    return groups


def build_composite_function(peak_model: str, 
                            background_method: str) -> Tuple[Callable, List[str]]:
    """
    Build a composite fitting function from peak and background models.
    
    Parameters:
        peak_model: Peak model name
        background_method: Background model name
        
    Returns:
        tuple: (function, parameter_names)
    """
    # This is a simplified version - in practice, you'd build more complex combinations
    if peak_model == 'gaussian' and background_method == 'quadratic':
        def func(x, amp, cent, sig, a, b, c):
            return gaussian(x, amp, cent, sig) + quadratic_background(x, a, b, c)
        param_names = ['amplitude', 'centroid', 'sigma', 'bg_a', 'bg_b', 'bg_c']
    else:
        # Default to Gaussian with linear background
        func = gaussian_with_background
        param_names = ['amplitude', 'centroid', 'sigma', 'bg_slope', 'bg_intercept']
    
    return func, param_names


def get_parameter_bounds(x_fit: np.ndarray,
                        y_fit: np.ndarray,
                        peak_model: str,
                        initial_params: tuple) -> Tuple[list, list]:
    """
    Get parameter bounds for fitting.
    
    Parameters:
        x_fit: Fitting region channels
        y_fit: Fitting region counts
        peak_model: Peak model name
        initial_params: Initial parameter estimates
        
    Returns:
        tuple: (lower_bounds, upper_bounds)
    """
    if peak_model == 'gaussian':
        lower = [0, x_fit[0], 0.1, -np.inf, 0]
        upper = [np.inf, x_fit[-1], (x_fit[-1]-x_fit[0])/2, np.inf, np.inf]
    else:
        # Generic bounds
        n_params = len(initial_params)
        lower = [-np.inf] * n_params
        upper = [np.inf] * n_params
        
        # Set some reasonable bounds
        lower[0] = 0  # Amplitude must be positive
        if n_params > 1:
            lower[1] = x_fit[0]  # Centroid within fit range
            upper[1] = x_fit[-1]
        if n_params > 2:
            lower[2] = 0.1  # Sigma must be positive
            upper[2] = (x_fit[-1] - x_fit[0]) / 2
    
    return lower, upper
