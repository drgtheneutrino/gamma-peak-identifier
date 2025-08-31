"""
Peak detection algorithms for gamma spectroscopy.

This module provides various methods for detecting peaks in gamma spectra,
including smoothing, background estimation, and peak finding algorithms.
"""

from typing import List, Tuple, Optional, Dict, Any
import warnings

import numpy as np
from scipy.signal import savgol_filter, find_peaks as scipy_find_peaks
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.stats import median_abs_deviation


def smooth_spectrum(counts: np.ndarray, 
                   window_length: int = 5,
                   polyorder: int = 2,
                   method: str = 'savgol') -> np.ndarray:
    """
    Apply smoothing filter to reduce statistical noise.
    
    Parameters:
        counts: Raw counts data
        window_length: Window size for filter
        polyorder: Polynomial order for Savitzky-Golay filter
        method: Smoothing method ('savgol', 'gaussian', 'moving_average', 'none')
        
    Returns:
        Smoothed counts array
    """
    if method == 'none' or window_length <= 1:
        return counts.copy()
    
    # Ensure we have enough points for the filter
    if len(counts) < window_length:
        warnings.warn(f"Spectrum too short for smoothing (length={len(counts)})")
        return counts.copy()
    
    # Ensure window_length is odd for Savitzky-Golay
    if method == 'savgol' and window_length % 2 == 0:
        window_length += 1
    
    try:
        if method == 'savgol':
            # Savitzky-Golay filter preserves peak shape better
            smoothed = savgol_filter(counts, window_length, polyorder, mode='edge')
        elif method == 'gaussian':
            # Gaussian filter with sigma based on window size
            sigma = window_length / 4.0
            smoothed = gaussian_filter1d(counts, sigma, mode='reflect')
        elif method == 'moving_average':
            # Simple moving average
            smoothed = uniform_filter1d(counts, size=window_length, mode='reflect')
        else:
            warnings.warn(f"Unknown smoothing method: {method}, returning original")
            smoothed = counts.copy()
    except Exception as e:
        warnings.warn(f"Smoothing failed: {e}, returning original")
        smoothed = counts.copy()
    
    # Ensure non-negative counts
    smoothed = np.maximum(smoothed, 0)
    
    return smoothed


def estimate_background(counts: np.ndarray, 
                       method: str = 'snip',
                       **kwargs) -> np.ndarray:
    """
    Estimate background in spectrum.
    
    Parameters:
        counts: Spectrum counts
        method: Background estimation method ('snip', 'rolling_ball', 'percentile')
        **kwargs: Method-specific parameters
        
    Returns:
        Estimated background array
    """
    if method == 'snip':
        return snip_background(counts, **kwargs)
    elif method == 'rolling_ball':
        return rolling_ball_background(counts, **kwargs)
    elif method == 'percentile':
        return percentile_background(counts, **kwargs)
    else:
        # Simple baseline estimation
        return np.full_like(counts, np.median(counts))


def snip_background(counts: np.ndarray, 
                    iterations: int = 20,
                    window: int = 20) -> np.ndarray:
    """
    SNIP (Statistics-sensitive Nonlinear Iterative Peak-clipping) background.
    
    Parameters:
        counts: Spectrum counts
        iterations: Number of iterations
        window: Window size for clipping
        
    Returns:
        Background estimate
    """
    # Apply log transform to handle Poisson statistics
    log_counts = np.log(np.maximum(counts, 1))
    background = log_counts.copy()
    
    for p in range(iterations, 0, -1):
        for i in range(p, len(counts) - p):
            # Compare with average of neighbors
            avg = (background[i - p] + background[i + p]) / 2
            background[i] = min(background[i], avg)
    
    # Transform back
    return np.exp(background)


def rolling_ball_background(counts: np.ndarray, radius: int = 50) -> np.ndarray:
    """
    Rolling ball algorithm for background estimation.
    
    Parameters:
        counts: Spectrum counts
        radius: Ball radius in channels
        
    Returns:
        Background estimate
    """
    background = counts.copy()
    
    for i in range(len(counts)):
        # Define window around current point
        start = max(0, i - radius)
        end = min(len(counts), i + radius)
        
        # Find minimum in window
        background[i] = np.min(counts[start:end])
    
    # Smooth the background
    background = gaussian_filter1d(background, sigma=radius/4)
    
    return background


def percentile_background(counts: np.ndarray, 
                         window: int = 100,
                         percentile: float = 10) -> np.ndarray:
    """
    Percentile-based background estimation.
    
    Parameters:
        counts: Spectrum counts
        window: Window size for local estimation
        percentile: Percentile to use (0-100)
        
    Returns:
        Background estimate
    """
    background = np.zeros_like(counts)
    half_window = window // 2
    
    for i in range(len(counts)):
        start = max(0, i - half_window)
        end = min(len(counts), i + half_window)
        
        # Use percentile of local window
        background[i] = np.percentile(counts[start:end], percentile)
    
    return background


def detect_peaks(counts: np.ndarray,
                min_prominence: float = 50,
                min_height: float = 10,
                min_distance: int = 3,
                method: str = 'scipy',
                **kwargs) -> np.ndarray:
    """
    Detect peaks in spectrum using various methods.
    
    Parameters:
        counts: Spectrum counts (preferably smoothed)
        min_prominence: Minimum peak prominence
        min_height: Minimum peak height
        min_distance: Minimum distance between peaks
        method: Detection method ('scipy', 'derivative', 'template')
        **kwargs: Additional method-specific parameters
        
    Returns:
        Array of peak indices
    """
    if method == 'scipy':
        return detect_peaks_scipy(counts, min_prominence, min_height, min_distance, **kwargs)
    elif method == 'derivative':
        return detect_peaks_derivative(counts, min_height, min_distance, **kwargs)
    elif method == 'template':
        return detect_peaks_template(counts, min_height, **kwargs)
    else:
        warnings.warn(f"Unknown detection method: {method}, using scipy")
        return detect_peaks_scipy(counts, min_prominence, min_height, min_distance)


def detect_peaks_scipy(counts: np.ndarray,
                       min_prominence: float = 50,
                       min_height: float = 10,
                       min_distance: int = 3,
                       rel_height: float = 0.5) -> np.ndarray:
    """
    Detect peaks using scipy's find_peaks with statistical filtering.
    
    Parameters:
        counts: Spectrum counts
        min_prominence: Minimum peak prominence
        min_height: Minimum peak height  
        min_distance: Minimum distance between peaks
        rel_height: Relative height for width calculation
        
    Returns:
        Array of peak indices
    """
    # Initial peak detection
    peaks, properties = scipy_find_peaks(
        counts,
        height=min_height,
        prominence=min_prominence,
        distance=min_distance,
        width=1,  # Minimum width
        rel_height=rel_height
    )
    
    if len(peaks) == 0:
        return np.array([])
    
    # Statistical filtering based on local noise
    filtered_peaks = []
    
    for i, peak in enumerate(peaks):
        # Define region around peak for noise estimation
        window = max(20, int(properties.get('widths', [10])[i] * 3))
        start = max(0, peak - window)
        end = min(len(counts), peak + window)
        
        # Exclude peak region for background estimation
        peak_start = max(0, peak - 3)
        peak_end = min(len(counts), peak + 3)
        
        # Get background region
        bg_region = np.concatenate([counts[start:peak_start], counts[peak_end:end]])
        
        if len(bg_region) > 5:
            # Estimate local background and noise
            background = np.median(bg_region)
            noise = median_abs_deviation(bg_region, scale='normal')
            
            # Check if peak is significant (default 3-sigma, but use prominence)
            threshold = background + max(3 * noise, min_prominence)
            
            if counts[peak] > threshold:
                filtered_peaks.append(peak)
        else:
            # Not enough data for statistical test, keep peak
            filtered_peaks.append(peak)
    
    return np.array(filtered_peaks, dtype=int)


def detect_peaks_derivative(counts: np.ndarray,
                           min_height: float = 10,
                           min_distance: int = 3,
                           smooth_derivative: bool = True) -> np.ndarray:
    """
    Detect peaks using derivative method (zero-crossings).
    
    Parameters:
        counts: Spectrum counts
        min_height: Minimum peak height
        min_distance: Minimum distance between peaks
        smooth_derivative: Whether to smooth the derivative
        
    Returns:
        Array of peak indices
    """
    # Calculate first derivative
    derivative = np.gradient(counts)
    
    if smooth_derivative:
        # Smooth the derivative to reduce noise
        derivative = savgol_filter(derivative, window_length=5, polyorder=2)
    
    # Find zero crossings (from positive to negative)
    zero_crossings = []
    for i in range(1, len(derivative) - 1):
        if derivative[i-1] > 0 and derivative[i+1] < 0:
            # Check if it's a peak (not just noise)
            if counts[i] > min_height:
                zero_crossings.append(i)
    
    peaks = np.array(zero_crossings, dtype=int)
    
    # Apply minimum distance constraint
    if len(peaks) > 1 and min_distance > 1:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        peaks = np.array(filtered_peaks, dtype=int)
    
    return peaks


def detect_peaks_template(counts: np.ndarray,
                         min_height: float = 10,
                         template_width: int = 10) -> np.ndarray:
    """
    Detect peaks using template matching with Gaussian template.
    
    Parameters:
        counts: Spectrum counts
        min_height: Minimum peak height
        template_width: Width of Gaussian template
        
    Returns:
        Array of peak indices
    """
    from scipy.signal import correlate
    
    # Create Gaussian template
    x = np.arange(-template_width, template_width + 1)
    template = np.exp(-x**2 / (2 * (template_width/3)**2))
    template = template / np.sum(template)
    
    # Correlate with template
    correlation = correlate(counts, template, mode='same')
    
    # Find peaks in correlation
    peaks = detect_peaks_scipy(correlation, min_height=min_height)
    
    return peaks


def refine_peak_positions(counts: np.ndarray, 
                         initial_peaks: np.ndarray,
                         method: str = 'centroid',
                         window: int = 5) -> np.ndarray:
    """
    Refine peak positions using various methods.
    
    Parameters:
        counts: Spectrum counts
        initial_peaks: Initial peak indices
        method: Refinement method ('centroid', 'parabolic', 'gaussian')
        window: Window size around peak
        
    Returns:
        Refined peak positions (can be fractional)
    """
    refined_peaks = []
    
    for peak in initial_peaks:
        # Define window around peak
        start = max(0, peak - window)
        end = min(len(counts), peak + window + 1)
        
        if method == 'centroid':
            # Centroid method
            indices = np.arange(start, end)
            weights = counts[start:end]
            if np.sum(weights) > 0:
                centroid = np.sum(indices * weights) / np.sum(weights)
                refined_peaks.append(centroid)
            else:
                refined_peaks.append(float(peak))
                
        elif method == 'parabolic':
            # Parabolic interpolation
            if peak > 0 and peak < len(counts) - 1:
                y1 = counts[peak - 1]
                y2 = counts[peak]
                y3 = counts[peak + 1]
                
                if y2 > y1 and y2 > y3:
                    # Parabolic peak position
                    delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                    refined_peaks.append(peak + delta)
                else:
                    refined_peaks.append(float(peak))
            else:
                refined_peaks.append(float(peak))
                
        elif method == 'gaussian':
            # Gaussian fit (simplified)
            indices = np.arange(start, end)
            weights = counts[start:end]
            
            if len(weights) > 3 and np.sum(weights) > 0:
                # Log transform for Gaussian
                log_weights = np.log(np.maximum(weights, 1e-10))
                
                # Fit parabola to log (equivalent to Gaussian)
                try:
                    coeffs = np.polyfit(indices, log_weights, 2)
                    if coeffs[0] < 0:  # Must be concave down
                        peak_pos = -coeffs[1] / (2 * coeffs[0])
                        if start <= peak_pos <= end:
                            refined_peaks.append(peak_pos)
                        else:
                            refined_peaks.append(float(peak))
                    else:
                        refined_peaks.append(float(peak))
                except:
                    refined_peaks.append(float(peak))
            else:
                refined_peaks.append(float(peak))
        else:
            refined_peaks.append(float(peak))
    
    return np.array(refined_peaks)


def group_peaks(peaks: np.ndarray, 
                min_separation: float = 5.0) -> List[List[int]]:
    """
    Group nearby peaks that might be multiplets.
    
    Parameters:
        peaks: Array of peak indices
        min_separation: Minimum separation to consider peaks separate
        
    Returns:
        List of peak groups
    """
    if len(peaks) == 0:
        return []
    
    # Sort peaks
    sorted_peaks = np.sort(peaks)
    
    # Group peaks
    groups = [[sorted_peaks[0]]]
    
    for peak in sorted_peaks[1:]:
        if peak - groups[-1][-1] <= min_separation:
            groups[-1].append(peak)
        else:
            groups.append([peak])
    
    return groups


def estimate_peak_width(counts: np.ndarray, peak_idx: int) -> float:
    """
    Estimate FWHM of a peak.
    
    Parameters:
        counts: Spectrum counts
        peak_idx: Peak index
        
    Returns:
        Estimated FWHM in channels
    """
    peak_height = counts[peak_idx]
    half_max = peak_height / 2
    
    # Search left
    left_idx = peak_idx
    while left_idx > 0 and counts[left_idx] > half_max:
        left_idx -= 1
    
    # Search right  
    right_idx = peak_idx
    while right_idx < len(counts) - 1 and counts[right_idx] > half_max:
        right_idx += 1
    
    # Calculate FWHM
    fwhm = right_idx - left_idx
    
    # Sanity check
    if fwhm < 1:
        fwhm = 1.0
    elif fwhm > len(counts) / 4:
        fwhm = 5.0  # Default for very wide peaks
    
    return float(fwhm)
