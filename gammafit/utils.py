"""
Utility functions for gamma spectroscopy analysis.

This module provides helper functions for logging, validation,
data processing, and other common operations.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings
import hashlib
import json

import numpy as np
from scipy.stats import poisson, chi2
from scipy.signal import find_peaks as scipy_find_peaks


def setup_logger(name: str = 'gammafit',
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Parameters:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_parameters(config: Dict[str, Any]) -> bool:
    """
    Validate analysis parameters.
    
    Parameters:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate detection parameters
    detection = config.get('detection', {})
    
    if detection.get('min_prominence', 50) <= 0:
        raise ValueError("min_prominence must be positive")
    
    if detection.get('min_height', 10) <= 0:
        raise ValueError("min_height must be positive")
    
    if detection.get('min_distance', 3) < 1:
        raise ValueError("min_distance must be at least 1")
    
    window = detection.get('smoothing_window', 5)
    if window < 0 or (window > 0 and window < 3):
        raise ValueError("smoothing_window must be 0 (disabled) or >= 3")
    
    # Validate fitting parameters
    fitting = config.get('fitting', {})
    
    if fitting.get('window_scale', 3.0) <= 0:
        raise ValueError("window_scale must be positive")
    
    valid_bg = ['linear', 'quadratic', 'step']
    if fitting.get('background_method', 'linear') not in valid_bg:
        raise ValueError(f"background_method must be one of {valid_bg}")
    
    # Validate calibration if present
    if 'calibration' in config and config['calibration']:
        if isinstance(config['calibration'], str):
            # Parse string calibration
            try:
                parts = config['calibration'].split(',')
                if len(parts) != 2:
                    raise ValueError
                float(parts[0])
                float(parts[1])
            except:
                raise ValueError("calibration must be 'a,b' format")
    
    return True


def estimate_fwhm(counts: np.ndarray, peak_idx: int) -> float:
    """
    Estimate Full Width at Half Maximum of a peak.
    
    Parameters:
        counts: Spectrum counts
        peak_idx: Peak index
        
    Returns:
        Estimated FWHM in channels
    """
    if peak_idx < 0 or peak_idx >= len(counts):
        return 3.0  # Default
    
    peak_height = counts[peak_idx]
    half_max = peak_height / 2
    
    # Search left
    left_idx = peak_idx
    while left_idx > 0 and counts[left_idx] > half_max:
        left_idx -= 1
    
    # Interpolate for better precision
    if left_idx > 0:
        y1 = counts[left_idx]
        y2 = counts[left_idx + 1]
        frac = (half_max - y1) / (y2 - y1)
        left_pos = left_idx + frac
    else:
        left_pos = left_idx
    
    # Search right
    right_idx = peak_idx
    while right_idx < len(counts) - 1 and counts[right_idx] > half_max:
        right_idx += 1
    
    # Interpolate
    if right_idx < len(counts) - 1:
        y1 = counts[right_idx - 1]
        y2 = counts[right_idx]
        frac = (half_max - y1) / (y2 - y1)
        right_pos = right_idx - 1 + frac
    else:
        right_pos = right_idx
    
    fwhm = right_pos - left_pos
    
    # Sanity check
    if fwhm < 1:
        fwhm = 1.0
    elif fwhm > len(counts) / 4:
        fwhm = 5.0  # Default for very wide peaks
    
    return float(fwhm)


def calculate_snr(peak_height: float, 
                 background: float,
                 background_std: Optional[float] = None) -> float:
    """
    Calculate signal-to-noise ratio for a peak.
    
    Parameters:
        peak_height: Peak height above background
        background: Background level
        background_std: Background standard deviation
        
    Returns:
        Signal-to-noise ratio
    """
    if background_std is None:
        # Assume Poisson statistics
        background_std = np.sqrt(max(background, 1))
    
    if background_std == 0:
        return float('inf') if peak_height > 0 else 0
    
    return peak_height / background_std


def find_peak_regions(counts: np.ndarray,
                     peaks: np.ndarray,
                     scale: float = 3.0) -> List[Tuple[int, int]]:
    """
    Find regions around peaks for fitting.
    
    Parameters:
        counts: Spectrum counts
        peaks: Peak indices
        scale: Scale factor for region width (in units of FWHM)
        
    Returns:
        List of (start, end) tuples for each peak
    """
    regions = []
    
    for peak_idx in peaks:
        # Estimate FWHM
        fwhm = estimate_fwhm(counts, peak_idx)
        
        # Define region
        half_width = int(scale * fwhm / 2)
        start = max(0, peak_idx - half_width)
        end = min(len(counts), peak_idx + half_width)
        
        regions.append((start, end))
    
    return regions


def merge_overlapping_regions(regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping regions.
    
    Parameters:
        regions: List of (start, end) tuples
        
    Returns:
        Merged regions
    """
    if not regions:
        return []
    
    # Sort by start position
    sorted_regions = sorted(regions, key=lambda x: x[0])
    
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        last = merged[-1]
        
        # Check for overlap
        if current[0] <= last[1]:
            # Merge
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap
            merged.append(current)
    
    return merged


def calculate_counting_statistics(counts: np.ndarray) -> Dict[str, Any]:
    """
    Calculate counting statistics for spectrum.
    
    Parameters:
        counts: Spectrum counts
        
    Returns:
        Dictionary with statistical measures
    """
    total_counts = np.sum(counts)
    non_zero = counts[counts > 0]
    
    stats = {
        'total_counts': total_counts,
        'channels': len(counts),
        'non_zero_channels': len(non_zero),
        'mean_counts': np.mean(counts),
        'std_counts': np.std(counts),
        'min_counts': np.min(counts),
        'max_counts': np.max(counts),
        'median_counts': np.median(counts)
    }
    
    # Estimate dead time if possible (simplified)
    if len(non_zero) > 0:
        # Check for deviation from Poisson statistics
        expected_std = np.sqrt(np.mean(non_zero))
        actual_std = np.std(non_zero)
        stats['poisson_deviation'] = (actual_std - expected_std) / expected_std
    
    return stats


def smooth_spectrum_adaptive(counts: np.ndarray,
                            target_snr: float = 10.0) -> np.ndarray:
    """
    Apply adaptive smoothing based on local statistics.
    
    Parameters:
        counts: Spectrum counts
        target_snr: Target signal-to-noise ratio
        
    Returns:
        Adaptively smoothed spectrum
    """
    smoothed = counts.copy()
    
    for i in range(len(counts)):
        # Estimate local noise
        window = 20
        start = max(0, i - window)
        end = min(len(counts), i + window)
        local_region = counts[start:end]
        
        if len(local_region) > 3:
            noise = np.std(local_region)
            
            # Determine smoothing needed
            if noise > 0:
                current_snr = counts[i] / noise
                
                if current_snr < target_snr:
                    # Need smoothing
                    smooth_window = int(target_snr / current_snr)
                    smooth_window = min(smooth_window, 10)  # Limit smoothing
                    
                    smooth_start = max(0, i - smooth_window // 2)
                    smooth_end = min(len(counts), i + smooth_window // 2 + 1)
                    
                    smoothed[i] = np.mean(counts[smooth_start:smooth_end])
    
    return smoothed


def generate_synthetic_spectrum(num_channels: int = 4096,
                              peaks: List[Tuple[float, float, float]] = None,
                              background_level: float = 10.0,
                              noise_level: float = 1.0,
                              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic gamma spectrum for testing.
    
    Parameters:
        num_channels: Number of channels
        peaks: List of (channel, amplitude, sigma) tuples
        background_level: Background count level
        noise_level: Noise amplitude
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (channels, counts)
    """
    if seed is not None:
        np.random.seed(seed)
    
    channels = np.arange(num_channels)
    
    # Generate background
    background = background_level * np.exp(-channels / (num_channels / 2))
    background += noise_level * np.random.randn(num_channels)
    
    # Add peaks
    counts = background.copy()
    
    if peaks is None:
        # Default peaks
        peaks = [
            (500, 1000, 5),    # Sharp peak
            (1000, 500, 10),   # Medium peak
            (1500, 300, 15),   # Broad peak
            (2000, 800, 8),    # Strong peak
            (2010, 400, 6),    # Overlapping peak
        ]
    
    for channel, amplitude, sigma in peaks:
        if 0 <= channel < num_channels:
            # Add Gaussian peak
            peak = amplitude * np.exp(-0.5 * ((channels - channel) / sigma) ** 2)
            counts += peak
    
    # Apply Poisson noise
    counts = np.maximum(counts, 0)
    counts = np.random.poisson(counts)
    
    return channels.astype(float), counts.astype(float)


def calculate_peak_area(channels: np.ndarray,
                        counts: np.ndarray,
                        peak_region: Tuple[int, int],
                        background_method: str = 'linear') -> Dict[str, float]:
    """
    Calculate peak area with background subtraction.
    
    Parameters:
        channels: Channel numbers
        counts: Counts data
        peak_region: (start, end) indices
        background_method: Background estimation method
        
    Returns:
        Dictionary with area and uncertainty
    """
    start, end = peak_region
    region_channels = channels[start:end]
    region_counts = counts[start:end]
    
    if len(region_counts) < 3:
        return {'area': 0, 'area_err': 0, 'background': 0}
    
    # Estimate background
    if background_method == 'linear':
        # Use endpoints for linear background
        bg_start = np.mean(counts[start:start+3])
        bg_end = np.mean(counts[end-3:end])
        
        # Linear interpolation
        bg_slope = (bg_end - bg_start) / (region_channels[-1] - region_channels[0])
        background = bg_start + bg_slope * (region_channels - region_channels[0])
        
    elif background_method == 'step':
        # Step function background
        background = np.full_like(region_counts, np.min(region_counts))
        
    else:
        # Constant background
        edges = np.concatenate([region_counts[:3], region_counts[-3:]])
        background = np.full_like(region_counts, np.mean(edges))
    
    # Calculate net area
    net_counts = region_counts - background
    net_counts = np.maximum(net_counts, 0)
    
    # Integrate
    area = np.trapz(net_counts, region_channels)
    
    # Estimate uncertainty (Poisson statistics)
    gross_area = np.trapz(region_counts, region_channels)
    area_err = np.sqrt(gross_area)
    
    return {
        'area': area,
        'area_err': area_err,
        'gross_area': gross_area,
        'background': np.mean(background)
    }


def find_calibration_peaks(counts: np.ndarray,
                          expected_ratios: List[float],
                          tolerance: float = 0.1) -> List[int]:
    """
    Find peaks matching expected energy ratios for calibration.
    
    Parameters:
        counts: Spectrum counts
        expected_ratios: Expected ratios between peak energies
        tolerance: Relative tolerance for ratio matching
        
    Returns:
        List of matched peak indices
    """
    from .detection import detect_peaks
    
    # Detect all significant peaks
    peaks = detect_peaks(counts, min_prominence=100)
    
    if len(peaks) < len(expected_ratios) + 1:
        return []
    
    # Try different combinations
    best_match = []
    best_score = float('inf')
    
    from itertools import combinations
    
    for peak_combo in combinations(peaks, len(expected_ratios) + 1):
        peak_combo = sorted(peak_combo)
        
        # Calculate ratios
        ratios = []
        for i in range(1, len(peak_combo)):
            ratio = peak_combo[i] / peak_combo[0]
            ratios.append(ratio)
        
        # Compare with expected ratios
        score = 0
        for obs_ratio, exp_ratio in zip(ratios, expected_ratios):
            relative_diff = abs(obs_ratio - exp_ratio) / exp_ratio
            if relative_diff > tolerance:
                score = float('inf')
                break
            score += relative_diff
        
        if score < best_score:
            best_score = score
            best_match = list(peak_combo)
    
    return best_match if best_score < float('inf') else []


def calculate_resolution(energy: float, fwhm: float) -> float:
    """
    Calculate energy resolution as percentage.
    
    Parameters:
        energy: Peak energy (keV)
        fwhm: Full width at half maximum (keV)
        
    Returns:
        Resolution in percent
    """
    if energy <= 0:
        return 0
    
    return (fwhm / energy) * 100


def estimate_detector_efficiency(energy: float,
                                detector_type: str = 'NaI',
                                detector_size: float = 3.0) -> float:
    """
    Estimate detector efficiency at given energy.
    
    Parameters:
        energy: Gamma energy in keV
        detector_type: Detector type ('NaI', 'HPGe', 'CsI')
        detector_size: Detector size in inches (for NaI) or cm³ (for HPGe)
        
    Returns:
        Relative efficiency (0-1)
    """
    # Simplified empirical models
    if detector_type == 'NaI':
        # NaI(Tl) detector efficiency curve
        # Peak around 100-200 keV, decreasing at higher energies
        if energy < 50:
            eff = 0.5
        elif energy < 200:
            eff = 0.9
        elif energy < 500:
            eff = 0.7
        elif energy < 1000:
            eff = 0.4
        elif energy < 2000:
            eff = 0.2
        else:
            eff = 0.1
        
        # Scale by detector size (assuming 3" as reference)
        eff *= (detector_size / 3.0) ** 0.5
        
    elif detector_type == 'HPGe':
        # HPGe detector - better at high energies
        if energy < 100:
            eff = 0.7
        elif energy < 500:
            eff = 0.5
        elif energy < 1000:
            eff = 0.3
        elif energy < 2000:
            eff = 0.2
        else:
            eff = 0.15
        
        # Scale by detector volume
        eff *= (detector_size / 100.0) ** 0.3
        
    else:
        # Generic detector
        eff = np.exp(-energy / 1000.0)
    
    return min(max(eff, 0), 1)


def calculate_activity(peak_area: float,
                      efficiency: float,
                      branching_ratio: float,
                      live_time: float) -> Dict[str, float]:
    """
    Calculate activity from peak area.
    
    Parameters:
        peak_area: Net peak area (counts)
        efficiency: Detector efficiency
        branching_ratio: Gamma emission probability
        live_time: Measurement live time (seconds)
        
    Returns:
        Dictionary with activity and uncertainty
    """
    if efficiency <= 0 or branching_ratio <= 0 or live_time <= 0:
        return {'activity': 0, 'activity_err': 0}
    
    # Activity = Area / (efficiency * branching_ratio * live_time)
    activity = peak_area / (efficiency * branching_ratio * live_time)
    
    # Simplified uncertainty (counting statistics only)
    activity_err = np.sqrt(peak_area) / (efficiency * branching_ratio * live_time)
    
    return {
        'activity': activity,
        'activity_err': activity_err,
        'activity_Bq': activity,
        'activity_uCi': activity / 37000.0  # Convert to µCi
    }


def save_analysis_metadata(output_dir: str,
                          config: Dict[str, Any],
                          results: Dict[str, Any]):
    """
    Save analysis metadata and configuration.
    
    Parameters:
        output_dir: Output directory path
        config: Analysis configuration
        results: Analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'version': get_version(),
        'configuration': config,
        'results_summary': {
            'peaks_found': len(results.get('peaks', [])),
            'calibrated': 'calibration' in config,
            'processing_time': results.get('processing_time', 0)
        }
    }
    
    # Save as JSON
    metadata_file = output_path / 'analysis_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save configuration separately
    config_file = output_path / 'analysis_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def get_version() -> str:
    """Get package version."""
    try:
        from . import __version__
        return __version__
    except:
        return 'unknown'


def calculate_spectrum_hash(counts: np.ndarray) -> str:
    """
    Calculate hash of spectrum for identification.
    
    Parameters:
        counts: Spectrum counts
        
    Returns:
        SHA256 hash string
    """
    # Convert to bytes
    counts_bytes = counts.astype(np.float32).tobytes()
    
    # Calculate hash
    hash_obj = hashlib.sha256(counts_bytes)
    
    return hash_obj.hexdigest()


def check_spectrum_quality(counts: np.ndarray) -> Dict[str, Any]:
    """
    Check spectrum quality metrics.
    
    Parameters:
        counts: Spectrum counts
        
    Returns:
        Dictionary with quality metrics
    """
    quality = {
        'total_counts': np.sum(counts),
        'max_counts': np.max(counts),
        'zero_channels': np.sum(counts == 0),
        'saturated_channels': np.sum(counts >= 65535),  # 16-bit saturation
        'noise_level': np.std(counts[counts < np.percentile(counts, 10)]),
        'dynamic_range': np.max(counts) / np.mean(counts[counts > 0]) if np.any(counts > 0) else 0
    }
    
    # Quality flags
    quality['is_empty'] = quality['total_counts'] == 0
    quality['has_saturation'] = quality['saturated_channels'] > 0
    quality['low_statistics'] = quality['total_counts'] < 10000
    quality['high_noise'] = quality['noise_level'] > np.sqrt(np.mean(counts)) * 2
    
    # Overall quality score (0-100)
    score = 100
    if quality['is_empty']:
        score = 0
    else:
        if quality['has_saturation']:
            score -= 20
        if quality['low_statistics']:
            score -= 30
        if quality['high_noise']:
            score -= 20
        if quality['zero_channels'] > len(counts) * 0.5:
            score -= 30
    
    quality['quality_score'] = max(score, 0)
    
    return quality


def format_uncertainty(value: float, uncertainty: float, 
                      precision: int = 2) -> str:
    """
    Format value with uncertainty in standard notation.
    
    Parameters:
        value: Central value
        uncertainty: Uncertainty
        precision: Number of significant figures for uncertainty
        
    Returns:
        Formatted string
    """
    if uncertainty <= 0:
        return f"{value:.{precision}f}"
    
    # Determine precision based on uncertainty
    if uncertainty >= 10:
        return f"{value:.0f} ± {uncertainty:.0f}"
    elif uncertainty >= 1:
        return f"{value:.1f} ± {uncertainty:.1f}"
    else:
        # Find first significant digit of uncertainty
        exp = int(np.floor(np.log10(uncertainty)))
        factor = 10 ** (-exp)
        unc_rounded = np.round(uncertainty * factor, precision - 1) / factor
        val_rounded = np.round(value, -exp)
        
        if exp >= 0:
            return f"{val_rounded:.0f} ± {unc_rounded:.0f}"
        else:
            return f"{val_rounded:.{-exp}f} ± {unc_rounded:.{-exp}f}"


# Constants for nuclear data
GAMMA_ENERGIES = {
    'K-40': 1460.82,
    'Co-60_1': 1173.23,
    'Co-60_2': 1332.49,
    'Cs-137': 661.66,
    'Ba-133_1': 356.01,
    'Ba-133_2': 80.99,
    'Na-22_1': 511.0,
    'Na-22_2': 1274.53,
    'Am-241': 59.54,
    'Eu-152_1': 121.78,
    'Eu-152_2': 344.28,
    'Eu-152_3': 1408.01,
}


def identify_isotope(energy: float, tolerance: float = 2.0) -> List[str]:
    """
    Identify possible isotopes from gamma energy.
    
    Parameters:
        energy: Gamma energy in keV
        tolerance: Energy tolerance in keV
        
    Returns:
        List of possible isotope identifications
    """
    matches = []
    
    for isotope, ref_energy in GAMMA_ENERGIES.items():
        if abs(energy - ref_energy) <= tolerance:
            matches.append(f"{isotope.split('_')[0]} ({ref_energy:.1f} keV)")
    
    return matches
