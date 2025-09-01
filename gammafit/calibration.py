"""
Energy calibration utilities for gamma spectroscopy.

This module provides functions for energy calibration, including
linear and polynomial calibration, automatic calibration from known peaks,
and calibration validation.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import linregress
import json


@dataclass
class CalibrationPoint:
    """Data class for calibration points."""
    channel: float
    energy: float
    isotope: Optional[str] = None
    uncertainty: Optional[float] = None
    weight: float = 1.0


class EnergyCalibration:
    """
    Energy calibration class for gamma spectroscopy.
    
    Supports linear and polynomial calibration models.
    """
    
    def __init__(self, model: str = 'linear'):
        """
        Initialize calibration.
        
        Parameters:
            model: Calibration model ('linear', 'quadratic', 'polynomial')
        """
        self.model = model
        self.coefficients = None
        self.covariance = None
        self.calibration_points = []
        self.fit_quality = {}
        
    def add_point(self, channel: float, energy: float, 
                  isotope: Optional[str] = None,
                  uncertainty: Optional[float] = None):
        """
        Add a calibration point.
        
        Parameters:
            channel: Channel number
            energy: Energy in keV
            isotope: Optional isotope name
            uncertainty: Optional energy uncertainty
        """
        point = CalibrationPoint(channel, energy, isotope, uncertainty)
        self.calibration_points.append(point)
    
    def fit(self, channels: Optional[np.ndarray] = None,
            energies: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit calibration model to data.
        
        Parameters:
            channels: Channel numbers (if not using stored points)
            energies: Energy values
            weights: Optional weights for fitting
            
        Returns:
            Dictionary with fit parameters
        """
        # Use provided data or stored calibration points
        if channels is None:
            if not self.calibration_points:
                raise ValueError("No calibration data provided")
            channels = np.array([p.channel for p in self.calibration_points])
            energies = np.array([p.energy for p in self.calibration_points])
            if weights is None:
                weights = np.array([p.weight for p in self.calibration_points])
        
        if len(channels) < 2:
            raise ValueError("At least 2 calibration points required")
        
        # Fit based on model
        if self.model == 'linear':
            self.coefficients = self._fit_linear(channels, energies, weights)
        elif self.model == 'quadratic':
            self.coefficients = self._fit_polynomial(channels, energies, 2, weights)
        elif self.model == 'polynomial':
            # Determine polynomial order based on number of points
            order = min(len(channels) - 1, 3)  # Max 3rd order
            self.coefficients = self._fit_polynomial(channels, energies, order, weights)
        else:
            raise ValueError(f"Unknown calibration model: {self.model}")
        
        # Calculate fit quality metrics
        self._calculate_fit_quality(channels, energies)
        
        return self.coefficients
    
    def _fit_linear(self, channels: np.ndarray, energies: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit linear calibration: E = a*channel + b
        
        Parameters:
            channels: Channel numbers
            energies: Energy values
            weights: Optional weights
            
        Returns:
            Dictionary with coefficients
        """
        if weights is not None:
            # Weighted linear regression
            def linear_func(x, a, b):
                return a * x + b
            
            popt, pcov = curve_fit(linear_func, channels, energies,
                                  sigma=1/weights if weights is not None else None)
            self.covariance = pcov
            return {'a': popt[0], 'b': popt[1], 'order': 1}
        else:
            # Standard linear regression
            slope, intercept, r_value, p_value, std_err = linregress(channels, energies)
            return {
                'a': slope,
                'b': intercept,
                'r_squared': r_value**2,
                'std_err': std_err,
                'order': 1
            }
    
    def _fit_polynomial(self, channels: np.ndarray, energies: np.ndarray,
                       order: int, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit polynomial calibration.
        
        Parameters:
            channels: Channel numbers
            energies: Energy values
            order: Polynomial order
            weights: Optional weights
            
        Returns:
            Dictionary with coefficients
        """
        if weights is not None:
            # Weighted polynomial fit
            coeffs = np.polyfit(channels, energies, order, w=weights)
        else:
            coeffs = np.polyfit(channels, energies, order)
        
        result = {'order': order}
        for i, coeff in enumerate(coeffs):
            result[f'c{order-i}'] = coeff
        
        return result
    
    def _calculate_fit_quality(self, channels: np.ndarray, energies: np.ndarray):
        """
        Calculate fit quality metrics.
        
        Parameters:
            channels: Channel numbers
            energies: Energy values
        """
        # Calculate fitted values
        fitted_energies = self.channel_to_energy(channels)
        
        # Calculate residuals
        residuals = energies - fitted_energies
        
        # Calculate metrics
        self.fit_quality = {
            'rms_error': np.sqrt(np.mean(residuals**2)),
            'max_error': np.max(np.abs(residuals)),
            'mean_error': np.mean(residuals),
            'std_error': np.std(residuals),
            'r_squared': 1 - np.sum(residuals**2) / np.sum((energies - np.mean(energies))**2)
        }
    
    def channel_to_energy(self, channels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert channels to energy using calibration.
        
        Parameters:
            channels: Channel number(s)
            
        Returns:
            Energy value(s) in keV
        """
        if self.coefficients is None:
            raise ValueError("Calibration not fitted yet")
        
        if self.model == 'linear':
            return self.coefficients['a'] * channels + self.coefficients['b']
        else:
            # Polynomial calibration
            order = self.coefficients['order']
            result = np.zeros_like(channels, dtype=float)
            for i in range(order + 1):
                coeff_key = f'c{i}'
                if coeff_key in self.coefficients:
                    result += self.coefficients[coeff_key] * (channels ** i)
            return result
    
    def energy_to_channel(self, energies: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert energy to channels using calibration (inverse).
        
        Parameters:
            energies: Energy value(s) in keV
            
        Returns:
            Channel number(s)
        """
        if self.coefficients is None:
            raise ValueError("Calibration not fitted yet")
        
        if self.model == 'linear':
            # Simple inverse for linear
            return (energies - self.coefficients['b']) / self.coefficients['a']
        else:
            # For polynomial, use numerical root finding
            # This is approximate and works best for monotonic calibrations
            warnings.warn("Inverse polynomial calibration is approximate")
            
            # Create a fine grid and interpolate
            min_ch = 0
            max_ch = 8192  # Typical max channels
            channels = np.linspace(min_ch, max_ch, 10000)
            energies_grid = self.channel_to_energy(channels)
            
            # Interpolate to find channels
            return np.interp(energies, energies_grid, channels)
    
    def get_uncertainty(self, channel: float) -> float:
        """
        Get energy uncertainty at a given channel.
        
        Parameters:
            channel: Channel number
            
        Returns:
            Energy uncertainty in keV
        """
        if self.covariance is None:
            # Return RMS error as estimate
            return self.fit_quality.get('rms_error', 0.5)
        
        if self.model == 'linear':
            # Error propagation for linear calibration
            var_a = self.covariance[0, 0]
            var_b = self.covariance[1, 1]
            cov_ab = self.covariance[0, 1]
            
            # Variance of E = a*ch + b
            variance = var_a * channel**2 + var_b + 2 * cov_ab * channel
            return np.sqrt(variance)
        else:
            # Simplified uncertainty for polynomial
            return self.fit_quality.get('rms_error', 0.5)
    
    def save(self, filepath: str):
        """
        Save calibration to file.
        
        Parameters:
            filepath: Output file path
        """
        cal_data = {
            'model': self.model,
            'coefficients': self.coefficients,
            'fit_quality': self.fit_quality,
            'calibration_points': [
                {
                    'channel': p.channel,
                    'energy': p.energy,
                    'isotope': p.isotope,
                    'uncertainty': p.uncertainty
                }
                for p in self.calibration_points
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(cal_data, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load calibration from file.
        
        Parameters:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            cal_data = json.load(f)
        
        self.model = cal_data['model']
        self.coefficients = cal_data['coefficients']
        self.fit_quality = cal_data.get('fit_quality', {})
        
        self.calibration_points = []
        for point_data in cal_data.get('calibration_points', []):
            self.add_point(
                point_data['channel'],
                point_data['energy'],
                point_data.get('isotope'),
                point_data.get('uncertainty')
            )


def apply_calibration(peaks: List[Dict[str, Any]],
                     calibration: Union[Dict[str, float], EnergyCalibration]) -> List[Dict[str, Any]]:
    """
    Apply energy calibration to fitted peaks.
    
    Parameters:
        peaks: List of peak dictionaries
        calibration: Calibration parameters or EnergyCalibration object
        
    Returns:
        Updated peak list with energy values
    """
    calibrated_peaks = []
    
    for peak in peaks:
        peak_copy = peak.copy()
        
        if isinstance(calibration, EnergyCalibration):
            # Use calibration object
            peak_copy['energy'] = calibration.channel_to_energy(peak['centroid'])
            peak_copy['energy_err'] = calibration.get_uncertainty(peak['centroid'])
            
            # Convert FWHM to energy units
            ch_low = peak['centroid'] - peak['fwhm'] / 2
            ch_high = peak['centroid'] + peak['fwhm'] / 2
            e_low = calibration.channel_to_energy(ch_low)
            e_high = calibration.channel_to_energy(ch_high)
            peak_copy['fwhm_energy'] = e_high - e_low
            
        elif isinstance(calibration, dict):
            # Simple linear calibration dictionary
            if 'a' in calibration and 'b' in calibration:
                peak_copy['energy'] = calibration['a'] * peak['centroid'] + calibration['b']
                peak_copy['energy_err'] = calibration['a'] * peak.get('centroid_err', 1.0)
                peak_copy['fwhm_energy'] = calibration['a'] * peak['fwhm']
            else:
                warnings.warn("Invalid calibration dictionary format")
        
        calibrated_peaks.append(peak_copy)
    
    return calibrated_peaks


def channel_to_energy(channel: Union[float, np.ndarray],
                      calibration: Dict[str, float]) -> Union[float, np.ndarray]:
    """
    Convert channel to energy using calibration parameters.
    
    Parameters:
        channel: Channel number(s)
        calibration: Calibration parameters
        
    Returns:
        Energy value(s) in keV
    """
    if 'order' in calibration:
        # Polynomial calibration
        order = calibration['order']
        result = np.zeros_like(channel, dtype=float)
        for i in range(order + 1):
            coeff_key = f'c{i}'
            if coeff_key in calibration:
                result += calibration[coeff_key] * (channel ** i)
        return result
    else:
        # Linear calibration
        return calibration['a'] * channel + calibration['b']


def energy_to_channel(energy: Union[float, np.ndarray],
                      calibration: Dict[str, float]) -> Union[float, np.ndarray]:
    """
    Convert energy to channel using calibration parameters (inverse).
    
    Parameters:
        energy: Energy value(s) in keV
        calibration: Calibration parameters
        
    Returns:
        Channel number(s)
    """
    if 'order' in calibration and calibration['order'] > 1:
        # Polynomial - use numerical approach
        warnings.warn("Inverse polynomial calibration is approximate")
        
        # Create lookup table
        channels = np.linspace(0, 8192, 10000)
        energies = channel_to_energy(channels, calibration)
        return np.interp(energy, energies, channels)
    else:
        # Linear calibration - simple inverse
        return (energy - calibration['b']) / calibration['a']


def auto_calibrate(spectrum_channels: np.ndarray,
                   spectrum_counts: np.ndarray,
                   known_peaks: List[Tuple[float, str]],
                   tolerance: float = 5.0) -> Optional[EnergyCalibration]:
    """
    Automatically calibrate spectrum using known peak energies.
    
    Parameters:
        spectrum_channels: Channel numbers
        spectrum_counts: Counts data
        known_peaks: List of (energy, isotope) tuples
        tolerance: Channel tolerance for peak matching
        
    Returns:
        EnergyCalibration object or None if failed
    """
    from .detection import detect_peaks
    from .fitting import fit_single_peak
    
    # Detect peaks in spectrum
    detected_peaks = detect_peaks(spectrum_counts, min_prominence=50)
    
    if len(detected_peaks) < 2:
        warnings.warn("Not enough peaks detected for auto-calibration")
        return None
    
    # Fit detected peaks to get precise centroids
    fitted_peaks = []
    for peak_idx in detected_peaks:
        try:
            result = fit_single_peak(spectrum_channels, spectrum_counts, peak_idx)
            if result['fit_success']:
                fitted_peaks.append(result)
        except:
            continue
    
    if len(fitted_peaks) < 2:
        warnings.warn("Not enough peaks fitted for auto-calibration")
        return None
    
    # Try to match known peaks to detected peaks
    calibration = EnergyCalibration(model='linear')
    
    # Sort known peaks by energy
    known_peaks_sorted = sorted(known_peaks, key=lambda x: x[0])
    
    # Simple matching algorithm (can be improved)
    for known_energy, isotope in known_peaks_sorted:
        # Find closest unmatched peak
        best_match = None
        best_distance = float('inf')
        
        for peak in fitted_peaks:
            if 'matched' not in peak:
                # Estimate expected channel (rough initial guess)
                if len(calibration.calibration_points) >= 2:
                    expected_ch = energy_to_channel(known_energy, calibration.coefficients)
                    distance = abs(peak['centroid'] - expected_ch)
                else:
                    # No calibration yet, use relative position
                    distance = abs(peak['centroid'])
                
                if distance < best_distance and distance < tolerance:
                    best_match = peak
                    best_distance = distance
        
        if best_match:
            calibration.add_point(best_match['centroid'], known_energy, isotope)
            best_match['matched'] = True
            
            # Refit calibration if we have enough points
            if len(calibration.calibration_points) >= 2:
                calibration.fit()
    
    if len(calibration.calibration_points) < 2:
        warnings.warn("Could not match enough peaks for calibration")
        return None
    
    # Final fit
    calibration.fit()
    
    return calibration


def validate_calibration(calibration: Union[Dict, EnergyCalibration],
                        test_peaks: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Validate calibration using test peaks.
    
    Parameters:
        calibration: Calibration parameters or object
        test_peaks: List of (channel, expected_energy) tuples
        
    Returns:
        Dictionary with validation metrics
    """
    channels = np.array([p[0] for p in test_peaks])
    expected_energies = np.array([p[1] for p in test_peaks])
    
    # Calculate calibrated energies
    if isinstance(calibration, EnergyCalibration):
        calculated_energies = calibration.channel_to_energy(channels)
    else:
        calculated_energies = channel_to_energy(channels, calibration)
    
    # Calculate errors
    errors = calculated_energies - expected_energies
    
    return {
        'rms_error': np.sqrt(np.mean(errors**2)),
        'max_error': np.max(np.abs(errors)),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'relative_errors': errors / expected_energies * 100  # Percentage
    }


# Common isotope libraries for auto-calibration
COMMON_ISOTOPES = {
    'Co-60': [1173.23, 1332.49],  # keV
    'Cs-137': [661.66],
    'Na-22': [511.0, 1274.53],
    'Ba-133': [80.99, 276.40, 302.85, 356.01, 383.85],
    'Eu-152': [121.78, 244.70, 344.28, 411.12, 443.96, 778.90, 867.38, 964.08, 1085.84, 1112.08, 1408.01],
    'Am-241': [59.54],
    'Th-228': [238.63, 583.19, 860.56, 2614.51],
}


def get_isotope_energies(isotope: str) -> List[float]:
    """
    Get known gamma energies for a given isotope.
    
    Parameters:
        isotope: Isotope name (e.g., 'Co-60')
        
    Returns:
        List of gamma energies in keV
    """
    return COMMON_ISOTOPES.get(isotope, [])
