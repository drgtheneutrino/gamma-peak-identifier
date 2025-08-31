"""
GammaFit - Gamma Spectroscopy Peak Detection and Fitting Library
================================================================

A Python package for automated detection, fitting, and analysis of peaks 
in gamma-ray spectra.

Repository: https://github.com/drgtheneutrino/gamma-peak-identifier

Basic Usage:
    from gammafit import load_spectrum, detect_peaks, fit_peaks
    
    # Load spectrum
    channels, counts = load_spectrum('spectrum.csv')
    
    # Detect peaks
    peak_indices = detect_peaks(counts)
    
    # Fit peaks
    fitted_peaks = fit_peaks(channels, counts, peak_indices)

Command Line Usage:
    python -m gammafit spectrum.csv --min-prominence 50
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__url__ = "https://github.com/drgtheneutrino/gamma-peak-identifier"

# Import main functions for easier access
from .io_module import load_spectrum, save_peaks, load_config
from .detection import detect_peaks, smooth_spectrum
from .fitting import fit_peaks, fit_single_peak, gaussian_with_background
from .output import plot_spectrum_with_fits, export_results
from .calibration import calibrate_energy, apply_calibration
from .utils import calculate_fwhm, estimate_background

# Define what gets imported with "from gammafit import *"
__all__ = [
    'load_spectrum',
    'save_peaks',
    'load_config',
    'detect_peaks',
    'smooth_spectrum',
    'fit_peaks',
    'fit_single_peak',
    'gaussian_with_background',
    'plot_spectrum_with_fits',
    'export_results',
    'calibrate_energy',
    'apply_calibration',
    'calculate_fwhm',
    'estimate_background',
]

# Package metadata
PACKAGE_DATA = {
    'name': 'gammafit',
    'version': __version__,
    'description': 'Automated gamma spectroscopy peak detection and fitting',
    'repository': __url__,
    'python_requires': '>=3.7',
    'install_requires': [
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'matplotlib>=3.3.0',
        'pandas>=1.1.0',
    ],
    'optional_requires': {
        'dev': ['pytest>=6.0', 'pytest-cov', 'black', 'flake8'],
        'docs': ['sphinx>=3.0', 'sphinx-rtd-theme'],
        'advanced': ['lmfit>=1.0', 'uncertainties>=3.1'],
    }
}
