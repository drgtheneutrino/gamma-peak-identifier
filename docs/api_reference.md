# API Reference

Complete API documentation for the GammaFit package.

## Table of Contents

- [gammafit](#gammafit-package)
- [gammafit.detection](#gammafitdetection)
- [gammafit.fitting](#gammafitfitting)
- [gammafit.calibration](#gammafitcalibration)
- [gammafit.io_module](#gammafitio_module)
- [gammafit.output](#gammafitoutput)
- [gammafit.utils](#gammafitutils)

---

## gammafit Package

Main package initialization and high-level functions.

### Functions

#### `detect_peaks(counts, min_prominence=50, min_height=10, min_distance=3, method='scipy', **kwargs)`

Detect peaks in gamma spectrum.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts data
- `min_prominence` (float): Minimum peak prominence above background
- `min_height` (float): Minimum absolute peak height
- `min_distance` (int): Minimum distance between peaks in channels
- `method` (str): Detection method ('scipy', 'derivative', 'template')
- `**kwargs`: Additional method-specific parameters

**Returns:**
- `np.ndarray`: Array of peak indices

**Example:**
```python
peaks = detect_peaks(counts, min_prominence=100, min_height=50)
```

---

#### `fit_peaks(channels, counts, peak_indices, peak_model='gaussian', background_method='linear', window_scale=3.0, parallel=False)`

Fit detected peaks with specified models.

**Parameters:**
- `channels` (np.ndarray): Channel numbers
- `counts` (np.ndarray): Counts data
- `peak_indices` (np.ndarray): Indices of detected peaks
- `peak_model` (str): Peak model ('gaussian', 'voigt', 'gaussian_tail')
- `background_method` (str): Background model ('linear', 'quadratic', 'step')
- `window_scale` (float): Scale factor for fitting window
- `parallel` (bool): Use parallel processing

**Returns:**
- `List[Dict]`: List of fitted peak dictionaries

**Example:**
```python
fitted_peaks = fit_peaks(channels, counts, peak_indices)
```

---

#### `smooth_spectrum(counts, window_length=5, polyorder=2, method='savgol')`

Apply smoothing filter to spectrum.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts
- `window_length` (int): Window size for filter
- `polyorder` (int): Polynomial order (for Savitzky-Golay)
- `method` (str): Smoothing method ('savgol', 'gaussian', 'moving_average', 'none')

**Returns:**
- `np.ndarray`: Smoothed counts

---

## gammafit.detection

Peak detection algorithms and background estimation.

### Functions

#### `estimate_background(counts, method='snip', **kwargs)`

Estimate background in spectrum.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts
- `method` (str): Background method ('snip', 'rolling_ball', 'percentile')
- `**kwargs`: Method-specific parameters

**Returns:**
- `np.ndarray`: Estimated background

**Methods:**

1. **SNIP** (Statistics-sensitive Nonlinear Iterative Peak-clipping)
   ```python
   background = estimate_background(counts, method='snip', iterations=20, window=20)
   ```

2. **Rolling Ball**
   ```python
   background = estimate_background(counts, method='rolling_ball', radius=50)
   ```

3. **Percentile**
   ```python
   background = estimate_background(counts, method='percentile', window=100, percentile=10)
   ```

---

#### `refine_peak_positions(counts, initial_peaks, method='centroid', window=5)`

Refine peak positions using various methods.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts
- `initial_peaks` (np.ndarray): Initial peak indices
- `method` (str): Refinement method ('centroid', 'parabolic', 'gaussian')
- `window` (int): Window size around peak

**Returns:**
- `np.ndarray`: Refined peak positions (can be fractional)

---

#### `group_peaks(peaks, min_separation=5.0)`

Group nearby peaks that might be multiplets.

**Parameters:**
- `peaks` (np.ndarray): Array of peak indices
- `min_separation` (float): Minimum separation to consider peaks separate

**Returns:**
- `List[List[int]]`: List of peak groups

---

## gammafit.fitting

Peak fitting models and algorithms.

### Peak Shape Functions

#### `gaussian(x, amplitude, centroid, sigma)`

Gaussian peak function.

**Parameters:**
- `x` (np.ndarray): Channel numbers
- `amplitude` (float): Peak amplitude
- `centroid` (float): Peak center position
- `sigma` (float): Peak width parameter (standard deviation)

**Returns:**
- `np.ndarray`: Gaussian peak values

---

#### `gaussian_with_background(x, amplitude, centroid, sigma, bg_slope, bg_intercept)`

Gaussian peak with linear background.

**Parameters:**
- `x` (np.ndarray): Channel numbers
- `amplitude` (float): Peak amplitude
- `centroid` (float): Peak center position
- `sigma` (float): Peak width parameter
- `bg_slope` (float): Background slope
- `bg_intercept` (float): Background intercept

**Returns:**
- `np.ndarray`: Model values (peak + background)

---

#### `voigt(x, amplitude, centroid, sigma, gamma)`

Voigt profile (convolution of Gaussian and Lorentzian).

**Parameters:**
- `x` (np.ndarray): Channel numbers
- `amplitude` (float): Peak amplitude
- `centroid` (float): Peak center position
- `sigma` (float): Gaussian width parameter
- `gamma` (float): Lorentzian width parameter

**Returns:**
- `np.ndarray`: Voigt profile values

---

### Fitting Functions

#### `fit_single_peak(channels, counts, peak_idx, peak_model='gaussian', background_method='linear', window_scale=3.0, max_iterations=5000)`

Fit a single peak with specified model.

**Parameters:**
- `channels` (np.ndarray): Channel numbers
- `counts` (np.ndarray): Counts data
- `peak_idx` (int): Index of peak maximum
- `peak_model` (str): Peak model type
- `background_method` (str): Background model
- `window_scale` (float): Scale factor for fitting window
- `max_iterations` (int): Maximum fitting iterations

**Returns:**
- `Dict`: Fitted peak parameters

**Return Dictionary Contains:**
```python
{
    'centroid': float,          # Peak center position
    'centroid_err': float,      # Centroid uncertainty
    'amplitude': float,         # Peak amplitude
    'amplitude_err': float,     # Amplitude uncertainty
    'sigma': float,             # Peak width parameter
    'sigma_err': float,         # Sigma uncertainty
    'area': float,              # Integrated peak area
    'area_err': float,          # Area uncertainty
    'fwhm': float,              # Full width at half maximum
    'fwhm_err': float,          # FWHM uncertainty
    'snr': float,               # Signal-to-noise ratio
    'resolution': float,        # Energy resolution (%)
    'chi_square': float,        # Reduced chi-square
    'fit_success': bool,        # Fit success flag
    'fit_params': array,        # Raw fit parameters
    'fit_region': tuple,        # (start, end) indices
}
```

---

#### `fit_multiplet(channels, counts, peak_indices, peak_model='gaussian', background_method='linear')`

Fit multiple overlapping peaks simultaneously.

**Parameters:**
- `channels` (np.ndarray): Channel numbers
- `counts` (np.ndarray): Counts data
- `peak_indices` (List[int]): List of peak indices in multiplet
- `peak_model` (str): Peak model to use
- `background_method` (str): Background model

**Returns:**
- `List[Dict]`: List of fitted peak dictionaries

---

#### `identify_overlapping_peaks(channels, counts, peak_indices, window_scale=3.0)`

Identify groups of overlapping peaks.

**Parameters:**
- `channels` (np.ndarray): Channel numbers
- `counts` (np.ndarray): Counts data
- `peak_indices` (np.ndarray): Array of peak indices
- `window_scale` (float): Scale factor for determining overlap

**Returns:**
- `List[List[int]]`: List of peak groups

---

## gammafit.calibration

Energy calibration classes and functions.

### Classes

#### `EnergyCalibration`

Energy calibration class for gamma spectroscopy.

**Constructor:**
```python
EnergyCalibration(model='linear')
```

**Parameters:**
- `model` (str): Calibration model ('linear', 'quadratic', 'polynomial')

**Methods:**

##### `add_point(channel, energy, isotope=None, uncertainty=None)`

Add a calibration point.

**Parameters:**
- `channel` (float): Channel number
- `energy` (float): Energy in keV
- `isotope` (str, optional): Isotope name
- `uncertainty` (float, optional): Energy uncertainty

---

##### `fit(channels=None, energies=None, weights=None)`

Fit calibration model to data.

**Parameters:**
- `channels` (np.ndarray, optional): Channel numbers
- `energies` (np.ndarray, optional): Energy values
- `weights` (np.ndarray, optional): Weights for fitting

**Returns:**
- `Dict`: Calibration coefficients

---

##### `channel_to_energy(channels)`

Convert channels to energy using calibration.

**Parameters:**
- `channels` (float or np.ndarray): Channel number(s)

**Returns:**
- `float or np.ndarray`: Energy value(s) in keV

---

##### `energy_to_channel(energies)`

Convert energy to channels (inverse calibration).

**Parameters:**
- `energies` (float or np.ndarray): Energy value(s) in keV

**Returns:**
- `float or np.ndarray`: Channel number(s)

---

##### `get_uncertainty(channel)`

Get energy uncertainty at a given channel.

**Parameters:**
- `channel` (float): Channel number

**Returns:**
- `float`: Energy uncertainty in keV

---

##### `save(filepath)`

Save calibration to JSON file.

**Parameters:**
- `filepath` (str): Output file path

---

##### `load(filepath)`

Load calibration from JSON file.

**Parameters:**
- `filepath` (str): Input file path

---

### Functions

#### `auto_calibrate(spectrum_channels, spectrum_counts, known_peaks, tolerance=5.0)`

Automatically calibrate spectrum using known peak energies.

**Parameters:**
- `spectrum_channels` (np.ndarray): Channel numbers
- `spectrum_counts` (np.ndarray): Counts data
- `known_peaks` (List[Tuple[float, str]]): List of (energy, isotope) tuples
- `tolerance` (float): Channel tolerance for peak matching

**Returns:**
- `EnergyCalibration or None`: Calibration object if successful

---

#### `validate_calibration(calibration, test_peaks)`

Validate calibration using test peaks.

**Parameters:**
- `calibration` (Dict or EnergyCalibration): Calibration parameters or object
- `test_peaks` (List[Tuple[float, float]]): List of (channel, expected_energy) tuples

**Returns:**
- `Dict`: Validation metrics including RMS error

---

#### `apply_calibration(peaks, calibration)`

Apply energy calibration to fitted peaks.

**Parameters:**
- `peaks` (List[Dict]): List of peak dictionaries
- `calibration` (Dict or EnergyCalibration): Calibration parameters or object

**Returns:**
- `List[Dict]`: Updated peak list with energy values

---

### Constants

#### `COMMON_ISOTOPES`

Dictionary of common isotope gamma energies.

```python
COMMON_ISOTOPES = {
    'Co-60': [1173.23, 1332.49],
    'Cs-137': [661.66],
    'Na-22': [511.0, 1274.53],
    'Ba-133': [80.99, 276.40, 302.85, 356.01, 383.85],
    'Eu-152': [121.78, 244.70, 344.28, ...],
    'Am-241': [59.54],
    'Th-228': [238.63, 583.19, 860.56, 2614.51],
}
```

---

## gammafit.io_module

File input/output operations.

### Functions

#### `load_spectrum(filepath)`

Load spectrum from file with automatic format detection.

**Parameters:**
- `filepath` (str): Path to spectrum file

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (channels, counts) arrays

**Supported Formats:**
- CSV (.csv, .txt, .dat)
- SPE (.spe) - IAEA format
- CHN (.chn) - Ortec binary format
- MCA (.mca) - MCA format

---

#### `save_peaks(peaks, filepath, format='csv')`

Save detected peaks to file.

**Parameters:**
- `peaks` (List[Dict]): List of peak dictionaries
- `filepath` (str): Output file path
- `format` (str): Output format ('csv', 'json', 'txt')

---

#### `load_config(filepath)`

Load configuration from JSON file.

**Parameters:**
- `filepath` (str): Path to configuration file

**Returns:**
- `Dict`: Configuration dictionary

---

#### `export_spectrum(channels, counts, filepath, format='csv', metadata=None)`

Export spectrum data to file.

**Parameters:**
- `channels` (np.ndarray): Channel numbers or energies
- `counts` (np.ndarray): Count data
- `filepath` (str): Output file path
- `format` (str): Output format ('csv', 'txt', 'spe')
- `metadata` (Dict, optional): Optional metadata dictionary

---

## gammafit.output

Visualization and report generation functions.

### Functions

#### `plot_spectrum_with_fits(channels, counts, smoothed_counts, fitted_peaks, output_file, calibration=None, plot_style='default', show_components=True, log_scale=True)`

Create comprehensive plot of spectrum with fitted peaks.

**Parameters:**
- `channels` (np.ndarray): Channel numbers
- `counts` (np.ndarray): Raw counts
- `smoothed_counts` (np.ndarray): Smoothed counts
- `fitted_peaks` (List[Dict]): List of fitted peak dictionaries
- `output_file` (str): Output file path
- `calibration` (Dict, optional): Energy calibration parameters
- `plot_style` (str): Plot style ('default', 'publication', 'presentation')
- `show_components` (bool): Whether to show individual peak components
- `log_scale` (bool): Whether to use log scale for y-axis

---

#### `export_results(fitted_peaks, output_file, include_energy=False, format='auto')`

Export fitted peak parameters to file.

**Parameters:**
- `fitted_peaks` (List[Dict]): List of fitted peak dictionaries
- `output_file` (str): Output file path
- `include_energy` (bool): Whether to include energy column
- `format` (str): Output format ('csv', 'json', 'excel', 'latex', 'auto')

---

#### `generate_html_report(fitted_peaks, spectrum_plot, output_file, metadata=None)`

Generate an HTML report with interactive elements.

**Parameters:**
- `fitted_peaks` (List[Dict]): List of fitted peak dictionaries
- `spectrum_plot` (str): Path to spectrum plot image
- `output_file` (str): Output HTML file path
- `metadata` (Dict, optional): Optional metadata about the analysis

---

#### `create_peak_comparison_plot(peak_lists, labels, output_file)`

Create comparison plot of peaks from multiple spectra.

**Parameters:**
- `peak_lists` (List[List[Dict]]): List of peak lists from different analyses
- `labels` (List[str]): Labels for each spectrum
- `output_file` (str): Output file path

---

## gammafit.utils

Utility functions for various operations.

### Functions

#### `generate_synthetic_spectrum(num_channels=4096, peaks=None, background_level=10.0, noise_level=1.0, seed=None)`

Generate synthetic gamma spectrum for testing.

**Parameters:**
- `num_channels` (int): Number of channels
- `peaks` (List[Tuple], optional): List of (channel, amplitude, sigma) tuples
- `background_level` (float): Background count level
- `noise_level` (float): Noise amplitude
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (channels, counts)

---

#### `calculate_counting_statistics(counts)`

Calculate counting statistics for spectrum.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts

**Returns:**
- `Dict`: Statistical measures

---

#### `check_spectrum_quality(counts)`

Check spectrum quality metrics.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts

**Returns:**
- `Dict`: Quality metrics including quality score (0-100)

---

#### `calculate_activity(peak_area, efficiency, branching_ratio, live_time)`

Calculate activity from peak area.

**Parameters:**
- `peak_area` (float): Net peak area (counts)
- `efficiency` (float): Detector efficiency
- `branching_ratio` (float): Gamma emission probability
- `live_time` (float): Measurement live time (seconds)

**Returns:**
- `Dict`: Activity in Bq and µCi with uncertainties

---

#### `identify_isotope(energy, tolerance=2.0)`

Identify possible isotopes from gamma energy.

**Parameters:**
- `energy` (float): Gamma energy in keV
- `tolerance` (float): Energy tolerance in keV

**Returns:**
- `List[str]`: List of possible isotope identifications

---

#### `estimate_fwhm(counts, peak_idx)`

Estimate Full Width at Half Maximum of a peak.

**Parameters:**
- `counts` (np.ndarray): Spectrum counts
- `peak_idx` (int): Peak index

**Returns:**
- `float`: Estimated FWHM in channels

---

#### `calculate_snr(peak_height, background, background_std=None)`

Calculate signal-to-noise ratio for a peak.

**Parameters:**
- `peak_height` (float): Peak height above background
- `background` (float): Background level
- `background_std` (float, optional): Background standard deviation

**Returns:**
- `float`: Signal-to-noise ratio

---

#### `setup_logger(name='gammafit', level=logging.INFO, log_file=None)`

Setup logger with console and optional file output.

**Parameters:**
- `name` (str): Logger name
- `level` (int): Logging level
- `log_file` (str, optional): Optional log file path

**Returns:**
- `logging.Logger`: Configured logger instance

---

#### `format_uncertainty(value, uncertainty, precision=2)`

Format value with uncertainty in standard notation.

**Parameters:**
- `value` (float): Central value
- `uncertainty` (float): Uncertainty
- `precision` (int): Number of significant figures for uncertainty

**Returns:**
- `str`: Formatted string (e.g., "511.0 ± 0.5")

---

## Data Classes

### Peak Dictionary Structure

All peak-related functions return dictionaries with the following structure:

```python
{
    # Core parameters
    'centroid': float,          # Peak center position (channels)
    'centroid_err': float,      # Centroid uncertainty
    'amplitude': float,         # Peak height above background
    'amplitude_err': float,     # Amplitude uncertainty
    'sigma': float,             # Gaussian width parameter
    'sigma_err': float,         # Sigma uncertainty
    
    # Derived quantities
    'area': float,              # Integrated peak area (counts)
    'area_err': float,          # Area uncertainty
    'fwhm': float,              # Full width at half maximum
    'fwhm_err': float,          # FWHM uncertainty
    'resolution': float,        # Energy resolution (%)
    
    # Quality metrics
    'snr': float,               # Signal-to-noise ratio
    'chi_square': float,        # Reduced chi-square of fit
    'fit_success': bool,        # Whether fit converged
    'fit_message': str,         # Fit status message
    
    # Background parameters
    'bg_slope': float,          # Background slope
    'bg_intercept': float,      # Background intercept
    
    # Calibration (if applied)
    'energy': float,            # Energy in keV
    'energy_err': float,        # Energy uncertainty
    'fwhm_energy': float,       # FWHM in energy units
    
    # Metadata
    'fit_region': tuple,        # (start, end) channel indices
    'peak_model': str,          # Model used for fitting
    'background_method': str,   # Background model used
    'multiplet': bool,          # Whether part of multiplet
    'multiplet_size': int,      # Number of peaks in multiplet
    
    # Raw data
    'fit_params': np.ndarray,  # Raw fit parameters
    'fit_errors': np.ndarray,  # Parameter uncertainties
    'covariance': np.ndarray,  # Covariance matrix
}
```

### Configuration Dictionary Structure

```python
{
    'detection': {
        'min_prominence': float,
        'min_height': float,
        'min_distance': int,
        'smoothing_window': int,
        'smoothing_method': str,
        'background_method': str
    },
    
    'fitting': {
        'peak_model': str,
        'background_method': str,
        'window_scale': float,
        'max_iterations': int
    },
    
    'calibration': {
        'model': str,
        'coefficients': dict,
        'reference_peaks': list
    },
    
    'output': {
        'directory': str,
        'prefix': str,
        'generate_plot': bool,
        'plot_format': str,
        'export_fits': bool,
        'generate_report': bool
    }
}
```

## Error Handling

### Custom Exceptions

The package uses standard Python exceptions with descriptive messages:

- `ValueError`: Invalid parameters or data
- `FileNotFoundError`: Missing input files
- `RuntimeError`: Fitting or processing failures
- `TypeError`: Type mismatches

### Error Handling Example

```python
from gammafit import load_spectrum, detect_peaks, fit_peaks
import logging

logger = logging.getLogger('gammafit')

try:
    # Load spectrum
    channels, counts = load_spectrum("spectrum.csv")
    
    # Detect peaks
    peaks = detect_peaks(counts)
    if len(peaks) == 0:
        raise ValueError("No peaks detected in spectrum")
    
    # Fit peaks
    fitted_peaks = fit_peaks(channels, counts, peaks)
    
    # Check fit quality
    failed_fits = [p for p in fitted_peaks if not p['fit_success']]
    if failed_fits:
        logger.warning(f"{len(failed_fits)} peaks failed to fit properly")
    
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except ValueError as e:
    logger.error(f"Invalid data: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

## Performance Considerations

### Memory Usage

- Spectra are stored as NumPy arrays (8 bytes per channel for float64)
- 4096-channel spectrum: ~32 KB
- 16384-channel spectrum: ~128 KB
- Fitting workspace: ~10x spectrum size

### Speed Optimization

```python
# Use vectorized operations
import numpy as np

# Slow
result = []
for x in data:
    result.append(x * 2)

# Fast
result = data * 2

# Use appropriate data types
channels = np.arange(4096, dtype=np.float32)  # Save memory
counts = np.array(counts, dtype=np.float64)    # Precision for fitting
```

### Parallel Processing

```python
from multiprocessing import Pool
from gammafit import fit_single_peak

def fit_peak_wrapper(args):
    channels, counts, peak_idx = args
    return fit_single_peak(channels, counts, peak_idx)

# Parallel fitting
with Pool() as pool:
    args = [(channels, counts, idx) for idx in peak_indices]
    fitted_peaks = pool.map(fit_peak_wrapper, args)
```

## Version History

### Version 0.1.0 (Current)
- Initial release
- Core peak detection and fitting
- Linear and polynomial calibration
- Multiple file format support
- Batch processing capabilities
- HTML report generation

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@software{gammafit2024,
  title = {GammaFit: Automated Gamma Spectroscopy Analysis},
  author = {drgtheneutrino},
  year = {2024},
  url = {https://github.com/drgtheneutrino/gamma-peak-identifier},
  version = {0.1.0}
}
```

---

*For examples and tutorials, see the [User Guide](user_guide.md) and [Examples](../examples/)*
