# User Guide

Complete guide to using GammaFit for gamma spectroscopy analysis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Analysis](#basic-analysis)
3. [Command Line Interface](#command-line-interface)
4. [Configuration](#configuration)
5. [File Formats](#file-formats)
6. [Peak Detection](#peak-detection)
7. [Peak Fitting](#peak-fitting)
8. [Energy Calibration](#energy-calibration)
9. [Batch Processing](#batch-processing)
10. [Visualization](#visualization)
11. [Advanced Features](#advanced-features)
12. [Examples](#examples)
13. [Best Practices](#best-practices)

## Quick Start

Get started with GammaFit in 5 minutes:

```python
from gammafit import load_spectrum, detect_peaks, fit_peaks, plot_spectrum_with_fits

# Load your spectrum
channels, counts = load_spectrum("your_spectrum.csv")

# Detect peaks automatically
peak_indices = detect_peaks(counts, min_prominence=50)

# Fit peaks with Gaussian model
fitted_peaks = fit_peaks(channels, counts, peak_indices)

# Generate visualization
plot_spectrum_with_fits(channels, counts, counts, fitted_peaks, "output.png")

# Print results
for peak in fitted_peaks:
    print(f"Peak at channel {peak['centroid']:.1f}: Area = {peak['area']:.0f}")
```

## Basic Analysis

### Step 1: Load Spectrum Data

GammaFit supports multiple file formats:

```python
from gammafit import load_spectrum

# CSV format (most common)
channels, counts = load_spectrum("spectrum.csv")

# SPE format (IAEA standard)
channels, counts = load_spectrum("spectrum.spe")

# CHN format (Ortec)
channels, counts = load_spectrum("spectrum.chn")

# MCA format
channels, counts = load_spectrum("spectrum.mca")
```

### Step 2: Preprocess Data

Apply smoothing to reduce noise:

```python
from gammafit import smooth_spectrum

# Savitzky-Golay filter (preserves peak shape)
smoothed = smooth_spectrum(counts, window_length=7, method='savgol')

# Gaussian filter
smoothed = smooth_spectrum(counts, window_length=7, method='gaussian')
```

### Step 3: Detect Peaks

Find peaks in your spectrum:

```python
from gammafit import detect_peaks

# Basic detection
peaks = detect_peaks(smoothed, min_prominence=50, min_height=30)

# Advanced detection with background subtraction
from gammafit.detection import estimate_background

background = estimate_background(counts, method='snip')
net_counts = counts - background
peaks = detect_peaks(net_counts, min_prominence=30)
```

### Step 4: Fit Peaks

Fit mathematical models to peaks:

```python
from gammafit import fit_peaks

# Simple Gaussian fitting
fitted_peaks = fit_peaks(channels, counts, peaks)

# Advanced fitting options
fitted_peaks = fit_peaks(
    channels, counts, peaks,
    peak_model='gaussian',       # or 'voigt', 'gaussian_tail'
    background_method='linear',   # or 'quadratic', 'step'
    window_scale=3.0             # Fitting window size
)
```

### Step 5: View Results

```python
# Print peak parameters
for i, peak in enumerate(fitted_peaks, 1):
    print(f"Peak {i}:")
    print(f"  Centroid: {peak['centroid']:.2f} ± {peak['centroid_err']:.2f}")
    print(f"  Area: {peak['area']:.0f} ± {peak['area_err']:.0f}")
    print(f"  FWHM: {peak['fwhm']:.2f}")
    print(f"  SNR: {peak['snr']:.1f}")
```

## Command Line Interface

### Basic Usage

```bash
# Analyze a spectrum
python -m gammafit.main spectrum.csv

# With options
python -m gammafit.main spectrum.csv \
    --min-prominence 50 \
    --min-height 30 \
    --calibration 0.5,0 \
    --output-dir results/
```

### CLI Options

```bash
python -m gammafit.main --help

Required arguments:
  spectrum              Path to spectrum file

Optional arguments:
  --config FILE         Configuration file
  --calibration A,B     Linear calibration (E = A*ch + B)
  --min-prominence N    Minimum peak prominence (default: 50)
  --min-height N        Minimum peak height (default: 30)
  --min-distance N      Minimum distance between peaks (default: 5)
  --smoothing-window N  Smoothing window size (default: 7)
  --smoothing-method M  Method: savgol, gaussian, none (default: savgol)
  --output-dir DIR      Output directory (default: current)
  --output-prefix PRE   Output file prefix
  --no-plot            Skip plot generation
  --plot-format FMT    Plot format: png, pdf, svg (default: png)
  --export-fits        Export detailed fit data
  --verbose            Enable verbose output
  --debug              Enable debug mode
```

### Examples

```bash
# Basic analysis
python -m gammafit.main data/na22.csv

# With energy calibration
python -m gammafit.main data/na22.csv --calibration 0.5,10

# Batch processing
for file in data/*.csv; do
    python -m gammafit.main "$file" --output-dir results/
done

# Using configuration file
python -m gammafit.main spectrum.csv --config config.json
```

## Configuration

### Configuration File Format

Create a `config.json` file:

```json
{
    "detection": {
        "min_prominence": 50,
        "min_height": 30,
        "min_distance": 5,
        "smoothing_window": 7,
        "smoothing_method": "savgol",
        "background_method": "snip"
    },
    
    "fitting": {
        "peak_model": "gaussian",
        "background_method": "linear",
        "window_scale": 3.0,
        "max_iterations": 5000
    },
    
    "calibration": {
        "model": "linear",
        "coefficients": {
            "a": 0.5,
            "b": 0.0
        }
    },
    
    "output": {
        "directory": "./results",
        "prefix": "analysis_",
        "generate_plot": true,
        "plot_format": "png",
        "export_fits": true,
        "generate_report": true
    }
}
```

### Using Configuration in Python

```python
from gammafit.io_module import load_config
from gammafit.main import process_spectrum

# Load configuration
config = load_config("config.json")

# Process with configuration
process_spectrum("spectrum.csv", config, logger)
```

## File Formats

### Input Formats

#### CSV Format
```csv
# Two columns: channel, counts
0,145
1,152
2,148
...
```

#### SPE Format (IAEA)
```
$SPEC_ID:
Sample Description
$DATA:
0 4095
counts...
$ROI:
...
```

### Output Formats

#### Peak List (CSV)
```csv
peak_number,centroid_channel,area,fwhm,snr,energy_keV
1,511.2,10523,5.3,45.2,511.0
2,1274.5,8234,7.1,38.1,1274.5
```

#### JSON Export
```json
{
    "peaks": [
        {
            "centroid": 511.2,
            "area": 10523,
            "fwhm": 5.3,
            "snr": 45.2,
            "fit_success": true
        }
    ],
    "metadata": {
        "total_counts": 50000,
        "analysis_time": "2024-01-01T12:00:00"
    }
}
```

## Peak Detection

### Detection Algorithms

#### 1. SciPy Method (Default)
```python
peaks = detect_peaks(counts, method='scipy')
```
- Uses local maxima detection
- Statistical significance testing
- Good for well-separated peaks

#### 2. Derivative Method
```python
peaks = detect_peaks(counts, method='derivative')
```
- Finds zero-crossings of derivative
- Good for overlapping peaks
- Sensitive to noise

#### 3. Template Matching
```python
peaks = detect_peaks(counts, method='template')
```
- Correlates with Gaussian template
- Robust to noise
- Good for weak peaks

### Detection Parameters

```python
peaks = detect_peaks(
    counts,
    min_prominence=50,      # Peak prominence above background
    min_height=30,          # Absolute minimum height
    min_distance=5,         # Minimum separation (channels)
    rel_height=0.5,         # Relative height for width calculation
    threshold=None,         # Absolute threshold
    width=(1, None)         # Peak width range
)
```

### Background Estimation

```python
from gammafit.detection import estimate_background

# SNIP method (recommended)
background = estimate_background(counts, method='snip', iterations=20)

# Rolling ball
background = estimate_background(counts, method='rolling_ball', radius=50)

# Percentile
background = estimate_background(counts, method='percentile', percentile=10)
```

## Peak Fitting

### Fitting Models

#### Gaussian (Default)
```python
fitted = fit_peaks(channels, counts, peaks, peak_model='gaussian')
```
Best for: Most gamma peaks, good resolution detectors

#### Gaussian with Tail
```python
fitted = fit_peaks(channels, counts, peaks, peak_model='gaussian_tail')
```
Best for: Peaks with low-energy tailing (detector effects)

#### Voigt Profile
```python
fitted = fit_peaks(channels, counts, peaks, peak_model='voigt')
```
Best for: High count rate, Doppler broadening

### Background Models

```python
# Linear background (default)
fitted = fit_peaks(channels, counts, peaks, background_method='linear')

# Quadratic background
fitted = fit_peaks(channels, counts, peaks, background_method='quadratic')

# Step function (for Compton edges)
fitted = fit_peaks(channels, counts, peaks, background_method='step')
```

### Multiplet Deconvolution

```python
from gammafit.fitting import fit_multiplet

# Identify overlapping peaks
groups = identify_overlapping_peaks(channels, counts, peaks)

# Fit multiplet
for group in groups:
    if len(group) > 1:
        multiplet_results = fit_multiplet(channels, counts, group)
```

## Energy Calibration

### Linear Calibration

```python
from gammafit.calibration import EnergyCalibration

# Create calibration
calibration = EnergyCalibration(model='linear')

# Add calibration points
calibration.add_point(channel=511, energy=511.0, isotope='Na-22')
calibration.add_point(channel=1323, energy=661.66, isotope='Cs-137')

# Fit calibration
calibration.fit()

# Apply to peaks
from gammafit import apply_calibration
calibrated_peaks = apply_calibration(fitted_peaks, calibration)
```

### Automatic Calibration

```python
from gammafit.calibration import auto_calibrate

# Define known peaks
known_peaks = [
    (511.0, 'Na-22'),
    (661.66, 'Cs-137'),
    (1274.53, 'Na-22')
]

# Auto-calibrate
calibration = auto_calibrate(channels, counts, known_peaks, tolerance=10)
```

### Polynomial Calibration

```python
# For non-linear detectors
calibration = EnergyCalibration(model='quadratic')
# Add multiple calibration points...
calibration.fit()
```

### Isotope Identification

```python
from gammafit.utils import identify_isotope

for peak in calibrated_peaks:
    if 'energy' in peak:
        isotopes = identify_isotope(peak['energy'], tolerance=2.0)
        print(f"Peak at {peak['energy']:.1f} keV: {isotopes}")
```

## Batch Processing

### Process Multiple Files

```python
from examples.batch_process import BatchProcessor

# Initialize processor
processor = BatchProcessor(config_file="config.json")

# Process directory
processor.process_directory("data/", pattern="*.csv")

# Export results
processor.export_results("results/")

# Generate report
processor.generate_report("results/")
```

### Parallel Processing

```python
from multiprocessing import Pool
from gammafit import load_spectrum, detect_peaks, fit_peaks

def process_file(filename):
    channels, counts = load_spectrum(filename)
    peaks = detect_peaks(counts)
    return fit_peaks(channels, counts, peaks)

# Process in parallel
with Pool() as pool:
    results = pool.map(process_file, file_list)
```

## Visualization

### Basic Plotting

```python
from gammafit import plot_spectrum_with_fits

plot_spectrum_with_fits(
    channels, counts, smoothed_counts, fitted_peaks,
    "output.png",
    calibration=calibration.coefficients,
    plot_style='default',      # or 'publication', 'presentation'
    show_components=True,       # Show individual peak components
    log_scale=True             # Logarithmic y-axis
)
```

### Custom Plots

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Spectrum plot
ax1.semilogy(channels, counts, 'b-', alpha=0.5, label='Raw')
ax1.semilogy(channels, smoothed, 'r-', label='Smoothed')

# Mark peaks
for peak in fitted_peaks:
    ax1.axvline(peak['centroid'], color='g', linestyle='--', alpha=0.5)

# Residuals
ax2.plot(channels, residuals, 'b-')
ax2.axhline(y=0, color='r')

plt.savefig('custom_plot.png')
```

### Interactive Plots

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add spectrum
fig.add_trace(go.Scatter(
    x=channels, y=counts,
    mode='lines',
    name='Spectrum'
))

# Add peaks
for i, peak in enumerate(fitted_peaks):
    fig.add_trace(go.Scatter(
        x=[peak['centroid']],
        y=[peak['amplitude']],
        mode='markers',
        name=f'Peak {i+1}'
    ))

fig.show()
```

## Advanced Features

### Custom Peak Models

```python
def custom_peak_model(x, amplitude, centroid, width, asymmetry):
    """Custom asymmetric peak model."""
    # Your model here
    return amplitude * np.exp(-((x - centroid) / width)**2) * (1 + asymmetry * (x - centroid))

# Use in fitting
from scipy.optimize import curve_fit
popt, pcov = curve_fit(custom_peak_model, x_data, y_data)
```

### Uncertainty Propagation

```python
from uncertainties import ufloat

# Create values with uncertainties
peak_area = ufloat(10000, 100)  # 10000 ± 100
efficiency = ufloat(0.15, 0.01)  # 0.15 ± 0.01

# Calculate activity with propagated uncertainty
activity = peak_area / efficiency
