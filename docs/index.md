# GammaFit Documentation

Welcome to the GammaFit documentation for gamma spectroscopy peak analysis.

![GitHub](https://img.shields.io/badge/github-gamma--peak--identifier-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

GammaFit is a comprehensive Python package for automatic detection, fitting, and analysis of peaks in gamma-ray spectroscopy data. It provides professional-grade tools for nuclear spectroscopy analysis with an easy-to-use interface.

## Key Features

- 🔍 **Automatic Peak Detection** - Multiple algorithms for reliable peak identification
- 📊 **Advanced Peak Fitting** - Gaussian, Voigt, and multiplet deconvolution
- 🎯 **Energy Calibration** - Linear and polynomial calibration with isotope libraries
- 📈 **Publication-Quality Plots** - Professional visualization tools
- 🔄 **Batch Processing** - Analyze multiple spectra efficiently
- 📁 **Multiple File Formats** - Support for CSV, SPE, CHN, MCA formats
- 📝 **Comprehensive Reports** - HTML and PDF report generation
- 🧪 **Isotope Identification** - Built-in nuclear data libraries

## Quick Navigation

### Getting Started
- [Installation Guide](installation.md) - Setup instructions and requirements
- [Quick Start Tutorial](user_guide.md#quick-start) - Get analyzing in 5 minutes
- [Example Gallery](user_guide.md#examples) - See what GammaFit can do

### User Documentation
- [User Guide](user_guide.md) - Complete usage instructions
- [Command Line Interface](user_guide.md#command-line-interface) - CLI reference
- [Configuration Options](user_guide.md#configuration) - Customization guide
- [File Formats](user_guide.md#file-formats) - Supported input/output formats

### Technical Documentation
- [API Reference](api_reference.md) - Complete API documentation
- [Algorithms](algorithms.md) - Mathematical methods and theory
- [Calibration Methods](algorithms.md#calibration) - Energy calibration techniques
- [Peak Fitting Models](algorithms.md#peak-fitting) - Mathematical models

### Examples & Tutorials
- [Basic Analysis](user_guide.md#basic-analysis) - Simple spectrum analysis
- [Batch Processing](user_guide.md#batch-processing) - Multiple spectra
- [Energy Calibration](user_guide.md#calibration) - Calibration workflows
- [Advanced Features](user_guide.md#advanced-features) - Power user tools

## Quick Example

```python
from gammafit import load_spectrum, detect_peaks, fit_peaks, plot_spectrum_with_fits

# Load spectrum
channels, counts = load_spectrum("spectrum.csv")

# Detect peaks
peaks = detect_peaks(counts, min_prominence=50)

# Fit peaks
fitted_peaks = fit_peaks(channels, counts, peaks)

# Generate plot
plot_spectrum_with_fits(channels, counts, counts, fitted_peaks, "output.png")

# Print results
for i, peak in enumerate(fitted_peaks, 1):
    print(f"Peak {i}: Channel {peak['centroid']:.1f}, Area {peak['area']:.0f}")
```

## System Requirements

- Python 3.7 or higher
- NumPy, SciPy, Matplotlib, Pandas
- 4GB RAM minimum (8GB recommended for large spectra)
- Windows, macOS, or Linux

## Project Structure

```
gamma-peak-identifier/
├── gammafit/              # Main package
│   ├── detection.py       # Peak detection algorithms
│   ├── fitting.py         # Peak fitting routines
│   ├── calibration.py     # Energy calibration
│   ├── io_module.py       # File I/O operations
│   └── output.py          # Visualization and export
├── examples/              # Example scripts and data
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Performance

GammaFit is optimized for both accuracy and speed:

- Analyze a 4096-channel spectrum in <1 second
- Batch process 100 spectra in <30 seconds
- Handle spectra up to 16k channels
- Detect peaks with SNR as low as 3:1

## Citation

If you use GammaFit in your research, please cite:

```bibtex
@software{gammafit2024,
  title = {GammaFit: Automated Gamma Spectroscopy Analysis},
  author = {drgtheneutrino},
  year = {2024},
  url = {https://github.com/drgtheneutrino/gamma-peak-identifier}
}
```

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/drgtheneutrino/gamma-peak-identifier/issues)
- **Documentation**: You're reading it!
- **Examples**: See the [examples/](https://github.com/drgtheneutrino/gamma-peak-identifier/tree/main/examples) directory

## License

GammaFit is released under the MIT License. See [LICENSE](https://github.com/drgtheneutrino/gamma-peak-identifier/blob/main/LICENSE) for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/drgtheneutrino/gamma-peak-identifier/blob/main/CONTRIBUTING.md) for details.

---

*Last updated: 2024*
