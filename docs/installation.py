# Installation Guide

This guide covers the installation of GammaFit on various platforms.

## Table of Contents
- [Requirements](#requirements)
- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [Development Installation](#development-installation)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## Requirements

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+, CentOS 7+)
- **Python**: 3.7 or higher (3.9+ recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 500MB for installation

### Python Dependencies
```
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
pandas>=1.1.0
```

### Optional Dependencies
```
lmfit>=1.0.0        # Advanced fitting models
uncertainties>=3.1.0 # Error propagation
pytest>=6.0.0       # Running tests
sphinx>=3.2.0       # Building documentation
jupyter>=1.0.0      # Interactive notebooks
```

## Quick Install

### Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install gammafit

# Or install from GitHub
pip install git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
```

### Using conda

```bash
# Create new environment
conda create -n gammafit python=3.9
conda activate gammafit

# Install dependencies
conda install numpy scipy matplotlib pandas

# Install GammaFit
pip install git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
```

## Detailed Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/drgtheneutrino/gamma-peak-identifier.git

# Navigate to the directory
cd gamma-peak-identifier
```

### Step 2: Create Virtual Environment (Recommended)

#### Using venv
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

#### Using conda
```bash
# Create conda environment
conda create -n gammafit python=3.9
conda activate gammafit
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install optional packages
pip install -r requirements-dev.txt  # For development
pip install -r requirements-docs.txt # For documentation
```

### Step 4: Install GammaFit

```bash
# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .
```

## Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/drgtheneutrino/gamma-peak-identifier.git
cd gamma-peak-identifier

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # Linux/macOS
# or
venv-dev\Scripts\activate  # Windows

# Install in development mode with all extras
pip install -e ".[dev,docs,test]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify
pytest tests/
```

## Platform-Specific Instructions

### Windows

1. **Install Python**:
   - Download from [python.org](https://python.org)
   - Check "Add Python to PATH" during installation

2. **Install Visual C++ Build Tools** (for some dependencies):
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install "Desktop development with C++"

3. **Install GammaFit**:
   ```cmd
   python -m pip install --upgrade pip
   pip install git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
   ```

### macOS

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**:
   ```bash
   brew install python@3.9
   ```

3. **Install GammaFit**:
   ```bash
   pip3 install git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
   ```

### Linux (Ubuntu/Debian)

1. **Install system dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-dev python3-venv
   sudo apt install gcc g++ gfortran  # For scipy compilation
   ```

2. **Install GammaFit**:
   ```bash
   pip3 install git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
   ```

### Linux (CentOS/RHEL/Fedora)

1. **Install system dependencies**:
   ```bash
   sudo yum install python3 python3-pip python3-devel
   sudo yum install gcc gcc-c++ gcc-gfortran
   ```

2. **Install GammaFit**:
   ```bash
   pip3 install git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
   ```

## Docker Installation

Use Docker for a containerized installation:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install .

CMD ["python", "-m", "gammafit.main", "--help"]
```

Build and run:
```bash
docker build -t gammafit .
docker run -v $(pwd)/data:/data gammafit python -m gammafit.main /data/spectrum.csv
```

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'gammafit'
**Solution**: Ensure GammaFit is installed and your virtual environment is activated.
```bash
pip show gammafit  # Check if installed
pip install -e .   # Reinstall
```

#### 2. NumPy/SciPy installation fails
**Solution**: Install pre-compiled wheels or use conda:
```bash
# Use wheels
pip install --only-binary :all: numpy scipy

# Or use conda
conda install numpy scipy
```

#### 3. Matplotlib backend issues
**Solution**: Set the backend explicitly:
```python
import matplotlib
matplotlib.use('Agg')  # For headless systems
import matplotlib.pyplot as plt
```

#### 4. Permission denied errors
**Solution**: Use user installation:
```bash
pip install --user git+https://github.com/drgtheneutrino/gamma-peak-identifier.git
```

#### 5. Memory errors with large spectra
**Solution**: Increase available memory or process in chunks:
```python
# Process in segments
for start in range(0, len(spectrum), chunk_size):
    segment = spectrum[start:start+chunk_size]
    # Process segment
```

### Environment Variables

Set these for custom configurations:

```bash
# Set data directory
export GAMMAFIT_DATA_DIR=/path/to/data

# Set configuration file
export GAMMAFIT_CONFIG=/path/to/config.json

# Set logging level
export GAMMAFIT_LOG_LEVEL=DEBUG
```

## Verification

### Basic Test

After installation, verify it works:

```python
# Python test
python -c "import gammafit; print(gammafit.__version__)"
```

### Command Line Test

```bash
# Check CLI
python -m gammafit.main --help

# Run example
cd examples
python run_example.py
```

### Run Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=gammafit

# Run specific test
pytest tests/test_detection.py
```

### Performance Test

```python
from gammafit.utils import generate_synthetic_spectrum
from gammafit import detect_peaks, fit_peaks
import time

# Generate test spectrum
channels, counts = generate_synthetic_spectrum(4096)

# Time analysis
start = time.time()
peaks = detect_peaks(counts)
fitted = fit_peaks(channels, counts, peaks)
elapsed = time.time() - start

print(f"Analysis completed in {elapsed:.2f} seconds")
print(f"Found {len(fitted)} peaks")
```

## Updating

### Update to Latest Version

```bash
# Using pip
pip install --upgrade git+https://github.com/drgtheneutrino/gamma-peak-identifier.git

# If installed in development mode
cd gamma-peak-identifier
git pull
pip install -e .
```

### Check Version

```python
import gammafit
print(gammafit.__version__)
```

## Uninstallation

```bash
# Uninstall package
pip uninstall gammafit

# Remove virtual environment
deactivate
rm -rf venv/  # Linux/macOS
# or
rmdir /s venv  # Windows
```

## Next Steps

- Read the [User Guide](user_guide.md) to learn how to use GammaFit
- Try the [Examples](../examples/) to see GammaFit in action
- Check the [API Reference](api_reference.md) for detailed documentation

---

*Need help? Open an issue on [GitHub](https://github.com/drgtheneutrino/gamma-peak-identifier/issues)*
