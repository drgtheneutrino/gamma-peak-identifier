"""
Input/Output operations for gamma spectroscopy data.

This module handles reading various spectrum file formats and saving results.
Supported formats: CSV, SPE, CHN, MCA (with appropriate parsers).
"""

import json
import struct
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd


def load_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from file with automatic format detection.
    
    Parameters:
        filepath: Path to spectrum file
        
    Returns:
        tuple: (channels, counts) as numpy arrays
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or data is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Spectrum file not found: {filepath}")
    
    # Detect file format by extension
    extension = filepath.suffix.lower()
    
    if extension in ['.csv', '.txt', '.dat']:
        return load_csv_spectrum(filepath)
    elif extension == '.spe':
        return load_spe_spectrum(filepath)
    elif extension == '.chn':
        return load_chn_spectrum(filepath)
    elif extension == '.mca':
        return load_mca_spectrum(filepath)
    else:
        # Try to load as CSV by default
        try:
            return load_csv_spectrum(filepath)
        except Exception:
            raise ValueError(f"Unsupported file format: {extension}")


def load_csv_spectrum(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from CSV file.
    
    Expected format: Two columns (channel/energy, counts)
    Can handle files with or without headers.
    
    Parameters:
        filepath: Path to CSV file
        
    Returns:
        tuple: (channels, counts) arrays
    """
    try:
        # Try to read with pandas, auto-detect delimiter
        data = pd.read_csv(filepath, header=None, comment='#')
        
        # Handle different CSV formats
        if data.shape[1] < 2:
            # Single column - assume it's counts only
            counts = data.iloc[:, 0].values
            channels = np.arange(len(counts))
        else:
            # Two or more columns - use first two
            channels = data.iloc[:, 0].values
            counts = data.iloc[:, 1].values
        
        # Validate data
        if len(channels) != len(counts):
            raise ValueError("Channel and counts arrays must have same length")
        if len(channels) == 0:
            raise ValueError("Empty spectrum file")
        
        # Ensure proper data types
        channels = channels.astype(float)
        counts = counts.astype(float)
        
        # Check for negative counts
        if np.any(counts < 0):
            print("Warning: Negative counts detected, setting to zero")
            counts = np.maximum(counts, 0)
        
        return channels, counts
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def load_spe_spectrum(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from IAEA SPE format file.
    
    Parameters:
        filepath: Path to SPE file
        
    Returns:
        tuple: (channels, counts) arrays
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse SPE format
        # Format: $SPEC_ID, $SPEC_REM, $DATE_MEA, $MEAS_TIM, $DATA, counts..., $ROI, $PRESETS, $ENER_FIT, $MCA_CAL
        
        in_data_section = False
        counts_list = []
        num_channels = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('$DATA'):
                in_data_section = True
                # Next line should be "start_channel end_channel"
                if i + 1 < len(lines):
                    data_range = lines[i + 1].strip().split()
                    if len(data_range) == 2:
                        start_ch = int(data_range[0])
                        end_ch = int(data_range[1])
                        num_channels = end_ch - start_ch + 1
                continue
            
            if in_data_section:
                if line.startswith('$'):
                    # End of data section
                    break
                try:
                    counts_list.append(float(line))
                except ValueError:
                    continue
        
        if not counts_list:
            raise ValueError("No data found in SPE file")
        
        counts = np.array(counts_list)
        channels = np.arange(len(counts))
        
        return channels, counts
        
    except Exception as e:
        raise ValueError(f"Error reading SPE file: {e}")


def load_chn_spectrum(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from Ortec CHN format file.
    
    Parameters:
        filepath: Path to CHN file
        
    Returns:
        tuple: (channels, counts) arrays
    """
    try:
        with open(filepath, 'rb') as f:
            # CHN format is binary
            # Read header (32 bytes)
            header = f.read(32)
            
            # Parse header to get number of channels
            # Bytes 4-5: number of channels (little-endian)
            num_channels = struct.unpack('<H', header[4:6])[0]
            
            # Read channel data (4 bytes per channel)
            counts = []
            for _ in range(num_channels):
                count = struct.unpack('<I', f.read(4))[0]
                counts.append(count)
        
        counts = np.array(counts, dtype=float)
        channels = np.arange(len(counts))
        
        return channels, counts
        
    except Exception as e:
        raise ValueError(f"Error reading CHN file: {e}")


def load_mca_spectrum(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from MCA format file.
    
    Parameters:
        filepath: Path to MCA file
        
    Returns:
        tuple: (channels, counts) arrays
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        in_data_section = False
        counts_list = []
        
        for line in lines:
            line = line.strip()
            
            if line == '<<DATA>>':
                in_data_section = True
                continue
            
            if line == '<<END>>':
                break
            
            if in_data_section and line:
                try:
                    counts_list.append(float(line))
                except ValueError:
                    continue
        
        if not counts_list:
            raise ValueError("No data found in MCA file")
        
        counts = np.array(counts_list)
        channels = np.arange(len(counts))
        
        return channels, counts
        
    except Exception as e:
        raise ValueError(f"Error reading MCA file: {e}")


def save_peaks(peaks: List[Dict], filepath: str, format: str = 'csv'):
    """
    Save detected peaks to file.
    
    Parameters:
        peaks: List of peak dictionaries
        filepath: Output file path
        format: Output format ('csv', 'json', 'txt')
    """
    filepath = Path(filepath)
    
    if format == 'csv':
        df = pd.DataFrame(peaks)
        df.to_csv(filepath, index=False)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(peaks, f, indent=2, default=str)
    elif format == 'txt':
        with open(filepath, 'w') as f:
            # Write header
            f.write("# Peak Analysis Results\n")
            f.write("#" + "="*50 + "\n")
            f.write(f"# {'Peak':<5} {'Centroid':<12} {'Area':<12} {'FWHM':<10} {'SNR':<8}\n")
            f.write("#" + "-"*50 + "\n")
            
            # Write data
            for i, peak in enumerate(peaks, 1):
                f.write(f"{i:<6} {peak['centroid']:<12.2f} {peak['area']:<12.0f} "
                       f"{peak['fwhm']:<10.2f} {peak['snr']:<8.2f}\n")
    else:
        raise ValueError(f"Unsupported output format: {format}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Parameters:
        filepath: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file.
    
    Parameters:
        config: Configuration dictionary
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def export_spectrum(channels: np.ndarray, counts: np.ndarray, 
                   filepath: str, format: str = 'csv',
                   metadata: Optional[Dict] = None):
    """
    Export spectrum data to file.
    
    Parameters:
        channels: Channel numbers or energies
        counts: Count data
        filepath: Output file path
        format: Output format ('csv', 'txt', 'spe')
        metadata: Optional metadata dictionary
    """
    filepath = Path(filepath)
    
    if format == 'csv':
        df = pd.DataFrame({'channel': channels, 'counts': counts})
        df.to_csv(filepath, index=False)
    
    elif format == 'txt':
        with open(filepath, 'w') as f:
            # Write header with metadata if provided
            if metadata:
                f.write("# Spectrum Data\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("#\n")
            
            # Write data
            for ch, cnt in zip(channels, counts):
                f.write(f"{ch:.1f}\t{cnt:.1f}\n")
    
    elif format == 'spe':
        # Write IAEA SPE format
        with open(filepath, 'w') as f:
            f.write("$SPEC_ID:\n")
            f.write("GammaFit Export\n")
            f.write("$SPEC_REM:\n")
            f.write("Exported spectrum\n")
            f.write("$DATA:\n")
            f.write(f"0 {len(counts)-1}\n")
            for count in counts:
                f.write(f"{int(count)}\n")
            f.write("$ROI:\n")
            f.write("0\n")
            f.write("$PRESETS:\n")
            f.write("None\n")
            f.write("$ENER_FIT:\n")
            f.write("0.0 1.0\n")
            f.write("$MCA_CAL:\n")
            f.write("2\n")
            f.write("0.0 1.0\n")
    
    else:
        raise ValueError(f"Unsupported export format: {format}")
