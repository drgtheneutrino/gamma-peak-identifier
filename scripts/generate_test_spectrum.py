#!/usr/bin/env python3
"""
Generate synthetic gamma spectra for testing and development.

This script creates realistic synthetic gamma spectra with:
- Known peak positions and intensities
- Realistic background
- Poisson noise
- Optional detector effects
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Optional, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit.utils import generate_synthetic_spectrum


def create_isotope_spectrum(isotope: str, 
                          num_channels: int = 4096,
                          calibration: Tuple[float, float] = (0.5, 0),
                          intensity_scale: float = 1000,
                          detector_resolution: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spectrum for a specific isotope.
    
    Parameters:
        isotope: Isotope name ('Na-22', 'Co-60', 'Cs-137', etc.)
        num_channels: Number of channels
        calibration: (a, b) for E = a*ch + b
        intensity_scale: Overall intensity scaling
        detector_resolution: Relative resolution (FWHM/E)
    
    Returns:
        Tuple of (channels, counts)
    """
    # Define isotope libraries
    isotope_data = {
        'Na-22': [
            (511.0, 1.80, 'Annihilation'),
            (1274.53, 0.999, 'Primary gamma')
        ],
        'Co-60': [
            (1173.23, 0.999, 'First gamma'),
            (1332.49, 0.999, 'Second gamma')
        ],
        'Cs-137': [
            (661.66, 0.85, 'Primary gamma')
        ],
        'Ba-133': [
            (80.99, 0.34, 'Ka X-ray'),
            (276.40, 0.071, 'Gamma'),
            (302.85, 0.183, 'Gamma'),
            (356.01, 0.621, 'Primary gamma'),
            (383.85, 0.089, 'Gamma')
        ],
        'Eu-152': [
            (121.78, 0.284, 'Primary gamma'),
            (244.70, 0.075, 'Gamma'),
            (344.28, 0.266, 'Primary gamma'),
            (778.90, 0.129, 'Gamma'),
            (964.08, 0.146, 'Gamma'),
            (1408.01, 0.208, 'Primary gamma')
        ],
        'Am-241': [
            (59.54, 0.359, 'Primary gamma')
        ],
        'Background': [
            (1460.82, 0.1, 'K-40'),
            (2614.51, 0.05, 'Tl-208')
        ]
    }
    
    if isotope not in isotope_data:
        raise ValueError(f"Unknown isotope: {isotope}. Available: {list(isotope_data.keys())}")
    
    # Initialize spectrum
    channels = np.arange(num_channels)
    counts = np.zeros(num_channels)
    
    # Add peaks for the isotope
    a, b = calibration
    peaks_added = []
    
    for energy, branching_ratio, description in isotope_data[isotope]:
        # Convert energy to channel
        channel = (energy - b) / a
        
        if 0 <= channel < num_channels:
            # Calculate peak width (detector resolution)
            sigma = (detector_resolution * energy) / (2.355 * a)
            
            # Calculate amplitude based on branching ratio
            amplitude = intensity_scale * branching_ratio
            
            # Add Gaussian peak
            peak = amplitude * np.exp(-0.5 * ((channels - channel) / sigma) ** 2)
            counts += peak
            
            peaks_added.append({
                'energy': energy,
                'channel': channel,
                'amplitude': amplitude,
                'sigma': sigma,
                'description': description
            })
    
    # Add realistic background
    # Exponential background
    background = 10 * np.exp(-channels / (num_channels / 2))
    # Add constant background
    background += 5
    # Add small random fluctuations
    background += np.random.randn(num_channels) * 0.5
    counts += background
    
    # Apply Poisson noise
    counts = np.maximum(counts, 0)  # Ensure non-negative
    counts = np.random.poisson(counts)
    
    # Print peak information
    print(f"Generated {isotope} spectrum with {len(peaks_added)} peaks:")
    for peak in peaks_added:
        print(f"  {peak['energy']:.2f} keV at channel {peak['channel']:.1f}: {peak['description']}")
    
    return channels.astype(float), counts.astype(float), peaks_added


def create_complex_spectrum(num_channels: int = 4096,
                          isotopes: List[str] = None,
                          background_level: float = 20,
                          noise_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create complex spectrum with multiple isotopes.
    
    Parameters:
        num_channels: Number of channels
        isotopes: List of isotope names
        background_level: Background level
        noise_scale: Noise scaling factor
    
    Returns:
        Tuple of (channels, counts)
    """
    if isotopes is None:
        isotopes = ['Na-22', 'Cs-137', 'Co-60']
    
    channels = np.arange(num_channels)
    counts = np.zeros(num_channels)
    
    # Calibration: approximately 0.5 keV/channel
    calibration = (0.5, 10)  # E = 0.5*ch + 10
    
    all_peaks = []
    
    # Add each isotope
    for isotope in isotopes:
        print(f"\nAdding {isotope}...")
        _, isotope_counts, peaks = create_isotope_spectrum(
            isotope,
            num_channels,
            calibration,
            intensity_scale=np.random.uniform(500, 2000)  # Random intensity
        )
        counts += isotope_counts
        all_peaks.extend(peaks)
    
    # Add additional background
    background = background_level * np.exp(-channels / (num_channels / 2))
    background += np.random.randn(num_channels) * noise_scale
    counts += background
    
    # Final Poisson noise
    counts = np.maximum(counts, 0)
    counts = np.random.poisson(counts)
    
    return channels.astype(float), counts.astype(float), all_peaks


def create_detector_effects_spectrum(base_spectrum: np.ndarray,
                                   detector_type: str = 'NaI',
                                   add_pileup: bool = True,
                                   add_escape_peaks: bool = True,
                                   add_backscatter: bool = True) -> np.ndarray:
    """
    Add realistic detector effects to spectrum.
    
    Parameters:
        base_spectrum: Input spectrum
        detector_type: Detector type ('NaI', 'HPGe', 'CZT')
        add_pileup: Add pile-up peaks
        add_escape_peaks: Add escape peaks
        add_backscatter: Add backscatter peak
    
    Returns:
        Modified spectrum
    """
    spectrum = base_spectrum.copy()
    channels = np.arange(len(spectrum))
    
    if detector_type == 'NaI':
        # NaI(Tl) specific effects
        
        if add_escape_peaks:
            # Single and double escape peaks for high-energy gammas
            # SE at E - 511 keV, DE at E - 1022 keV
            print("Adding escape peaks...")
            for i in range(len(spectrum)):
                if i > 1022/0.5 and spectrum[i] > 100:  # High energy peak
                    # Single escape
                    se_channel = int(i - 511/0.5)
                    if se_channel > 0:
                        spectrum[se_channel] += spectrum[i] * 0.1
                    
                    # Double escape
                    de_channel = int(i - 1022/0.5)
                    if de_channel > 0:
                        spectrum[de_channel] += spectrum[i] * 0.05
        
        if add_backscatter:
            # Backscatter peak around 200-250 keV
            print("Adding backscatter peak...")
            bs_channel = int(220/0.5)  # ~220 keV
            bs_width = 20
            backscatter = 50 * np.exp(-0.5 * ((channels - bs_channel) / bs_width) ** 2)
            spectrum += backscatter
    
    elif detector_type == 'HPGe':
        # HPGe specific effects - generally cleaner
        pass
    
    elif detector_type == 'CZT':
        # CdZnTe specific effects - tailing
        print("Adding low-energy tailing...")
        for i in range(len(spectrum)):
            if spectrum[i] > 100:
                # Add exponential tail
                for j in range(max(0, i-50), i):
                    spectrum[j] += spectrum[i] * 0.001 * np.exp((j-i)/10)
    
    if add_pileup:
        # Simulate pile-up (sum peaks)
        print("Adding pile-up effects...")
        major_peaks = np.where(spectrum > np.max(spectrum) * 0.1)[0]
        for i in major_peaks[:3]:  # Use first 3 major peaks
            for j in major_peaks[:3]:
                pileup_channel = min(i + j, len(spectrum) - 1)
                spectrum[pileup_channel] += np.sqrt(spectrum[i] * spectrum[j]) * 0.01
    
    return spectrum


def save_spectrum(channels: np.ndarray, 
                 counts: np.ndarray,
                 output_file: str,
                 format: str = 'csv',
                 metadata: Dict = None):
    """
    Save spectrum to file.
    
    Parameters:
        channels: Channel array
        counts: Counts array
        output_file: Output file path
        format: Output format ('csv', 'spe', 'json')
        metadata: Optional metadata
    """
    output_path = Path(output_file)
    
    if format == 'csv':
        df = pd.DataFrame({'channel': channels, 'counts': counts})
        df.to_csv(output_path, index=False, header=False)
        print(f"Saved CSV spectrum to {output_path}")
        
    elif format == 'spe':
        # IAEA SPE format
        with open(output_path, 'w') as f:
            f.write("$SPEC_ID:\n")
            f.write("Synthetic spectrum\n")
            f.write("$SPEC_REM:\n")
            f.write("Generated by generate_test_spectrum.py\n")
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
            f.write("$DATE_MEA:\n")
            f.write("01-01-2024 00:00:00\n")
            f.write("$MEAS_TIM:\n")
            f.write("1000 1000\n")
            f.write("$DATA:\n")
            f.write(f"0 {len(counts)-1}\n")
            for count in counts:
                f.write(f"{int(count)}\n")
            f.write("$ROI:\n")
            f.write("0\n")
            f.write("$PRESETS:\n")
            f.write("None\n")
            f.write("$ENER_FIT:\n")
            f.write("0.0 0.5\n")
            f.write("$MCA_CAL:\n")
            f.write("2\n")
            f.write("0.5 0.0\n")
        print(f"Saved SPE spectrum to {output_path}")
        
    elif format == 'json':
        data = {
            'channels': channels.tolist(),
            'counts': counts.tolist(),
            'metadata': metadata or {}
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON spectrum to {output_path}")
    
    # Save metadata separately if provided
    if metadata:
        meta_file = output_path.with_suffix('.meta.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic gamma spectra for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Na-22 spectrum
  python generate_test_spectrum.py --isotope Na-22 --output na22_test.csv
  
  # Generate complex spectrum with multiple isotopes
  python generate_test_spectrum.py --complex --isotopes Na-22 Cs-137 Co-60
  
  # Generate spectrum with detector effects
  python generate_test_spectrum.py --isotope Co-60 --detector NaI --effects
  
  # Batch generate multiple spectra
  python generate_test_spectrum.py --batch 10 --output-dir test_spectra/

Available isotopes:
  Na-22, Co-60, Cs-137, Ba-133, Eu-152, Am-241, Background
        """
    )
    
    parser.add_argument('--output', '-o', type=str, default='test_spectrum.csv',
                       help='Output file name')
    parser.add_argument('--format', '-f', choices=['csv', 'spe', 'json'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--channels', '-n', type=int, default=4096,
                       help='Number of channels (default: 4096)')
    
    # Spectrum type
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--isotope', type=str,
                       help='Generate single isotope spectrum')
    group.add_argument('--complex', action='store_true',
                       help='Generate complex multi-isotope spectrum')
    group.add_argument('--custom', action='store_true',
                       help='Generate custom spectrum from peak list')
    
    # Isotope options
    parser.add_argument('--isotopes', nargs='+', 
                       default=['Na-22', 'Cs-137', 'Co-60'],
                       help='Isotopes for complex spectrum')
    
    # Detector effects
    parser.add_argument('--detector', choices=['NaI', 'HPGe', 'CZT'], default='NaI',
                       help='Detector type for effects simulation')
    parser.add_argument('--effects', action='store_true',
                       help='Add detector effects (escape peaks, pileup, etc.)')
    
    # Calibration
    parser.add_argument('--calibration', type=str, default='0.5,0',
                       help='Energy calibration as "a,b" for E=a*ch+b')
    
    # Intensity
    parser.add_argument('--intensity', type=float, default=1000,
                       help='Overall intensity scale (default: 1000)')
    parser.add_argument('--background', type=float, default=20,
                       help='Background level (default: 20)')
    
    # Batch generation
    parser.add_argument('--batch', type=int,
                       help='Generate batch of spectra')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for batch generation')
    
    # Random seed
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed:
        np.random.seed(args.seed)
    
    # Parse calibration
    try:
        a, b = map(float, args.calibration.split(','))
        calibration = (a, b)
    except:
        print("Invalid calibration format. Using default: E = 0.5*ch + 0")
        calibration = (0.5, 0)
    
    # Generate spectra
    if args.batch:
        # Batch generation
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Generating {args.batch} test spectra...")
        
        for i in range(args.batch):
            # Randomly select isotope
            isotope = np.random.choice(['Na-22', 'Co-60', 'Cs-137', 'Ba-133', 'Eu-152'])
            
            channels, counts, peaks = create_isotope_spectrum(
                isotope,
                args.channels,
                calibration,
                intensity_scale=args.intensity * np.random.uniform(0.5, 1.5)
            )
            
            # Add detector effects randomly
            if np.random.random() > 0.5:
                counts = create_detector_effects_spectrum(counts, args.detector)
            
            # Save
            output_file = output_dir / f"test_{isotope}_{i:03d}.{args.format}"
            metadata = {
                'isotope': isotope,
                'calibration_a': calibration[0],
                'calibration_b': calibration[1],
                'num_peaks': len(peaks)
            }
            save_spectrum(channels, counts, str(output_file), args.format, metadata)
        
        print(f"Batch generation complete. Files saved to {output_dir}")
        
    elif args.complex:
        # Complex multi-isotope spectrum
        print("Generating complex spectrum...")
        channels, counts, peaks = create_complex_spectrum(
            args.channels,
            args.isotopes,
            args.background
        )
        
        if args.effects:
            counts = create_detector_effects_spectrum(counts, args.detector)
        
        metadata = {
            'type': 'complex',
            'isotopes': args.isotopes,
            'num_peaks': len(peaks),
            'calibration': f"{calibration[0]},{calibration[1]}"
        }
        
        save_spectrum(channels, counts, args.output, args.format, metadata)
        
    elif args.isotope:
        # Single isotope spectrum
        print(f"Generating {args.isotope} spectrum...")
        channels, counts, peaks = create_isotope_spectrum(
            args.isotope,
            args.channels,
            calibration,
            args.intensity
        )
        
        if args.effects:
            counts = create_detector_effects_spectrum(counts, args.detector)
        
        metadata = {
            'isotope': args.isotope,
            'calibration_a': calibration[0],
            'calibration_b': calibration[1],
            'intensity_scale': args.intensity,
            'num_peaks': len(peaks),
            'detector': args.detector if args.effects else 'ideal'
        }
        
        save_spectrum(channels, counts, args.output, args.format, metadata)
        
    else:
        # Default: simple test spectrum
        print("Generating simple test spectrum...")
        channels, counts = generate_synthetic_spectrum(
            args.channels,
            peaks=[(500, 1000, 5), (1000, 800, 8), (1500, 600, 10)],
            background_level=args.background
        )
        
        save_spectrum(channels, counts, args.output, args.format)
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
