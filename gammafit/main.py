#!/usr/bin/env python3
"""
Main command-line interface for GammaFit peak detection and fitting.

This module provides the CLI entry point for the gamma-peak-identifier package.
Repository: https://github.com/drgtheneutrino/gamma-peak-identifier
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime

from .io_module import load_spectrum, save_peaks, load_config
from .detection import detect_peaks, smooth_spectrum
from .fitting import fit_peaks
from .output import plot_spectrum_with_fits, export_results
from .calibration import parse_calibration, apply_calibration
from .utils import print_summary, validate_input


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GammaFit - Automated Gamma Spectroscopy Peak Detection and Fitting',
        epilog='For more information, visit: https://github.com/drgtheneutrino/gamma-peak-identifier',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'spectrum',
        type=str,
        help='Path to spectrum file (CSV, SPE, or CHN format)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--calibration',
        type=str,
        default=None,
        help='Linear calibration coefficients as "a,b" where Energy = a*Channel + b'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON configuration file'
    )
    
    # Peak detection parameters
    detection_group = parser.add_argument_group('peak detection parameters')
    detection_group.add_argument(
        '--min-prominence',
        type=float,
        default=50,
        help='Minimum peak prominence for detection (default: 50)'
    )
    
    detection_group.add_argument(
        '--min-height',
        type=float,
        default=10,
        help='Minimum peak height for detection (default: 10)'
    )
    
    detection_group.add_argument(
        '--min-distance',
        type=int,
        default=3,
        help='Minimum distance between peaks in channels (default: 3)'
    )
    
    # Smoothing parameters
    smoothing_group = parser.add_argument_group('smoothing parameters')
    smoothing_group.add_argument(
        '--smoothing-window',
        type=int,
        default=5,
        help='Window size for Savitzky-Golay filter (default: 5)'
    )
    
    smoothing_group.add_argument(
        '--smoothing-method',
        type=str,
        choices=['savgol', 'gaussian', 'none'],
        default='savgol',
        help='Smoothing method to use (default: savgol)'
    )
    
    smoothing_group.add_argument(
        '--polyorder',
        type=int,
        default=2,
        help='Polynomial order for Savitzky-Golay filter (default: 2)'
    )
    
    # Output parameters
    output_group = parser.add_argument_group('output parameters')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for results (default: current directory)'
    )
    
    output_group.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output files (default: none)'
    )
    
    output_group.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plot'
    )
    
    output_group.add_argument(
        '--plot-format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Format for output plot (default: png)'
    )
    
    output_group.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for output plot (default: 150)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    return parser.parse_args()


def load_configuration(args):
    """
    Load and merge configuration from file and command line.
    
    Parameters:
        args: Command line arguments
        
    Returns:
        dict: Merged configuration
    """
    config = {}
    
    # Load configuration file if provided
    if args.config:
        try:
            config = load_config(args.config)
            if not args.quiet:
                print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Override with command line arguments
    cli_config = {
        'detection': {
            'min_prominence': args.min_prominence,
            'min_height': args.min_height,
            'min_distance': args.min_distance,
        },
        'smoothing': {
            'method': args.smoothing_method,
            'window_length': args.smoothing_window,
            'polyorder': args.polyorder,
        },
        'output': {
            'directory': args.output_dir,
            'prefix': args.output_prefix,
            'generate_plot': not args.no_plot,
            'plot_format': args.plot_format,
            'dpi': args.dpi,
        },
        'calibration': parse_calibration(args.calibration) if args.calibration else None,
        'verbose': args.verbose,
        'quiet': args.quiet,
    }
    
    # Merge configurations (CLI overrides file)
    for key, value in cli_config.items():
        if value is not None:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    return config


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_configuration(args)
    
    # Set up verbosity
    verbose = config.get('verbose', False)
    quiet = config.get('quiet', False)
    
    if not quiet:
        print("=" * 60)
        print("GammaFit - Gamma Spectroscopy Peak Analysis")
        print("Repository: https://github.com/drgtheneutrino/gamma-peak-identifier")
        print("=" * 60)
    
    # Validate input file
    if not validate_input(args.spectrum):
        sys.exit(1)
    
    # Load spectrum
    if verbose:
        print(f"\nLoading spectrum from {args.spectrum}...")
    
    try:
        channels, counts = load_spectrum(args.spectrum)
        if not quiet:
            print(f"✓ Loaded {len(channels)} channels")
            print(f"  Total counts: {np.sum(counts):.0f}")
            print(f"  Max counts: {np.max(counts):.0f}")
    except Exception as e:
        print(f"✗ Error loading spectrum: {e}")
        sys.exit(1)
    
    # Apply smoothing
    smoothing_config = config.get('smoothing', {})
    if smoothing_config.get('method', 'savgol') != 'none':
        if verbose:
            print(f"\nApplying {smoothing_config.get('method', 'savgol')} smoothing...")
        
        smoothed_counts = smooth_spectrum(
            counts,
            window_length=smoothing_config.get('window_length', 5),
            polyorder=smoothing_config.get('polyorder', 2),
            method=smoothing_config.get('method', 'savgol')
        )
        if not quiet:
            print(f"✓ Smoothing applied")
    else:
        smoothed_counts = counts
    
    # Detect peaks
    detection_config = config.get('detection', {})
    if verbose:
        print(f"\nDetecting peaks...")
        print(f"  Min prominence: {detection_config.get('min_prominence', 50)}")
        print(f"  Min height: {detection_config.get('min_height', 10)}")
    
    peak_indices = detect_peaks(
        smoothed_counts,
        min_prominence=detection_config.get('min_prominence', 50),
        min_height=detection_config.get('min_height', 10),
        min_distance=detection_config.get('min_distance', 3)
    )
    
    if not quiet:
        print(f"✓ Found {len(peak_indices)} peaks")
    
    if len(peak_indices) == 0:
        print("\n⚠ No peaks detected. Try adjusting detection parameters:")
        print("  - Decrease --min-prominence")
        print("  - Decrease --min-height")
        print("  - Check spectrum file for valid data")
        sys.exit(0)
    
    # Fit peaks
    if verbose:
        print(f"\nFitting {len(peak_indices)} peaks...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_peaks = fit_peaks(channels, counts, peak_indices)
    
    if not quiet:
        print(f"✓ Fitted {len(fitted_peaks)} peaks successfully")
    
    # Apply calibration if provided
    calibration = config.get('calibration')
    if calibration:
        if not quiet:
            print(f"✓ Applied energy calibration: E = {calibration['a']:.3f} × Channel + {calibration['b']:.3f}")
        fitted_peaks = apply_calibration(fitted_peaks, calibration)
    
    # Create output directory
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('directory', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file names
    prefix = output_config.get('prefix', '')
    if prefix and not prefix.endswith('_'):
        prefix += '_'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    peaks_file = output_dir / f'{prefix}peaks_{timestamp}.csv'
    
    # Export results
    if verbose:
        print(f"\nExporting results to {output_dir}...")
    
    export_results(fitted_peaks, peaks_file, calibration is not None)
    if not quiet:
        print(f"✓ Peak list saved to {peaks_file}")
    
    # Generate plot
    if output_config.get('generate_plot', True):
        plot_format = output_config.get('plot_format', 'png')
        plot_file = output_dir / f'{prefix}spectrum_{timestamp}.{plot_format}'
        
        if verbose:
            print(f"Generating plot...")
        
        plot_spectrum_with_fits(
            channels, counts, smoothed_counts, fitted_peaks,
            plot_file, calibration,
            dpi=output_config.get('dpi', 150)
        )
        if not quiet:
            print(f"✓ Plot saved to {plot_file}")
    
    # Print summary
    if not quiet:
        print_summary(fitted_peaks, calibration)
    
    if not quiet:
        print("\n✓ Analysis complete!")
        print("=" * 60)
    
    return 0


def entry_point():
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    entry_point()
