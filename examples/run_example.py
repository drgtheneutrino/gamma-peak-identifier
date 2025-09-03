#!/usr/bin/env python3
"""
Example script demonstrating the use of GammaFit package for spectrum analysis.

This script shows:
1. Loading a gamma spectrum
2. Detecting peaks
3. Fitting peaks with Gaussian models
4. Applying energy calibration
5. Exporting results
6. Creating publication-quality plots
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path if running from examples folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit import (
    load_spectrum,
    smooth_spectrum,
    detect_peaks,
    fit_peaks,
    plot_spectrum_with_fits,
    export_results,
    apply_calibration
)
from gammafit.calibration import EnergyCalibration
from gammafit.utils import calculate_counting_statistics, identify_isotope


def main():
    """Run example analysis on test spectrum."""
    
    print("=" * 60)
    print("GammaFit Example Analysis")
    print("=" * 60)
    
    # Define file paths
    example_dir = Path(__file__).parent
    spectrum_file = example_dir / "example_spectrum.csv"
    output_dir = example_dir / "example_output"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load spectrum
    print("\n1. Loading spectrum...")
    try:
        channels, counts = load_spectrum(str(spectrum_file))
        print(f"   Loaded {len(channels)} channels")
        
        # Calculate statistics
        stats = calculate_counting_statistics(counts)
        print(f"   Total counts: {stats['total_counts']:.0f}")
        print(f"   Max counts: {stats['max_counts']:.0f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {spectrum_file}")
        print("Creating synthetic spectrum instead...")
        
        # Generate synthetic spectrum as fallback
        from gammafit.utils import generate_synthetic_spectrum
        channels, counts = generate_synthetic_spectrum(
            num_channels=2048,
            peaks=[
                (511, 2000, 6),    # 511 keV
                (661, 1500, 7),    # 661 keV
                (1274, 1000, 9),   # 1274 keV
            ],
            background_level=30,
            noise_level=1,
            seed=42
        )
    
    # Step 2: Smooth spectrum
    print("\n2. Smoothing spectrum...")
    smoothed = smooth_spectrum(
        counts,
        window_length=7,
        method='savgol'
    )
    print("   Applied Savitzky-Golay filter")
    
    # Step 3: Detect peaks
    print("\n3. Detecting peaks...")
    peak_indices = detect_peaks(
        smoothed,
        min_prominence=50,
        min_height=30,
        min_distance=5
    )
    print(f"   Found {len(peak_indices)} peaks at channels: {peak_indices}")
    
    if len(peak_indices) == 0:
        print("   No peaks detected! Adjusting parameters...")
        peak_indices = detect_peaks(
            smoothed,
            min_prominence=20,
            min_height=10
        )
        print(f"   Found {len(peak_indices)} peaks with relaxed criteria")
    
    # Step 4: Fit peaks
    print("\n4. Fitting peaks...")
    fitted_peaks = fit_peaks(
        channels,
        counts,
        peak_indices,
        peak_model='gaussian',
        background_method='linear',
        window_scale=3.0
    )
    
    print(f"   Successfully fitted {len(fitted_peaks)} peaks")
    
    # Display fit results
    print("\n   Fit Results:")
    print("   " + "-" * 50)
    print(f"   {'Peak':<6} {'Channel':<10} {'Area':<12} {'FWHM':<8} {'SNR':<8}")
    print("   " + "-" * 50)
    
    for i, peak in enumerate(fitted_peaks, 1):
        print(f"   {i:<6} {peak['centroid']:<10.1f} {peak['area']:<12.0f} "
              f"{peak['fwhm']:<8.2f} {peak['snr']:<8.1f}")
    
    # Step 5: Energy calibration
    print("\n5. Applying energy calibration...")
    
    # Create calibration using known peaks
    calibration = EnergyCalibration(model='linear')
    
    # Add calibration points (adjust based on your spectrum)
    # These are example values - replace with your actual calibration
    if len(fitted_peaks) >= 2:
        # Use first two peaks as calibration points
        # Assuming they are 511 and 1274 keV peaks
        calibration.add_point(fitted_peaks[0]['centroid'], 511.0, 'Na-22')
        
        if len(fitted_peaks) >= 3:
            calibration.add_point(fitted_peaks[2]['centroid'], 1274.53, 'Na-22')
        else:
            # Use estimated position
            calibration.add_point(1274, 1274.53, 'Na-22')
        
        # Fit calibration
        calibration.fit()
        print(f"   Calibration: E = {calibration.coefficients['a']:.4f} * Ch + {calibration.coefficients['b']:.2f}")
        
        # Apply calibration to peaks
        calibrated_peaks = apply_calibration(fitted_peaks, calibration)
        
        # Try to identify isotopes
        print("\n   Isotope Identification:")
        for i, peak in enumerate(calibrated_peaks, 1):
            if 'energy' in peak:
                isotopes = identify_isotope(peak['energy'], tolerance=3.0)
                if isotopes:
                    print(f"   Peak {i} ({peak['energy']:.1f} keV): {', '.join(isotopes)}")
    else:
        print("   Not enough peaks for calibration")
        calibrated_peaks = fitted_peaks
        calibration = None
    
    # Step 6: Export results
    print("\n6. Exporting results...")
    
    # Save peak list
    peaks_file = output_dir / "peaks.csv"
    export_results(
        calibrated_peaks,
        str(peaks_file),
        include_energy=(calibration is not None)
    )
    print(f"   Peak list saved to: {peaks_file}")
    
    # Step 7: Generate plots
    print("\n7. Generating plots...")
    
    # Main spectrum plot
    plot_file = output_dir / "spectrum.png"
    plot_spectrum_with_fits(
        channels,
        counts,
        smoothed,
        calibrated_peaks,
        str(plot_file),
        calibration.coefficients if calibration else None,
        plot_style='default',
        show_components=True,
        log_scale=True
    )
    print(f"   Spectrum plot saved to: {plot_file}")
    
    # Create additional analysis plots
    create_analysis_plots(
        channels, counts, smoothed, 
        fitted_peaks, output_dir
    )
    
    # Step 8: Generate report
    print("\n8. Generating HTML report...")
    generate_report(
        channels, counts, calibrated_peaks,
        calibration, output_dir
    )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved in: {output_dir}")
    print("=" * 60)


def create_analysis_plots(channels, counts, smoothed, peaks, output_dir):
    """Create additional analysis plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Raw vs Smoothed
    ax1 = axes[0, 0]
    ax1.plot(channels, counts, 'b-', alpha=0.5, linewidth=0.5, label='Raw')
    ax1.plot(channels, smoothed, 'r-', linewidth=1, label='Smoothed')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Counts')
    ax1.set_title('Raw vs Smoothed Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak positions
    ax2 = axes[0, 1]
    ax2.plot(channels, smoothed, 'k-', linewidth=1)
    for peak in peaks:
        ax2.axvline(peak['centroid'], color='r', linestyle='--', alpha=0.7)
        ax2.plot(peak['centroid'], peak['amplitude'], 'ro', markersize=8)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Counts')
    ax2.set_title('Detected Peaks')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Logarithmic scale
    ax3 = axes[1, 0]
    ax3.semilogy(channels, counts, 'b-', linewidth=0.5)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Counts (log scale)')
    ax3.set_title('Logarithmic View')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Peak areas
    ax4 = axes[1, 1]
    if peaks:
        peak_channels = [p['centroid'] for p in peaks]
        peak_areas = [p['area'] for p in peaks]
        ax4.bar(range(len(peaks)), peak_areas, color='green', alpha=0.7)
        ax4.set_xlabel('Peak Number')
        ax4.set_ylabel('Area (counts)')
        ax4.set_title('Peak Areas')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, area in enumerate(peak_areas):
            ax4.text(i, area, f'{area:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    analysis_plot = output_dir / "analysis_plots.png"
    plt.savefig(analysis_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Analysis plots saved to: {analysis_plot}")


def generate_report(channels, counts, peaks, calibration, output_dir):
    """Generate HTML report of analysis results."""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gamma Spectrum Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                text-align: left;
            }
            td {
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .plot-container {
                text-align: center;
                margin: 20px 0;
            }
            .plot-container img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .summary {
                background-color: #e8f5e9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .footer {
                text-align: center;
                color: #666;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <h1>Gamma Spectrum Analysis Report</h1>
        
        <div class="container summary">
            <h2>Summary</h2>
            <p><strong>Total Channels:</strong> {n_channels}</p>
            <p><strong>Total Counts:</strong> {total_counts:.0f}</p>
            <p><strong>Peaks Detected:</strong> {n_peaks}</p>
            <p><strong>Calibration:</strong> {calibration_info}</p>
        </div>
        
        <div class="container">
            <h2>Spectrum Plot</h2>
            <div class="plot-container">
                <img src="spectrum.png" alt="Gamma Spectrum">
            </div>
        </div>
        
        <div class="container">
            <h2>Peak Analysis Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Peak #</th>
                        <th>Channel</th>
                        {energy_header}
                        <th>Area</th>
                        <th>FWHM</th>
                        <th>SNR</th>
                    </tr>
                </thead>
                <tbody>
                    {peak_rows}
                </tbody>
            </table>
        </div>
        
        <div class="container">
            <h2>Additional Analysis</h2>
            <div class="plot-container">
                <img src="analysis_plots.png" alt="Analysis Plots">
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by GammaFit - Gamma Spectroscopy Analysis Package</p>
            <p>Repository: <a href="https://github.com/drgtheneutrino/gamma-peak-identifier">
                https://github.com/drgtheneutrino/gamma-peak-identifier</a></p>
        </div>
    </body>
    </html>
    """
    
    # Fill in template values
    n_channels = len(channels)
    total_counts = np.sum(counts)
    n_peaks = len(peaks)
    
    if calibration:
        calibration_info = f"E = {calibration.coefficients['a']:.4f} × Ch + {calibration.coefficients['b']:.2f} keV"
        energy_header = "<th>Energy (keV)</th>"
    else:
        calibration_info = "Not calibrated"
        energy_header = ""
    
    # Generate peak rows
    peak_rows = ""
    for i, peak in enumerate(peaks, 1):
        energy_cell = ""
        if 'energy' in peak:
            energy_cell = f"<td>{peak['energy']:.2f}</td>"
        
        peak_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{peak['centroid']:.1f}</td>
            {energy_cell}
            <td>{peak['area']:.0f}</td>
            <td>{peak['fwhm']:.2f}</td>
            <td>{peak['snr']:.1f}</td>
        </tr>
        """
    
    # Format HTML
    html = html_content.format(
        n_channels=n_channels,
        total_counts=total_counts,
        n_peaks=n_peaks,
        calibration_info=calibration_info,
        energy_header=energy_header,
        peak_rows=peak_rows
    )
    
    # Save report
    report_file = output_dir / "report.html"
    with open(report_file, 'w') as f:
        f.write(html)
    
    print(f"   HTML report saved to: {report_file}")


if __name__ == "__main__":
    main()
