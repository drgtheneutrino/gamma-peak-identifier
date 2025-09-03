#!/usr/bin/env python3
"""
Energy calibration example for gamma spectroscopy.

This script demonstrates:
1. Manual calibration using known peaks
2. Automatic calibration with isotope library
3. Polynomial calibration for non-linear detectors
4. Calibration validation and quality assessment
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit import load_spectrum, detect_peaks, fit_peaks
from gammafit.calibration import (
    EnergyCalibration,
    auto_calibrate,
    validate_calibration,
    get_isotope_energies,
    COMMON_ISOTOPES
)
from gammafit.utils import generate_synthetic_spectrum


def manual_calibration_example():
    """Demonstrate manual energy calibration."""
    
    print("\n" + "=" * 60)
    print("MANUAL CALIBRATION EXAMPLE")
    print("=" * 60)
    
    # Generate spectrum with known peaks
    print("\nGenerating test spectrum with known isotopes...")
    channels = np.arange(4096)
    counts = np.zeros(4096)
    
    # Add peaks at known positions
    # Using approximate 0.5 keV/channel calibration
    peak_data = [
        (1022, 511.0, "Na-22", 3000),    # 511 keV annihilation
        (1323, 661.66, "Cs-137", 2500),  # 661.66 keV
        (2346, 1173.23, "Co-60", 2000),  # 1173.23 keV
        (2549, 1274.53, "Na-22", 1800),  # 1274.53 keV
        (2665, 1332.49, "Co-60", 1700),  # 1332.49 keV
    ]
    
    for channel, energy, isotope, amplitude in peak_data:
        peak = amplitude * np.exp(-0.5 * ((channels - channel) / 6) ** 2)
        counts += peak
    
    # Add background and noise
    counts += 20 * np.exp(-channels / 2000) + 5
    counts = np.random.poisson(counts)
    
    # Detect and fit peaks
    print("Detecting and fitting peaks...")
    peak_indices = detect_peaks(counts, min_prominence=100)
    fitted_peaks = fit_peaks(channels, counts, peak_indices)
    
    print(f"Found {len(fitted_peaks)} peaks")
    
    # Create calibration
    print("\nPerforming manual calibration...")
    calibration = EnergyCalibration(model='linear')
    
    # Match fitted peaks to known energies
    for peak in fitted_peaks[:5]:  # Use first 5 peaks
        centroid = peak['centroid']
        
        # Find closest known peak
        closest_peak = min(peak_data, key=lambda p: abs(p[0] - centroid))
        if abs(closest_peak[0] - centroid) < 20:  # Within 20 channels
            calibration.add_point(
                centroid,
                closest_peak[1],
                closest_peak[2]
            )
            print(f"  Matched channel {centroid:.1f} to {closest_peak[1]} keV ({closest_peak[2]})")
    
    # Fit calibration
    if len(calibration.calibration_points) >= 2:
        calibration.fit()
        
        print(f"\nCalibration results:")
        print(f"  Model: {calibration.model}")
        print(f"  Equation: E = {calibration.coefficients['a']:.6f} × Ch + {calibration.coefficients['b']:.3f}")
        print(f"  R²: {calibration.fit_quality['r_squared']:.6f}")
        print(f"  RMS error: {calibration.fit_quality['rms_error']:.3f} keV")
        
        # Plot calibration
        plot_calibration(calibration)
        
        # Test calibration
        test_channels = [1022, 1323, 2346, 2665]
        print("\nTesting calibration:")
        for ch in test_channels:
            energy = calibration.channel_to_energy(ch)
            print(f"  Channel {ch} → {energy:.2f} keV")
        
        return calibration
    else:
        print("Not enough calibration points!")
        return None


def automatic_calibration_example():
    """Demonstrate automatic calibration using isotope library."""
    
    print("\n" + "=" * 60)
    print("AUTOMATIC CALIBRATION EXAMPLE")
    print("=" * 60)
    
    # Generate spectrum with Co-60
    print("\nGenerating Co-60 spectrum...")
    channels = np.arange(4096)
    counts = np.zeros(4096)
    
    # Add Co-60 peaks (assuming ~0.5 keV/channel)
    co60_peaks = [
        (2346, 1173.23, 3000),
        (2665, 1332.49, 2800)
    ]
    
    for channel, energy, amplitude in co60_peaks:
        peak = amplitude * np.exp(-0.5 * ((channels - channel) / 7) ** 2)
        counts += peak
    
    # Add background
    counts += 15 * np.exp(-channels / 2500) + 3
    counts = np.random.poisson(counts)
    
    # Get known Co-60 energies
    print("Using Co-60 reference energies:")
    co60_energies = get_isotope_energies('Co-60')
    for energy in co60_energies:
        print(f"  {energy:.2f} keV")
    
    # Perform auto-calibration
    print("\nPerforming automatic calibration...")
    known_peaks = [(e, 'Co-60') for e in co60_energies]
    
    calibration = auto_calibrate(
        channels,
        counts,
        known_peaks,
        tolerance=30  # Channel tolerance for matching
    )
    
    if calibration:
        print("Auto-calibration successful!")
        print(f"  Equation: E = {calibration.coefficients['a']:.6f} × Ch + {calibration.coefficients['b']:.3f}")
        print(f"  Matched {len(calibration.calibration_points)} peaks")
        
        # Validate calibration
        test_peaks = [(2346, 1173.23), (2665, 1332.49)]
        validation = validate_calibration(calibration, test_peaks)
        
        print(f"\nValidation results:")
        print(f"  RMS error: {validation['rms_error']:.3f} keV")
        print(f"  Max error: {validation['max_error']:.3f} keV")
        
        return calibration
    else:
        print("Auto-calibration failed!")
        return None


def polynomial_calibration_example():
    """Demonstrate polynomial calibration for non-linear response."""
    
    print("\n" + "=" * 60)
    print("POLYNOMIAL CALIBRATION EXAMPLE")
    print("=" * 60)
    
    # Create non-linear calibration data
    print("\nCreating non-linear calibration data...")
    
    # Simulate non-linear detector response
    true_channels = np.array([100, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    true_energies = np.array([59.5, 320.1, 661.7, 1064.2, 1460.8, 1836.1, 2204.2, 2614.5])
    
    # Add some measurement uncertainty
    np.random.seed(42)
    measured_channels = true_channels + np.random.randn(len(true_channels)) * 2
    
    # Linear calibration
    print("\nLinear calibration:")
    linear_cal = EnergyCalibration(model='linear')
    for ch, en in zip(measured_channels, true_energies):
        linear_cal.add_point(ch, en)
    linear_cal.fit()
    
    print(f"  Equation: E = {linear_cal.coefficients['a']:.6f} × Ch + {linear_cal.coefficients['b']:.3f}")
    print(f"  R²: {linear_cal.fit_quality['r_squared']:.6f}")
    print(f"  RMS error: {linear_cal.fit_quality['rms_error']:.3f} keV")
    
    # Quadratic calibration
    print("\nQuadratic calibration:")
    quad_cal = EnergyCalibration(model='quadratic')
    for ch, en in zip(measured_channels, true_energies):
        quad_cal.add_point(ch, en)
    quad_cal.fit()
    
    print(f"  Order: {quad_cal.coefficients['order']}")
    print(f"  Coefficients: c2={quad_cal.coefficients.get('c2', 0):.9f}, "
          f"c1={quad_cal.coefficients.get('c1', 0):.6f}, "
          f"c0={quad_cal.coefficients.get('c0', 0):.3f}")
    print(f"  R²: {quad_cal.fit_quality['r_squared']:.6f}")
    print(f"  RMS error: {quad_cal.fit_quality['rms_error']:.3f} keV")
    
    # Compare calibrations
    plot_calibration_comparison(linear_cal, quad_cal, true_channels, true_energies)
    
    return quad_cal


def plot_calibration(calibration):
    """Plot calibration curve and residuals."""
    
    if not calibration.calibration_points:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Get calibration points
    channels = np.array([p.channel for p in calibration.calibration_points])
    energies = np.array([p.energy for p in calibration.calibration_points])
    
    # Plot calibration curve
    ax1.scatter(channels, energies, s=100, c='blue', label='Calibration points', zorder=5)
    
    # Plot fitted line
    x_fit = np.linspace(0, max(channels) * 1.1, 100)
    y_fit = calibration.channel_to_energy(x_fit)
    ax1.plot(x_fit, y_fit, 'r-', label='Fitted calibration', linewidth=2)
    
    # Add isotope labels
    for point in calibration.calibration_points:
        if point.isotope:
            ax1.annotate(point.isotope, 
                        (point.channel, point.energy),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=8)
    
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Energy (keV)')
    ax1.set_title('Energy Calibration Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot residuals
    fitted_energies = calibration.channel_to_energy(channels)
    residuals = energies - fitted_energies
    
    ax2.scatter(channels, residuals, s=100, c='blue')
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=2)
    ax2.axhline(y=calibration.fit_quality['rms_error'], color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=-calibration.fit_quality['rms_error'], color='g', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Residual (keV)')
    ax2.set_title('Calibration Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add RMS text
    ax2.text(0.02, 0.98, f"RMS: {calibration.fit_quality['rms_error']:.3f} keV",
            transform=ax2.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nCalibration plot saved as 'calibration_curve.png'")


def plot_calibration_comparison(linear_cal, poly_cal, channels, energies):
    """Compare linear and polynomial calibrations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot both calibrations
    ax1 = axes[0, 0]
    ax1.scatter(channels, energies, s=100, c='black', label='Data points', zorder=5)
    
    x_fit = np.linspace(0, max(channels) * 1.1, 200)
    y_linear = linear_cal.channel_to_energy(x_fit)
    y_poly = poly_cal.channel_to_energy(x_fit)
    
    ax1.plot(x_fit, y_linear, 'b-', label='Linear', linewidth=2)
    ax1.plot(x_fit, y_poly, 'r-', label='Polynomial', linewidth=2)
    
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Energy (keV)')
    ax1.set_title('Calibration Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals for linear
    ax2 = axes[0, 1]
    linear_fitted = linear_cal.channel_to_energy(channels)
    linear_res = energies - linear_fitted
    
    ax2.scatter(channels, linear_res, s=100, c='blue')
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=2)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Residual (keV)')
    ax2.set_title(f'Linear Residuals (RMS: {linear_cal.fit_quality["rms_error"]:.3f} keV)')
    ax2.grid(True, alpha=0.3)
    
    # Residuals for polynomial
    ax3 = axes[1, 1]
    poly_fitted = poly_cal.channel_to_energy(channels)
    poly_res = energies - poly_fitted
    
    ax3.scatter(channels, poly_res, s=100, c='red')
    ax3.axhline(y=0, color='r', linestyle='-', linewidth=2)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Residual (keV)')
    ax3.set_title(f'Polynomial Residuals (RMS: {poly_cal.fit_quality["rms_error"]:.3f} keV)')
    ax3.grid(True, alpha=0.3)
    
    # Difference between models
    ax4 = axes[1, 0]
    diff = y_poly - y_linear
    ax4.plot(x_fit, diff, 'g-', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Polynomial - Linear (keV)')
    ax4.set_title('Model Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved as 'calibration_comparison.png'")


def isotope_library_example():
    """Demonstrate use of isotope library for calibration."""
    
    print("\n" + "=" * 60)
    print("ISOTOPE LIBRARY EXAMPLE")
    print("=" * 60)
    
    print("\nAvailable isotopes in library:")
    for isotope, energies in COMMON_ISOTOPES.items():
        print(f"\n{isotope}:")
        for energy in energies:
            print(f"  {energy:.2f} keV")
    
    # Create mixed isotope spectrum
    print("\nGenerating mixed isotope spectrum...")
    channels = np.arange(4096)
    counts = np.zeros(4096)
    
    # Add peaks from multiple isotopes
    # Assuming 0.5 keV/channel calibration
    isotope_peaks = [
        ('Na-22', 511.0, 1022, 2000),
        ('Na-22', 1274.53, 2549, 1500),
        ('Cs-137', 661.66, 1323, 1800),
        ('Co-60', 1173.23, 2346, 1600),
        ('Co-60', 1332.49, 2665, 1400),
        ('Ba-133', 356.01, 712, 1000),
        ('Am-241', 59.54, 119, 800),
    ]
    
    for isotope, energy, channel, amplitude in isotope_peaks:
        peak = amplitude * np.exp(-0.5 * ((channels - channel) / 6) ** 2)
        counts += peak
        print(f"  Added {isotope} peak at {energy} keV (channel {channel})")
    
    # Add background
    counts += 25 * np.exp(-channels / 2000) + 10
    counts = np.random.poisson(counts)
    
    # Try auto-calibration with different isotopes
    print("\nAttempting calibration with different isotope combinations...")
    
    isotope_combinations = [
        ['Na-22'],
        ['Cs-137', 'Co-60'],
        ['Na-22', 'Cs-137', 'Co-60'],
        ['Am-241', 'Ba-133', 'Na-22'],
    ]
    
    best_calibration = None
    best_rms = float('inf')
    
    for isotopes in isotope_combinations:
        print(f"\nTrying: {', '.join(isotopes)}")
        
        # Collect known peaks
        known_peaks = []
        for isotope in isotopes:
            for energy in get_isotope_energies(isotope):
                known_peaks.append((energy, isotope))
        
        # Attempt calibration
        cal = auto_calibrate(channels, counts, known_peaks, tolerance=20)
        
        if cal and cal.fit_quality['rms_error'] < best_rms:
            best_calibration = cal
            best_rms = cal.fit_quality['rms_error']
            print(f"  Success! RMS error: {cal.fit_quality['rms_error']:.3f} keV")
            print(f"  Matched {len(cal.calibration_points)} peaks")
        elif cal:
            print(f"  Success, but higher RMS: {cal.fit_quality['rms_error']:.3f} keV")
        else:
            print("  Failed to calibrate")
    
    if best_calibration:
        print(f"\nBest calibration achieved with RMS error: {best_rms:.3f} keV")
        print(f"Equation: E = {best_calibration.coefficients['a']:.6f} × Ch + {best_calibration.coefficients['b']:.3f}")
        
        # Test on known peaks
        print("\nVerifying calibration on known peaks:")
        for isotope, energy, channel, _ in isotope_peaks[:5]:
            calc_energy = best_calibration.channel_to_energy(channel)
            error = calc_energy - energy
            print(f"  {isotope} {energy:.1f} keV: calculated {calc_energy:.1f} keV (error: {error:+.2f} keV)")
        
        return best_calibration
    
    return None


def calibration_uncertainty_example():
    """Demonstrate calibration uncertainty propagation."""
    
    print("\n" + "=" * 60)
    print("CALIBRATION UNCERTAINTY EXAMPLE")
    print("=" * 60)
    
    # Create calibration with uncertainty
    print("\nCreating calibration with uncertainties...")
    
    calibration = EnergyCalibration(model='linear')
    
    # Add calibration points with uncertainties
    cal_points = [
        (511, 511.0, 'Na-22', 0.1),
        (1323, 661.66, 'Cs-137', 0.05),
        (2549, 1274.53, 'Na-22', 0.1),
    ]
    
    for channel, energy, isotope, uncertainty in cal_points:
        calibration.add_point(channel, energy, isotope, uncertainty)
        print(f"  Added: Ch {channel} = {energy}±{uncertainty} keV ({isotope})")
    
    # Fit calibration
    calibration.fit()
    
    print(f"\nCalibration: E = {calibration.coefficients['a']:.6f} × Ch + {calibration.coefficients['b']:.3f}")
    print(f"Fit quality: RMS = {calibration.fit_quality['rms_error']:.3f} keV")
    
    # Calculate uncertainties at different channels
    print("\nEnergy uncertainties at various channels:")
    test_channels = [100, 500, 1000, 1500, 2000, 2500, 3000]
    
    for ch in test_channels:
        energy = calibration.channel_to_energy(ch)
        uncertainty = calibration.get_uncertainty(ch)
        print(f"  Channel {ch:4}: {energy:7.2f} ± {uncertainty:.3f} keV")
    
    # Plot uncertainty band
    plot_calibration_uncertainty(calibration)
    
    return calibration


def plot_calibration_uncertainty(calibration):
    """Plot calibration with uncertainty band."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get calibration points
    if calibration.calibration_points:
        channels = np.array([p.channel for p in calibration.calibration_points])
        energies = np.array([p.energy for p in calibration.calibration_points])
        
        # Plot points
        ax.scatter(channels, energies, s=100, c='blue', label='Calibration points', zorder=5)
    
    # Plot calibration line with uncertainty band
    x_range = np.linspace(0, 4000, 200)
    y_center = calibration.channel_to_energy(x_range)
    
    # Calculate uncertainty band
    y_uncertainties = [calibration.get_uncertainty(x) for x in x_range]
    y_upper = y_center + np.array(y_uncertainties)
    y_lower = y_center - np.array(y_uncertainties)
    
    # Plot
    ax.plot(x_range, y_center, 'r-', label='Calibration', linewidth=2)
    ax.fill_between(x_range, y_lower, y_upper, alpha=0.3, color='red', label='Uncertainty band')
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Energy (keV)')
    ax.set_title('Calibration with Uncertainty Band')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nUncertainty plot saved as 'calibration_uncertainty.png'")


def save_and_load_calibration_example():
    """Demonstrate saving and loading calibration."""
    
    print("\n" + "=" * 60)
    print("SAVE/LOAD CALIBRATION EXAMPLE")
    print("=" * 60)
    
    # Create a calibration
    print("\nCreating calibration...")
    cal1 = EnergyCalibration(model='linear')
    cal1.add_point(511, 511.0, 'Na-22')
    cal1.add_point(1323, 661.66, 'Cs-137')
    cal1.add_point(2549, 1274.53, 'Na-22')
    cal1.fit()
    
    print(f"Original: E = {cal1.coefficients['a']:.6f} × Ch + {cal1.coefficients['b']:.3f}")
    
    # Save calibration
    cal_file = 'calibration.json'
    print(f"\nSaving calibration to {cal_file}...")
    cal1.save(cal_file)
    
    # Load calibration
    print(f"Loading calibration from {cal_file}...")
    cal2 = EnergyCalibration()
    cal2.load(cal_file)
    
    print(f"Loaded: E = {cal2.coefficients['a']:.6f} × Ch + {cal2.coefficients['b']:.3f}")
    
    # Verify they're the same
    test_ch = 1000
    energy1 = cal1.channel_to_energy(test_ch)
    energy2 = cal2.channel_to_energy(test_ch)
    
    print(f"\nVerification:")
    print(f"  Channel {test_ch}:")
    print(f"    Original: {energy1:.3f} keV")
    print(f"    Loaded:   {energy2:.3f} keV")
    print(f"    Match: {abs(energy1 - energy2) < 0.001}")
    
    # Clean up
    import os
    if os.path.exists(cal_file):
        os.remove(cal_file)
        print(f"\nCleaned up {cal_file}")


def main():
    """Run all calibration examples."""
    
    print("=" * 60)
    print("GAMMAFIT CALIBRATION EXAMPLES")
    print("=" * 60)
    
    # Run examples
    examples = [
        ("Manual Calibration", manual_calibration_example),
        ("Automatic Calibration", automatic_calibration_example),
        ("Polynomial Calibration", polynomial_calibration_example),
        ("Isotope Library", isotope_library_example),
        ("Calibration Uncertainty", calibration_uncertainty_example),
        ("Save/Load Calibration", save_and_load_calibration_example),
    ]
    
    calibrations = []
    
    for name, func in examples:
        try:
            print(f"\nRunning: {name}")
            cal = func()
            if cal:
                calibrations.append((name, cal))
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    
    if calibrations:
        print(f"\nSuccessfully created {len(calibrations)} calibrations:")
        for name, cal in calibrations:
            if hasattr(cal, 'fit_quality'):
                print(f"  {name}: RMS error = {cal.fit_quality.get('rms_error', 'N/A'):.3f} keV")
    
    print("\n" + "=" * 60)
    print("All calibration examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
