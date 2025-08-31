"""
Output and visualization functions for gamma spectroscopy analysis.

This module handles plotting spectra with fitted peaks, generating reports,
and exporting results in various formats.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from .fitting import gaussian_with_background, gaussian


def plot_spectrum_with_fits(channels: np.ndarray,
                           counts: np.ndarray,
                           smoothed_counts: np.ndarray,
                           fitted_peaks: List[Dict[str, Any]],
                           output_file: str,
                           calibration: Optional[Dict[str, float]] = None,
                           plot_style: str = 'default',
                           show_components: bool = True,
                           log_scale: bool = True):
    """
    Create comprehensive plot of spectrum with fitted peaks.
    
    Parameters:
        channels: Channel numbers
        counts: Raw counts
        smoothed_counts: Smoothed counts
        fitted_peaks: List of fitted peak dictionaries
        output_file: Output file path
        calibration: Energy calibration parameters
        plot_style: Plot style ('default', 'publication', 'presentation')
        show_components: Whether to show individual peak components
        log_scale: Whether to use log scale for y-axis
    """
    # Apply plot style
    apply_plot_style(plot_style)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
    
    # Main spectrum plot
    ax1 = fig.add_subplot(gs[0])
    
    # Residuals plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Peak information plot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Convert to energy if calibrated
    x_data = channels
    x_label = 'Channel'
    if calibration:
        x_data = calibration['a'] * channels + calibration['b']
        x_label = 'Energy (keV)'
    
    # Plot main spectrum
    ax1.plot(x_data, counts, 'b-', alpha=0.3, linewidth=0.5, label='Raw data')
    ax1.plot(x_data, smoothed_counts, 'k-', linewidth=1, label='Smoothed data')
    
    # Initialize total fit array
    total_fit = np.zeros_like(counts)
    peak_components = []
    
    # Plot fitted peaks
    for i, peak in enumerate(fitted_peaks):
        if peak.get('fit_params') is not None and peak.get('fit_success', False):
            start, end = peak['fit_region']
            x_fit = channels[start:end]
            
            # Convert to energy if needed
            x_fit_plot = x_fit
            if calibration:
                x_fit_plot = calibration['a'] * x_fit + calibration['b']
            
            # Calculate fitted curve
            if peak.get('peak_model') == 'gaussian':
                y_fit = gaussian_with_background(x_fit, *peak['fit_params'])
                total_fit[start:end] = y_fit
                
                # Extract peak component (without background)
                if show_components and len(peak['fit_params']) >= 5:
                    peak_only = gaussian(x_fit, 
                                        peak['fit_params'][0],
                                        peak['fit_params'][1],
                                        peak['fit_params'][2])
                    peak_components.append((x_fit_plot, peak_only, i))
            else:
                # Generic fit plotting
                y_fit = counts[start:end]  # Fallback
                total_fit[start:end] = y_fit
            
            # Plot total fit
            ax1.plot(x_fit_plot, y_fit, 'r-', linewidth=1.5, alpha=0.8)
            
            # Mark peak centroid
            peak_x = peak.get('energy', peak['centroid']) if calibration else peak['centroid']
            ax1.axvline(peak_x, color='g', linestyle='--', alpha=0.3, linewidth=0.5)
            
            # Add peak label
            y_pos = peak['amplitude'] + peak.get('bg_intercept', 0)
            ax1.annotate(f'{i+1}', 
                        xy=(peak_x, y_pos),
                        xytext=(peak_x, y_pos * 1.2),
                        fontsize=8,
                        ha='center',
                        color='darkgreen')
    
    # Plot individual peak components if requested
    if show_components:
        for x_comp, y_comp, idx in peak_components:
            ax1.fill_between(x_comp, 0, y_comp, alpha=0.2, 
                            label=f'Peak {idx+1}' if idx < 3 else None)
    
    # Configure main plot
    ax1.set_ylabel('Counts', fontsize=11)
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=0.5)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('Gamma Spectrum Analysis', fontsize=13, fontweight='bold')
    
    # Plot residuals
    residuals = counts - total_fit
    ax2.plot(x_data, residuals, 'b-', linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=1)
    ax2.fill_between(x_data, -3*np.sqrt(np.abs(counts)), 3*np.sqrt(np.abs(counts)),
                     alpha=0.2, color='gray', label='±3σ')
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Plot peak markers and info
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Peak Map', fontsize=11)
    ax3.set_xlabel(x_label, fontsize=11)
    
    # Add peak markers on bottom panel
    for i, peak in enumerate(fitted_peaks):
        if peak.get('fit_success', False):
            peak_x = peak.get('energy', peak['centroid']) if calibration else peak['centroid']
            fwhm = peak['fwhm']
            if calibration:
                fwhm *= calibration['a']
            
            # Draw peak as rectangle
            rect = mpatches.Rectangle((peak_x - fwhm/2, 0.2), fwhm, 0.6,
                                     alpha=0.5, facecolor=f'C{i%10}')
            ax3.add_patch(rect)
            
            # Add peak number
            ax3.text(peak_x, 0.5, str(i+1), ha='center', va='center',
                    fontsize=8, fontweight='bold')
    
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Remove x-axis labels from upper plots
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save in requested format
    output_path = Path(output_file)
    if output_path.suffix.lower() == '.pdf':
        with PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    
    plt.close(fig)


def export_results(fitted_peaks: List[Dict[str, Any]],
                  output_file: str,
                  include_energy: bool = False,
                  format: str = 'auto'):
    """
    Export fitted peak parameters to file.
    
    Parameters:
        fitted_peaks: List of fitted peak dictionaries
        output_file: Output file path
        include_energy: Whether to include energy column
        format: Output format ('csv', 'json', 'excel', 'latex', 'auto')
    """
    output_path = Path(output_file)
    
    # Auto-detect format from extension
    if format == 'auto':
        ext = output_path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.tex': 'latex',
            '.txt': 'text'
        }
        format = format_map.get(ext, 'csv')
    
    # Prepare data for export
    export_data = []
    for i, peak in enumerate(fitted_peaks, 1):
        row = {
            'peak_number': i,
            'centroid_channel': peak['centroid'],
            'centroid_err': peak.get('centroid_err', 0),
            'area': peak['area'],
            'area_err': peak.get('area_err', 0),
            'fwhm': peak['fwhm'],
            'fwhm_err': peak.get('fwhm_err', 0),
            'resolution_%': peak.get('resolution', 0),
            'snr': peak['snr'],
            'chi_square': peak.get('chi_square', 0),
            'fit_success': peak.get('fit_success', True)
        }
        
        if include_energy and 'energy' in peak:
            row['energy_keV'] = peak['energy']
            row['energy_err'] = peak.get('energy_err', 0)
        
        # Add multiplet information if present
        if peak.get('multiplet', False):
            row['multiplet'] = True
            row['multiplet_size'] = peak.get('multiplet_size', 0)
        
        export_data.append(row)
    
    # Export based on format
    if format == 'csv':
        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False, float_format='%.4f')
    
    elif format == 'json':
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    elif format == 'excel':
        df = pd.DataFrame(export_data)
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Peaks', index=False)
            
            # Add a summary sheet
            summary_data = {
                'Total Peaks': len(fitted_peaks),
                'Successful Fits': sum(1 for p in fitted_peaks if p.get('fit_success', False)),
                'Multiplets': sum(1 for p in fitted_peaks if p.get('multiplet', False)),
                'Average SNR': np.mean([p['snr'] for p in fitted_peaks]),
                'Average Resolution': np.mean([p.get('resolution', 0) for p in fitted_peaks])
            }
            summary_df = pd.DataFrame(summary_data, index=[0])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    elif format == 'latex':
        df = pd.DataFrame(export_data)
        latex_str = df.to_latex(index=False, float_format='%.2f', escape=False)
        with open(output_file, 'w') as f:
            f.write(latex_str)
    
    elif format == 'text':
        write_text_report(fitted_peaks, output_file, include_energy)
    
    else:
        warnings.warn(f"Unknown format: {format}, using CSV")
        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False)


def write_text_report(fitted_peaks: List[Dict[str, Any]],
                     output_file: str,
                     include_energy: bool = False):
    """
    Write a formatted text report of peak analysis results.
    
    Parameters:
        fitted_peaks: List of fitted peak dictionaries
        output_file: Output file path
        include_energy: Whether to include energy information
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("="*70 + "\n")
        f.write("GAMMA SPECTROSCOPY PEAK ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Write summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Total peaks detected: {len(fitted_peaks)}\n")
        
        successful_fits = sum(1 for p in fitted_peaks if p.get('fit_success', False))
        f.write(f"Successful fits: {successful_fits}/{len(fitted_peaks)}\n")
        
        multiplets = sum(1 for p in fitted_peaks if p.get('multiplet', False))
        if multiplets > 0:
            f.write(f"Multiplets identified: {multiplets}\n")
        
        if fitted_peaks:
            avg_snr = np.mean([p['snr'] for p in fitted_peaks])
            f.write(f"Average SNR: {avg_snr:.2f}\n")
            
            avg_res = np.mean([p.get('resolution', 0) for p in fitted_peaks])
            if avg_res > 0:
                f.write(f"Average resolution: {avg_res:.2f}%\n")
        
        f.write("\n")
        
        # Write detailed peak information
        f.write("DETAILED PEAK INFORMATION\n")
        f.write("-"*70 + "\n")
        
        # Create header line
        if include_energy:
            header = f"{'Peak':<5} {'Energy (keV)':<12} {'Channel':<10} {'Area':<12} {'FWHM':<8} {'SNR':<8}"
        else:
            header = f"{'Peak':<5} {'Channel':<12} {'Area':<12} {'FWHM':<10} {'SNR':<8} {'Chi²':<8}"
        f.write(header + "\n")
        f.write("-"*70 + "\n")
        
        # Write peak data
        for i, peak in enumerate(fitted_peaks, 1):
            if include_energy and 'energy' in peak:
                line = f"{i:<5} {peak['energy']:<12.2f} {peak['centroid']:<10.2f} "
            else:
                line = f"{i:<5} {peak['centroid']:<12.2f} "
            
            line += f"{peak['area']:<12.0f} {peak['fwhm']:<10.2f} {peak['snr']:<8.2f}"
            
            if not include_energy:
                chi2 = peak.get('chi_square', 0)
                line += f" {chi2:<8.2f}"
            
            # Add markers for special cases
            markers = []
            if not peak.get('fit_success', True):
                markers.append('*')
            if peak.get('multiplet', False):
                markers.append('M')
            
            if markers:
                line += f"  {''.join(markers)}"
            
            f.write(line + "\n")
        
        # Write legend
        f.write("\n")
        f.write("-"*70 + "\n")
        f.write("Legend: * = Fit failed, M = Multiplet\n")
        f.write("="*70 + "\n")


def generate_html_report(fitted_peaks: List[Dict[str, Any]],
                        spectrum_plot: str,
                        output_file: str,
                        metadata: Optional[Dict[str, Any]] = None):
    """
    Generate an HTML report with interactive elements.
    
    Parameters:
        fitted_peaks: List of fitted peak dictionaries
        spectrum_plot: Path to spectrum plot image
        output_file: Output HTML file path
        metadata: Optional metadata about the analysis
    """
    import base64
    
    # Read plot image and encode as base64
    with open(spectrum_plot, 'rb') as f:
        plot_data = base64.b64encode(f.read()).decode()
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gamma Spectroscopy Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            h2 {
                color: #555;
                margin-top: 30px;
            }
            .summary {
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
            .plot {
                text-align: center;
                margin: 20px 0;
            }
            .plot img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .metadata {
                background-color: #e8f5e9;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Gamma Spectroscopy Analysis Report</h1>
    """
    
    # Add metadata if provided
    if metadata:
        html_content += '<div class="metadata">'
        html_content += '<h3>Analysis Information</h3>'
        for key, value in metadata.items():
            html_content += f'<p><strong>{key}:</strong> {value}</p>'
        html_content += '</div>'
    
    # Add summary statistics
    html_content += '<div class="summary">'
    html_content += '<h2>Summary Statistics</h2>'
    html_content += f'<p><strong>Total peaks:</strong> {len(fitted_peaks)}</p>'
    
    successful = sum(1 for p in fitted_peaks if p.get('fit_success', False))
    html_content += f'<p><strong>Successful fits:</strong> {successful}/{len(fitted_peaks)}</p>'
    
    if fitted_peaks:
        avg_snr = np.mean([p['snr'] for p in fitted_peaks])
        html_content += f'<p><strong>Average SNR:</strong> {avg_snr:.2f}</p>'
    
    html_content += '</div>'
    
    # Add spectrum plot
    html_content += '<div class="plot">'
    html_content += '<h2>Spectrum with Fitted Peaks</h2>'
    html_content += f'<img src="data:image/png;base64,{plot_data}" alt="Spectrum Plot">'
    html_content += '</div>'
    
    # Add peak table
    html_content += '<h2>Peak Analysis Results</h2>'
    html_content += '<table>'
    html_content += '<thead><tr>'
    html_content += '<th>Peak</th><th>Centroid</th><th>Area</th>'
    html_content += '<th>FWHM</th><th>SNR</th><th>Status</th>'
    html_content += '</tr></thead>'
    html_content += '<tbody>'
    
    for i, peak in enumerate(fitted_peaks, 1):
        status = 'OK' if peak.get('fit_success', False) else 'Failed'
        if peak.get('multiplet', False):
            status += ' (M)'
        
        html_content += '<tr>'
        html_content += f'<td>{i}</td>'
        html_content += f'<td>{peak["centroid"]:.2f}</td>'
        html_content += f'<td>{peak["area"]:.0f}</td>'
        html_content += f'<td>{peak["fwhm"]:.2f}</td>'
        html_content += f'<td>{peak["snr"]:.2f}</td>'
        html_content += f'<td>{status}</td>'
        html_content += '</tr>'
    
    html_content += '</tbody></table>'
    
    # Close HTML
    html_content += '</body></html>'
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)


def plot_calibration_curve(calibration_points: List[Tuple[float, float]],
                          fitted_params: Optional[Dict[str, float]],
                          output_file: str):
    """
    Plot energy calibration curve.
    
    Parameters:
        calibration_points: List of (channel, energy) tuples
        fitted_params: Fitted calibration parameters
        output_file: Output file path
    """
    if not calibration_points:
        return
    
    channels = np.array([p[0] for p in calibration_points])
    energies = np.array([p[1] for p in calibration_points])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot calibration points and fit
    ax1.scatter(channels, energies, s=50, c='blue', label='Calibration points')
    
    if fitted_params:
        # Plot fitted line
        x_fit = np.linspace(channels.min(), channels.max(), 100)
        if 'a' in fitted_params and 'b' in fitted_params:
            y_fit = fitted_params['a'] * x_fit + fitted_params['b']
            ax1.plot(x_fit, y_fit, 'r-', label=f'Fit: E = {fitted_params["a"]:.4f}×Ch + {fitted_params["b"]:.2f}')
    
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Energy (keV)')
    ax1.set_title('Energy Calibration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot residuals
    if fitted_params and 'a' in fitted_params:
        fitted_energies = fitted_params['a'] * channels + fitted_params['b']
        residuals = energies - fitted_energies
        
        ax2.scatter(channels, residuals, s=50, c='blue')
        ax2.axhline(y=0, color='r', linestyle='-', linewidth=1)
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Residual (keV)')
        ax2.set_title('Calibration Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        rms = np.sqrt(np.mean(residuals**2))
        ax2.text(0.02, 0.98, f'RMS: {rms:.3f} keV',
                transform=ax2.transAxes, va='top')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def apply_plot_style(style: str = 'default'):
    """
    Apply matplotlib style settings.
    
    Parameters:
        style: Plot style ('default', 'publication', 'presentation')
    """
    if style == 'publication':
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 13,
            'lines.linewidth': 1.5,
            'lines.markersize': 5
        })
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 20,
            'lines.linewidth': 2,
            'lines.markersize': 8
        })
    else:
        # Default style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5
        })


def create_peak_comparison_plot(peak_lists: List[List[Dict]],
                               labels: List[str],
                               output_file: str):
    """
    Create comparison plot of peaks from multiple spectra.
    
    Parameters:
        peak_lists: List of peak lists from different analyses
        labels: Labels for each spectrum
        output_file: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(peak_lists)))
    
    for i, (peaks, label) in enumerate(zip(peak_lists, labels)):
        centroids = [p['centroid'] for p in peaks]
        amplitudes = [p['amplitude'] for p in peaks]
        
        ax.stem(centroids, amplitudes, linefmt=f'C{i}-', 
                markerfmt=f'C{i}o', basefmt=' ', label=label)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Peak Amplitude')
    ax.set_title('Peak Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()"Average resolution: {avg_res:.2f}%\n")
