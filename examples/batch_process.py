#!/usr/bin/env python3
"""
Batch processing example for multiple gamma spectra.

This script demonstrates:
1. Processing multiple spectrum files
2. Comparing results across spectra
3. Generating summary reports
4. Creating comparison plots
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit import (
    load_spectrum,
    smooth_spectrum,
    detect_peaks,
    fit_peaks,
    export_results
)
from gammafit.calibration import EnergyCalibration
from gammafit.output import create_peak_comparison_plot


class BatchProcessor:
    """Class for batch processing multiple spectra."""
    
    def __init__(self, config_file=None):
        """
        Initialize batch processor.
        
        Parameters:
            config_file: Path to configuration file
        """
        self.config = self.load_config(config_file)
        self.results = []
        self.summary = {}
        
    def load_config(self, config_file):
        """Load configuration from file or use defaults."""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'detection': {
                    'min_prominence': 50,
                    'min_height': 30,
                    'min_distance': 5,
                    'smoothing_window': 7
                },
                'fitting': {
                    'peak_model': 'gaussian',
                    'background_method': 'linear'
                },
                'output': {
                    'generate_plots': True,
                    'export_format': 'csv'
                }
            }
    
    def process_spectrum(self, spectrum_file):
        """
        Process a single spectrum file.
        
        Parameters:
            spectrum_file: Path to spectrum file
            
        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing: {spectrum_file}")
        
        result = {
            'file': spectrum_file,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Load spectrum
            channels, counts = load_spectrum(str(spectrum_file))
            result['channels'] = len(channels)
            result['total_counts'] = float(np.sum(counts))
            
            # Smooth spectrum
            smoothed = smooth_spectrum(
                counts,
                window_length=self.config['detection']['smoothing_window']
            )
            
            # Detect peaks
            peaks = detect_peaks(
                smoothed,
                min_prominence=self.config['detection']['min_prominence'],
                min_height=self.config['detection']['min_height'],
                min_distance=self.config['detection']['min_distance']
            )
            
            result['peaks_detected'] = len(peaks)
            
            # Fit peaks
            fitted_peaks = fit_peaks(
                channels,
                counts,
                peaks,
                peak_model=self.config['fitting']['peak_model'],
                background_method=self.config['fitting']['background_method']
            )
            
            result['peaks_fitted'] = len(fitted_peaks)
            result['peak_data'] = fitted_peaks
            result['success'] = True
            
            # Store spectrum data for comparison
            result['spectrum_data'] = {
                'channels': channels.tolist(),
                'counts': counts.tolist(),
                'smoothed': smoothed.tolist()
            }
            
            print(f"  ✓ Found {len(fitted_peaks)} peaks")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"  ✗ Error: {e}")
        
        return result
    
    def process_directory(self, directory, pattern="*.csv"):
        """
        Process all spectrum files in a directory.
        
        Parameters:
            directory: Directory path
            pattern: File pattern to match
        """
        directory = Path(directory)
        spectrum_files = list(directory.glob(pattern))
        
        if not spectrum_files:
            print(f"No files matching '{pattern}' found in {directory}")
            return
        
        print(f"Found {len(spectrum_files)} spectrum files")
        print("=" * 60)
        
        for spectrum_file in spectrum_files:
            result = self.process_spectrum(spectrum_file)
            self.results.append(result)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate summary statistics from all processed spectra."""
        successful = [r for r in self.results if r['success']]
        
        self.summary = {
            'total_files': len(self.results),
            'successful': len(successful),
            'failed': len(self.results) - len(successful),
            'total_peaks': sum(r['peaks_fitted'] for r in successful),
            'avg_peaks_per_spectrum': np.mean([r['peaks_fitted'] for r in successful]) if successful else 0,
            'total_counts': sum(r['total_counts'] for r in successful)
        }
        
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.summary['total_files']}")
        print(f"Successful: {self.summary['successful']}")
        print(f"Failed: {self.summary['failed']}")
        print(f"Total peaks found: {self.summary['total_peaks']}")
        print(f"Average peaks per spectrum: {self.summary['avg_peaks_per_spectrum']:.1f}")
        print(f"Total counts processed: {self.summary['total_counts']:.0f}")
    
    def export_results(self, output_dir):
        """
        Export batch processing results.
        
        Parameters:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export summary
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'summary': self.summary,
                'config': self.config,
                'results': [
                    {k: v for k, v in r.items() if k != 'spectrum_data'}
                    for r in self.results
                ]
            }, f, indent=2, default=str)
        
        print(f"\nSummary saved to: {summary_file}")
        
        # Export combined peak list
        all_peaks = []
        for result in self.results:
            if result['success'] and 'peak_data' in result:
                for peak in result['peak_data']:
                    peak_entry = peak.copy()
                    peak_entry['source_file'] = Path(result['file']).name
                    all_peaks.append(peak_entry)
        
        if all_peaks:
            peaks_df = pd.DataFrame(all_peaks)
            peaks_file = output_dir / "all_peaks.csv"
            peaks_df.to_csv(peaks_file, index=False)
            print(f"Combined peak list saved to: {peaks_file}")
        
        # Generate comparison plots
        if self.config['output']['generate_plots']:
            self.create_comparison_plots(output_dir)
    
    def create_comparison_plots(self, output_dir):
        """
        Create comparison plots for batch results.
        
        Parameters:
            output_dir: Output directory path
        """
        successful = [r for r in self.results if r['success']]
        
        if len(successful) < 2:
            print("Need at least 2 successful analyses for comparison plots")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Overlay of all spectra
        ax1 = axes[0, 0]
        for result in successful[:5]:  # Limit to 5 for clarity
            if 'spectrum_data' in result:
                channels = result['spectrum_data']['channels']
                counts = result['spectrum_data']['counts']
                label = Path(result['file']).stem
                ax1.semilogy(channels, counts, alpha=0.7, label=label)
        
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Counts (log scale)')
        ax1.set_title('Spectrum Overlay')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Peak count comparison
        ax2 = axes[0, 1]
        file_names = [Path(r['file']).stem for r in successful]
        peak_counts = [r['peaks_fitted'] for r in successful]
        
        bars = ax2.bar(range(len(successful)), peak_counts, color='green', alpha=0.7)
        ax2.set_xlabel('Spectrum')
        ax2.set_ylabel('Number of Peaks')
        ax2.set_title('Peaks Detected per Spectrum')
        ax2.set_xticks(range(len(successful)))
        ax2.set_xticklabels(file_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, peak_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom')
        
        # Plot 3: Total counts comparison
        ax3 = axes[1, 0]
        total_counts = [r['total_counts'] for r in successful]
        
        bars = ax3.bar(range(len(successful)), total_counts, color='blue', alpha=0.7)
        ax3.set_xlabel('Spectrum')
        ax3.set_ylabel('Total Counts')
        ax3.set_title('Total Counts per Spectrum')
        ax3.set_xticks(range(len(successful)))
        ax3.set_xticklabels(file_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Peak centroid distribution
        ax4 = axes[1, 1]
        all_centroids = []
        all_labels = []
        
        for result in successful:
            if 'peak_data' in result:
                for peak in result['peak_data']:
                    all_centroids.append(peak['centroid'])
                    all_labels.append(Path(result['file']).stem)
        
        if all_centroids:
            # Create histogram of peak positions
            ax4.hist(all_centroids, bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Channel')
            ax4.set_ylabel('Peak Count')
            ax4.set_title('Distribution of Peak Positions')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        plot_file = output_dir / "batch_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to: {plot_file}")
        
        # Create peak comparison plot
        self.create_peak_alignment_plot(output_dir)
    
    def create_peak_alignment_plot(self, output_dir):
        """
        Create plot showing peak alignment across spectra.
        
        Parameters:
            output_dir: Output directory path
        """
        successful = [r for r in self.results if r['success'] and 'peak_data' in r]
        
        if len(successful) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot peaks from each spectrum as horizontal lines
        y_positions = []
        labels = []
        
        for i, result in enumerate(successful):
            y_pos = i
            y_positions.append(y_pos)
            labels.append(Path(result['file']).stem)
            
            for peak in result['peak_data']:
                centroid = peak['centroid']
                amplitude = peak['amplitude']
                
                # Scale amplitude for visualization
                marker_size = np.log10(amplitude + 1) * 20
                
                ax.scatter(centroid, y_pos, s=marker_size, alpha=0.6,
                          c='red', edgecolors='black', linewidth=0.5)
        
        # Add grid lines for common peak positions
        common_channels = [511, 661, 1173, 1274, 1332]  # Common gamma energies
        for ch in common_channels:
            ax.axvline(ch, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            ax.text(ch, len(successful), f'{ch}', rotation=90,
                   va='bottom', ha='right', fontsize=8, color='gray')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Channel')
        ax.set_title('Peak Alignment Across Spectra\n(circle size = log(amplitude))')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(-0.5, len(successful) - 0.5)
        
        plt.tight_layout()
        
        # Save figure
        plot_file = output_dir / "peak_alignment.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Peak alignment plot saved to: {plot_file}")
    
    def find_common_peaks(self, tolerance=5):
        """
        Find peaks that appear in multiple spectra.
        
        Parameters:
            tolerance: Channel tolerance for matching peaks
            
        Returns:
            List of common peak groups
        """
        successful = [r for r in self.results if r['success'] and 'peak_data' in r]
        
        if len(successful) < 2:
            return []
        
        # Collect all peaks with source info
        all_peaks = []
        for result in successful:
            source = Path(result['file']).stem
            for peak in result['peak_data']:
                all_peaks.append({
                    'source': source,
                    'centroid': peak['centroid'],
                    'area': peak['area']
                })
        
        # Group peaks by position
        peak_groups = []
        used_peaks = set()
        
        for i, peak1 in enumerate(all_peaks):
            if i in used_peaks:
                continue
            
            group = [peak1]
            used_peaks.add(i)
            
            for j, peak2 in enumerate(all_peaks[i+1:], i+1):
                if j not in used_peaks:
                    if abs(peak1['centroid'] - peak2['centroid']) <= tolerance:
                        group.append(peak2)
                        used_peaks.add(j)
            
            if len(group) > 1:  # Only keep groups with multiple spectra
                peak_groups.append(group)
        
        # Filter for peaks appearing in multiple files
        common_peaks = []
        for group in peak_groups:
            sources = set(p['source'] for p in group)
            if len(sources) > 1:
                avg_centroid = np.mean([p['centroid'] for p in group])
                common_peaks.append({
                    'centroid': avg_centroid,
                    'sources': list(sources),
                    'count': len(sources),
                    'peaks': group
                })
        
        return sorted(common_peaks, key=lambda x: x['count'], reverse=True)
    
    def generate_report(self, output_dir):
        """
        Generate comprehensive HTML report for batch processing.
        
        Parameters:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        
        # Find common peaks
        common_peaks = self.find_common_peaks()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1, h2 {{ color: #333; }}
                .container {{ background-color: white; padding: 20px; border-radius: 10px; 
                             box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background-color: #4CAF50; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Batch Processing Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="container">
                <h2>Summary</h2>
                <p><strong>Total Files:</strong> {self.summary['total_files']}</p>
                <p><strong>Successful:</strong> <span class="success">{self.summary['successful']}</span></p>
                <p><strong>Failed:</strong> <span class="failed">{self.summary['failed']}</span></p>
                <p><strong>Total Peaks:</strong> {self.summary['total_peaks']}</p>
                <p><strong>Average Peaks per Spectrum:</strong> {self.summary['avg_peaks_per_spectrum']:.1f}</p>
            </div>
            
            <div class="container">
                <h2>File Processing Results</h2>
                <table>
                    <tr>
                        <th>File</th>
                        <th>Status</th>
                        <th>Channels</th>
                        <th>Total Counts</th>
                        <th>Peaks Detected</th>
                        <th>Peaks Fitted</th>
                    </tr>
        """
        
        for result in self.results:
            status = "✓" if result['success'] else "✗"
            status_class = "success" if result['success'] else "failed"
            
            html_content += f"""
                    <tr>
                        <td>{Path(result['file']).name}</td>
                        <td class="{status_class}">{status}</td>
                        <td>{result.get('channels', '-')}</td>
                        <td>{result.get('total_counts', '-'):.0f if result.get('total_counts') else '-'}</td>
                        <td>{result.get('peaks_detected', '-')}</td>
                        <td>{result.get('peaks_fitted', '-')}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Add common peaks section
        if common_peaks:
            html_content += """
            <div class="container">
                <h2>Common Peaks</h2>
                <p>Peaks appearing in multiple spectra (within 5 channels tolerance):</p>
                <table>
                    <tr>
                        <th>Channel</th>
                        <th>Appearances</th>
                        <th>Files</th>
                    </tr>
            """
            
            for peak in common_peaks[:10]:  # Show top 10
                files = ", ".join(peak['sources'][:3])
                if len(peak['sources']) > 3:
                    files += f" (+{len(peak['sources'])-3} more)"
                
                html_content += f"""
                    <tr>
                        <td>{peak['centroid']:.1f}</td>
                        <td>{peak['count']}</td>
                        <td>{files}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add plots
        html_content += """
            <div class="container">
                <h2>Comparison Plots</h2>
                <div class="plot">
                    <img src="batch_comparison.png" alt="Batch Comparison">
                </div>
                <div class="plot">
                    <img src="peak_alignment.png" alt="Peak Alignment">
                </div>
            </div>
            
            <div class="container">
                <p style="text-align: center; color: #666;">
                    Generated by GammaFit Batch Processor<br>
                    <a href="https://github.com/drgtheneutrino/gamma-peak-identifier">
                        https://github.com/drgtheneutrino/gamma-peak-identifier
                    </a>
                </p>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_file = output_dir / "batch_report.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {report_file}")


def main():
    """Main function for batch processing example."""
    
    print("=" * 60)
    print("GammaFit Batch Processing Example")
    print("=" * 60)
    
    # Setup paths
    example_dir = Path(__file__).parent
    output_dir = example_dir / "batch_output"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize batch processor
    config_file = example_dir / "example_config.json"
    processor = BatchProcessor(config_file)
    
    # Process all CSV files in examples directory
    print(f"\nSearching for spectrum files in: {example_dir}")
    processor.process_directory(example_dir, pattern="*.csv")
    
    # If no real files found, generate synthetic ones
    if not processor.results:
        print("\nNo spectrum files found. Generating synthetic spectra...")
        generate_test_spectra(example_dir)
        processor.process_directory(example_dir, pattern="test_*.csv")
    
    # Export results
    processor.export_results(output_dir)
    
    # Generate report
    processor.generate_report(output_dir)
    
    # Find and display common peaks
    common_peaks = processor.find_common_peaks()
    if common_peaks:
        print(f"\nFound {len(common_peaks)} common peak positions")
        print("\nTop 5 most common peaks:")
        for peak in common_peaks[:5]:
            print(f"  Channel {peak['centroid']:.1f}: appears in {peak['count']} spectra")
    
    print("\n" + "=" * 60)
    print("Batch processing complete!")
    print(f"Results saved in: {output_dir}")
    print("=" * 60)


def generate_test_spectra(output_dir):
    """Generate test spectra for demonstration."""
    from gammafit.utils import generate_synthetic_spectrum
    
    # Generate 3 similar spectra with variations
    base_peaks = [
        (511, 2000, 6),
        (661, 1500, 7),
        (1274, 1000, 9)
    ]
    
    for i in range(3):
        # Add random variations
        peaks = []
        for ch, amp, sigma in base_peaks:
            ch_varied = ch + np.random.randint(-5, 5)
            amp_varied = amp * (1 + np.random.uniform(-0.2, 0.2))
            peaks.append((ch_varied, amp_varied, sigma))
        
        # Generate spectrum
        channels, counts = generate_synthetic_spectrum(
            num_channels=2048,
            peaks=peaks,
            background_level=30 + i * 5,
            noise_level=1,
            seed=42 + i
        )
        
        # Save to file
        file_path = output_dir / f"test_spectrum_{i+1}.csv"
        df = pd.DataFrame({'channel': channels, 'counts': counts})
        df.to_csv(file_path, index=False, header=False)
        
        print(f"  Generated: {file_path.name}")


if __name__ == "__main__":
    main()
