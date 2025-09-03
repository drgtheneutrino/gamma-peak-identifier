#!/usr/bin/env python3
"""
Batch processing script for multiple gamma spectra.

Features:
- Process multiple files in parallel
- Apply consistent analysis parameters
- Generate comparison reports
- Export consolidated results
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
import multiprocessing as mp
from typing import List, Dict, Any, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit import load_spectrum, detect_peaks, fit_peaks, smooth_spectrum
from gammafit.calibration import EnergyCalibration, auto_calibrate
from gammafit.output import export_results, create_peak_comparison_plot
from gammafit.utils import check_spectrum_quality, setup_logger


class BatchAnalyzer:
    """Batch analyzer for multiple spectra."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize batch analyzer.
        
        Parameters:
            config: Analysis configuration dictionary
        """
        self.config = config or self.default_config()
        self.results = []
        self.failed_files = []
        self.logger = setup_logger('batch_analyzer')
        
    @staticmethod
    def default_config():
        """Get default configuration."""
        return {
            'detection': {
                'min_prominence': 50,
                'min_height': 30,
                'min_distance': 5,
                'smoothing_window': 7,
                'smoothing_method': 'savgol'
            },
            'fitting': {
                'peak_model': 'gaussian',
                'background_method': 'linear',
                'window_scale': 3.0
            },
            'calibration': {
                'auto_calibrate': False,
                'reference_isotope': 'Cs-137',
                'coefficients': None
            },
            'quality': {
                'min_total_counts': 1000,
                'min_peaks': 1,
                'max_chi_square': 10
            },
            'output': {
                'save_individual': True,
                'save_plots': False,
                'export_format': 'csv'
            }
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single spectrum file.
        
        Parameters:
            file_path: Path to spectrum file
            
        Returns:
            Processing results dictionary
        """
        result = {
            'file': file_path,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None
        }
        
        try:
            # Load spectrum
            channels, counts = load_spectrum(file_path)
            
            # Quality check
            quality = check_spectrum_quality(counts)
            result['quality'] = quality
            
            if quality['total_counts'] < self.config['quality']['min_total_counts']:
                raise ValueError(f"Low statistics: {quality['total_counts']} counts")
            
            # Smooth spectrum
            smoothed = smooth_spectrum(
                counts,
                window_length=self.config['detection']['smoothing_window'],
                method=self.config['detection']['smoothing_method']
            )
            
            # Detect peaks
            peaks = detect_peaks(
                smoothed,
                min_prominence=self.config['detection']['min_prominence'],
                min_height=self.config['detection']['min_height'],
                min_distance=self.config['detection']['min_distance']
            )
            
            result['peaks_detected'] = len(peaks)
            
            if len(peaks) < self.config['quality']['min_peaks']:
                raise ValueError(f"Too few peaks: {len(peaks)}")
            
            # Fit peaks
            fitted_peaks = fit_peaks(
                channels,
                counts,
                peaks,
                peak_model=self.config['fitting']['peak_model'],
                background_method=self.config['fitting']['background_method'],
                window_scale=self.config['fitting']['window_scale']
            )
            
            result['peaks_fitted'] = len(fitted_peaks)
            result['peak_data'] = fitted_peaks
            
            # Check fit quality
            poor_fits = [p for p in fitted_peaks 
                        if p.get('chi_square', 0) > self.config['quality']['max_chi_square']]
            if poor_fits:
                result['poor_fits'] = len(poor_fits)
                self.logger.warning(f"{file_path}: {len(poor_fits)} poor fits")
            
            # Apply calibration if configured
            if self.config['calibration']['coefficients']:
                from gammafit import apply_calibration
                calibration = EnergyCalibration(model='linear')
                calibration.coefficients = self.config['calibration']['coefficients']
                fitted_peaks = apply_calibration(fitted_peaks, calibration)
                result['calibrated'] = True
            
            # Statistics
            result['statistics'] = {
                'total_counts': float(np.sum(counts)),
                'max_counts': float(np.max(counts)),
                'mean_snr': float(np.mean([p['snr'] for p in fitted_peaks])),
                'mean_resolution': float(np.mean([p.get('resolution', 0) for p in fitted_peaks]))
            }
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Failed to process {file_path}: {e}")
            
        return result
    
    def process_directory(self, 
                         directory: str,
                         pattern: str = "*.csv",
                         parallel: bool = True,
                         max_workers: Optional[int] = None) -> List[Dict]:
        """
        Process all files in directory.
        
        Parameters:
            directory: Directory path
            pattern: File pattern
            parallel: Use parallel processing
            max_workers: Maximum number of workers
            
        Returns:
            List of results
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))
        
        if not files:
            self.logger.warning(f"No files matching {pattern} in {directory}")
            return []
        
        self.logger.info(f"Found {len(files)} files to process")
        
        start_time = time.time()
        
        if parallel and len(files) > 1:
            # Parallel processing
            max_workers = max_workers or mp.cpu_count()
            self.logger.info(f"Using {max_workers} workers for parallel processing")
            
            with mp.Pool(max_workers) as pool:
                results = pool.map(self.process_file, [str(f) for f in files])
        else:
            # Sequential processing
            results = []
            for i, file in enumerate(files, 1):
                self.logger.info(f"Processing {i}/{len(files)}: {file.name}")
                result = self.process_file(str(file))
                results.append(result)
                
                # Progress indication
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(files) - i) / rate
                    self.logger.info(f"Progress: {i}/{len(files)} ({rate:.1f} files/s, ETA: {eta:.0f}s)")
        
        # Store results
        self.results = results
        
        # Separate successful and failed
        successful = [r for r in results if r['success']]
        self.failed_files = [r for r in results if not r['success']]
        
        elapsed = time.time() - start_time
        
        self.logger.info(f"Processing complete in {elapsed:.1f}s")
        self.logger.info(f"Successful: {len(successful)}/{len(files)}")
        
        if self.failed_files:
            self.logger.warning(f"Failed files: {len(self.failed_files)}")
            for failed in self.failed_files:
                self.logger.warning(f"  {Path(failed['file']).name}: {failed['error']}")
        
        return results
    
    def find_common_peaks(self, tolerance: float = 5.0) -> List[Dict]:
        """
        Find peaks common across multiple spectra.
        
        Parameters:
            tolerance: Channel tolerance for matching
            
        Returns:
            List of common peak groups
        """
        successful = [r for r in self.results if r['success']]
        
        if len(successful) < 2:
            return []
        
        # Collect all peaks with file info
        all_peaks = []
        for result in successful:
            file_name = Path(result['file']).stem
            for peak in result.get('peak_data', []):
                all_peaks.append({
                    'file': file_name,
                    'centroid': peak['centroid'],
                    'area': peak['area'],
                    'energy': peak.get('energy')
                })
        
        # Sort by centroid
        all_peaks.sort(key=lambda p: p['centroid'])
        
        # Group similar peaks
        common_peaks = []
        current_group = [all_peaks[0]]
        
        for peak in all_peaks[1:]:
            if abs(peak['centroid'] - current_group[0]['centroid']) <= tolerance:
                current_group.append(peak)
            else:
                if len(set(p['file'] for p in current_group)) > 1:
                    # Peak appears in multiple files
                    common_peaks.append({
                        'centroid': np.mean([p['centroid'] for p in current_group]),
                        'centroid_std': np.std([p['centroid'] for p in current_group]),
                        'files': list(set(p['file'] for p in current_group)),
                'count': len(set(p['file'] for p in current_group)),
                'peaks': current_group
            })
        
        # Sort by frequency
        common_peaks.sort(key=lambda p: p['count'], reverse=True)
        
        return common_peaks
    
    def generate_summary_report(self, output_file: str):
        """
        Generate summary report of batch processing.
        
        Parameters:
            output_file: Output file path
        """
        output_path = Path(output_file)
        
        # Prepare summary data
        successful = [r for r in self.results if r['success']]
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_files': len(self.results),
            'successful': len(successful),
            'failed': len(self.failed_files),
            'configuration': self.config,
            'statistics': {}
        }
        
        if successful:
            # Calculate aggregate statistics
            summary['statistics'] = {
                'total_peaks': sum(r['peaks_fitted'] for r in successful),
                'avg_peaks_per_file': np.mean([r['peaks_fitted'] for r in successful]),
                'std_peaks_per_file': np.std([r['peaks_fitted'] for r in successful]),
                'total_counts': sum(r['statistics']['total_counts'] for r in successful),
                'avg_snr': np.mean([r['statistics']['mean_snr'] for r in successful]),
            }
            
            # Find common peaks
            common_peaks = self.find_common_peaks()
            summary['common_peaks'] = common_peaks[:10]  # Top 10
        
        # Save based on format
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        elif output_path.suffix == '.html':
            self._generate_html_report(summary, output_path)
        
        else:
            # Default to text report
            with open(output_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("BATCH PROCESSING SUMMARY REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Date: {summary['processing_date']}\n")
                f.write(f"Total files: {summary['total_files']}\n")
                f.write(f"Successful: {summary['successful']}\n")
                f.write(f"Failed: {summary['failed']}\n\n")
                
                if summary['statistics']:
                    f.write("STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    for key, value in summary['statistics'].items():
                        f.write(f"{key}: {value:.2f}\n")
                
                if 'common_peaks' in summary and summary['common_peaks']:
                    f.write("\nCOMMON PEAKS\n")
                    f.write("-" * 40 + "\n")
                    for peak in summary['common_peaks'][:5]:
                        f.write(f"Channel {peak['centroid']:.1f}: appears in {peak['count']} files\n")
        
        self.logger.info(f"Report saved to {output_path}")
    
    def _generate_html_report(self, summary: Dict, output_path: Path):
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th {{ background-color: #4CAF50; color: white; padding: 10px; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Batch Processing Report</h1>
            <p>Generated: {summary['processing_date']}</p>
            
            <h2>Summary</h2>
            <p>Total Files: {summary['total_files']}</p>
            <p class="success">Successful: {summary['successful']}</p>
            <p class="failed">Failed: {summary['failed']}</p>
            
            <h2>Results</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>Status</th>
                    <th>Peaks</th>
                    <th>Total Counts</th>
                </tr>
        """
        
        for result in self.results:
            status = "✓" if result['success'] else "✗"
            peaks = result.get('peaks_fitted', '-')
            counts = result.get('statistics', {}).get('total_counts', '-')
            if isinstance(counts, float):
                counts = f"{counts:.0f}"
            
            html += f"""
                <tr>
                    <td>{Path(result['file']).name}</td>
                    <td>{status}</td>
                    <td>{peaks}</td>
                    <td>{counts}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def export_consolidated_results(self, output_dir: str):
        """
        Export all results to directory.
        
        Parameters:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Export individual peak lists
        if self.config['output']['save_individual']:
            peaks_dir = output_path / 'peak_lists'
            peaks_dir.mkdir(exist_ok=True)
            
            for result in self.results:
                if result['success'] and 'peak_data' in result:
                    file_stem = Path(result['file']).stem
                    peak_file = peaks_dir / f"{file_stem}_peaks.csv"
                    export_results(result['peak_data'], str(peak_file))
        
        # Export consolidated peak table
        all_peaks = []
        for result in self.results:
            if result['success'] and 'peak_data' in result:
                for peak in result['peak_data']:
                    peak_entry = peak.copy()
                    peak_entry['source_file'] = Path(result['file']).name
                    all_peaks.append(peak_entry)
        
        if all_peaks:
            df = pd.DataFrame(all_peaks)
            df.to_csv(output_path / 'all_peaks.csv', index=False)
        
        # Export summary
        self.generate_summary_report(str(output_path / 'summary.json'))
        
        # Generate plots if requested
        if self.config['output']['save_plots']:
            self._generate_comparison_plots(output_path)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def _generate_comparison_plots(self, output_dir: Path):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        successful = [r for r in self.results if r['success']]
        
        if not successful:
            return
        
        # Plot 1: Peak count distribution
        ax1 = axes[0, 0]
        peak_counts = [r['peaks_fitted'] for r in successful]
        ax1.hist(peak_counts, bins=max(10, len(set(peak_counts))), edgecolor='black')
        ax1.set_xlabel('Number of Peaks')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Peak Count Distribution')
        
        # Plot 2: Total counts distribution
        ax2 = axes[0, 1]
        total_counts = [r['statistics']['total_counts'] for r in successful]
        ax2.hist(total_counts, bins=20, edgecolor='black')
        ax2.set_xlabel('Total Counts')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Total Counts Distribution')
        
        # Plot 3: SNR distribution
        ax3 = axes[1, 0]
        mean_snrs = [r['statistics']['mean_snr'] for r in successful]
        ax3.hist(mean_snrs, bins=20, edgecolor='black')
        ax3.set_xlabel('Mean SNR')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Signal-to-Noise Ratio Distribution')
        
        # Plot 4: Success rate
        ax4 = axes[1, 1]
        success_rate = len(successful) / len(self.results) * 100
        ax4.pie([len(successful), len(self.failed_files)],
                labels=['Success', 'Failed'],
                autopct='%1.1f%%',
                colors=['green', 'red'])
        ax4.set_title(f'Processing Success Rate: {success_rate:.1f}%')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'batch_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Batch process multiple gamma spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSV files in directory
  python batch_process.py data/ -o results/
  
  # Process with configuration file
  python batch_process.py data/ -c config.json -o results/
  
  # Process specific pattern with parallel execution
  python batch_process.py data/ -p "na22_*.csv" --parallel -w 4
  
  # Generate detailed report
  python batch_process.py data/ -o results/ --report html --plots
        """
    )
    
    parser.add_argument('directory', help='Directory containing spectrum files')
    parser.add_argument('-o', '--output', default='batch_results',
                       help='Output directory (default: batch_results)')
    parser.add_argument('-p', '--pattern', default='*.csv',
                       help='File pattern to match (default: *.csv)')
    parser.add_argument('-c', '--config', help='Configuration file (JSON)')
    
    # Processing options
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    parser.add_argument('-w', '--workers', type=int,
                       help='Number of parallel workers')
    
    # Analysis parameters
    parser.add_argument('--min-prominence', type=float,
                       help='Minimum peak prominence')
    parser.add_argument('--min-height', type=float,
                       help='Minimum peak height')
    parser.add_argument('--smoothing', type=int,
                       help='Smoothing window size')
    
    # Output options
    parser.add_argument('--report', choices=['txt', 'json', 'html'], default='json',
                       help='Report format (default: json)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual peak lists')
    
    # Quality filters
    parser.add_argument('--min-counts', type=float, default=1000,
                       help='Minimum total counts required')
    parser.add_argument('--min-peaks', type=int, default=1,
                       help='Minimum peaks required')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = BatchAnalyzer.default_config()
    
    # Override with command line arguments
    if args.min_prominence:
        config['detection']['min_prominence'] = args.min_prominence
    if args.min_height:
        config['detection']['min_height'] = args.min_height
    if args.smoothing:
        config['detection']['smoothing_window'] = args.smoothing
    
    config['quality']['min_total_counts'] = args.min_counts
    config['quality']['min_peaks'] = args.min_peaks
    config['output']['save_individual'] = args.save_individual
    config['output']['save_plots'] = args.plots
    
    # Create analyzer
    analyzer = BatchAnalyzer(config)
    
    # Process directory
    print(f"Processing files in {args.directory}")
    results = analyzer.process_directory(
        args.directory,
        pattern=args.pattern,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Export results
    analyzer.export_consolidated_results(args.output)
    
    # Generate report
    report_file = Path(args.output) / f"report.{args.report}"
    analyzer.generate_summary_report(str(report_file))
    
    # Find common peaks
    common_peaks = analyzer.find_common_peaks()
    if common_peaks:
        print(f"\nFound {len(common_peaks)} common peak positions")
        print("Top 5 most common peaks:")
        for peak in common_peaks[:5]:
            print(f"  Channel {peak['centroid']:.1f} (±{peak['centroid_std']:.2f}): "
                  f"appears in {peak['count']} files")
    
    print(f"\nResults saved to {args.output}")
    print("Processing complete!")


if __name__ == "__main__":
    main()(set(p['file'] for p in current_group)),
                        'count': len(set(p['file'] for p in current_group)),
                        'peaks': current_group
                    })
                current_group = [peak]
        
        # Check last group
        if len(set(p['file'] for p in current_group)) > 1:
            common_peaks.append({
                'centroid': np.mean([p['centroid'] for p in current_group]),
                'centroid_std': np.std([p['centroid'] for p in current_group]),
                'files': list
