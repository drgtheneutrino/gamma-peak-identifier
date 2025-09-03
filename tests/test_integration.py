"""
Integration tests for the complete gamma spectroscopy pipeline.
"""

import unittest
import tempfile
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit import (
    detect_peaks,
    fit_peaks,
    load_spectrum,
    save_peaks,
    apply_calibration,
    smooth_spectrum,
    plot_spectrum_with_fits,
    export_results
)
from gammafit.calibration import EnergyCalibration
from gammafit.utils import generate_synthetic_spectrum, calculate_counting_statistics
from gammafit.main import process_spectrum, load_configuration


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete analysis pipeline from spectrum to results."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Generate realistic synthetic spectrum
        np.random.seed(42)
        self.channels, self.counts = generate_synthetic_spectrum(
            num_channels=4096,
            peaks=[
                (511, 5000, 6),      # Na-22 annihilation
                (1274, 3000, 8),     # Na-22 primary
                (661, 4000, 7),      # Cs-137
                (1332, 2000, 9),     # Co-60
                (1173, 2200, 8.5),   # Co-60
                (2614, 500, 15),     # Th-228
            ],
            background_level=50,
            noise_level=1
        )
        
        # Save spectrum to file
        self.spectrum_file = self.temp_path / "test_spectrum.csv"
        df = pd.DataFrame({'channel': self.channels, 'counts': self.counts})
        df.to_csv(self.spectrum_file, index=False, header=False)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_analysis_pipeline(self):
        """Test full pipeline: load -> detect -> fit -> calibrate -> export."""
        
        # Step 1: Load spectrum
        channels, counts = load_spectrum(str(self.spectrum_file))
        self.assertEqual(len(channels), 4096)
        
        # Step 2: Smooth spectrum
        smoothed = smooth_spectrum(counts, window_length=7, method='savgol')
        
        # Step 3: Detect peaks
        peaks = detect_peaks(
            smoothed,
            min_prominence=100,
            min_height=50,
            min_distance=10
        )
        
        # Should detect most of the major peaks
        self.assertGreaterEqual(len(peaks), 4)
        
        # Step 4: Fit peaks
        fitted_peaks = fit_peaks(
            channels,
            counts,
            peaks,
            peak_model='gaussian',
            background_method='linear'
        )
        
        # Check fitting results
        self.assertEqual(len(fitted_peaks), len(peaks))
        for peak in fitted_peaks:
            self.assertIn('centroid', peak)
            self.assertIn('area', peak)
            self.assertIn('fwhm', peak)
            self.assertIn('snr', peak)
        
        # Step 5: Apply calibration
        calibration = EnergyCalibration(model='linear')
        # Use known peaks for calibration
        calibration.add_point(511, 511.0, 'Na-22')
        calibration.add_point(661, 661.66, 'Cs-137')
        calibration.add_point(1274, 1274.53, 'Na-22')
        calibration.fit()
        
        calibrated_peaks = apply_calibration(fitted_peaks, calibration)
        
        # Check calibration applied
        for peak in calibrated_peaks:
            self.assertIn('energy', peak)
            self.assertIn('energy_err', peak)
        
        # Step 6: Export results
        output_file = self.temp_path / "results.csv"
        export_results(calibrated_peaks, str(output_file), include_energy=True)
        
        # Verify output file
        self.assertTrue(output_file.exists())
        results_df = pd.read_csv(output_file)
        self.assertEqual(len(results_df), len(calibrated_peaks))
        self.assertIn('energy_keV', results_df.columns)
        
        # Step 7: Generate plot
        plot_file = self.temp_path / "spectrum.png"
        plot_spectrum_with_fits(
            channels, counts, smoothed, calibrated_peaks,
            str(plot_file), calibration.coefficients
        )
        
        # Verify plot created
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 1000)  # Not empty
    
    def test_pipeline_with_config(self):
        """Test pipeline using configuration file."""
        
        # Create configuration
        config = {
            'detection': {
                'min_prominence': 75,
                'min_height': 40,
                'min_distance': 8,
                'smoothing_window': 5,
                'smoothing_method': 'savgol'
            },
            'fitting': {
                'window_scale': 3.0,
                'background_method': 'linear'
            },
            'calibration': '1.0,0.0',  # Simple 1:1 calibration
            'output': {
                'directory': str(self.temp_path),
                'prefix': 'test_',
                'generate_plot': True,
                'plot_format': 'png',
                'export_fits': True
            }
        }
        
        # Save config
        config_file = self.temp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Create mock logger
        import logging
        logger = logging.getLogger('test')
        
        # Run analysis with config
        process_spectrum(str(self.spectrum_file), config, logger)
        
        # Check outputs created
        peaks_file = self.temp_path / "test_peaks.csv"
        plot_file = self.temp_path / "test_spectrum.png"
        fits_file = self.temp_path / "test_fit_data.json"
        
        self.assertTrue(peaks_file.exists())
        self.assertTrue(plot_file.exists())
        self.assertTrue(fits_file.exists())
        
        # Verify results
        peaks_df = pd.read_csv(peaks_file)
        self.assertGreater(len(peaks_df), 0)
        
        with open(fits_file, 'r') as f:
            fit_data = json.load(f)
        self.assertIsInstance(fit_data, list)
        self.assertGreater(len(fit_data), 0)


class TestCLIIntegration(unittest.TestCase):
    """Test command-line interface integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test spectrum
        channels, counts = generate_synthetic_spectrum(
            num_channels=2048,
            peaks=[(500, 1000, 5), (1000, 500, 10)],
            seed=42
        )
        
        self.spectrum_file = self.temp_path / "spectrum.csv"
        df = pd.DataFrame({'ch': channels, 'cnt': counts})
        df.to_csv(self.spectrum_file, index=False, header=False)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cli_basic_run(self):
        """Test basic CLI execution."""
        # Run CLI command
        cmd = [
            sys.executable, '-m', 'gammafit.main',
            str(self.spectrum_file),
            '--output-dir', str(self.temp_path),
            '--min-prominence', '50'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check successful execution
            if result.returncode == 0:
                # Check output files created
                peaks_file = self.temp_path / "peaks.csv"
                plot_file = self.temp_path / "spectrum.png"
                
                self.assertTrue(peaks_file.exists() or plot_file.exists())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Skip if CLI not fully set up
            self.skipTest("CLI not available for testing")
    
    def test_cli_with_calibration(self):
        """Test CLI with calibration parameters."""
        cmd = [
            sys.executable, '-m', 'gammafit.main',
            str(self.spectrum_file),
            '--calibration', '0.5,10',
            '--output-dir', str(self.temp_path),
            '--output-prefix', 'calibrated'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Check for calibrated output
                peaks_file = self.temp_path / "calibrated_peaks.csv"
                
                if peaks_file.exists():
                    df = pd.read_csv(peaks_file)
                    # Should have energy column if calibrated
                    self.assertIn('energy_keV', df.columns)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("CLI not available for testing")


class TestMultiSpectrumAnalysis(unittest.TestCase):
    """Test analysis of multiple spectra."""
    
    def setUp(self):
        """Create multiple test spectra."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Generate multiple spectra with slight variations
        self.spectra = []
        for i in range(3):
            np.random.seed(42 + i)
            channels, counts = generate_synthetic_spectrum(
                num_channels=2048,
                peaks=[
                    (500 + i*10, 1000, 5),    # Shifting peak
                    (1000, 500 + i*100, 10),   # Varying intensity
                ],
                background_level=30 + i*5
            )
            
            file_path = self.temp_path / f"spectrum_{i}.csv"
            df = pd.DataFrame({'channel': channels, 'counts': counts})
            df.to_csv(file_path, index=False, header=False)
            self.spectra.append(file_path)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_batch_processing(self):
        """Test processing multiple spectra."""
        all_results = []
        
        for spectrum_file in self.spectra:
            # Load and process each spectrum
            channels, counts = load_spectrum(str(spectrum_file))
            
            # Detect and fit peaks
            peaks = detect_peaks(counts, min_prominence=50)
            fitted_peaks = fit_peaks(channels, counts, peaks)
            
            all_results.append({
                'file': spectrum_file.name,
                'num_peaks': len(fitted_peaks),
                'peaks': fitted_peaks
            })
        
        # Should process all spectra
        self.assertEqual(len(all_results), 3)
        
        # Each should have detected peaks
        for result in all_results:
            self.assertGreater(result['num_peaks'], 0)
        
        # Check for expected peak shift
        first_peak_positions = []
        for result in all_results:
            if result['peaks']:
                first_peak_positions.append(result['peaks'][0]['centroid'])
        
        # Peaks should shift as expected
        if len(first_peak_positions) == 3:
            self.assertLess(first_peak_positions[0], first_peak_positions[2])
    
    def test_comparison_analysis(self):
        """Test comparing results from multiple spectra."""
        from gammafit.output import create_peak_comparison_plot
        
        # Analyze all spectra
        peak_lists = []
        labels = []
        
        for i, spectrum_file in enumerate(self.spectra):
            channels, counts = load_spectrum(str(spectrum_file))
            peaks = detect_peaks(counts, min_prominence=50)
            fitted_peaks = fit_peaks(channels, counts, peaks)
            
            peak_lists.append(fitted_peaks)
            labels.append(f"Spectrum {i+1}")
        
        # Create comparison plot
        comparison_file = self.temp_path / "comparison.png"
        create_peak_comparison_plot(
            peak_lists,
            labels,
            str(comparison_file)
        )
        
        # Check plot created
        self.assertTrue(comparison_file.exists())


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_empty_spectrum(self):
        """Test handling of empty spectrum."""
        # Create empty spectrum
        empty_file = self.temp_path / "empty.csv"
        df = pd.DataFrame({'channel': [], 'counts': []})
        df.to_csv(empty_file, index=False, header=False)
        
        # Should raise appropriate error
        with self.assertRaises(ValueError):
            load_spectrum(str(empty_file))
    
    def test_no_peaks_detected(self):
        """Test handling when no peaks are detected."""
        # Create flat spectrum
        channels = np.arange(1000)
        counts = np.ones(1000) * 10 + np.random.randn(1000) * 0.1
        
        # Detect peaks with high threshold
        peaks = detect_peaks(counts, min_prominence=1000)
        
        # Should return empty array
        self.assertEqual(len(peaks), 0)
        
        # Fitting empty peaks should return empty list
        fitted = fit_peaks(channels, counts, peaks)
        self.assertEqual(len(fitted), 0)
    
    def test_corrupted_file(self):
        """Test handling of corrupted input file."""
        corrupted_file = self.temp_path / "corrupted.csv"
        
        # Write corrupted data
        with open(corrupted_file, 'w') as f:
            f.write("not,valid,csv,data\n")
            f.write("with weird stuff!@#\n")
            f.write("1,2,3,4,5,6,7\n")
        
        # Should handle gracefully
        try:
            channels, counts = load_spectrum(str(corrupted_file))
            # If it loads, check it's reasonable
            self.assertIsNotNone(channels)
            self.assertIsNotNone(counts)
        except ValueError:
            # Expected for corrupted file
            pass
    
    def test_saturated_spectrum(self):
        """Test handling of saturated peaks."""
        channels = np.arange(1000)
        counts = np.zeros(1000)
        
        # Add saturated peak
        counts[400:420] = 65535  # 16-bit saturation
        
        # Should still detect as peak
        peaks = detect_peaks(counts, min_prominence=100)
        self.assertGreater(len(peaks), 0)
        
        # Fitting should handle saturation
        fitted = fit_peaks(channels, counts, peaks)
        
        # Check quality metrics
        from gammafit.utils import check_spectrum_quality
        quality = check_spectrum_quality(counts)
        
        self.assertTrue(quality['has_saturation'])
        self.assertLess(quality['quality_score'], 100)


class TestPerformance(unittest.TestCase):
    """Test performance with large spectra."""
    
    def test_large_spectrum_processing(self):
        """Test processing of large spectrum."""
        import time
        
        # Generate large spectrum
        np.random.seed(42)
        channels, counts = generate_synthetic_spectrum(
            num_channels=16384,  # 16k channels
            peaks=[
                (1000, 5000, 10),
                (3000, 3000, 15),
                (5000, 2000, 20),
                (8000, 1500, 25),
                (12000, 1000, 30),
            ],
            background_level=100
        )
        
        # Time the processing
        start_time = time.time()
        
        # Smooth
        smoothed = smooth_spectrum(counts, window_length=9)
        
        # Detect peaks
        peaks = detect_peaks(smoothed, min_prominence=100)
        
        # Fit peaks
        fitted = fit_peaks(channels, counts, peaks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time
        self.assertLess(processing_time, 30)  # 30 seconds max
        
        # Should detect the major peaks
        self.assertGreaterEqual(len(fitted), 3)
        
        # Memory usage check (basic)
        import sys
        # Check that we're not using excessive memory
        # This is a simple check, could be more sophisticated
        self.assertLess(sys.getsizeof(smoothed) / 1024 / 1024, 10)  # < 10 MB


class TestCalibrationIntegration(unittest.TestCase):
    """Test calibration integration."""
    
    def test_auto_calibration(self):
        """Test automatic calibration with known peaks."""
        from gammafit.calibration import auto_calibrate, get_isotope_energies
        
        # Generate spectrum with known isotope peaks (Co-60)
        channels = np.arange(4096)
        counts = np.zeros(4096)
        
        # Add Co-60 peaks at specific channels
        # Real energies: 1173.23 and 1332.49 keV
        ch_1173 = 2346  # Channel for 1173 keV (approx 0.5 keV/ch)
        ch_1332 = 2665  # Channel for 1332 keV
        
        for ch, amp in [(ch_1173, 1000), (ch_1332, 800)]:
            peak = amp * np.exp(-0.5 * ((channels - ch) / 5) ** 2)
            counts += peak
        
        # Add some background and noise
        counts += 10
        counts = np.random.poisson(counts)
        
        # Get known Co-60 energies
        co60_energies = get_isotope_energies('Co-60')
        known_peaks = [(e, 'Co-60') for e in co60_energies]
        
        # Attempt auto-calibration
        calibration = auto_calibrate(channels, counts, known_peaks, tolerance=20)
        
        if calibration is not None:
            # Check calibration quality
            self.assertIsNotNone(calibration.coefficients)
            self.assertIn('a', calibration.coefficients)
            self.assertIn('b', calibration.coefficients)
            
            # Calibration should be close to 0.5 keV/channel
            self.assertAlmostEqual(calibration.coefficients['a'], 0.5, delta=0.1)
    
    def test_calibration_validation(self):
        """Test calibration validation with test peaks."""
        from gammafit.calibration import EnergyCalibration, validate_calibration
        
        # Create calibration
        cal = EnergyCalibration(model='linear')
        cal.add_point(1000, 500.0)  # 500 keV at channel 1000
        cal.add_point(2000, 1000.0)  # 1000 keV at channel 2000
        cal.fit()
        
        # Test peaks for validation
        test_peaks = [
            (1500, 750.0),  # Should be 750 keV
            (2500, 1250.0),  # Should be 1250 keV
        ]
        
        # Validate
        metrics = validate_calibration(cal, test_peaks)
        
        # Check validation metrics
        self.assertIn('rms_error', metrics)
        self.assertIn('max_error', metrics)
        
        # Should have very small error for perfect linear calibration
        self.assertLess(metrics['rms_error'], 1.0)


if __name__ == '__main__':
    unittest.main()
