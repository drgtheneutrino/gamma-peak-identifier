"""
Unit tests for peak detection module.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit.detection import (
    smooth_spectrum,
    detect_peaks,
    detect_peaks_scipy,
    detect_peaks_derivative,
    estimate_background,
    snip_background,
    rolling_ball_background,
    refine_peak_positions,
    group_peaks,
    estimate_peak_width
)
from gammafit.utils import generate_synthetic_spectrum


class TestSmoothing(unittest.TestCase):
    """Test spectrum smoothing functions."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.channels = np.arange(1000)
        # Create noisy spectrum
        self.counts = 100 * np.exp(-self.channels / 200) + 10 * np.random.randn(1000)
        self.counts = np.maximum(self.counts, 0)
    
    def test_savgol_smoothing(self):
        """Test Savitzky-Golay smoothing."""
        smoothed = smooth_spectrum(self.counts, window_length=5, method='savgol')
        
        # Check shape preserved
        self.assertEqual(len(smoothed), len(self.counts))
        
        # Check smoothing reduces noise
        original_std = np.std(np.diff(self.counts))
        smoothed_std = np.std(np.diff(smoothed))
        self.assertLess(smoothed_std, original_std)
        
        # Check non-negative
        self.assertTrue(np.all(smoothed >= 0))
    
    def test_gaussian_smoothing(self):
        """Test Gaussian smoothing."""
        smoothed = smooth_spectrum(self.counts, window_length=5, method='gaussian')
        
        self.assertEqual(len(smoothed), len(self.counts))
        self.assertTrue(np.all(smoothed >= 0))
    
    def test_no_smoothing(self):
        """Test with smoothing disabled."""
        smoothed = smooth_spectrum(self.counts, window_length=0, method='none')
        np.testing.assert_array_equal(smoothed, self.counts)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Very short spectrum
        short_counts = np.array([1, 2, 3])
        smoothed = smooth_spectrum(short_counts, window_length=5)
        self.assertEqual(len(smoothed), 3)
        
        # Single point
        single = np.array([10])
        smoothed = smooth_spectrum(single, window_length=3)
        self.assertEqual(len(smoothed), 1)


class TestPeakDetection(unittest.TestCase):
    """Test peak detection algorithms."""
    
    def setUp(self):
        """Create synthetic spectrum with known peaks."""
        np.random.seed(42)
        # Generate spectrum with known peaks
        self.channels, self.counts = generate_synthetic_spectrum(
            num_channels=2048,
            peaks=[
                (500, 1000, 5),   # Sharp strong peak
                (800, 200, 10),   # Weak broad peak
                (1200, 500, 8),   # Medium peak
                (1210, 300, 6),   # Overlapping peak
            ],
            background_level=10,
            noise_level=0.5
        )
        self.known_peaks = [500, 800, 1200, 1210]
    
    def test_scipy_detection(self):
        """Test scipy peak detection."""
        peaks = detect_peaks_scipy(
            self.counts,
            min_prominence=50,
            min_height=20,
            min_distance=3
        )
        
        # Should detect at least the strong peaks
        self.assertGreater(len(peaks), 0)
        
        # Check if known strong peak is detected
        self.assertTrue(any(abs(p - 500) < 10 for p in peaks))
    
    def test_derivative_detection(self):
        """Test derivative-based peak detection."""
        # Smooth first for derivative method
        smoothed = smooth_spectrum(self.counts, window_length=5)
        peaks = detect_peaks_derivative(
            smoothed,
            min_height=20,
            min_distance=3
        )
        
        self.assertGreater(len(peaks), 0)
    
    def test_detection_parameters(self):
        """Test effect of detection parameters."""
        # High threshold - fewer peaks
        peaks_high = detect_peaks(
            self.counts,
            min_prominence=200,
            min_height=100
        )
        
        # Low threshold - more peaks
        peaks_low = detect_peaks(
            self.counts,
            min_prominence=20,
            min_height=10
        )
        
        self.assertLess(len(peaks_high), len(peaks_low))
    
    def test_overlapping_peaks(self):
        """Test detection of overlapping peaks."""
        # The peaks at 1200 and 1210 are close
        peaks = detect_peaks(
            self.counts,
            min_prominence=50,
            min_distance=5  # Small distance to allow close peaks
        )
        
        # Check if both overlapping peaks are detected
        close_peaks = [p for p in peaks if 1190 < p < 1220]
        self.assertGreaterEqual(len(close_peaks), 1)  # At least one detected
    
    def test_no_peaks(self):
        """Test with spectrum containing no peaks."""
        flat_spectrum = np.ones(1000) * 10 + np.random.randn(1000) * 0.1
        peaks = detect_peaks(flat_spectrum, min_prominence=50)
        self.assertEqual(len(peaks), 0)


class TestBackgroundEstimation(unittest.TestCase):
    """Test background estimation methods."""
    
    def setUp(self):
        """Create test spectrum with background."""
        self.channels = np.arange(1000)
        # Exponential background + peaks
        self.background_true = 100 * np.exp(-self.channels / 300) + 10
        self.counts = self.background_true.copy()
        
        # Add peaks
        for center in [200, 500, 800]:
            peak = 500 * np.exp(-0.5 * ((self.channels - center) / 10) ** 2)
            self.counts += peak
    
    def test_snip_background(self):
        """Test SNIP background estimation."""
        bg_estimated = snip_background(self.counts, iterations=20)
        
        # Check shape
        self.assertEqual(len(bg_estimated), len(self.counts))
        
        # Background should be lower than spectrum
        self.assertTrue(np.all(bg_estimated <= self.counts + 1))  # Small tolerance
        
        # Check in peak-free regions
        bg_region = slice(350, 450)  # Between peaks
        bg_diff = np.abs(bg_estimated[bg_region] - self.background_true[bg_region])
        self.assertLess(np.mean(bg_diff), 20)  # Reasonable accuracy
    
    def test_rolling_ball_background(self):
        """Test rolling ball background estimation."""
        bg_estimated = rolling_ball_background(self.counts, radius=50)
        
        self.assertEqual(len(bg_estimated), len(self.counts))
        # Should be smooth
        self.assertLess(np.std(np.diff(bg_estimated)), np.std(np.diff(self.counts)))
    
    def test_percentile_background(self):
        """Test percentile-based background."""
        from gammafit.detection import percentile_background
        
        bg_estimated = percentile_background(self.counts, window=100, percentile=10)
        
        self.assertEqual(len(bg_estimated), len(self.counts))
        # Should be lower than most points
        self.assertLess(np.mean(bg_estimated), np.mean(self.counts))


class TestPeakRefinement(unittest.TestCase):
    """Test peak position refinement."""
    
    def setUp(self):
        """Create spectrum with precise peak."""
        self.channels = np.arange(100)
        # Peak at non-integer position
        true_center = 50.3
        self.counts = 100 * np.exp(-0.5 * ((self.channels - true_center) / 3) ** 2)
        self.counts += np.random.randn(100) * 0.1  # Small noise
        self.true_center = true_center
    
    def test_centroid_refinement(self):
        """Test centroid refinement method."""
        initial_peak = np.argmax(self.counts)
        refined = refine_peak_positions(
            self.counts,
            np.array([initial_peak]),
            method='centroid',
            window=10
        )
        
        # Should be close to true center
        self.assertAlmostEqual(refined[0], self.true_center, places=1)
    
    def test_parabolic_refinement(self):
        """Test parabolic interpolation."""
        initial_peak = np.argmax(self.counts)
        refined = refine_peak_positions(
            self.counts,
            np.array([initial_peak]),
            method='parabolic'
        )
        
        # Should improve position estimate
        initial_error = abs(initial_peak - self.true_center)
        refined_error = abs(refined[0] - self.true_center)
        self.assertLess(refined_error, initial_error)
    
    def test_gaussian_refinement(self):
        """Test Gaussian fit refinement."""
        initial_peak = np.argmax(self.counts)
        refined = refine_peak_positions(
            self.counts,
            np.array([initial_peak]),
            method='gaussian',
            window=10
        )
        
        # Should be reasonably close
        self.assertAlmostEqual(refined[0], self.true_center, delta=1.0)


class TestPeakGrouping(unittest.TestCase):
    """Test peak grouping for multiplets."""
    
    def test_single_peaks(self):
        """Test grouping of isolated peaks."""
        peaks = np.array([100, 300, 500, 700])
        groups = group_peaks(peaks, min_separation=50)
        
        # Each peak should be in its own group
        self.assertEqual(len(groups), 4)
        for group in groups:
            self.assertEqual(len(group), 1)
    
    def test_multiplet_grouping(self):
        """Test grouping of close peaks."""
        peaks = np.array([100, 105, 110, 300, 500, 505])
        groups = group_peaks(peaks, min_separation=10)
        
        # Should have 3 groups
        self.assertEqual(len(groups), 3)
        
        # First group should have 3 peaks
        self.assertEqual(len(groups[0]), 3)
        
        # Last group should have 2 peaks
        self.assertEqual(len(groups[2]), 2)
    
    def test_empty_peaks(self):
        """Test with no peaks."""
        peaks = np.array([])
        groups = group_peaks(peaks, min_separation=10)
        self.assertEqual(len(groups), 0)


class TestPeakWidth(unittest.TestCase):
    """Test peak width estimation."""
    
    def test_gaussian_peak_width(self):
        """Test FWHM estimation for Gaussian peak."""
        channels = np.arange(100)
        sigma = 5
        fwhm_true = 2.355 * sigma
        
        # Create perfect Gaussian
        counts = 1000 * np.exp(-0.5 * ((channels - 50) / sigma) ** 2)
        
        # Estimate FWHM
        estimated_fwhm = estimate_peak_width(counts, 50)
        
        # Should be close to true value
        self.assertAlmostEqual(estimated_fwhm, fwhm_true, delta=2)
    
    def test_narrow_peak(self):
        """Test with very narrow peak."""
        counts = np.zeros(100)
        counts[50] = 100
        counts[49] = 40
        counts[51] = 40
        
        fwhm = estimate_peak_width(counts, 50)
        
        # Should return reasonable minimum
        self.assertGreaterEqual(fwhm, 1.0)
        self.assertLessEqual(fwhm, 5.0)
    
    def test_edge_peak(self):
        """Test peak at spectrum edge."""
        counts = np.exp(-0.5 * ((np.arange(100) - 5) / 3) ** 2) * 100
        
        # Peak near edge
        fwhm = estimate_peak_width(counts, 5)
        
        # Should handle edge case
        self.assertGreater(fwhm, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for detection module."""
    
    def test_full_detection_pipeline(self):
        """Test complete detection pipeline."""
        # Generate complex spectrum
        channels, counts = generate_synthetic_spectrum(
            num_channels=4096,
            peaks=[
                (500, 2000, 5),
                (1000, 1000, 10),
                (1500, 500, 15),
                (2000, 1500, 8),
                (2500, 300, 20),
            ]
        )
        
        # Full pipeline
        smoothed = smooth_spectrum(counts, window_length=7, method='savgol')
        background = estimate_background(smoothed, method='snip')
        net_counts = smoothed - background
        net_counts = np.maximum(net_counts, 0)
        
        peaks = detect_peaks(net_counts, min_prominence=30, min_height=10)
        
        # Should detect most peaks
        self.assertGreaterEqual(len(peaks), 3)
        
        # Refine positions
        refined = refine_peak_positions(counts, peaks, method='centroid')
        
        # Should maintain same number of peaks
        self.assertEqual(len(refined), len(peaks))
        
        # Group peaks
        groups = group_peaks(peaks, min_separation=50)
        
        # Should have reasonable grouping
        self.assertGreaterEqual(len(groups), 1)


if __name__ == '__main__':
    unittest.main()
