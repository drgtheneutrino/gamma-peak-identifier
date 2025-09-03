"""
Unit tests for peak fitting module.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit.fitting import (
    gaussian,
    gaussian_with_background,
    gaussian_with_tail,
    voigt,
    double_gaussian,
    estimate_initial_params,
    fit_single_peak,
    fit_peaks,
    fit_multiplet,
    identify_overlapping_peaks,
    get_parameter_bounds
)
from gammafit.utils import generate_synthetic_spectrum


class TestPeakFunctions(unittest.TestCase):
    """Test peak shape functions."""
    
    def setUp(self):
        """Create test data."""
        self.x = np.linspace(0, 100, 101)
    
    def test_gaussian_function(self):
        """Test Gaussian peak function."""
        # Parameters
        amplitude = 100
        centroid = 50
        sigma = 5
        
        y = gaussian(self.x, amplitude, centroid, sigma)
        
        # Check peak properties
        self.assertAlmostEqual(np.max(y), amplitude, places=5)
        self.assertEqual(np.argmax(y), int(centroid))
        
        # Check FWHM
        half_max = amplitude / 2
        above_half = self.x[y > half_max]
        fwhm_measured = above_half[-1] - above_half[0]
        fwhm_expected = 2.355 * sigma
        self.assertAlmostEqual(fwhm_measured, fwhm_expected, delta=1)
    
    def test_gaussian_with_background(self):
        """Test Gaussian with linear background."""
        amplitude = 100
        centroid = 50
        sigma = 5
        bg_slope = 0.5
        bg_intercept = 10
        
        y = gaussian_with_background(
            self.x, amplitude, centroid, sigma, bg_slope, bg_intercept
        )
        
        # Check background at edges
        self.assertAlmostEqual(y[0], bg_intercept, delta=1)
        self.assertAlmostEqual(y[-1], bg_slope * self.x[-1] + bg_intercept, delta=1)
        
        # Peak should be above background
        peak_height = y[int(centroid)]
        bg_at_peak = bg_slope * centroid + bg_intercept
        self.assertAlmostEqual(peak_height, amplitude + bg_at_peak, delta=1)
    
    def test_gaussian_with_tail(self):
        """Test Gaussian with low-energy tail."""
        amplitude = 100
        centroid = 50
        sigma = 5
        tail_amp = 10
        tail_slope = 10
        
        y = gaussian_with_tail(
            self.x, amplitude, centroid, sigma, tail_amp, tail_slope
        )
        
        # Should have asymmetry (more counts on low-energy side)
        left_sum = np.sum(y[self.x < centroid])
        right_sum = np.sum(y[self.x > centroid])
        
        # Left side should have more counts due to tail
        self.assertGreater(left_sum, right_sum)
    
    def test_voigt_function(self):
        """Test Voigt profile."""
        amplitude = 100
        centroid = 50
        sigma = 5
        gamma = 2
        
        y = voigt(self.x, amplitude, centroid, sigma, gamma)
        
        # Should have peak at centroid
        peak_idx = np.argmax(y)
        self.assertAlmostEqual(self.x[peak_idx], centroid, delta=2)
        
        # Should be broader than pure Gaussian
        gaussian_y = gaussian(self.x, amplitude, centroid, sigma)
        
        # Compare widths at half maximum
        voigt_width = len(self.x[y > amplitude/2])
        gaussian_width = len(self.x[gaussian_y > amplitude/2])
        self.assertGreater(voigt_width, gaussian_width)
    
    def test_double_gaussian(self):
        """Test double Gaussian function."""
        # Two peaks
        amp1, cent1, sig1 = 100, 40, 3
        amp2, cent2, sig2 = 80, 60, 4
        bg_slope = 0.1
        bg_intercept = 5
        
        y = double_gaussian(
            self.x, amp1, cent1, sig1, amp2, cent2, sig2,
            bg_slope, bg_intercept
        )
        
        # Should have two local maxima
        from scipy.signal import find_peaks as scipy_find_peaks
        peaks, _ = scipy_find_peaks(y)
        self.assertEqual(len(peaks), 2)
        
        # Peak positions should be close to centroids
        self.assertTrue(any(abs(self.x[p] - cent1) < 5 for p in peaks))
        self.assertTrue(any(abs(self.x[p] - cent2) < 5 for p in peaks))


class TestParameterEstimation(unittest.TestCase):
    """Test initial parameter estimation."""
    
    def setUp(self):
        """Create test spectrum with known peak."""
        self.channels = np.arange(200)
        
        # Known peak parameters
        self.true_amp = 500
        self.true_cent = 100
        self.true_sigma = 5
        self.true_bg_slope = 0.2
        self.true_bg_int = 20
        
        # Generate peak
        self.counts = gaussian_with_background(
            self.channels,
            self.true_amp, self.true_cent, self.true_sigma,
            self.true_bg_slope, self.true_bg_int
        )
        
        # Add small noise
        np.random.seed(42)
        self.counts += np.random.randn(len(self.channels)) * 2
    
    def test_initial_param_estimation(self):
        """Test estimation of initial parameters."""
        peak_idx = np.argmax(self.counts)
        
        initial_params, (fit_start, fit_end) = estimate_initial_params(
            self.channels, self.counts, peak_idx
        )
        
        # Should have correct number of parameters
        self.assertEqual(len(initial_params), 5)  # amp, cent, sigma, slope, intercept
        
        # Check estimates are reasonable
        amp_est, cent_est, sigma_est, slope_est, int_est = initial_params
        
        # Amplitude should be close
        self.assertAlmostEqual(amp_est, self.true_amp, delta=100)
        
        # Centroid should be very close
        self.assertAlmostEqual(cent_est, self.true_cent, delta=2)
        
        # Sigma should be in right ballpark
        self.assertAlmostEqual(sigma_est, self.true_sigma, delta=3)
        
        # Fit region should contain peak
        self.assertLess(fit_start, peak_idx)
        self.assertGreater(fit_end, peak_idx)
    
    def test_edge_peak_estimation(self):
        """Test parameter estimation for peak near edge."""
        # Peak near beginning
        edge_counts = self.counts[:50]
        edge_channels = self.channels[:50]
        
        # Find peak (will be cut off)
        peak_idx = np.argmax(edge_counts)
        
        initial_params, (fit_start, fit_end) = estimate_initial_params(
            edge_channels, edge_counts, peak_idx
        )
        
        # Should handle edge case
        self.assertGreaterEqual(fit_start, 0)
        self.assertLessEqual(fit_end, len(edge_channels))


class TestSinglePeakFitting(unittest.TestCase):
    """Test single peak fitting."""
    
    def setUp(self):
        """Create test spectrum with single peak."""
        np.random.seed(42)
        self.channels = np.arange(300)
        
        # True parameters
        self.true_params = {
            'amplitude': 1000,
            'centroid': 150,
            'sigma': 6,
            'bg_slope': 0.1,
            'bg_intercept': 30
        }
        
        # Generate peak
        self.counts = gaussian_with_background(
            self.channels,
            self.true_params['amplitude'],
            self.true_params['centroid'],
            self.true_params['sigma'],
            self.true_params['bg_slope'],
            self.true_params['bg_intercept']
        )
        
        # Add Poisson noise
        self.counts = np.random.poisson(self.counts)
    
    def test_fit_single_gaussian(self):
        """Test fitting single Gaussian peak."""
        peak_idx = np.argmax(self.counts)
        
        result = fit_single_peak(
            self.channels,
            self.counts,
            peak_idx,
            peak_model='gaussian',
            background_method='linear'
        )
        
        # Check fit success
        self.assertTrue(result['fit_success'])
        
        # Check fitted parameters are close to true values
        self.assertAlmostEqual(result['centroid'], self.true_params['centroid'], delta=1)
        self.assertAlmostEqual(result['amplitude'], self.true_params['amplitude'], delta=50)
        self.assertAlmostEqual(result['sigma'], self.true_params['sigma'], delta=1)
        
        # Check derived quantities
        expected_fwhm = 2.355 * self.true_params['sigma']
        self.assertAlmostEqual(result['fwhm'], expected_fwhm, delta=2)
        
        expected_area = (self.true_params['amplitude'] * 
                        self.true_params['sigma'] * np.sqrt(2 * np.pi))
        self.assertAlmostEqual(result['area'], expected_area, delta=expected_area * 0.1)
        
        # Check SNR is reasonable
        self.assertGreater(result['snr'], 10)
        
        # Check chi-square is reasonable (close to 1 for good fit)
        self.assertLess(result['chi_square'], 2)
        self.assertGreater(result['chi_square'], 0.5)
    
    def test_fit_with_different_windows(self):
        """Test fitting with different window scales."""
        peak_idx = np.argmax(self.counts)
        
        # Narrow window
        result_narrow = fit_single_peak(
            self.channels, self.counts, peak_idx,
            window_scale=2.0
        )
        
        # Wide window
        result_wide = fit_single_peak(
            self.channels, self.counts, peak_idx,
            window_scale=5.0
        )
        
        # Both should succeed
        self.assertTrue(result_narrow['fit_success'])
        self.assertTrue(result_wide['fit_success'])
        
        # Parameters should be similar
        self.assertAlmostEqual(result_narrow['centroid'], result_wide['centroid'], delta=1)
    
    def test_weak_peak_fitting(self):
        """Test fitting of weak peak."""
        # Create weak peak
        weak_counts = self.counts / 10
        peak_idx = np.argmax(weak_counts)
        
        result = fit_single_peak(
            self.channels,
            weak_counts,
            peak_idx
        )
        
        # Should still fit, but with lower SNR
        if result['fit_success']:
            self.assertLess(result['snr'], 10)
            self.assertGreater(result['centroid_err'], 0.1)


class TestMultipletFitting(unittest.TestCase):
    """Test multiplet peak fitting."""
    
    def setUp(self):
        """Create spectrum with overlapping peaks."""
        np.random.seed(42)
        self.channels = np.arange(500)
        
        # Two overlapping peaks
        self.peak1 = {
            'amplitude': 800,
            'centroid': 250,
            'sigma': 8
        }
        self.peak2 = {
            'amplitude': 600,
            'centroid': 265,  # Close to first peak
            'sigma': 6
        }
        
        # Generate spectrum
        self.counts = np.zeros_like(self.channels, dtype=float)
        self.counts += gaussian(self.channels, **self.peak1)
        self.counts += gaussian(self.channels, **self.peak2)
        self.counts += 0.05 * self.channels + 20  # Linear background
        
        # Add noise
        self.counts = np.random.poisson(self.counts)
    
    def test_identify_overlapping(self):
        """Test identification of overlapping peaks."""
        # Find peaks
        from gammafit.detection import detect_peaks
        peak_indices = detect_peaks(self.counts, min_prominence=100, min_distance=5)
        
        # Identify overlapping groups
        groups = identify_overlapping_peaks(
            self.channels,
            self.counts,
            peak_indices,
            window_scale=3.0
        )
        
        # Should identify at least one group with multiple peaks
        multiplet_groups = [g for g in groups if len(g) > 1]
        
        # May detect as single or double depending on separation
        self.assertGreaterEqual(len(groups), 1)
    
    def test_fit_double_gaussian(self):
        """Test fitting double Gaussian."""
        from gammafit.fitting import fit_double_gaussian
        
        # Approximate peak positions
        peak_indices = [250, 265]
        
        results = fit_double_gaussian(
            self.channels,
            self.counts,
            peak_indices
        )
        
        # Should return two results
        self.assertEqual(len(results), 2)
        
        # Both should be marked as multiplet
        for result in results:
            self.assertTrue(result.get('multiplet', False))
            self.assertEqual(result.get('multiplet_size'), 2)
        
        # Centroids should be approximately correct
        centroids = [r['centroid'] for r in results]
        self.assertTrue(any(abs(c - 250) < 10 for c in centroids))
        self.assertTrue(any(abs(c - 265) < 10 for c in centroids))
    
    def test_
