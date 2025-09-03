"""
Unit tests for I/O operations module.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammafit.io_module import (
    load_spectrum,
    load_csv_spectrum,
    load_spe_spectrum,
    load_chn_spectrum,
    load_mca_spectrum,
    save_peaks,
    load_config,
    save_config,
    export_spectrum
)


class TestCSVIO(unittest.TestCase):
    """Test CSV file I/O operations."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv_two_columns(self):
        """Test loading CSV with two columns."""
        # Create test CSV
        test_file = self.temp_path / "test_spectrum.csv"
        data = pd.DataFrame({
            'channel': np.arange(100),
            'counts': np.random.poisson(50, 100)
        })
        data.to_csv(test_file, index=False, header=False)
        
        # Load spectrum
        channels, counts = load_csv_spectrum(test_file)
        
        # Verify
        self.assertEqual(len(channels), 100)
        self.assertEqual(len(counts), 100)
        np.testing.assert_array_equal(channels, np.arange(100))
    
    def test_load_csv_single_column(self):
        """Test loading CSV with single column (counts only)."""
        test_file = self.temp_path / "counts_only.csv"
        counts_data = np.random.poisson(100, 50)
        pd.DataFrame(counts_data).to_csv(test_file, index=False, header=False)
        
        # Load spectrum
        channels, counts = load_csv_spectrum(test_file)
        
        # Should create channel numbers
        self.assertEqual(len(channels), 50)
        self.assertEqual(len(counts), 50)
        np.testing.assert_array_equal(channels, np.arange(50))
        np.testing.assert_array_equal(counts, counts_data)
    
    def test_load_csv_with_header(self):
        """Test loading CSV with header row."""
        test_file = self.temp_path / "with_header.csv"
        data = pd.DataFrame({
            'Channel': np.arange(10),
            'Counts': np.arange(10, 20)
        })
        data.to_csv(test_file, index=False)
        
        # Should handle header automatically
        channels, counts = load_csv_spectrum(test_file)
        
        self.assertEqual(len(channels), 10)
        # First data row should be read, not header
        self.assertEqual(channels[0], 0)
        self.assertEqual(counts[0], 10)
    
    def test_load_csv_with_comments(self):
        """Test loading CSV with comment lines."""
        test_file = self.temp_path / "with_comments.csv"
        with open(test_file, 'w') as f:
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
            f.write("0,100\n")
            f.write("1,101\n")
            f.write("2,102\n")
        
        channels, counts = load_csv_spectrum(test_file)
        
        self.assertEqual(len(channels), 3)
        self.assertEqual(counts[0], 100)
    
    def test_negative_counts_handling(self):
        """Test handling of negative counts."""
        test_file = self.temp_path / "negative.csv"
        data = pd.DataFrame({
            'channel': [0, 1, 2],
            'counts': [100, -10, 50]  # Negative count
        })
        data.to_csv(test_file, index=False, header=False)
        
        channels, counts = load_csv_spectrum(test_file)
        
        # Negative counts should be set to zero
        self.assertEqual(counts[1], 0)
        self.assertEqual(counts[0], 100)
        self.assertEqual(counts[2], 50)


class TestSPEFormat(unittest.TestCase):
    """Test SPE format I/O."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_spe_file(self):
        """Test loading SPE format file."""
        test_file = self.temp_path / "test.spe"
        
        # Create minimal SPE file
        with open(test_file, 'w') as f:
            f.write("$SPEC_ID:\n")
            f.write("Test spectrum\n")
            f.write("$SPEC_REM:\n")
            f.write("Test remarks\n")
            f.write("$DATA:\n")
            f.write("0 99\n")  # 100 channels
            for i in range(100):
                f.write(f"{100 + i}\n")
            f.write("$ROI:\n")
            f.write("0\n")
            f.write("$PRESETS:\n")
            f.write("None\n")
            f.write("$ENER_FIT:\n")
            f.write("0.0 1.0\n")
            f.write("$MCA_CAL:\n")
            f.write("2\n")
            f.write("0.0 1.0\n")
        
        # Load spectrum
        channels, counts = load_spe_spectrum(test_file)
        
        # Verify
        self.assertEqual(len(channels), 100)
        self.assertEqual(len(counts), 100)
        self.assertEqual(counts[0], 100)
        self.assertEqual(counts[99], 199)
    
    def test_export_spe_format(self):
        """Test exporting to SPE format."""
        test_file = self.temp_path / "export.spe"
        
        channels = np.arange(50)
        counts = np.random.poisson(100, 50)
        
        export_spectrum(channels, counts, str(test_file), format='spe')
        
        # Verify file exists and has SPE structure
        self.assertTrue(test_file.exists())
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        self.assertIn("$SPEC_ID:", content)
        self.assertIn("$DATA:", content)
        self.assertIn("0 49", content)  # Channel range


class TestCHNFormat(unittest.TestCase):
    """Test CHN format I/O."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_chn_file(self):
        """Test loading CHN binary format."""
        import struct
        
        test_file = self.temp_path / "test.chn"
        
        # Create minimal CHN file (binary)
        num_channels = 256
        with open(test_file, 'wb') as f:
            # Write header (32 bytes)
            header = bytearray(32)
            # Bytes 4-5: number of channels (little-endian)
            header[4:6] = struct.pack('<H', num_channels)
            f.write(header)
            
            # Write channel data (4 bytes per channel)
            for i in range(num_channels):
                count = 100 + i
                f.write(struct.pack('<I', count))
        
        # Load spectrum
        channels, counts = load_chn_spectrum(test_file)
        
        # Verify
        self.assertEqual(len(channels), num_channels)
        self.assertEqual(len(counts), num_channels)
        self.assertEqual(counts[0], 100)
        self.assertEqual(counts[255], 355)


class TestMCAFormat(unittest.TestCase):
    """Test MCA format I/O."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_mca_file(self):
        """Test loading MCA format file."""
        test_file = self.temp_path / "test.mca"
        
        # Create MCA file
        with open(test_file, 'w') as f:
            f.write("<<PMCA SPECTRUM>>\n")
            f.write("TAG - Test\n")
            f.write("DESCRIPTION - Test spectrum\n")
            f.write("GAIN - 2\n")
            f.write("THRESHOLD - 0\n")
            f.write("LIVE_MODE - 0\n")
            f.write("PRESET_TIME - 0\n")
            f.write("LIVE_TIME - 100\n")
            f.write("REAL_TIME - 100\n")
            f.write("START_TIME - 01/01/2024 00:00:00\n")
            f.write("SERIAL_NUMBER - 12345\n")
            f.write("<<DATA>>\n")
            for i in range(100):
                f.write(f"{50 + i}\n")
            f.write("<<END>>\n")
        
        # Load spectrum
        channels, counts = load_mca_spectrum(test_file)
        
        # Verify
        self.assertEqual(len(channels), 100)
        self.assertEqual(len(counts), 100)
        self.assertEqual(counts[0], 50)
        self.assertEqual(counts[99], 149)


class TestPeakIO(unittest.TestCase):
    """Test peak data I/O."""
    
    def setUp(self):
        """Create temporary directory and test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test peak data
        self.test_peaks = [
            {
                'centroid': 511.0,
                'area': 10000,
                'fwhm': 5.2,
                'snr': 45.3,
                'energy': 511.0
            },
            {
                'centroid': 1274.5,
                'area': 5000,
                'fwhm': 7.8,
                'snr': 25.1,
                'energy': 1274.5
            }
        ]
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_peaks_csv(self):
        """Test saving peaks to CSV."""
        test_file = self.temp_path / "peaks.csv"
        
        save_peaks(self.test_peaks, str(test_file), format='csv')
        
        # Verify file exists
        self.assertTrue(test_file.exists())
        
        # Load and check
        df = pd.read_csv(test_file)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.iloc[0]['centroid'], 511.0)
        self.assertAlmostEqual(df.iloc[1]['area'], 5000)
    
    def test_save_peaks_json(self):
        """Test saving peaks to JSON."""
        test_file = self.temp_path / "peaks.json"
        
        save_peaks(self.test_peaks, str(test_file), format='json')
        
        # Load and verify
        with open(test_file, 'r') as f:
            loaded_peaks = json.load(f)
        
        self.assertEqual(len(loaded_peaks), 2)
        self.assertEqual(loaded_peaks[0]['centroid'], 511.0)
        self.assertEqual(loaded_peaks[1]['fwhm'], 7.8)
    
    def test_save_peaks_text(self):
        """Test saving peaks to text format."""
        test_file = self.temp_path / "peaks.txt"
        
        save_peaks(self.test_peaks, str(test_file), format='txt')
        
        # Verify file exists and has content
        self.assertTrue(test_file.exists())
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Check for expected content
        self.assertIn("Peak Analysis Results", content)
        self.assertIn("511", content)
        self.assertIn("1274", content)


class TestConfigIO(unittest.TestCase):
    """Test configuration file I/O."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.test_config = {
            'detection': {
                'min_prominence': 100,
                'min_height': 50,
                'smoothing_window': 7
            },
            'fitting': {
                'peak_model': 'gaussian',
                'background_method': 'linear'
            },
            'calibration': {
                'a': 0.5,
                'b': 0.0
            }
        }
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        test_file = self.temp_path / "config.json"
        
        # Save config
        save_config(self.test_config, str(test_file))
        
        # Load config
        loaded_config = load_config(str(test_file))
        
        # Verify
        self.assertEqual(loaded_config['detection']['min_prominence'], 100)
        self.assertEqual(loaded_config['fitting']['peak_model'], 'gaussian')
        self.assertAlmostEqual(loaded_config['calibration']['a'], 0.5)
    
    def test_load_invalid_config(self):
        """Test loading invalid configuration file."""
        test_file = self.temp_path / "invalid.json"
        
        # Create invalid JSON
        with open(test_file, 'w') as f:
            f.write("{ invalid json }")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            load_config(str(test_file))
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent file."""
        test_file = self.temp_path / "nonexistent.json"
        
        with self.assertRaises(FileNotFoundError):
            load_config(str(test_file))


class TestAutoFormatDetection(unittest.TestCase):
    """Test automatic format detection."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_auto_detect_csv(self):
        """Test automatic detection of CSV format."""
        test_file = self.temp_path / "spectrum.csv"
        
        # Create CSV file
        data = pd.DataFrame({
            'ch': np.arange(10),
            'cnt': np.arange(10, 20)
        })
        data.to_csv(test_file, index=False, header=False)
        
        # Load with auto-detection
        channels, counts = load_spectrum(str(test_file))
        
        self.assertEqual(len(channels), 10)
        self.assertEqual(counts[0], 10)
    
    def test_auto_detect_by_content(self):
        """Test format detection when extension doesn't match."""
        test_file = self.temp_path / "spectrum.dat"  # .dat extension
        
        # But content is CSV
        data = pd.DataFrame({
            'ch': np.arange(5),
            'cnt': [100, 200, 300, 400, 500]
        })
        data.to_csv(test_file, index=False, header=False)
        
        # Should still load correctly
        channels, counts = load_spectrum(str(test_file))
        
        self.assertEqual(len(channels), 5)
        self.assertEqual(counts[2], 300)


if __name__ == '__main__':
    unittest.main()
