#!/usr/bin/env python3
"""
Convert between different gamma spectrum file formats.

Supported formats:
- CSV (Comma-Separated Values)
- SPE (IAEA Standard)
- CHN (Ortec binary)
- MCA (Multichannel Analyzer)
- JSON (JavaScript Object Notation)
- TXT (Tab-delimited text)
"""

import argparse
import sys
import struct
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

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
    export_spectrum
)


class SpectrumConverter:
    """Convert between spectrum file formats."""
    
    SUPPORTED_FORMATS = {
        'csv': 'Comma-separated values',
        'txt': 'Tab-delimited text',
        'spe': 'IAEA SPE format',
        'chn': 'Ortec CHN binary format',
        'mca': 'MCA text format',
        'json': 'JSON format'
    }
    
    def __init__(self):
        """Initialize converter."""
        self.metadata = {}
        self.channels = None
        self.counts = None
    
    def load(self, file_path: str, format: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load spectrum from file.
        
        Parameters:
            file_path: Input file path
            format: Force specific format (auto-detect if None)
            
        Returns:
            Tuple of (channels, counts)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(file_path)
        
        print(f"Loading {format.upper()} file: {file_path}")
        
        # Load based on format
        if format == 'csv':
            self.channels, self.counts = load_csv_spectrum(file_path)
        elif format == 'spe':
            self.channels, self.counts = load_spe_spectrum(file_path)
            self._extract_spe_metadata(file_path)
        elif format == 'chn':
            self.channels, self.counts = load_chn_spectrum(file_path)
        elif format == 'mca':
            self.channels, self.counts = load_mca_spectrum(file_path)
        elif format == 'json':
            self.channels, self.counts = self._load_json_spectrum(file_path)
        elif format == 'txt':
            self.channels, self.counts = self._load_txt_spectrum(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Loaded {len(self.channels)} channels, {np.sum(self.counts):.0f} total counts")
        
        return self.channels, self.counts
    
    def save(self, file_path: str, format: str, **kwargs):
        """
        Save spectrum to file.
        
        Parameters:
            file_path: Output file path
            format: Output format
            **kwargs: Additional format-specific options
        """
        if self.channels is None or self.counts is None:
            raise ValueError("No spectrum loaded")
        
        file_path = Path(file_path)
        print(f"Saving to {format.upper()} format: {file_path}")
        
        # Save based on format
        if format == 'csv':
            self._save_csv(file_path, **kwargs)
        elif format == 'spe':
            self._save_spe(file_path, **kwargs)
        elif format == 'chn':
            self._save_chn(file_path, **kwargs)
        elif format == 'mca':
            self._save_mca(file_path, **kwargs)
        elif format == 'json':
            self._save_json(file_path, **kwargs)
        elif format == 'txt':
            self._save_txt(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved successfully")
    
    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect file format."""
        ext = file_path.suffix.lower()
        
        # Check by extension
        if ext == '.csv':
            return 'csv'
        elif ext == '.spe':
            return 'spe'
        elif ext == '.chn':
            return 'chn'
        elif ext == '.mca':
            return 'mca'
        elif ext == '.json':
            return 'json'
        elif ext in ['.txt', '.dat']:
            # Could be CSV or tab-delimited
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    return 'txt'
                else:
                    return 'csv'
        else:
            # Try to detect by content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if first_line.startswith('$'):
                        return 'spe'
                    elif first_line.startswith('<<'):
                        return 'mca'
                    elif first_line.startswith('{'):
                        return 'json'
                    else:
                        return 'csv'  # Default
            except:
                # Binary file - assume CHN
                return 'chn'
    
    def _load_json_spectrum(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load JSON format spectrum."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'channels' in data and 'counts' in data:
            channels = np.array(data['channels'])
            counts = np.array(data['counts'])
        elif 'spectrum' in data:
            # Alternative format
            spectrum = data['spectrum']
            if isinstance(spectrum, dict):
                channels = np.array(list(spectrum.keys()), dtype=float)
                counts = np.array(list(spectrum.values()), dtype=float)
            else:
                counts = np.array(spectrum)
                channels = np.arange(len(counts))
        else:
            # Assume it's just an array of counts
            counts = np.array(data)
            channels = np.arange(len(counts))
        
        # Store metadata
        self.metadata = {k: v for k, v in data.items() 
                        if k not in ['channels', 'counts', 'spectrum']}
        
        return channels, counts
    
    def _load_txt_spectrum(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load tab-delimited text spectrum."""
        # Try to read with pandas
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, comment='#')
            
            if df.shape[1] == 1:
                # Single column - counts only
                counts = df.iloc[:, 0].values
                channels = np.arange(len(counts))
            else:
                # Two columns - channels and counts
                channels = df.iloc[:, 0].values
                counts = df.iloc[:, 1].values
            
            return channels.astype(float), counts.astype(float)
        
        except Exception as e:
            raise ValueError(f"Error reading text file: {e}")
    
    def _save_csv(self, file_path: Path, **kwargs):
        """Save as CSV."""
        df = pd.DataFrame({
            'channel': self.channels,
            'counts': self.counts
        })
        
        header = kwargs.get('header', False)
        df.to_csv(file_path, index=False, header=header)
    
    def _save_spe(self, file_path: Path, **kwargs):
        """Save as SPE format."""
        with open(file_path, 'w') as f:
            # Write SPE header
            f.write("$SPEC_ID:\n")
            f.write(kwargs.get('spec_id', 'Converted spectrum') + "\n")
            
            f.write("$SPEC_REM:\n")
            f.write(kwargs.get('spec_rem', f'Converted by convert_formats.py on {datetime.now()}') + "\n")
            
            f.write("$DATE_MEA:\n")
            f.write(kwargs.get('date_mea', datetime.now().strftime('%m/%d/%Y %H:%M:%S')) + "\n")
            
            f.write("$MEAS_TIM:\n")
            live_time = kwargs.get('live_time', 1000)
            real_time = kwargs.get('real_time', 1000)
            f.write(f"{live_time} {real_time}\n")
            
            f.write("$DATA:\n")
            f.write(f"0 {len(self.counts)-1}\n")
            
            for count in self.counts:
                f.write(f"{int(count)}\n")
            
            f.write("$ROI:\n")
            f.write("0\n")
            
            f.write("$PRESETS:\n")
            f.write("None\n")
            
            # Energy calibration
            f.write("$ENER_FIT:\n")
            a = kwargs.get('energy_cal_a', 1.0)
            b = kwargs.get('energy_cal_b', 0.0)
            f.write(f"{b} {a}\n")
            
            f.write("$MCA_CAL:\n")
            f.write("2\n")
            f.write(f"{a} {b}\n")
    
    def _save_chn(self, file_path: Path, **kwargs):
        """Save as CHN binary format."""
        with open(file_path, 'wb') as f:
            # CHN header (32 bytes)
            header = bytearray(32)
            
            # File type identifier (usually -1)
            header[0:2] = struct.pack('<h', -1)
            
            # MCA number (usually 1)
            header[2:4] = struct.pack('<h', 1)
            
            # Segment number (usually 1)
            header[4:6] = struct.pack('<h', 1)
            
            # ASCII seconds
            seconds = kwargs.get('seconds', '00000000')
            header[6:14] = seconds[:8].encode('ascii')
            
            # Real time (in 20ms units)
            real_time = kwargs.get('real_time', 50000)
            header[14:18] = struct.pack('<I', real_time)
            
            # Live time (in 20ms units)
            live_time = kwargs.get('live_time', 50000)
            header[18:22] = struct.pack('<I', live_time)
            
            # Start date
            header[22:30] = kwargs.get('date', '01JAN00').encode('ascii')[:8]
            
            # Start time
            header[30:32] = struct.pack('<H', kwargs.get('time', 0))
            
            f.write(header)
            
            # Channel data (4 bytes per channel)
            for count in self.counts:
                f.write(struct.pack('<I', int(count)))
    
    def _save_mca(self, file_path: Path, **kwargs):
        """Save as MCA format."""
        with open(file_path, 'w') as f:
            f.write("<<PMCA SPECTRUM>>\n")
            f.write(f"TAG - {kwargs.get('tag', 'Converted')}\n")
            f.write(f"DESCRIPTION - {kwargs.get('description', 'Converted spectrum')}\n")
            f.write(f"GAIN - {kwargs.get('gain', 2)}\n")
            f.write(f"THRESHOLD - {kwargs.get('threshold', 0)}\n")
            f.write(f"LIVE_MODE - {kwargs.get('live_mode', 0)}\n")
            f.write(f"PRESET_TIME - {kwargs.get('preset_time', 0)}\n")
            f.write(f"LIVE_TIME - {kwargs.get('live_time', 1000)}\n")
            f.write(f"REAL_TIME - {kwargs.get('real_time', 1000)}\n")
            f.write(f"START_TIME - {kwargs.get('start_time', datetime.now().strftime('%m/%d/%Y %H:%M:%S'))}\n")
            f.write(f"SERIAL_NUMBER - {kwargs.get('serial', '00000')}\n")
            
            f.write("<<DATA>>\n")
            for count in self.counts:
                f.write(f"{int(count)}\n")
            f.write("<<END>>\n")
            
            # Optional calibration
            if kwargs.get('calibration'):
                f.write("<<CALIBRATION>>\n")
                f.write("LABEL - keV\n")
                cal = kwargs['calibration']
                f.write(f"{cal.get('a', 1.0)} {cal.get('b', 0.0)}\n")
    
    def _save_json(self, file_path: Path, **kwargs):
        """Save as JSON format."""
        data = {
            'format_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'channels': self.channels.tolist(),
            'counts': self.counts.tolist(),
            'metadata': self.metadata
        }
        
        # Add additional metadata
        if kwargs.get('calibration'):
            data['calibration'] = kwargs['calibration']
        
        if kwargs.get('description'):
            data['description'] = kwargs['description']
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_txt(self, file_path: Path, **kwargs):
        """Save as tab-delimited text."""
        with open(file_path, 'w') as f:
            # Write header if requested
            if kwargs.get('header'):
                f.write("# Spectrum data\n")
                f.write(f"# Generated: {datetime.now()}\n")
                f.write(f"# Channels: {len(self.channels)}\n")
                f.write(f"# Total counts: {np.sum(self.counts):.0f}\n")
                f.write("#\n")
                f.write("# Channel\tCounts\n")
            
            # Write data
            for ch, cnt in zip(self.channels, self.counts):
                f.write(f"{ch:.1f}\t{cnt:.1f}\n")
    
    def _extract_spe_metadata(self, file_path: Path):
        """Extract metadata from SPE file."""
        metadata = {}
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line == '$SPEC_ID:' and i + 1 < len(lines):
                metadata['spec_id'] = lines[i + 1].strip()
            elif line == '$DATE_MEA:' and i + 1 < len(lines):
                metadata['date'] = lines[i + 1].strip()
            elif line == '$MEAS_TIM:' and i + 1 < len(lines):
                times = lines[i + 1].strip().split()
                if len(times) == 2:
                    metadata['live_time'] = times[0]
                    metadata['real_time'] = times[1]
        
        self.metadata = metadata
    
    def print_info(self):
        """Print spectrum information."""
        if self.channels is None or self.counts is None:
            print("No spectrum loaded")
            return
        
        print("\n" + "=" * 50)
        print("SPECTRUM INFORMATION")
        print("=" * 50)
        print(f"Channels: {len(self.channels)}")
        print(f"Channel range: {self.channels[0]:.1f} - {self.channels[-1]:.1f}")
        print(f"Total counts: {np.sum(self.counts):.0f}")
        print(f"Maximum counts: {np.max(self.counts):.0f}")
        print(f"Mean counts: {np.mean(self.counts):.2f}")
        print(f"Non-zero channels: {np.count_nonzero(self.counts)}")
        
        if self.metadata:
            print("\nMetadata:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
        
        print("=" * 50)


def batch_convert(input_dir: str, 
                  output_dir: str,
                  input_format: str,
                  output_format: str,
                  pattern: str = "*") -> int:
    """
    Batch convert multiple files.
    
    Parameters:
        input_dir: Input directory
        output_dir: Output directory
        input_format: Input format
        output_format: Output format
        pattern: File pattern to match
        
    Returns:
        Number of files converted
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find matching files
    if input_format:
        files = list(input_path.glob(f"{pattern}.{input_format}"))
    else:
        files = list(input_path.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return 0
    
    print(f"Found {len(files)} files to convert")
    
    converter = SpectrumConverter()
    converted = 0
    
    for file in files:
        try:
            print(f"\nProcessing: {file.name}")
            
            # Load
            converter.load(str(file), format=input_format)
            
            # Generate output filename
            output_name = file.stem + f".{output_format}"
            output_file = output_path / output_name
            
            # Save
            converter.save(str(output_file), format=output_format)
            
            converted += 1
            
        except Exception as e:
            print(f"Error converting {file.name}: {e}")
    
    print(f"\nConverted {converted}/{len(files)} files")
    return converted


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Convert between gamma spectrum file formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported formats:
  csv  - Comma-separated values
  txt  - Tab-delimited text
  spe  - IAEA SPE format
  chn  - Ortec CHN binary format
  mca  - MCA text format
  json - JSON format

Examples:
  # Convert single file
  python convert_formats.py input.csv output.spe
  
  # Convert with explicit formats
  python convert_formats.py input.dat output.json -i csv -o json
  
  # Batch convert directory
  python convert_formats.py --batch input_dir/ output_dir/ -i csv -o spe
  
  # Add calibration to SPE file
  python convert_formats.py spectrum.csv calibrated.spe --cal 0.5,10
  
  # Print file information
  python convert_formats.py spectrum.spe --info
        """
    )
    
    # Positional arguments
    parser.add_argument('input', nargs='?', help='Input file or directory')
    parser.add_argument('output', nargs='?', help='Output file or directory')
    
    # Format options
    parser.add_argument('-i', '--input-format', 
                       choices=SpectrumConverter.SUPPORTED_FORMATS.keys(),
                       help='Input format (auto-detect if not specified)')
    parser.add_argument('-o', '--output-format',
                       choices=SpectrumConverter.SUPPORTED_FORMATS.keys(),
                       help='Output format (required for conversion)')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Batch convert directory')
    parser.add_argument('-p', '--pattern', default='*',
                       help='File pattern for batch conversion')
    
    # Calibration
    parser.add_argument('--cal', '--calibration', type=str,
                       help='Energy calibration as "a,b" for E=a*ch+b')
    
    # Metadata
    parser.add_argument('--description', type=str,
                       help='Add description to output file')
    parser.add_argument('--live-time', type=float, default=1000,
                       help='Live time in seconds')
    parser.add_argument('--real-time', type=float, default=1000,
                       help='Real time in seconds')
    
    # Other options
    parser.add_argument('--info', action='store_true',
                       help='Print file information only')
    parser.add_argument('--header', action='store_true',
                       help='Include header in output file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check arguments
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    # Information mode
    if args.info:
        converter = SpectrumConverter()
        converter.load(args.input, format=args.input_format)
        converter.print_info()
        sys.exit(0)
    
    # Batch mode
    if args.batch:
        if not args.output:
            print("Error: Output directory required for batch conversion")
            sys.exit(1)
        if not args.output_format:
            print("Error: Output format required for batch conversion")
            sys.exit(1)
        
        converted = batch_convert(
            args.input,
            args.output,
            args.input_format,
            args.output_format,
            args.pattern
        )
        
        sys.exit(0 if converted > 0 else 1)
    
    # Single file conversion
    if not args.output:
        print("Error: Output file required for conversion")
        sys.exit(1)
    
    # Determine output format
    output_format = args.output_format
    if not output_format:
        # Try to detect from output filename
        output_ext = Path(args.output).suffix.lower()[1:]
        if output_ext in SpectrumConverter.SUPPORTED_FORMATS:
            output_format = output_ext
        else:
            print("Error: Cannot determine output format. Please specify with -o")
            sys.exit(1)
    
    # Convert file
    converter = SpectrumConverter()
    
    try:
        # Load
        converter.load(args.input, format=args.input_format)
        
        if args.verbose:
            converter.print_info()
        
        # Prepare kwargs for saving
        save_kwargs = {
            'header': args.header
        }
        
        if args.calibration:
            try:
                a, b = map(float, args.calibration.split(','))
                save_kwargs['calibration'] = {'a': a, 'b': b}
                save_kwargs['energy_cal_a'] = a
                save_kwargs['energy_cal_b'] = b
            except:
                print("Warning: Invalid calibration format")
        
        if args.description:
            save_kwargs['description'] = args.description
            save_kwargs['spec_rem'] = args.description
        
        save_kwargs['live_time'] = args.live_time
        save_kwargs['real_time'] = args.real_time
        
        # Save
        converter.save(args.output, format=output_format, **save_kwargs)
        
        print(f"\nConversion complete: {args.input} -> {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
