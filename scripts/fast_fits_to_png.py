#!/usr/bin/env python3
"""
Fast FITS to PNG Converter

Optimized script to convert existing FITS files to PNG format
without any download overhead.

Usage:
    python scripts/fast_fits_to_png.py --input-dir /path/to/fits --parallel-jobs 16
"""

import os
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

def convert_fits_to_png_fast(fits_path, output_dir=None, quality=95):
    """
    Ultra-fast FITS to PNG conversion using PIL
    
    Parameters:
    -----------
    fits_path : str
        Path to FITS file
    output_dir : str
        Output directory (default: same as input)
    quality : int
        PNG quality (not used for PNG, kept for compatibility)
    """
    try:
        fits_path = Path(fits_path)
        
        if output_dir is None:
            output_dir = fits_path.parent
        else:
            output_dir = Path(output_dir)
        
        # Create PNG filename
        png_path = output_dir / f"{fits_path.stem}.png"
        
        # Skip if PNG already exists
        if png_path.exists():
            return f"SKIP: {png_path.name}"
        
        # Read FITS data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            
        if data is None:
            return f"ERROR: No data in {fits_path.name}"
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Fast normalization (1st and 99th percentiles)
        vmin, vmax = np.percentile(data, [1, 99])
        if vmax > vmin:
            data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            data_norm = np.zeros_like(data)
        
        # Convert to 8-bit
        data_8bit = (data_norm * 255).astype(np.uint8)
        
        # Convert to PIL Image and save
        img = Image.fromarray(data_8bit, mode='L')
        img.save(png_path, 'PNG')
        
        return f"SUCCESS: {png_path.name}"
        
    except Exception as e:
        return f"ERROR: {fits_path.name} - {str(e)}"

def find_fits_files(directory, pattern="*.fits"):
    """Find all FITS files in directory"""
    directory = Path(directory)
    return list(directory.glob(pattern))

def convert_batch(fits_files, output_dir=None, parallel_jobs=8):
    """Convert multiple FITS files to PNG in parallel"""
    
    print(f"Converting {len(fits_files)} FITS files to PNG...")
    print(f"Using {parallel_jobs} parallel workers")
    
    results = {
        'success': 0,
        'skipped': 0,
        'errors': 0,
        'error_details': []
    }
    
    # Progress bar
    with tqdm(total=len(fits_files), desc="Converting") as pbar:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(convert_fits_to_png_fast, fits_file, output_dir): fits_file
                for fits_file in fits_files
            }
            
            # Process results
            for future in as_completed(future_to_file):
                result = future.result()
                pbar.update(1)
                
                if result.startswith("SUCCESS"):
                    results['success'] += 1
                elif result.startswith("SKIP"):
                    results['skipped'] += 1
                else:  # ERROR
                    results['errors'] += 1
                    results['error_details'].append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Fast FITS to PNG converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all FITS in current directory
    python scripts/fast_fits_to_png.py --input-dir ./images
    
    # Convert with 16 parallel workers
    python scripts/fast_fits_to_png.py --input-dir ./images --parallel-jobs 16
    
    # Convert COSMOS FITS files
    python scripts/fast_fits_to_png.py --input-dir /pscratch/sd/s/sihany/galaxy-sommelier-data/hubble/images --pattern "cosmos_*.fits"
        """
    )
    
    parser.add_argument('--input-dir', required=True,
                       help='Directory containing FITS files')
    parser.add_argument('--output-dir',
                       help='Output directory for PNG files (default: same as input)')
    parser.add_argument('--pattern', default='*.fits',
                       help='File pattern to match (default: *.fits)')
    parser.add_argument('--parallel-jobs', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        parser.error(f"Input directory not found: {input_dir}")
    
    # Find FITS files
    fits_files = find_fits_files(input_dir, args.pattern)
    
    if not fits_files:
        print(f"No FITS files found in {input_dir} matching pattern '{args.pattern}'")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Convert files
    results = convert_batch(
        fits_files, 
        output_dir=args.output_dir,
        parallel_jobs=args.parallel_jobs
    )
    
    # Print results
    print("\n" + "="*50)
    print("CONVERSION RESULTS")
    print("="*50)
    print(f"Total files: {len(fits_files)}")
    print(f"Successfully converted: {results['success']}")
    print(f"Already existed (skipped): {results['skipped']}")
    print(f"Errors: {results['errors']}")
    
    if results['errors'] > 0:
        print(f"\nSuccess rate: {results['success']/(len(fits_files)-results['skipped'])*100:.1f}%")
        print("\nFirst 5 errors:")
        for error in results['error_details'][:5]:
            print(f"  {error}")
    else:
        print(f"\nSuccess rate: 100%")

if __name__ == '__main__':
    main() 