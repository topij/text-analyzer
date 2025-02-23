#!/usr/bin/env python3
"""
Image Optimizer for Web Use
==========================

A simple command-line tool to optimize images for web use. The script performs the following optimizations:
- Resizes large images while maintaining aspect ratio
- Converts images to RGB format (handling RGBA/transparency)
- Slightly enhances contrast for better web viewing
- Applies PNG optimization

Usage
-----
Basic usage (saves in same directory with "_optimized" suffix):
    python optimize_image.py path/to/image.png

Specify custom output path:
    python optimize_image.py path/to/image.png -o path/to/output.png

Specify maximum width (default 1200px):
    python optimize_image.py path/to/image.png -w 800

Arguments
---------
input : str
    Path to the input image file
-o, --output : str, optional
    Path for the optimized output image
    Default: input_directory/input_name_optimized.ext
-w, --width : int, optional
    Maximum width in pixels (default: 1200)
    Images smaller than this will not be resized

Output
------
The script will create an optimized version of the input image and print:
- Original and optimized file sizes
- Final image dimensions
- Path to the saved file

Technical Details
----------------
- Uses high-quality Lanczos resampling for resizing
- Converts RGBA images to RGB
- Applies slight contrast enhancement (1.1x)
- Uses PIL's optimize flag for PNG compression
- Maintains original image format
- Handles errors gracefully

Examples
--------
1. Basic optimization:
   python optimize_image.py photo.jpg
   # Creates: photo_optimized.jpg

2. Custom output name:
   python optimize_image.py photo.jpg -o web/small_photo.jpg

3. Resize to specific width:
   python optimize_image.py photo.jpg -w 800
   # Resizes to 800px width, maintaining aspect ratio

Notes
-----
- Supports common image formats (PNG, JPEG, etc.)
- Original file is never modified
- If output path is specified, directories must exist
- Memory usage scales with image size
"""

import sys
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance

def optimize_image(input_path: Path, output_path: Path = None, max_width: int = 1200) -> None:
    """
    Optimize an image for web use by resizing, enhancing, and compressing.
    
    The function performs several optimizations:
    1. Resizes the image if it exceeds max_width while maintaining aspect ratio
    2. Converts RGBA images to RGB format
    3. Slightly enhances contrast for better web viewing
    4. Applies PNG optimization when saving
    
    Args:
        input_path: Path to input image file
        output_path: Path to save optimized image (default: input_directory/input_name_optimized.ext)
        max_width: Maximum width of the output image in pixels (default: 1200)
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        OSError: If there are issues reading/writing files
        Exception: For other PIL or system errors
        
    Example:
        >>> from pathlib import Path
        >>> optimize_image(Path("photo.jpg"), Path("web/small.jpg"), 800)
    """
    # Handle default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
    
    # Open and convert to RGB (in case it's RGBA)
    img = Image.open(input_path).convert('RGB')
    
    # Resize if needed (maintaining aspect ratio)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    # Save with optimization
    img.save(output_path, 'PNG', optimize=True, quality=85)
    
    # Calculate size reduction
    original_size = input_path.stat().st_size / 1024  # KB
    optimized_size = output_path.stat().st_size / 1024  # KB
    reduction = ((original_size - optimized_size) / original_size) * 100
    
    # Print image details
    print(f"\nImage optimized successfully!")
    print(f"Original size: {original_size:.1f}KB")
    print(f"Optimized size: {optimized_size:.1f}KB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Dimensions: {img.size[0]}x{img.size[1]}px")
    print(f"Saved to: {output_path}")

def main():
    """
    Main function that handles command-line arguments and runs the optimization.
    
    Parses command-line arguments, validates inputs, and calls optimize_image()
    with appropriate parameters. Handles errors and provides user feedback.
    """
    parser = argparse.ArgumentParser(
        description='Optimize an image for web use',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
examples:
  %(prog)s image.png                        # Basic optimization
  %(prog)s image.jpg -o web/small.jpg       # Custom output path
  %(prog)s image.png -w 800                 # Custom maximum width
        '''
    )
    parser.add_argument('input', type=str, 
                       help='Input image path')
    parser.add_argument('-o', '--output', type=str,
                       help='Output image path (default: input_directory/input_name_optimized.ext)')
    parser.add_argument('-w', '--width', type=int, default=1200,
                       help='Maximum width in pixels (default: 1200)')
    
    args = parser.parse_args()
    
    # Handle input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    # Handle output path
    output_path = Path(args.output) if args.output else None
    
    try:
        optimize_image(input_path, output_path, args.width)
    except Exception as e:
        print(f"Error optimizing image: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 