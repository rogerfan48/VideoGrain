#!/usr/bin/env python3
"""
Convert JPG mask files to PNG format for VideoGrain compatibility
"""

import os
from PIL import Image

def convert_masks_to_png():
    # TODO: Assign the directory need to convert
    masks_base_dir = "/work/rogerfan48/VideoGrain/data/07_shakehand/layout_masks"

    mask_dirs = [d for d in os.listdir(masks_base_dir)
                 if os.path.isdir(os.path.join(masks_base_dir, d))]
    
    print(f"Found mask directories: {mask_dirs}")

    for mask_dir in mask_dirs:
        mask_path = os.path.join(masks_base_dir, mask_dir)
            
        print(f"Converting masks in {mask_dir}/...")
        
        # Get all JPG files
        jpg_files = [f for f in os.listdir(mask_path) if f.endswith('.jpg')]
        
        for jpg_file in jpg_files:
            jpg_path = os.path.join(mask_path, jpg_file)
            png_file = jpg_file.replace('.jpg', '.png')
            png_path = os.path.join(mask_path, png_file)
            
            # Open JPG and save as PNG
            with Image.open(jpg_path) as img:
                # Convert to grayscale if not already
                if img.mode != 'L':
                    img = img.convert('L')
                img.save(png_path, 'PNG')
            
            # Delete the original JPG file
            os.remove(jpg_path)
            
            print(f"Converted {jpg_file} -> {png_file}")
    
    print("All mask files converted to PNG format!")

if __name__ == "__main__":
    convert_masks_to_png()
