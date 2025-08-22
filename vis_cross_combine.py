import os
import re
import shutil
from collections import defaultdict
from typing import Tuple

from PIL import Image

FINAL_OUTPUT_DIR_NAME = '_grid_output'
TEMP_DIR_NAME = 'temp_stacked_images'

VALID_SUBDIR_PREFIXES: Tuple[str, ...] = ('down_', 'mid_', 'up_')
GRID_PADDING = 60
GRID_BACKGROUND_COLOR = 'white'


def perform_stacking_step(source_dir: str, temp_output_dir: str) -> bool:
    """
    Performs the first stage of the pipeline: stacking frame images vertically.
    Reads from subdirectories in `source_dir` and writes to `temp_output_dir`.
    """
    print("\n--- Step 1: Stacking Frame Images ---")

    # 1a. Identify target subdirectories
    try:
        all_entries = os.listdir(source_dir)
        target_subdirs = [
            d for d in all_entries if os.path.isdir(os.path.join(source_dir, d)) and d.startswith(VALID_SUBDIR_PREFIXES)
        ]
    except OSError as e:
        print(f"❌ Error: Could not read source directory '{source_dir}'. Details: {e}")
        return False

    if not target_subdirs:
        print(f"⚠️ Warning: No subdirectories with valid prefixes {VALID_SUBDIR_PREFIXES} found in '{source_dir}'.")
        return False

    print(f"Found {len(target_subdirs)} subdirectories to process.")

    # 1b. Iterate and process each subdirectory
    for subdir_name in target_subdirs:
        subdir_path = os.path.join(source_dir, subdir_name)
        try:
            image_filenames = [f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')]

            # Sort images by frame number
            def extract_frame_number(filename: str) -> int:
                match = re.search(r'frame_(\d+)', filename, re.IGNORECASE)
                return int(match.group(1)) if match else 0

            sorted_filenames = sorted(image_filenames, key=extract_frame_number)

            image_objects = [Image.open(os.path.join(subdir_path, f)) for f in sorted_filenames]

            # Calculate dimensions and create stacked image
            widths, heights = zip(*(img.size for img in image_objects))
            stacked_canvas = Image.new('RGB', (max(widths), sum(heights)))
            y_offset = 0
            for img in image_objects:
                stacked_canvas.paste(img, (0, y_offset))
                y_offset += img.height
                img.close()

            # Save to temporary directory
            output_filepath = os.path.join(temp_output_dir, f"{subdir_name}.jpg")
            stacked_canvas.save(output_filepath)

        except Exception as e:
            print(f"  - ❌ Error processing '{subdir_name}': {e}")

    print("✅ Step 1 completed successfully.")
    return True


def perform_gridding_step(temp_source_dir: str, final_output_dir: str) -> bool:
    """
    Performs the second stage: composing stacked images into 2x4 grids.
    Reads from `temp_source_dir` and writes to `final_output_dir`.
    """
    print("\n--- Step 2: Generating Final Grids ---")

    # 2a. Scan and group intermediate files
    image_groups = defaultdict(list)
    try:
        filenames = [f for f in os.listdir(temp_source_dir) if f.lower().endswith('.jpg')]
        for filename in filenames:
            if '_head_' in filename:
                base_name = filename.split('_head_')[0]
                image_groups[base_name].append(filename)
    except OSError as e:
        print(f"❌ Error: Could not read temporary directory '{temp_source_dir}'. Details: {e}")
        return False

    if not image_groups:
        print(f"⚠️ Warning: No intermediate images found in '{temp_source_dir}' to generate grids from.")
        return False

    print(f"Found {len(image_groups)} image groups to assemble into grids.")

    # 2b. Process each image group
    for base_name, filenames in image_groups.items():
        if len(filenames) != 8:
            print(f"  - Skipping '{base_name}': Expected 8 images for grid, found {len(filenames)}.")
            continue

        try:
            # Sort images by head index
            def get_head_index(f: str) -> int:
                match = re.search(r'_head_(\d+)\.jpg', f, re.IGNORECASE)
                return int(match.group(1)) if match else -1

            sorted_filenames = sorted(filenames, key=get_head_index)

            # Calculate grid dimensions from a sample image
            sample_image_path = os.path.join(temp_source_dir, sorted_filenames[0])
            with Image.open(sample_image_path) as sample_image:
                img_width, img_height = sample_image.size

            grid_width = (img_width * 4) + (GRID_PADDING * 3)
            grid_height = (img_height * 2) + GRID_PADDING

            grid_canvas = Image.new('RGB', (grid_width, grid_height), GRID_BACKGROUND_COLOR)

            # Composite images onto grid
            for index, filename in enumerate(sorted_filenames):
                row, col = divmod(index, 4)
                x_pos = col * (img_width + GRID_PADDING)
                y_pos = row * (img_height + GRID_PADDING)
                with Image.open(os.path.join(temp_source_dir, filename)) as img:
                    grid_canvas.paste(img, (x_pos, y_pos))

            # Save final grid image
            output_filepath = os.path.join(final_output_dir, f"{base_name}.jpg")
            grid_canvas.save(output_filepath)

        except Exception as e:
            print(f"  - ❌ Error processing grid for '{base_name}': {e}")

    print("✅ Step 2 completed successfully.")
    return True


def main():
    """
    Main function to orchestrate the entire image processing pipeline.
    """

    # 1. Get source directory from user input
    source_dir = input("Enter the source directory path: ").strip()
    if not os.path.isdir(source_dir):
        print("🛑 Error: The provided path is not a valid directory.")
        return
    print()

    # 2. Define and create pipeline directories
    temp_dir_path = os.path.join(source_dir, TEMP_DIR_NAME)
    final_dir_path = os.path.join(source_dir, FINAL_OUTPUT_DIR_NAME)

    os.makedirs(temp_dir_path, exist_ok=True)
    os.makedirs(final_dir_path, exist_ok=True)
    print(f"Temporary directory: {temp_dir_path}")
    print(f"Final output directory: {final_dir_path}")

    try:
        # 3. Execute Step 1: Stacking
        success_step1 = perform_stacking_step(source_dir, temp_dir_path)

        if not success_step1:
            print("\nPipeline halted due to issues in Step 1.")
            return

        # 4. Execute Step 2: Gridding
        success_step2 = perform_gridding_step(temp_dir_path, final_dir_path)

        if not success_step2:
            print("\nPipeline halted or completed with errors in Step 2.")

    finally:
        # 5. Cleanup: Always remove the temporary directory
        if os.path.exists(temp_dir_path):
            print(f"\nCleaning up temporary directory: '{temp_dir_path}'...")
            try:
                shutil.rmtree(temp_dir_path)
                print("Cleanup successful.")
            except OSError as e:
                print(f"❌ Error during cleanup: Could not remove temporary directory. Details: {e}")

    print("\n🎉 Pipeline finished.")


if __name__ == '__main__':
    main()
