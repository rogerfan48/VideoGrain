import imageio
import os

def images_to_video(image_folder, output_video_name, start_frame=0, end_frame=None, sample_rate=1, fps=10):
    # Get all images and sort them by file name
    filenames = sorted([os.path.join(image_folder, image) for image in os.listdir(image_folder) if image.endswith(".png") or image.endswith(".jpg")])

    # Ensure that images were found
    if not filenames:
        raise ValueError("No images found in the specified directory!")

    # If end_frame is not specified, default to the last image
    if end_frame is None or end_frame > len(filenames):
        end_frame = len(filenames)

    # Select images based on start_frame, end_frame, and sample_rate
    selected_filenames = filenames[start_frame:end_frame:sample_rate]

    # Ensure that some images have been selected
    if not selected_filenames:
        raise ValueError("No images selected based on the provided range and sample rate!")

    # Read the selected images
    images = [imageio.imread(filename) for filename in selected_filenames]

    # Write the video file
    imageio.mimwrite(output_video_name, images, fps=fps)

    print(f"Video created successfully and saved to {output_video_name}")

if __name__ == "__main__":
    source_image_folder = ''  # Replace with the path to your original image folder
    output_video_name = ''  # The desired output video name

    # Specify the start frame, end frame, and sample rate
    start_frame = 0  # Starting frame
    end_frame = 15  # Ending frame, adjust as needed
    sample_rate = 1  # Frame sampling rate
    fps = 10  # Frames per second

    # Create the video
    images_to_video(source_image_folder, output_video_name, start_frame=start_frame, end_frame=end_frame, sample_rate=sample_rate, fps=fps)
