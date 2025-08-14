import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory {output_dir} created.")
    else:
        print(f"Output directory {output_dir} already exists.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: The video file at {video_path} could not be opened.")
        exit()

    # Initialize frame count
    frame_count = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print(f"Error reading frame {frame_count}. Stopping capture.")
            break

        # Check if frame is None
        if frame is None:
            print(f"Frame {frame_count} is None. Stopping capture.")
            break

        # Save each frame to output directory
        output_path = os.path.join(output_dir, f'{frame_count:05d}.jpg')
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_count} to {output_path}")

        frame_count += 1

    # When everything is done, release the video capture object
    cap.release()
    print(f"All frames ({frame_count} frames) are saved successfully in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them as images.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the extracted frames will be saved")

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir)
