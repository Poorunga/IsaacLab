import argparse
import os
from moviepy.editor import VideoFileClip, clips_array

def combine_videos(input_dir, output_dir, nums):
    """
    Combine four video clips into a single 2x2 grid for each demo.

    Parameters:
        input_dir (str): Path to the directory containing raw video files.
        output_dir (str): Path to save the combined output videos.
        nums (int): Number of demo sets to process, from 0 to nums-1.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(nums):
        # Construct the file paths for the current demo
        persr_cam_path = os.path.join(input_dir, f"demo_{i}_persr_cam.mp4")
        head_cam_path = os.path.join(input_dir, f"demo_{i}_head_cam.mp4")
        left_cam_path = os.path.join(input_dir, f"demo_{i}_left_cam.mp4")
        right_cam_path = os.path.join(input_dir, f"demo_{i}_right_cam.mp4")

        # Check if all files exist
        if not os.path.exists(persr_cam_path) or not os.path.exists(head_cam_path) or not os.path.exists(left_cam_path) or not os.path.exists(right_cam_path):
            print(f"Skipping demo {i} because some video files are missing.")
            continue
        
        # Load the video files
        persr_cam = VideoFileClip(persr_cam_path)
        head_cam = VideoFileClip(head_cam_path)
        left_cam = VideoFileClip(left_cam_path)
        right_cam = VideoFileClip(right_cam_path)
        
        # Resize videos to ensure they fit within the grid (640x480 as per your specification)
        width, height = persr_cam.size  # Assuming all videos have the same size (640x480)
        
        # Resize the smaller videos (head_cam, left_cam, right_cam) to fit 2x2 grid
        head_cam = head_cam.resize(newsize=(width // 2, height // 2))
        left_cam = left_cam.resize(newsize=(width // 2, height // 2))
        right_cam = right_cam.resize(newsize=(width // 2, height // 2))
        
        # Arrange the videos into the 2x2 grid as specified:
        # persr_cam (top left), head_cam (top right), left_cam (bottom left), right_cam (bottom right)
        final_video = clips_array([
            [persr_cam, head_cam],
            [left_cam, right_cam]
        ])
        
        # Construct the output file path
        output_path = os.path.join(output_dir, f"combined_demo_{i}.mp4")
        
        # Write the final combined video to the output path with fps=30
        final_video.write_videofile(output_path, codec='mpeg4', fps=30)
        print(f"Successfully created combined_demo_{i}.mp4")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Combine 4 MP4 videos into a 2x2 grid for each demo.")
    parser.add_argument('input_dir', type=str, help="Directory containing raw video files (demo_x_persr_cam.mp4, etc.).")
    parser.add_argument('output_dir', type=str, default='combine_videos', help="Directory to save combined output videos.")
    parser.add_argument('nums', type=int, help="Number of demos to process, from 0 to nums-1.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to combine the videos
    combine_videos(args.input_dir, args.output_dir, args.nums)

if __name__ == "__main__":
    main()
