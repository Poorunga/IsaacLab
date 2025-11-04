import numpy as np
import imageio
import os
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process depth map and save as PNG images.")
    parser.add_argument('--input_npy', type=str, required=True, help="Path to the input .npy file containing the depth map.")
    parser.add_argument('--output_png', type=str, default=None, help="Directory where output PNG images will be saved. If not provided, output will be saved in the same directory as the input.")

    # Parse arguments
    args = parser.parse_args()

    # Load the depth map
    depth_map = np.load(args.input_npy)

    # Print the shape of the depth map to verify it
    print(f"Loaded depth map with shape: {depth_map.shape}")

    # Normalize depth map to range [0, 1]
    depth_map = depth_map / depth_map.max()

    # Remove the last dimension (if shape is (480, 640, 1), convert to (480, 640))
    depth_map = np.squeeze(depth_map)

    # Determine output directory: If --output_png is provided, use that; otherwise, use the input directory
    output_dir = args.output_png if args.output_png else os.path.dirname(args.input_npy)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the base name of the input file without the extension
    base_name = os.path.basename(args.input_npy).replace(".npy", "")

    # Save the depth map as a PNG image in the output directory
    output_path = os.path.join(output_dir, f"{base_name}.png")
    imageio.imwrite(output_path, (depth_map * 255).astype(np.uint8))  # Save as PNG


if __name__ == "__main__":
    main()
