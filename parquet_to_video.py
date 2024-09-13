import pandas as pd
import imageio
import io
from PIL import Image
import os
import numpy

def convert_parquet_to_video(parquet_file, output_video):
    # Read the parquet file
    df = pd.read_parquet(parquet_file)
    
    # Extract state images and convert to PIL Images
    images = [Image.open(io.BytesIO(state)) for state in df['state']]
    
    # Convert PIL Images to numpy arrays
    frames = [numpy.array(img) for img in images]
    
    # Write frames to video file
    imageio.mimsave(output_video, frames, fps=30)

def process_parquet_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            parquet_path = os.path.join(directory, filename)
            video_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.mp4")
            convert_parquet_to_video(parquet_path, video_path)
            print(f"Converted {filename} to video: {video_path}")

# Specify the directory containing the parquet files
parquet_directory = '.'

# Process all parquet files in the directory
process_parquet_files(parquet_directory)

