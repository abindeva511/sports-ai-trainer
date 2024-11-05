import os
from utils import save_nth_frame
# Ensure 'save_nth_frame()' is defined elsewhere in your code and works as intended.

def save_and_render_frames(parent_folder, dataframe, output_folder, col1='Indices of array1', col2='Indices of array2'):
    """
    Extracts and saves frames from two videos located in the parent folder based on indices specified in a DataFrame.

    Parameters:
    - parent_folder: Directory containing the two video files.
    - dataframe: DataFrame containing frame indices for both videos.
    - output_folder: Directory where extracted frames will be saved.
    - col1: Column name in DataFrame for indices of the first video.
    - col2: Column name in DataFrame for indices of the second video.

    Returns:
    - frame_paths: List of paths to the saved frame images.
    """
    # List all files in the parent folder
    all_files = os.listdir(parent_folder)
    
    # Filter out video files (assuming common video file extensions)
    video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv'))]
    
    # Ensure there are exactly two video files
    if len(video_files) != 2:
        raise ValueError("The parent folder must contain exactly two video files.")
    
    # Assign video paths
    video_path1 = os.path.join(parent_folder, video_files[0])
    video_path2 = os.path.join(parent_folder, video_files[1])
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare a list to store the paths of images for rendering in the template
    frame_paths = []

    # Iterate through the DataFrame and save frames based on column values
    for index, row in dataframe.iterrows():
        frame_idx1 = row[col1]
        frame_idx2 = row[col2]
        
        # Save images from the first video
        output_path1 = os.path.join(output_folder, f'frame_video1_{frame_idx1}.jpg')
        
        output_path1_2 = os.path.join('static/frames', f'frame_video1_{frame_idx1}.jpg')
        save_nth_frame(video_path1, frame_idx1, output_path1)
        
        
        # Save images from the second video
        output_path2 = os.path.join(output_folder, f'frame_video2_{frame_idx2}.jpg')

        output_path2_2 = os.path.join('static/frames', f'frame_video2_{frame_idx2}.jpg')
        save_nth_frame(video_path2, frame_idx2, output_path2)

        frame_paths.append([output_path1_2, output_path2_2])

    
    return frame_paths

    
    # Render the output.html with the paths of the saved frames

# Example usage:
# video_path should be the path to your video file (e.g., './uploads/eby.mp4')
# dataframe should be your DataFrame containing columns 'Indices of array1' and 'Indices of array2'

