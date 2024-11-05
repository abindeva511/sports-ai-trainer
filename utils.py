import os
import cv2
import os
from cvzone.PoseModule import PoseDetector

import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def pose_detection():
    folder_path = './uploads'  # Make sure this path is correct

    # List all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    # Initialize PoseDetector
    detector = PoseDetector()

    # Dictionary to store pose data for each video
    all_pose_data = {}

    # Loop through each video file
    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(folder_path, video_file))
        posList = []
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = detector.findPose(img)
            lmList, bboxInfo = detector.findPosition(img)
            
            if bboxInfo:
                frame_data = []
                for lm in lmList:
                    frame_data.append((lm[0], img.shape[0] - lm[1], lm[2]))
                posList.append(frame_data)
            
            #cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            
            if key == 27:  # Press 'ESC' to exit early
                break
        
        # Store the pose data for the current video
        all_pose_data[video_file] = posList
        cap.release()

        cv2.destroyAllWindows()

    def flatten_and_limit(tuples, limit=7):
        flattened = [item for sublist in tuples for item in sublist]
        # Truncate the list to the first 7 elements
        return flattened[:]

    all_pose_flattened_data = {key: [flatten_and_limit(sequence) for sequence in value] for key, value in all_pose_data.items()}
    return all_pose_flattened_data


def filtering(all_pose_flattened_data):
    # Sample data structure for the provided data
    video_pose_data = all_pose_flattened_data

    # Indices to be removed, where each index represents a keypoint (removing x, y, z for each)
    indices_to_remove = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 20, 22, 21, 19, 17]

    # Convert the list of indices to remove to sets of triplets (x, y, z positions in the flattened list)
    remove_indices = set()
    for index in indices_to_remove:
        remove_indices.update(range(index * 3, index * 3 + 3))

    # Function to filter out the specified indices from each video's frame data
    def filter_pose_data(pose_data, remove_indices):
        filtered_data = []
        for frame_data in pose_data:
            filtered_frame = [value for i, value in enumerate(frame_data) if i not in remove_indices]
            filtered_data.append(filtered_frame)
        return filtered_data

    # Apply filtering to each video
    filtered_video_pose_data = {
        video: filter_pose_data(frames, remove_indices)
        for video, frames in video_pose_data.items()
    }
    return filtered_video_pose_data



def dtw_mapping_and_distances(array1, array2):
    """
    Computes the DTW alignment between two 3-dimensional arrays and returns a DataFrame
    with the indices and distances of the matched points.
    
    Parameters:
    - array1: First 3-dimensional array (numpy array)
    - array2: Second 3-dimensional array (numpy array)
    
    Returns:
    - DataFrame with columns 'Indices of array1', 'Indices of array2', and 'Distances'
    """
    # Compute the DTW alignment using fastdtw
    distance, path = fastdtw(array1, array2, dist=euclidean)
    
    # Path contains the mapping, each element in path is a tuple (index_in_array1, index_in_array2)
    indices1, indices2 = zip(*path)
    
    # Calculate the distance between each pair of points
    distances = [euclidean(array1[i], array2[j]) for i, j in path]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Indices of array1': indices1,
        'Indices of array2': indices2,
        'Distances': distances
    })
    
    return df


def named_positions_at_timeframes(array1, array2):
    df1 = pd.DataFrame(array1)
    columns_to_remove = [col for col in df1.columns if (col + 1) % 3 == 0]
    filtered_df1 = df1.drop(columns=columns_to_remove)

    df2 = pd.DataFrame(array2)
    columns_to_remove = [col for col in df2.columns if (col + 1) % 3 == 0]
    filtered_df2 = df2.drop(columns=columns_to_remove)


    filtered_df1.columns = ['Head_x coordinate', 'Head_y coordinate', 'Left Shoulder_x coordinate', 'Left Shoulder_y coordinate', 'Right Shoulder_x coordinate', 'Right Shoulder_y coordinate', 'Left Elbow_x coordinate', 'Left Elbow_y coordinate', 'Right Elbow_x coordinate', 'Right Elbow_y coordinate', 'Left Wrist_x coordinate', 'Left Wrist_y coordinate', 'Right Wrist_x coordinate', 'Right Wrist_y coordinate', 'Left Hip_x coordinate', 'Left Hip_y coordinate', 'Right Hip_x coordinate', 'Right Hip_y coordinate', 'Left Knee_x coordinate', 'Left Knee_y coordinate', 'Right Knee_x coordinate', 'Right Knee_y coordinate', 'Left Ankle_x coordinate', 'Left Ankle_y coordinate', 'Right Ankle_x coordinate', 'Right Ankle_y coordinate', 'Left Heel_x coordinate', 'Left Heel_y coordinate', 'Right Heel_x coordinate', 'Right Heel_y coordinate', 'Left Foot Index_x coordinate', 'Left Foot Index_y coordinate', 'Right Foot Index_x coordinate', 'Right Foot Index_y coordinate']
    filtered_df2.columns = ['Head_x coordinate', 'Head_y coordinate', 'Left Shoulder_x coordinate', 'Left Shoulder_y coordinate', 'Right Shoulder_x coordinate', 'Right Shoulder_y coordinate', 'Left Elbow_x coordinate', 'Left Elbow_y coordinate', 'Right Elbow_x coordinate', 'Right Elbow_y coordinate', 'Left Wrist_x coordinate', 'Left Wrist_y coordinate', 'Right Wrist_x coordinate', 'Right Wrist_y coordinate', 'Left Hip_x coordinate', 'Left Hip_y coordinate', 'Right Hip_x coordinate', 'Right Hip_y coordinate', 'Left Knee_x coordinate', 'Left Knee_y coordinate', 'Right Knee_x coordinate', 'Right Knee_y coordinate', 'Left Ankle_x coordinate', 'Left Ankle_y coordinate', 'Right Ankle_x coordinate', 'Right Ankle_y coordinate', 'Left Heel_x coordinate', 'Left Heel_y coordinate', 'Right Heel_x coordinate', 'Right Heel_y coordinate', 'Left Foot Index_x coordinate', 'Left Foot Index_y coordinate', 'Right Foot Index_x coordinate', 'Right Foot Index_y coordinate']
    return filtered_df1 , filtered_df2 

def single_string(instance_to_intstance_analysis):
    result_str = ""
    for key, value_list in instance_to_intstance_analysis.items():
        for value in value_list:
            # Concatenate the key with the value
            result_str += f"At {key} , the differences are : {value}\n\n"
    return result_str


def save_nth_frame(video_path, n, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)  # OpenCV uses 0-based indexing for frames
    success, img = cap.read()
    
    if success:
        # Save the frame as an image file
        cv2.imwrite(output_path, img)
        # print(f"Frame {n} saved as {output_path}")
 
    
    cap.release()

