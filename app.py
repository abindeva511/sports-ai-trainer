import streamlit as st
from utils import pose_detection, filtering, dtw_mapping_and_distances, named_positions_at_timeframes, single_string, save_nth_frame
import os
from embedding import summarize_results_with_llm, external_source_info, chunk_text, find_similar_chunks, get_relevant_chunks
import numpy as np
from llm_summarisation import instance_to_intstance, overall_summarize_results_with_llm
import pickle
from sentence_transformers import SentenceTransformer
from testing import save_and_render_frames

# Load the embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2', trust_remote_code=True)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Streamlit UI
st.title("Sports AI Trainer")
st.write("Upload a video to compare the shot put stances.")

# File upload section
uploaded_file = st.file_uploader("Choose a video file", type=list(ALLOWED_EXTENSIONS))
question = "Compare the shot put stances of the first thrower and the second thrower. Note down the key differences."

if uploaded_file is not None:
    # Save the uploaded file
    filename = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())
    
    # Step 1: Pose detection
    all_pose_flattened_data = pose_detection()
    
    # Step 2: Filtering
    filtered_video_pose_data = filtering(all_pose_flattened_data)
    
    # Step 3: DTW mapping and distances
    array1 = filtered_video_pose_data['7.mp4']  # Placeholder, replace with your file logic
    array2 = filtered_video_pose_data['eby.mp4']
    df = dtw_mapping_and_distances(array1, array2)
    
    # Step 4: Named positions at timeframes
    filtered_df1, filtered_df2 = named_positions_at_timeframes(array1, array2)
    results = {'a': filtered_df1, 'b': filtered_df2}
    
    # Step 5: Summarize results with LLM
    contextualisation = summarize_results_with_llm(results)
    
    # Step 6: External source info
    external_source = external_source_info()
    
    # Step 7: Chunking text
    chunks_technique = chunk_text(external_source)
    chunks_input = chunk_text(contextualisation[0]['summary'])
    
    # Step 8: Embedding
    embeddings_technique = embedding_model.encode(chunks_technique)
    embeddings_input = embedding_model.encode(chunks_input)
    
    # Step 9: Find similar chunks
    similar_chunks = find_similar_chunks(embeddings_technique, embeddings_input, threshold=0.8)
    relevant_chunks_from_embeddings1 = get_relevant_chunks(chunks_technique, similar_chunks)
    
    # Step 10: Instance-to-instance analysis
    instance_to_instance_analysis = instance_to_intstance(results, df, relevant_chunks_from_embeddings1, question)
    
    # Step 11: Convert to a single string
    result_str = single_string(instance_to_instance_analysis)
    
    # Step 12: Overall summary
    overall_summary = overall_summarize_results_with_llm(result_str, question)
    
    # Display results in Streamlit
    st.write("### Analysis Summary")
    st.write(overall_summary)
    
    # Save and render frames (optional, depending on requirements)
    output_folder = 'static/frames'
    os.makedirs(output_folder, exist_ok=True)
    
    frame_paths = save_and_render_frames(
        './uploads', df, './static/frames', col1='Indices of array1', col2='Indices of array2'
    )

    if frame_paths and instance_to_instance_analysis:
        st.write("### Saved Frames and Analysis:")

    # Iterate over each pair of frame paths and corresponding analysis
        for paths, analysis in zip(frame_paths, instance_to_instance_analysis.values()):
            # Ensure there are exactly two images in paths for each row
            if len(paths) == 2:
                col1, col2, col3 = st.columns([3, 3, 6])  # Adjust column widths as needed

                # Display the two images in the first two columns
                with col1:
                    st.image(paths[0], caption="Saved Frame 1", use_column_width=True)
                with col2:
                    st.image(paths[1], caption="Saved Frame 2", use_column_width=True)

                # Display the analysis in the third column
                with col3:
                    st.write("**Analysis:**")
                    st.write(analysis)
            else:
                st.write("Warning: Each analysis should have exactly 2 frames.")
    else:
        st.write("No frames and analysis available.")

