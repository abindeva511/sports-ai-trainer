# Function to summarize results using OpenAI's GPT API
import openai
import requests
from scipy.spatial.distance import cosine 
from llamaapi import LlamaAPI
import streamlit as st
import os



# Access the variables
api_key = st.secrets["API_KEY"]

def summarize_results_with_llm(results):
    resul = []
    
    # Prepare the text to be summarized

    # Initialize the Llama API with your API key
    llama = LlamaAPI(api_key)

    # Prepare input text with player coordinates
    input_text = ""
    input_text += f"\nFirst Player Body position coordinates of shot put at a time frame 't': {results['a']}\n"
    input_text += f"\nSecond Player Body position coordinates of shot put at a time frame 't': {results['b']}\n"

    # Build the API request for Llama
    api_request_json = {
        "model": "llama3.1-70b",  # Specify the Llama model version
        "messages": [
            {
                "role": "user",
                "content": (
                    "The input text has 2 person body coordinates while throwing. Don't compare them. Extract the information on their head position, "
                    "hip position, knee position, elbow, hand, etc. Give a detailed description of their position based on the shot put throw. "
                    "Mine on how shot put is thrown. Don't give the coordinates. Just a summarization is enough.\n\n"
                    f"{input_text}.\n"
                    "You can give the output like 'head is facing this or that side', 'right hip is raised', 'upper body is bent', etc. "
                    "Expand the summary for various body parts. Give me at least 500 words for this. Values look like this for the "
                    "first player's body position coordinates of shot put at a time frame 't': e.g., 34 Head_x coordinate, 281 Head_y coordinate, "
                    "185 Left Shoulder_x coordinate, 275. Feel free to make any suggestions even if they're wrong; just provide suggestions."
                )
            }
        ],
        "stream": False  # Set to True if you want streaming
    }

    # Execute the request
    response = llama.run(api_request_json)

    # Extract and append the summary to the results list
    summary = response.json()['choices'][0]['message']['content']
    resul.append({'summary': summary})

    return resul


def external_source_info():
    

    # URL of the raw file
    url = "https://raw.githubusercontent.com/abindeva511/country-and-states-map/main/a.txt"

    # Sending a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        content = response.text
    return content



def chunk_text(text, max_chunk_size=512):
    """
    Split text into smaller chunks to avoid exceeding the model's token limit.
    
    Args:
    text (str): The input text.
    max_chunk_size (int): Maximum number of characters per chunk. Default is 512.
    
    Returns:
    list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        # Check if adding the next word exceeds the max_chunk_size
        if len(" ".join(current_chunk + [word])) <= max_chunk_size:
            current_chunk.append(word)
        else:
            # Add the current chunk to the list and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def find_similar_chunks(embeddings1, embeddings2, threshold=0.8):
    """
    Find similar chunks from embeddings2 that are relevant to embeddings1.
    
    Args:
    embeddings1 (list): List of embeddings from the first set.
    embeddings2 (list): List of embeddings from the second set.
    threshold (float): Cosine similarity threshold to consider as similar. Default is 0.8.
    
    Returns:
    list: List of tuples containing indices of similar pairs and their similarity score.
    """
    similar_chunks = []

    # Compare each embedding from embeddings1 with each embedding from embeddings2
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            # Calculate cosine similarity
            similarity = 1 - cosine(emb1, emb2)
            
            # If the similarity is above the threshold, consider it relevant
            if similarity >= threshold:
                similar_chunks.append((i, j, similarity))

    return similar_chunks


def get_relevant_chunks(chunks1, similar_chunks):
    """
    Retrieve the chunks from the first set that are similar to any chunk in the second set.
    
    Args:
    chunks1 (list): List of text chunks corresponding to 'embeddings1'.
    similar_chunks (list): List of tuples containing indices of similar pairs and their similarity score.
    
    Returns:
    list: List of relevant chunks from 'chunks1' that matched with 'embeddings2'.
    """
    # Get unique indices of the relevant chunks from 'embeddings1'
    relevant_indices = {i for i, _, _ in similar_chunks}
    
    # Retrieve the corresponding chunks from 'chunks1'
    relevant_chunks = [chunks1[i] for i in relevant_indices]
    
    return relevant_chunks