# Function to summarize results using OpenAI's GPT API
import openai
import requests
from scipy.spatial.distance import cosine
def summarize_results_with_llm(results):
    resul = []
    # Prepare the text to be summarized

    input_text = ""
    input_text += f"\nFirst Player Body position coordinates of shot put at a time frame 't': {results['a']}\n"
    input_text += f"\nSecond Player Body position coordinates of shot put at a time frame 't': {results['b']}\n"

    # Call OpenAI API to summarize
    openai.api_key = '''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'''  # Add your OpenAI API key here
    client = openai.OpenAI(api_key=openai.api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"The input text has 2 person body coordinates while throwing. Dont commpare them. Extract the information on their head position, hip position, knee position , elbow, hand , etc. "\
             "give a detailed descirption on their position based on the shot put throw. Mine on how shot put is thrown .Dont give the coordinates. Just the summarisation is enough. \n\n{input_text}."\
                    "You can give the output like head is facing to this or that side. Rigght hip is raised, upper body is bent etc. Exapnd the summary like thise for the various parts. Give me atleast 500 words for this. It  "\
                        "Values look like this First Player Body posiiton coordiantes of shot put at a time frame 't':   eg:     34 Head_x coordinate   281 Head_y coordinate              185 Left Shoulder_x coordinate     275"\
                        "feel free to say any suggestions even if its wrong. Just give the suggestions"}
        ],
        model="gpt-3.5-turbo",  # Or use "gpt-4" if available
    )


    resul.append({'summary':chat_completion.choices[0].message.content   })
    # Extract and return the summary (Accessing the text via .choices[0].message.content)

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