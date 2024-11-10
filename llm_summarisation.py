

# Send the formatted input to LLM (example with OpenAI API as before)
# Function to summarize results using OpenAI's GPT API
import openai
import pandas as pd
import time
import textwrap
import time
import textwrap
from llamaapi import LlamaAPI
import streamlit as st
import os

# Access the variable from Streamlit secrets
api_key = st.secrets["API_KEY"]







# Initialize the Llama API
llama = LlamaAPI(api_key)

def summarize_results_with_llm(results, relevant_chunks_from_embeddings1, question):
    resul = []
    # Load environment variables from .env file



    # Initialize the Llama API
    llama = LlamaAPI(api_key)
    
    # Prepare the input text for summarization
    input_text = ""
    input_text += f"\nFirst thrower Body position coordinates of shot put at a time frame 't': {results['a']}\n"
    input_text += f"\nSecond thrower Body position coordinates of shot put at a time frame 't': {results['b']}\n"
    input_text += f"\nTechniques of shot put throw is here : 't': {relevant_chunks_from_embeddings1}\n"
    
    # Build the API request for Llama
    api_request_json = {
        "model": "llama3.1-70b",  # Specify the Llama model
        "messages": [
            {
                "role": "user",
                "content": (
                    f" The input text contains body coordinates of two throwers while throwing shot put. "
                    "Look at their body parts, make hypotheses on their position. Each of thrower's coordinates are given in the input text. "
                    "Also go through the relevant chunks to get the context. "
                    f"\n\n{input_text}."
                    f"Based on the input text, answer the question: {question}. "
                    "Do not provide any coordinates in the answer; just give a solution for the question. "
                    "Be concise about the question. Provide detailed insights on the key aspects of the question."
                    "Feel free to offer suggestions even if they might be incorrect, but don't add any coordinates."
                )
            }
        ],
        "stream": False
    }
    
    # Execute the request and retrieve the response
    response = llama.run(api_request_json)
    summary = response.json()['choices'][0]['message']['content']
    resul.append(summary)

    return resul

def instance_to_instance(results, df, relevant_chunks_from_embeddings1, question):
    instance_to_instance_analysis = {}
    # Load environment variables from .env file
    api_key = st.secrets["API_KEY"]


    # Initialize the Llama API
    llama = LlamaAPI(api_key)
    for i in range(len(df)):
        instance_to_instance_analysis[f"time_frame_{i}"] = summarize_results_with_llm(results, relevant_chunks_from_embeddings1, question)
        time.sleep(1)  # Adding a delay to avoid rate limiting
    return instance_to_instance_analysis

def overall_summarize_results_with_llm(results, question):
    resul = []
    # Load environment variables from .env file
    api_key = st.secrets["API_KEY"]


    # Initialize the Llama API
    llama = LlamaAPI(api_key)
    
    # Build the API request for overall summarization
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A text will be added. It is the comparison of shotput thrower 1 and thrower 2 and their technical differences. "
                    "Give an overall summary of their techniques. Since this is time series data, consider the time aspect in your summary. "
                    f"Here is the text: {results}"
                )
            }
        ],
        "stream": False
    }
    
    # Execute the request and retrieve the response
    response = llama.run(api_request_json)
    summary = response.json()['choices'][0]['message']['content']
    resul.append(summary)

    # Format the output text
    text = resul[0]
    wrapped_text = textwrap.fill(text, width=80)
    return wrapped_text
