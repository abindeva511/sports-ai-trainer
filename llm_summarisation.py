

# Send the formatted input to LLM (example with OpenAI API as before)
# Function to summarize results using OpenAI's GPT API
import openai
import pandas as pd
import time
import textwrap
def summarize_results_with_llm(results, relevant_chunks_from_embeddings1, question):
    
    resul = []
    # Prepare the text to be summarized

    input_text = ""
    input_text += f"\nFirst thrower Body position coordinates of shot put at a time frame 't': {results['a']}\n"
    input_text += f"\nSecond thrower Body position coordinates of shot put at a time frame 't': {results['b']}\n"
    input_text += f"\nTechniques of shot put throw is here : 't': {relevant_chunks_from_embeddings1}\n"

    # Call OpenAI API to summarize
    openai.api_key = '''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'''  # Add your OpenAI API key here
    client = openai.OpenAI(api_key=openai.api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
    "role": "user",
    "content": (
        f" The input text contains body coordinates of two throwers while throwing shot put. "
        "Look at their body parts, make hypotheses on their position. Each of throwers coordinates are given in the input text. Also go through the relevant chunks to get the context. "\
            "Attaching the intput text here : "
        "\n\n{input_text}."
        "Based on the input textx, answer the question : {question}. "
        "Do not provide any coordinates in the answer; just give a solution for the question. "
        "Be concise about the question. Give a detailed answer to the crux of the questions"
        "Feel free to offer suggestions even if they might be incorrect. But dont add any coordinates"
    )
}

        ],
        model="gpt-3.5-turbo",  # Or use "gpt-4" if available
    )


    resul.append({chat_completion.choices[0].message.content   })
    # Extract and return the summary (Accessing the text via .choices[0].message.content)

    return resul

def instance_to_intstance(results , df, relevant_chunks_from_embeddings1,question ):
    instance_to_intstance_analysis = {}
    for i in range(len(df)):
        
        instance_to_intstance_analysis[f"time_frame_{i}"] = summarize_results_with_llm(results, relevant_chunks_from_embeddings1, question)
        time.sleep(1)
    return instance_to_intstance_analysis


def overall_summarize_results_with_llm(results, question):
    resul = []
    # Prepare the text to be summarized



    # Call OpenAI API to summarize
    openai.api_key = '''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'''  # Add your OpenAI API key here
    client = openai.OpenAI(api_key=openai.api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
    "role": "user",
    "content": (
        f" A text will added. It is the comparison of shotput thrower 1 and thrower 2 and there techincal difference. Give an overall summary of their techinquer. Since this time seres data, you can think in that way too. Attaching the text here "\
            "{results}"
    )
}

        ],
        model="gpt-3.5-turbo",  # Or use "gpt-4" if available
    )


    resul.append({chat_completion.choices[0].message.content   })
    # Extract and return the summary (Accessing the text via .choices[0].message.content)
    text = next(iter(resul[0]))

    


    wrapped_text = textwrap.fill(text, width=80)
    return wrapped_text
