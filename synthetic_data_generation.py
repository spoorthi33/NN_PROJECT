import openai
import json
import os
from typing import List, Dict
import random

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_synthetic_data(method: str, num_samples: int) -> List[Dict[str, str]]:
    """
    Generate synthetic data using the specified method.
    
    :param method: The generation method ('prompt', 'incontext', 'instruction', 'topic')
    :param num_samples: Number of samples to generate
    :return: List of dictionaries containing generated text and label
    """
    prompts = {
        'prompt': "Generate a movie review and its sentiment (positive or negative). The review should be one sentence long.\n\nReview: {review}\nSentiment: {sentiment}",
        'incontext': "Here are some movie reviews and their sentiments:\n\nReview: A heartwarming tale that touches the soul.\nSentiment: positive\n\nReview: The plot was confusing and the acting was terrible.\nSentiment: negative\n\nReview: An action-packed thrill ride from start to finish.\nSentiment: positive\n\nNow generate a new movie review and its sentiment:\n\nReview: {review}\nSentiment: {sentiment}",
        'instruction': "You are a movie critic. Your task is to write a one-sentence movie review and determine its sentiment (positive or negative). Provide your response in the following format:\n\nReview: [Your one-sentence review]\nSentiment: [positive or negative]",
        'topic': "Generate a one-sentence movie review about [TOPIC] and determine its sentiment (positive or negative). Provide your response in the following format:\n\nReview: [Your one-sentence review]\nSentiment: [positive or negative]"
    }
    
    synthetic_data = []
    
    for _ in range(num_samples):
        if method == 'topic':
            # For topic-guided, we'll randomly choose a movie genre as the topic
            topics = ["action", "comedy", "drama", "horror", "sci-fi", "romance"]
            prompt = prompts[method].replace("[TOPIC]", random.choice(topics))
        else:
            prompt = prompts[method]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        generated_text = response.choices[0].message.content.strip()
        review, sentiment = generated_text.split("\nSentiment: ")
        review = review.replace("Review: ", "").strip()
        sentiment = sentiment.strip().lower()
        
        synthetic_data.append({"text": review, "label": sentiment})
    
    return synthetic_data

def save_synthetic_data(data: List[Dict[str, str]], filename: str):
    """
    Save the generated synthetic data to a JSON file.
    
    :param data: List of dictionaries containing generated text and label
    :param filename: Name of the file to save the data
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    methods = ['prompt', 'incontext', 'instruction', 'topic']
    num_samples = 1000  # Adjust this based on the paper's specifications
    
    for method in methods:
        synthetic_data = generate_synthetic_data(method, num_samples)
        save_synthetic_data(synthetic_data, f"sst2_synthetic_{method}.json")
        print(f"Generated {len(synthetic_data)} samples using {method} method.")
