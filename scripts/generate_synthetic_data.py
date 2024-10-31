"""
Synthetic data generation script for sentiment analysis using various prompting methods.
Based on prompting strategies from Li et al. (2024) https://arxiv.org/html/2407.12813v2
"""

import json
import os
import random
from datetime import datetime
from tqdm import tqdm
import openai

# Constants
NUM_SAMPLES = 10
MODEL_CONFIG = {
    "name": "meta-llama-3.1-8b-instruct",
    "base_url": "http://127.0.0.1:1234/v1",
    "api_key": "lm-studio"  # Any string works for LM Studio
}

# OpenAI client configuration
client = openai.OpenAI(
    base_url=MODEL_CONFIG["base_url"],
    api_key=MODEL_CONFIG["api_key"]
)

# Schema for structured output
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_response",
        "schema": {
            "type": "object",
            "properties": {
                "sentence": {"type": "string"},
                "label": {"type": "integer", "enum": [0, 1]}
            },
            "required": ["sentence", "label"]
        }
    }
}

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def generate_synthetic_samples(prompt, generation_method, num_samples=NUM_SAMPLES):
    """Generate synthetic samples using the specified prompt."""
    synthetic_samples = []
    for _ in tqdm(range(num_samples), desc=f"Generating {generation_method} samples"):
        try:
            response = client.chat.completions.create(
                model=MODEL_CONFIG["name"],
                messages=[
                    {"role": "system", "content": "You are an AI assistant that generates synthetic data for sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format=RESPONSE_SCHEMA,
                temperature=0.7,
                max_tokens=50
            )
            synthetic_samples.append(json.loads(response.choices[0].message.content))
        except Exception as e:
            print(f"Error during generation: {e}")
            continue
    return synthetic_samples

def generate_zero_shot():
    """Zero-shot generation with basic task description."""
    sentiment = random.choice(["positive", "negative"])
    prompt = f"Please generate a sentence that contains a {sentiment} sentiment. Sentence:"
    return generate_synthetic_samples(prompt, "zero_shot")

def generate_zero_shot_topic():
    """Zero-shot generation with topic specification."""
    topics = ["movies", "restaurants", "products", "services", "entertainment"]
    sentiment = random.choice(["positive", "negative"])
    prompt = f"Please consider this topic for generation: {random.choice(topics)}. Please generate a sentence that contains a {sentiment} sentiment. Sentence:"
    return generate_synthetic_samples(prompt, "zero_shot_topic")

def generate_one_shot(train_data):
    """One-shot generation with a single example."""
    example = random.choice(train_data)
    sentiment = "positive" if example["label"] == 1 else "negative"
    prompt = f"""The task is to predict whether the following sentence is positive or negative sentiment.
Sentence: {example['sentence']}
Label: {sentiment}
Please generate a similar example on the same topic, including a Sentence and a Label. Sentence:"""
    return generate_synthetic_samples(prompt, "one_shot")

def generate_few_shot(train_data, num_examples):
    """Few-shot generation with multiple examples."""
    examples = random.sample(train_data, num_examples)
    examples_text = "\n".join([
        f"Sentence: {ex['sentence']}\nLabel: {'positive' if ex['label'] == 1 else 'negative'}"
        for ex in examples
    ])
    prompt = f"""The task is to predict whether the following sentence is positive or negative sentiment.
{examples_text}
Please generate a similar example, including a Sentence and a Label."""
    return generate_synthetic_samples(prompt, f"few_shot_{num_examples}")

def main():
    # Setup paths
    data_paths = {
        'train': '../data/prepared/pretrained_train.json',
        'val': '../data/prepared/pretrained_val.json',
        'output': '../synthetic_data/'
    }
    os.makedirs(data_paths['output'], exist_ok=True)

    # Load training data
    train_data = load_json(data_paths['train'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate data using different methods
    generation_methods = {
        "zero_shot": lambda: generate_zero_shot(),
        "zero_shot_topic": lambda: generate_zero_shot_topic(),
        "one_shot": lambda: generate_one_shot(train_data),
        "few_shot_3": lambda: generate_few_shot(train_data, 3),
        "few_shot_5": lambda: generate_few_shot(train_data, 5)
    }

    for method_name, generate_fn in generation_methods.items():
        output_path = os.path.join(data_paths['output'], f'sst2_synthetic_{method_name}_{timestamp}.json')
        save_json(generate_fn(), output_path)

    print("Synthetic data generation completed.")

if __name__ == "__main__":
    main()