# scripts/prepare_sst2_dataset.py

import json
import os
import datasets
import pandas as pd

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def download_sst2():
    """Download SST-2 dataset and convert to JSON format"""
    # Load SST2 dataset from HuggingFace datasets
    dataset = datasets.load_dataset('glue', 'sst2')
    
    # Convert to desired format
    def convert_to_json_format(split_data):
        df = pd.DataFrame(split_data)
        return [{"sentence": row['sentence'], "label": row['label']} 
                for _, row in df.iterrows()]

    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Convert and save each split
    train_data = convert_to_json_format(dataset['train'])
    val_data = convert_to_json_format(dataset['validation'])
    test_data = convert_to_json_format(dataset['test'])
    
    save_json(train_data, '../data/sst2_train.json')
    save_json(val_data, '../data/sst2_val.json')
    save_json(test_data, '../data/sst2_test.json')

def prepare_datasets(train_path, val_path, test_path, output_dir):
    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)

    # Example preprocessing steps
    # This can be customized based on requirements
    def preprocess(data):
        return [{"sentence": item["sentence"], "label": item["label"]} for item in data]

    pretrained_train = preprocess(train_data)
    pretrained_val = preprocess(val_data)
    pretrained_test = preprocess(test_data)

    os.makedirs(output_dir, exist_ok=True)
    save_json(pretrained_train, os.path.join(output_dir, 'pretrained_train.json'))
    save_json(pretrained_val, os.path.join(output_dir, 'pretrained_val.json'))
    save_json(pretrained_test, os.path.join(output_dir, 'pretrained_test.json'))

if __name__ == "__main__":
    # First download and extract SST2 dataset
    print("Downloading SST-2 dataset...")
    download_sst2()
    print("Download complete.")

    # Then prepare the datasets
    train_path = '../data/sst2_train.json'
    val_path = '../data/sst2_val.json'
    test_path = '../data/sst2_test.json'
    output_dir = '../data/prepared/'
    prepare_datasets(train_path, val_path, test_path, output_dir)
    print("Datasets prepared successfully.")