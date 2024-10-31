"""
Training script for SST-2 sentiment classification model using synthetic data.
Based on experiments from Li et al. (2024) [https://arxiv.org/html/2407.12813v2]

Key details from paper for SST-2 (Section 4):
- Model: RoBERTa-base 
- Training hyperparameters:
  - Batch size: 16
  - Learning rate: Default from transformers library 
  - Number of epochs: 3
  - Optimizer: AdamW with weight decay 0.01
  - Warmup steps: 500
  - Evaluation strategy: Per epoch
  - Save strategy: Per epoch
  - Load best model at end: True

- Data (Section 3):
  - Original SST-2 dataset combined with synthetic data
  - Synthetic data generated using:
    - Zero-shot prompting: Task description only
    - Zero-shot with topic specification: Task description + topic
    - One-shot prompting: Task description + 1 example
    - Few-shot prompting: Task description + 3 or 5 examples
  - Synthetic data size: Equal to original training set

- Evaluation (Section 4):
  - Metrics: Accuracy and F1 score
  - Evaluated on original SST-2 validation and test sets
  - Results saved to evaluation_metrics.json

Key findings for SST-2 (Section 1):
- Zero-shot with topic specification performed best among synthetic methods
- Combining synthetic with original data improved performance
- Model achieved 76% accuracy with 6000 synthetic samples vs 88% with human-labeled data
- Cost comparison (from Section 1):
  - Human labeling: $221-300 USD, ~1000 minutes for 3000 samples
  - GPT-3 generation: $14.37 USD, 46 minutes for 3000 samples

To run this script, use the following command:
python train_model.py --synthetic_method <method_name>

possible method names (Section 3):
- zero_shot: Zero-shot in-context generation
- zero_shot_topic: Zero-shot topic in-context generation  
- one_shot: One-shot in-context generation
- few_shot_3: Few-shot (3 examples) in-context generation
- few_shot_5: Few-shot (5 examples) in-context generation
"""

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import torch
import json
import os
import argparse

# Replace the current device configuration with:
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit for memory allocations
device = "cpu"
torch.set_num_threads(4)  # Allow more CPU threads for better performance
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
# Remove the MPS setting as it's not needed


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_dataset(data):
    return Dataset.from_pandas(pd.DataFrame(data))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_method', type=str, required=True,
                       choices=['zero_shot', 'zero_shot_topic', 'one_shot', 'few_shot_3', 'few_shot_5'],
                       help='Method used to generate synthetic data')
    args = parser.parse_args()

    output_dir = f'../models/roberta_base_{args.synthetic_method}/'
    os.makedirs(output_dir, exist_ok=True)

    # Load original and synthetic data
    synthetic_data_path = f'../synthetic_data/sst2_synthetic_{args.synthetic_method}.json'
    trained_data_path = '../data/prepared/pretrained_train.json'
    val_data_path = '../data/prepared/pretrained_val.json'
    test_data_path = '../data/prepared/pretrained_test.json'

    synthetic_data = load_json(synthetic_data_path)
    trained_data = load_json(trained_data_path)
    val_data = load_json(val_data_path)
    test_data = load_json(test_data_path)

    # Combine original and synthetic data
    combined_train = trained_data + synthetic_data

    # Print dataset sizes
    print(f"Original training data size: {len(trained_data)}")
    print(f"Synthetic data size: {len(synthetic_data)}")
    print(f"Combined training data size: {len(combined_train)}")

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(combined_train))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Tokenizer and encoding
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Model initialization
    num_labels = 2
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../logs/',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=True,
    )

    # Metrics
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test Results: {test_results}")

    # Save evaluation metrics
    results_dir = '../results/'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'evaluation_metrics_{args.synthetic_method}.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    import pandas as pd  # Make sure to import pandas
    main()