"""
Evaluation script for SST-2 sentiment classification model.
Based on experiments from Li et al. (2024) [https://arxiv.org/html/2407.12813v2]

This script evaluates models trained with synthetic data on the SST-2 test set.

Key details from paper (Section 4):
- Model: RoBERTa-base
- Evaluation metrics: Accuracy and F1 score
- Results saved to test_evaluation_metrics.json

To run this script, use the following command:
python evaluate_model.py --synthetic_method <method_name>

possible method names (Section 3):
- zero_shot: Zero-shot in-context generation
- zero_shot_topic: Zero-shot topic in-context generation  
- one_shot: One-shot in-context generation
- few_shot_3: Few-shot (3 examples) in-context generation
- few_shot_5: Few-shot (5 examples) in-context generation
"""

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import json
import os
import argparse

# Device configuration
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit for memory allocations
device = "cpu"
torch.set_num_threads(4)  # Allow more CPU threads for better performance
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_method', type=str, required=True,
                       choices=['zero_shot', 'zero_shot_topic', 'one_shot', 'few_shot_3', 'few_shot_5'],
                       help='Method used to generate synthetic data')
    args = parser.parse_args()

    model_dir = f'../models/roberta_base_{args.synthetic_method}/'
    test_data_path = '../data/prepared/pretrained_test.json'
    results_dir = '../results/'
    os.makedirs(results_dir, exist_ok=True)

    # Load test data
    test_data = load_json(test_data_path)
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)

    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Metrics
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=32,
        logging_dir='../logs/',
        no_cuda=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    test_results = trainer.evaluate(tokenized_test)
    print(f"Test Results: {test_results}")

    # Save evaluation metrics
    with open(os.path.join(results_dir, f'test_evaluation_metrics_{args.synthetic_method}.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    import pandas as pd  # Make sure to import pandas
    main()