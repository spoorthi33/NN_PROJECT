import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=128)
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'label': torch.tensor(0 if item['label'] == 'negative' else 1)
        }

def load_data(filename: str) -> List[Dict[str, str]]:
    with open(filename, 'r') as f:
        return json.load(f)

def train_and_evaluate(train_data: List[Dict[str, str]], test_data: List[Dict[str, str]]):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    train_dataset = SST2Dataset(train_data, tokenizer)
    test_dataset = SST2Dataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

if __name__ == "__main__":
    methods = ['prompt', 'incontext', 'instruction', 'topic']
    test_data = load_data("sst2_test.json")  # Load the original SST-2 test set
    val_data = load_data("sst2_val.json")  # Load the original SST-2 validation set

    for method in methods:
        train_data = load_data(f"sst2_synthetic_{method}.json")
        val_accuracy = train_and_evaluate(train_data, val_data)
        test_accuracy = train_and_evaluate(train_data, test_data)
        print(f"Validation accuracy using {method} method: {val_accuracy:.4f}")
        print(f"Test accuracy using {method} method: {test_accuracy:.4f}")
