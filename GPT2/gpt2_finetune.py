import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Initialization
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
batch_size = 32
max_length = 2048
num_epochs = 5
learning_rate = 5e-5
tokenizer.pad_token = tokenizer.eos_token


# Model with specified number of classes for classification
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)



# Load Data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

train_data = load_data('../NLPClassifierTool/data/bias_combined-data_train.json')
test_data = load_data('../NLPClassifierTool/data/bias_combined-data_test.json')
val_data = load_data('../NLPClassifierTool/data/bias_combined-data_val.json')

# Encoding Data
def encode_data(data):
    return tokenizer(data["doc_token"], truncation=True, padding='max_length', max_length=max_length)

train_encodings = [encode_data(doc) for doc in train_data]
test_encodings = [encode_data(doc) for doc in test_data]
val_encodings = [encode_data(doc) for doc in val_data]

# Create label map and encode labels
unique_labels = set(doc['doc_label'][0] for doc in train_data)
label_map = {label: idx for idx, label in enumerate(unique_labels)}

train_labels = [label_map[doc['doc_label'][0]] for doc in train_data]
test_labels = [label_map[doc['doc_label'][0]] for doc in test_data]
val_labels = [label_map[doc['doc_label'][0]] for doc in val_data]

# Dataset Class
class GPT2Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Data Loaders
train_dataset = GPT2Dataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GPT2Dataset(val_encodings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_progress = tqdm(train_loader, desc='Training', leave=False)
    for batch in train_progress:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_progress.set_postfix({'loss': loss.item()})

# Evaluation
def evaluate(model, data_loader):
    model.eval()
    eval_progress = tqdm(data_loader, desc='Evaluating', leave=False)
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in eval_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            true_labels.extend(batch['labels'].tolist())

    return accuracy_score(true_labels, predictions)

val_accuracy = evaluate(model, val_loader)
print(f"Validation Accuracy: {val_accuracy}")
