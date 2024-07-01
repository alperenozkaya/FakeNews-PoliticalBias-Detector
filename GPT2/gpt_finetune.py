import numpy
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import torch.optim as optim
import torch

num_labels = 3

model_name = "gpt2"  # Choose a pre-trained model name
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Hyperparameters
learning_rate = 2e-5
batch_size = 4
num_epochs = 3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''
classifier_head = torch.nn.Linear(model.config.hidden_size, num_labels)
model.lm_head = classifier_head'''
fpath = '../JsonParser/formatted_datasets/bias_combined.csv'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Fine-tuning GPT-2 model: {model_name}")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {num_epochs}")
print(f"Number of Classes: {num_labels}")


import pandas as pd


df = pd.read_csv(fpath)
df = df.drop('c_words', axis=1)
df = df.drop('word_count', axis=1)

def tokenize_data(df, tokenizer, max_length=512):
    # Initialize lists to hold tokenized results
    input_ids = []
    attention_masks = []

    # Iterate over each text, tokenize, and append results
    for text in df['Text']:
        tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length,
                                       truncation=True, padding='max_length', return_tensors='pt')
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])

    # Convert lists to tensors and stack for proper dimensionality
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

df_s = df.sample(n=200, random_state=42)
label_map = {'left': 0, 'center': 1, 'right': 2}
df_s['Label'] = df_s['Label'].map(label_map)

input_ids, attention_masks = tokenize_data(df_s, tokenizer)



import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, _input_ids, _attention_masks, _labels):
        self.input_ids = _input_ids
        self.attention_masks = _attention_masks
        self.labels = _labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx]
        }

# Splitting the data
def split_data(input_ids, attention_masks, labels, test_size=0.15, val_size=0.15):
    # Train-test split
    train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = train_test_split(
        input_ids, attention_masks, labels, test_size=test_size, random_state=42)

    # Train-validation split
    train_ids, val_ids, train_masks, val_masks, train_labels, val_labels = train_test_split(
        train_ids, train_masks, train_labels, test_size=val_size, random_state=42)

    return (train_ids, train_masks, train_labels), (val_ids, val_masks, val_labels), (test_ids, test_masks, test_labels)

# Labels array
labels_array = df_s['Label'].tolist()

# Perform the split
(train_ids, train_masks, train_labels), (val_ids, val_masks, val_labels), (test_ids, test_masks, test_labels) = split_data(input_ids, attention_masks, labels_array)

# Create data loaders
def create_data_loaders(train_data, val_data, test_data, batch_size):
    train_ds = TextDataset(*train_data)
    val_ds = TextDataset(*val_data)
    test_ds = TextDataset(*test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Set up data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    (train_ids, train_masks, train_labels),
    (val_ids, val_masks, val_labels),
    (test_ids, test_masks, test_labels),
    batch_size
)
'''train_df, val_df, test_df = split_data(df_s, attention_masks, labels_array)
train_loader, val_loader, test_loader = create_data_loaders(train_df, val_df, test_df, batch_size)'''
from tqdm import tqdm
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()
import numpy as np
def train(model, train_loader, optimizer, criterion, num_epochs, device):
  model.to(device)
  model.train()
  total_loss = 0

  progress_bar = tqdm(train_loader, desc="Training", leave=False)
  for batch in progress_bar:
    input_idx = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    outputs = model(input_ids=input_idx, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    progress_bar.set_postfix({"loss": loss.item()})


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            texts = batch['text']
            labels = batch['label']

            outputs = model(**texts, labels=labels).logits
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})  # Update progress bar with current loss

    return total_loss / len(val_loader)


train(model, train_loader, optimizer, criterion, num_epochs, device)
evaluate(model, val_loader, criterion, device)