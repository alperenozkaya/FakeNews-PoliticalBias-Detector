import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import BertTokenizer, AutoTokenizer


class PoliticalBiasClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_prob1, dropout_prob2):
        super(PoliticalBiasClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_prob1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out[:, -1, :]
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
fpath = '../JsonParser/formatted_datasets/bias_combined.csv'

df = pd.read_csv(fpath)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

def clean_tokenize(text):
  text = text.lower()

  text = ''.join([c for c in text if c not in string.punctuation])
  tokens = word_tokenize(text)

  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token not in stop_words]

  return tokens

# take random 200 samples from the dataset
df = df.sample(200)

df['Tokens'] = df['Text'].apply(clean_tokenize)




from collections import defaultdict
class TextDataset(Dataset):
    def __init__(self, dataframe, max_seq_len=None):
        self.data = dataframe
        self.vocab = self.build_vocab(self.data['Tokens'])
        self.label_map = self.build_label_map(self.data['Label'])
        self.max_seq_len = max_seq_len  # Maximum sequence length

    def build_vocab(self, tokens_list):
        vocab = defaultdict(lambda: len(vocab))
        vocab['<PAD>'] = 0
        for tokens in tokens_list:
            for token in tokens:
                vocab[token]
        return dict(vocab)

    def build_label_map(self, labels):
        unique_labels = set(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        return label_map

    def encode_tokens(self, tokens):
        # Truncate or pad the token sequence
        if self.max_seq_len is not None:
            encoded = [self.vocab.get(token, 0) for token in tokens[:self.max_seq_len]]
            encoded += [self.vocab['<PAD>']] * (self.max_seq_len - len(encoded))
        else:
            encoded = [self.vocab.get(token, 0) for token in tokens]
        return encoded

    def encode_label(self, label):
        return self.label_map[label]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.iloc[idx]['Tokens']
        label = self.data.iloc[idx]['Label']
        encoded_tokens = self.encode_tokens(tokens)
        encoded_label = self.encode_label(label)
        return torch.tensor(encoded_tokens, dtype=torch.long), torch.tensor(encoded_label, dtype=torch.long)




from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df)
test_dataset = TextDataset(test_df)

pad_value = 0
batch_size = 2
learning_rate = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout_1 = 0.2
dropout_2 = 0.3
embedding_dim = 128
hidden_dim = 128
output_dim = 3

vocab_size = train_dataset.vocab_size
print("Vocabulary Size:", vocab_size)
print(f'Device: {device}')
model = PoliticalBiasClassifier(
    vocab_size = vocab_size,
    embedding_dim = embedding_dim,
    hidden_dim = hidden_dim,
    output_dim = output_dim,
    dropout_prob1 = dropout_1,
    dropout_prob2 = dropout_2
)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()





def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for tokens_batch, labels_batch in progress_bar:
        tokens_batch, labels_batch = tokens_batch.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(tokens_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for tokens_batch, labels_batch in progress_bar:
            tokens_batch, labels_batch = tokens_batch.to(device), labels_batch.to(device)
            outputs = model(tokens_batch)
            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
    return total_loss / len(dataloader)


from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch, max_seq_len=2048):
    if not batch:
        return torch.tensor([]), torch.tensor([])

    text_list, label_list = zip(*batch)
    token_tensors = [torch.tensor(text) for text in text_list]
    label_tensors = [torch.tensor(label) for label in label_list]

    token_tensors_padded = pad_sequence(token_tensors, batch_first=True, padding_value=pad_value)

    if token_tensors_padded.size(1) > max_seq_len:
        token_tensors_padded = token_tensors_padded[:, :max_seq_len]
    else:
        padding_size = max_seq_len - token_tensors_padded.size(1)
        padding = torch.full((token_tensors_padded.size(0), padding_size), pad_value, dtype=torch.long)
        token_tensors_padded = torch.cat([token_tensors_padded, padding], dim=1)

    label_tensors = torch.stack(label_tensors)

    return token_tensors_padded, label_tensors



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
