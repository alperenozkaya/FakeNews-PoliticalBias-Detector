import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Define the fake news classifier model
class FakeNewsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_prob):
        super(FakeNewsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
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

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = [self.vocab.get(word, 0) for word in item["doc_token"]]
        label = 1 if item["doc_label"][0] == "real" else 0
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.float)

    # Add a collate function to handle padding
    @staticmethod
    def collate_fn(batch):
        texts, labels = zip(*batch)
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # Padding
        labels = torch.tensor(labels, dtype=torch.float)
        return texts_padded, labels

# Function to create a vocabulary
def build_vocab(data, min_freq=1):
    counter = Counter()
    for item in data:
        counter.update(item["doc_token"])
    vocab = {word: i+1 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = 0
    return vocab

