import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from tqdm import tqdm
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Hyperparameters
avg_type = 'macro' # for classification, use 'macro' or 'micro' for multi-class and 'binary' for binary
word_embeddings_dict = {}
batch_size = 32
learning_rate = 0.001
dropout = 0.1
num_epochs = 10
d_model = 128
n_head = 4 # attention heads
num_encoder_layers = 2 # number of transformer layers
data_path = '../JsonParser/formatted_datasets_json/bias_combined.json'
embedding_path = '../NLPClassifierTool/r_bias_embeddings.pkl'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
embedding_dim = 128
output_dim = 3 #1 => binary classification.

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

with open (embedding_path, 'rb') as file:
    word_embeddings_dict = pickle.load(file)

with open(data_path, 'r') as file:
    data = [json.loads(line) for line in file]

class TextDataset(Dataset):
    def __init__(self, data, word_embeddings_dict, max_seq_length=2048):
        self.data = data
        self.word_embeddings_dict = word_embeddings_dict
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["doc_token"]
        label_v = self.data[idx]["doc_label"][0]
        if label_v == 'left':
            label = 0
        elif label_v == 'center':
            label = 1
        else:
            label = 2

        embeddings = [torch.tensor(self.word_embeddings_dict.get(token, np.zeros(128)), dtype=torch.float) for token in tokens]

        # Pad or truncate the sequence to the max_seq_length
        if len(embeddings) > self.max_seq_length:
            embeddings = embeddings[:self.max_seq_length]
        elif len(embeddings) < self.max_seq_length:
            embeddings.extend([torch.zeros(128, dtype=torch.float) for _ in range(self.max_seq_length - len(embeddings))])

        embeddings = torch.stack(embeddings)
        return embeddings, label


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim) # Assuming binary classification

    def forward(self, src):
        src = self.transformer_encoder(src)
        src = src.mean(dim=1) # Pooling
        output = self.fc(src)
        # output_sig = torch.sigmoid(output) # yields better results in binary classification
        return output


def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
    labels = labels.long()  # Ensure labels are long integers

    accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels.cpu().numpy(), predicted.cpu().numpy(), average=avg_type)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }
    return metrics



def print_metrics(epoch, phase, metrics, loss):
    metrics_message = (f"{phase} Metrics in Epoch {epoch}: "
                       f"Accuracy: {metrics['accuracy']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, "
                       f"F1 Score: {metrics['f1_score']:.4f}, "
                       f"Loss: {loss:.4f}")
    print(metrics_message)

# Create dataset
dataset = TextDataset(data, word_embeddings_dict)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Model, loss, optimizer
model = TransformerModel().to(device)
criterion = nn.CrossEntropyLoss() # good candidates are: BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    train_outputs = []
    train_labels = []
    train_loss = 0
    train_subset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)//4))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    model.train()
    for embeddings, labels in tqdm(train_loader, desc=f"Training, Epoch {epoch}"):
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_outputs.append(outputs.detach())
        train_labels.append(labels.detach())

    train_outputs = torch.cat(train_outputs).squeeze()
    train_labels = torch.cat(train_labels)

    # Calculate training metrics
    train_metrics = calculate_metrics(train_outputs, train_labels)
    train_loss = train_loss / len(train_loader)
    print_metrics(epoch, "Training", train_metrics, train_loss)

    # Validation loop
    val_subset = torch.utils.data.Subset(val_dataset, torch.randperm(len(val_dataset)//4))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    val_outputs = []
    val_labels = []
    with torch.no_grad():  # No need to track gradients
        for embeddings, labels in tqdm(val_loader, desc=f"Validation, Epoch {epoch}"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_outputs.append(outputs.detach())
            val_labels.append(labels.detach())

    val_outputs = torch.cat(val_outputs).squeeze()
    val_labels = torch.cat(val_labels)

    # Calculate validation metrics
    val_metrics = calculate_metrics(val_outputs, val_labels)
    val_loss = val_loss / len(val_loader)
    print_metrics(epoch, "Validation", val_metrics, val_loss)

    # Test loop
    model.eval()  # Ensure the model is in evaluation mode
    test_subset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)//4))
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    test_outputs = []
    test_labels = []
    test_loss = 0
    with torch.no_grad():
        for embeddings, labels in tqdm(test_loader, desc=f"Testing, Epoch {epoch}"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            test_outputs.append(outputs.detach())
            test_labels.append(labels.detach())

    test_outputs = torch.cat(test_outputs).squeeze()
    test_labels = torch.cat(test_labels)
    # Calculate testing metrics
    test_metrics = calculate_metrics(test_outputs, test_labels)
    test_loss = test_loss / len(test_loader)
    print_metrics(epoch, "Testing", test_metrics, test_loss)

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss}")


# Save model
torch.save(model.state_dict(), 'transformer_model_low_layer.pth')

