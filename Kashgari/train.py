import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from bilstm import FakeNewsClassifier, NewsDataset, build_vocab
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score

# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        train_labels = []
        train_predictions = []

        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            outputs = outputs.squeeze()

            # Ensure outputs and labels are of the same dimension
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            predictions = outputs.round()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_labels.extend(labels.cpu().tolist())

            # Check if predictions is a tensor and extend the list, else append
            if isinstance(predictions, torch.Tensor):
                train_predictions.extend(predictions.detach().cpu().tolist())
            else:
                train_predictions.append(predictions.detach().cpu().item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_precision = precision_score(train_labels, train_predictions, zero_division=1)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for texts, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                texts, labels = texts.to(device), labels.to(device)

                outputs = model(texts)
                outputs = outputs.squeeze()

                # Convert outputs to a one-element tensor if it's a single value
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                predictions = outputs.round()

                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                val_labels.extend(labels.cpu().tolist())
                # Check if predictions is a tensor and extend the list, else append
                if isinstance(predictions, torch.Tensor):
                    val_predictions.extend(predictions.detach().cpu().tolist())
                else:
                    val_predictions.append(predictions.detach().cpu().item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions, zero_division=1)

        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}")
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}")


# Load datasets from the respective JSON files
train_data = load_data('combined_dataset-data_train.json')
val_data = load_data('combined_dataset-data_val.json')
test_data = load_data('combined_dataset-data_test.json')

# Build vocabulary from the training data
vocab = build_vocab(train_data)
vocab_size = len(vocab)  # Using the same constant as before

# Create DataLoaders for each dataset
train_loader = DataLoader(NewsDataset(train_data, vocab), batch_size=4, shuffle=True, collate_fn=NewsDataset.collate_fn)
val_loader = DataLoader(NewsDataset(val_data, vocab), batch_size=4, shuffle=False, collate_fn=NewsDataset.collate_fn)
test_loader = DataLoader(NewsDataset(test_data, vocab), batch_size=4, shuffle=False, collate_fn=NewsDataset.collate_fn)

# Initialize the model with the same constants as before
# model = FakeNewsClassifier(vocab_size, embedding_dim=100, hidden_dim=128, output_dim=1, dropout_prob=0.3) # ORIGINAL
model = FakeNewsClassifier(vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1, dropout_prob=0.3)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 4  # Keeping the same number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
torch.save(model, 'model.pt')
