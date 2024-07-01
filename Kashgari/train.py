import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from bilstm import FakeNewsClassifier, NewsDataset, build_vocab
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
            texts, labels = texts.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(texts)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_labels.extend(labels.cpu().tolist())
            train_predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())

        avg_train_loss = total_train_loss / len(train_loader)

        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_precision = precision_score(train_labels, train_predictions, zero_division=1, average=avg_type)
        train_recall = recall_score(train_labels, train_predictions, average=avg_type)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for texts, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                texts, labels = texts.to(device), labels.to(device).long()
                outputs = model(texts)

                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                val_labels.extend(labels.cpu().tolist())
                val_predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions, zero_division=1, average=avg_type)
        val_recall = recall_score(val_labels, val_predictions, average=avg_type)

        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

# Load datasets from the respective JSON files
train_data = load_data('bert_combined-data_train.json')
val_data = load_data('bert_combined-data_val.json')
test_data = load_data('bert_combined-data_test.json')


# PARAMS
avg_type = 'micro'  # Define average type for precision and recall calculations
sz_batch = 8
epochs = 4
criterion = nn.CrossEntropyLoss()  # Updated for multi-class classification
# Build vocabulary from the training data
vocab = build_vocab(train_data)
vocab_size = len(vocab)

# Create DataLoaders for each dataset
train_loader = DataLoader(NewsDataset(train_data, vocab), batch_size=sz_batch, shuffle=True, collate_fn=NewsDataset.collate_fn)
val_loader = DataLoader(NewsDataset(val_data, vocab), batch_size=sz_batch, shuffle=False, collate_fn=NewsDataset.collate_fn)
test_loader = DataLoader(NewsDataset(test_data, vocab), batch_size=sz_batch, shuffle=False, collate_fn=NewsDataset.collate_fn)


# Initialize the model
model = FakeNewsClassifier(vocab_size, embedding_dim=128, hidden_dim=128, output_dim=3, dropout_prob=0.3)

# Define loss and optimizer

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

# Save the trained model
torch.save(model.state_dict(), 'model_bias.pt')  # Save model weights instead of the whole model
