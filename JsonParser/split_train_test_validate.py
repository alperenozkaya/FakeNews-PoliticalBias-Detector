import json
from sklearn.model_selection import train_test_split

# Function to load JSON data line by line
def load_json_line_by_line(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Function to sanitize text by removing non-ASCII characters
def sanitize_text(text):
    return text.encode('ascii', 'ignore').decode('ascii')

# Function to rename labels and sanitize text
def preprocess_data(data):
    for entry in data:
        entry['doc_token'] = [sanitize_text(token) for token in entry['doc_token']]
        if entry['doc_label'] == [0]:
            entry['doc_label'] = ['fake']
        elif entry['doc_label'] == [1]:
            entry['doc_label'] = ['real']

# Load and preprocess JSON data
data = load_json_line_by_line('combined_dataset_line_by_line.json')
preprocess_data(data)

# Split the data into training and a combined test/validation set
train_data, test_val_data = train_test_split(data, test_size=0.3, random_state=42)

# Split the combined test/validation set into separate test and validation sets
test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

# Function to save data line by line
def save_json_line_by_line(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

# Save the datasets to new JSON files
save_json_line_by_line(train_data, 'data_train.json')
save_json_line_by_line(val_data, 'data_val.json')
save_json_line_by_line(test_data, 'data_test.json')

print("Train/Validation/Test split complete with label renaming and text sanitization. Data saved in 'data_train.json', 'data_val.json', and 'data_test.json'.")
