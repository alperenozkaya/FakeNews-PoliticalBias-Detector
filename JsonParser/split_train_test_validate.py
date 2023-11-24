import json
import sys
from sklearn.model_selection import train_test_split

# Function to save data line by line
def save_json_line_by_line(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')


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
# TODO: Create a function to process all files in a specificied directory...
data = load_json_line_by_line(
        'formatted_datasets_json/PolitiFact_fake_news_content.json')  # choose a file from formatted_data_sets_json
preprocess_data(data)

def train_test_validate_split(data, test_size=0.4, val_size=0.375, random_state=42):

    # 0.7 => train, 0.3 => test/val combined
    # Split the data into training and a combined test/validation set
    train_data, test_val_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # 0.15 => test, 0.15 =>val
    # Split the combined test/validation set into separate test and validation sets
    test_data, val_data = train_test_split(test_val_data, test_size=val_size, random_state=random_state)
    return train_data, test_data, val_data


i = 1
if i == 0:
    train_data, test_data, val_data = train_test_validate_split(data)
else:
    test_data = data
    save_json_line_by_line(test_data, '../NLPClassifierTool/data/data_test.json')
    sys.exit("No test/val split performed. Data saved in 'data_test.json'.")


# Save the datasets to new JSON files
save_json_line_by_line(train_data, '../NLPClassifierTool/data/data_train.json')
save_json_line_by_line(val_data, '../NLPClassifierTool/data/data_val.json')
save_json_line_by_line(test_data, '../NLPClassifierTool/data/data_test.json')

print("Train/Validation/Test split complete with label renaming and text sanitization. Data saved in 'data_train.json', 'data_val.json', and 'data_test.json'.")
