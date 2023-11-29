import json
import os
import sys
from sklearn.model_selection import train_test_split
from config import Config


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


def train_test_validate_split(data, test_size=0.4, val_size=0.375, random_state=42):

    # 0.7 => train, 0.3 => test/val combined
    # Split the data into training and a combined test/validation set
    train_data, test_val_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # 0.15 => test, 0.15 =>val
    # Split the combined test/validation set into separate test and validation sets
    test_data, val_data = train_test_split(test_val_data, test_size=val_size, random_state=random_state)
    return train_data, test_data, val_data


def update_train_config(dataset_name):
    # read train.json config file
    with open('../NLPClassifierTool/conf/train.json', 'r') as file:
        data = json.load(file)

    # change train, test, val file paths

    data['data']['train_json_files'][0] = f'data/{dataset_name}-data_train.json'
    data['data']['validate_json_files'][0] = f'data/{dataset_name}-data_val.json'
    data['data']['test_json_files'][0] = f'data/{dataset_name}-data_test.json'
    # add dataset name to json file
    data['dataset_name'] = dataset_name

    with open('../NLPClassifierTool/conf/train.json', 'w') as file:
        json.dump(data, file, indent=4)


def main():
    # Load and preprocess JSON data
    # TODO: Create a function to process all files in a specificied directory...
    dataset_name = config.json_parser.dataset_name
    data = load_json_line_by_line(
        f'formatted_datasets_json/{dataset_name}')  # choose a file from formatted_data_sets_json
    preprocess_data(data)

    i = 0
    if i == 0:
        train_data, test_data, val_data = train_test_validate_split(data)
    else:
        test_data = data
        save_json_line_by_line(test_data, '../NLPClassifierTool/data/data_test.json')
        sys.exit("No test/val split performed. Data saved in 'data_test.json'.")

    dataset_name_split = os.path.splitext(os.path.basename(dataset_name))
    # Save the datasets to new JSON files
    save_json_line_by_line(train_data, f'../NLPClassifierTool/data/{dataset_name_split[0]}-data_train.json')
    save_json_line_by_line(val_data, f'../NLPClassifierTool/data/{dataset_name_split[0]}-data_val.json')
    save_json_line_by_line(test_data, f'../NLPClassifierTool/data/{dataset_name_split[0]}-data_test.json')
    print(
        "Train/Validation/Test split complete with label renaming and text sanitization. Data saved in 'data_train.json', 'data_val.json', and 'data_test.json'.")

    # update train, test, val file paths
    update_train_config(dataset_name_split[0])


if __name__ == '__main__':
    config = Config(config_file='../config/dataset_modifier.json')
    main()