import os
import pandas as pd
import nltk
import json
from config import Config
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Ensure NLTK tokenizers are downloaded
nltk.download('punkt')


# Function to tokenize text, convert to lowercase, and filter out non-alphabetic characters
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [token.lower() for token in tokens if token.isalpha()]


def bert_tokenize(text):
    # remove any emojis/chinese/japanese characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    tokens = tokenizer.tokenize(text)
    return tokens

def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Text']).drop_duplicates(subset=['Text'])

    output_file = os.path.splitext(os.path.basename(file_path))[0] + '.json'  # [0]: name [1]: .csv
    output_path = os.path.join(output_dir, output_file)

    # Process each row and write to JSON file
    with open(output_path, 'w') as file:
        for _, row in df.iterrows():
            # Assuming 'Label' field contains a single label per row (1 or 0)
            doc_label = [row['Label']]
            doc_token = bert_tokenize(row['Text'])

            # Create a dictionary for each row and write it as a JSON string
            json.dump({
                'doc_label': doc_label,
                'doc_token': doc_token,
                'doc_keyword': [],  # Empty list for doc_keyword
                'doc_topic': []  # Empty list for doc_topic
            }, file)
            file.write('\n')  # Write a newline character after each JSON string


def main():
    # Load CSV files
    input_dir = config.json_parser.input_dir
    output_dir = config.json_parser.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, output_dir)

    print("Conversion complete, converted files are saved in formatted_data_sets_json file '.")


if __name__ == '__main__':
    config = Config(config_file='../config/dataset_modifier.json')
    main()