import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Ensure you have the necessary package
nltk.download('punkt')

def tokenize_text(text):
    # Tokenizes the text using NLTK
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return nltk.word_tokenize(text, language='turkish') #yurtdaş, türkçe konuş!

import json
import random

def load_and_shuffle_json(file_path):
    # Load JSON objects from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    # Shuffle the data
    random.shuffle(data)

    return data

def save_json(data, file_path):
    # Save the shuffled data back to a file
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

def main():
    input_path = 'dataset.json'  # Path to the original JSON file
    output_path = 'shuffled_dataset.json'  # Path to save the shuffled data

    shuffled_data = load_and_shuffle_json(input_path)
    save_json(shuffled_data, output_path)

if __name__ == "__main__":
    main()
