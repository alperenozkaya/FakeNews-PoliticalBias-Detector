import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import random

# Ensure you have the necessary package
nltk.download('punkt')

def tokenize_text(text):
    # Tokenizes the text using NLTK
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return nltk.word_tokenize(text, language='turkish') #yurttaş, türkçe konuş!

# Ensure you have the necessary package
nltk.download('punkt')


def process_file(filepath, doc_label):
    # Process a single file and return its JSON representation
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = tokenize_text(text)

        return {
            'doc_label': [doc_label],
            'doc_token': tokens,
            'doc_keyword': [],  # Empty list for doc_keyword
            'doc_topic': []     # Empty list for doc_topic
        }

def process_directory(directory):
    # Process each file in the given directory
    documents = []
    doc_label = os.path.basename(directory)
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            document = process_file(filepath, doc_label)
            documents.append(document)
    return documents

def main():
    base_directories = ['real', 'fake']
    all_documents = []

    # Process each directory and store the documents
    for directory in base_directories:
        documents = process_directory(directory)
        all_documents.extend(documents)

    # Shuffle the documents
    random.shuffle(all_documents)

    # Write the shuffled documents to a JSON file
    with open('shuffled_dataset.json', 'w', encoding='utf-8') as output_file:
        for document in all_documents:
            json.dump(document, output_file)
            output_file.write('\n')

if __name__ == "__main__":
    main()
