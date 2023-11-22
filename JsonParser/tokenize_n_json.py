import pandas as pd
import nltk
import json

# Ensure NLTK tokenizers are downloaded
nltk.download('punkt')

# Function to tokenize text, convert to lowercase, and filter out non-alphabetic characters
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [token.lower() for token in tokens if token.isalpha()]

# Load CSV file
df = pd.read_csv('combined_dataset.csv')
df = df.dropna(subset=['Text']).drop_duplicates(subset=['Text'])

# Process each row and write to JSON file
with open('combined_dataset_line_by_line.json', 'w') as file:
    for _, row in df.iterrows():
        # Assuming 'Label' field contains a single label per row (1 or 0)
        doc_label = [row['Label']]
        doc_token = tokenize(row['Text'])

        # Create a dictionary for each row and write it as a JSON string
        json.dump({
            'doc_label': doc_label,
            'doc_token': doc_token,
            'doc_keyword': [],  # Empty list for doc_keyword
            'doc_topic': []     # Empty list for doc_topic
        }, file)
        file.write('\n')  # Write a newline character after each JSON string

print("Conversion complete. Data saved in 'combined_dataset_line_by_line.json'.")
