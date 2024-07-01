import json
import pickle

# Load the embeddings dictionary
with open('../pretrained_embeddings/token_embeddings.pkl', 'rb') as file:
    embeddings_dict = pickle.load(file)

# Load the dataset
with open('../JsonParser/formatted_datasets_json/shuffled_dataset_bert.json', 'r') as file:
    dataset = [json.loads(line) for line in file]

# Extract words from the dataset
# Assuming dataset is a list of dictionaries with a key 'doc_token' that contains the list of words
words_in_dataset = set([token for item in dataset for token in item['doc_token']])

# Check for unmatched words
unmatched_words = set()
matched_words = set()
for word in words_in_dataset:
    if word not in embeddings_dict:
        unmatched_words.add(word)
    else:
        matched_words.add(word)

# Output the results
unmatched_count = len(unmatched_words)
print(f"Number of unmatched words: {unmatched_count}")

# For debugging: Inspect the unmatched_words set
# This set can be inspected in debug mode to analyze patterns or issues
