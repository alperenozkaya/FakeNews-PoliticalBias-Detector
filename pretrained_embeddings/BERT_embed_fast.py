import torch
from transformers import BertModel, BertTokenizer
import json
import pickle
from tqdm import tqdm


# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to read data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return [item['doc_token'] for item in data]

# Function to handle pre-tokenized input
def process_tokenized_input(input_data):
    return [" ".join(tokens) for tokens in input_data]

# Batch processing function
def get_embeddings(token_list, batch_size=32):
    model.eval()
    embeddings_dict = {}
    for i in tqdm(range(0, len(token_list), batch_size), desc="Processing Batches"):
        batch_tokens = token_list[i:i+batch_size]
        encoded_input = tokenizer(batch_tokens, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = model(**encoded_input)
        embeddings = outputs.last_hidden_state[:,0,:].squeeze()  # Take the embedding of the first token ([CLS])
        for token, embedding in zip(batch_tokens, embeddings):
            embeddings_dict[token] = embedding.numpy()
    return embeddings_dict


# Load your data
text_path = '../JsonParser/formatted_datasets_json/bert_combined.json'
input_data = load_data(text_path)

unique_tokens = set([token for sublist in input_data for token in sublist])
# Generate embeddings
embeddings = get_embeddings(list(unique_tokens), batch_size=32)

with open('token_embeddings_nltk-bert.pkl', 'wb') as fout:
    pickle.dump(embeddings, fout)

# embeddings now contains the embeddings for your data
print('Embeddings generated!')