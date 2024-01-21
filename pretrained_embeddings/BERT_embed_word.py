from transformers import BertTokenizer, BertModel
import torch
import json
import pickle
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc


class TokenizedTextDataset(Dataset):
    def __init__(self, tokenized_texts, tokenizer_dl, max_len=512):
        self.tokenized_texts = tokenized_texts
        self.tokenizer = tokenizer_dl
        self.max_len = max_len

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        encoded_list = [self.tokenizer.encode(token, add_special_tokens=True) for token in tokens]
        input_idx = [item for sublist in encoded_list for item in sublist][:self.max_len]  # Flatten and truncate
        attention_mask_dl = [1] * len(input_idx)

        # Padding if necessary
        padding_length = self.max_len - len(input_idx)
        if padding_length > 0:
            input_idx = input_idx + ([0] * padding_length)
            attention_mask_dl = attention_mask_dl + ([0] * padding_length)

        return {
            'input_ids': torch.tensor(input_idx),
            'attention_mask': torch.tensor(attention_mask_dl)
        }


# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
text_path = '../JsonParser/formatted_datasets_json/bert_combined.json'
model.to(device)

# Load the data
with open(text_path, 'r') as file:
    data = [json.loads(line) for line in file]

input_data = [item['doc_token'] for item in data]

# Dictionary to hold embeddings
dataset = TokenizedTextDataset(input_data, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

embeddings_sum_dict = {}
word_embeddings_dict = {}
token_count_dict = {}

# Flatten the input_data to align with the batches
flat_input_data = [token for sublist in input_data for token in sublist]

# Dictionary to track the current position in flat_input_data
current_position = 0

for batch in tqdm(loader, desc="Processing Batches"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # Get the embeddings for all tokens
        embeddings = outputs.last_hidden_state.cpu().numpy()

    # Process each token in the batch
    for i in range(embeddings.shape[0]):
        seq_len = sum(batch['attention_mask'][i])
        for j in range(1, seq_len - 1):
            token_id = input_ids[i][j].item()
            token = tokenizer.convert_ids_to_tokens(token_id)
            token_embedding = embeddings[i, j, :]

            # Store or update the embeddings for the token
            if token not in ['[CLS]', '[SEP]']:  # Exclude CLS and SEP tokens
                token_embedding = token_embedding.astype(np.float32)
                if token in word_embeddings_dict:
                    #word_embeddings_dict[token].append(token_embedding)
                    embeddings_sum_dict[token] += token_embedding
                    token_count_dict[token] += 1

                else:
                    #word_embeddings_dict[token] = [token_embedding]
                    embeddings_sum_dict[token] = token_embedding
                    token_count_dict[token] = 1

    # After processing each batch, explicitly free up memory
    del input_ids, attention_mask, outputs, embeddings
    torch.cuda.empty_cache()
    gc.collect()


# Average the embeddings
for token in embeddings_sum_dict:
    embeddings_sum_dict[token] /= token_count_dict[token]


# Save the dictionary using pickle
with open('../NLPClassifierTool/embeddings_dict_nltk.pkl', 'wb') as file:
    pickle.dump(embeddings_sum_dict, file)
