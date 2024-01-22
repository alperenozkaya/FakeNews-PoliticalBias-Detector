from transformers import AutoTokenizer, AutoModel
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
        encoded_list = [self.tokenizer.encode(token, add_special_tokens=False) for token in tokens]
        input_idx = [item for sublist in encoded_list for item in sublist]

        if len(input_idx) > self.max_len:
            input_idx = input_idx[:self.max_len - 1] + [self.tokenizer.sep_token_id]
        attention_mask_dl = [1] * len(input_idx)

        padding_length = self.max_len - len(input_idx)
        if padding_length > 0:
            input_idx += [self.tokenizer.pad_token_id] * padding_length
            attention_mask_dl += [0] * padding_length

        return {
            'input_ids': torch.tensor(input_idx),
            'attention_mask': torch.tensor(attention_mask_dl)
        }

# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/convbert-base-turkish-mc4-uncased")
model = AutoModel.from_pretrained("dbmdz/convbert-base-turkish-mc4-uncased")

text_path = '../TurkishDataset/shuffled_dataset_bert.json'
model.to(device)

# Load the data
with open(text_path, 'r') as file:
    data = [json.loads(line) for line in file]

input_data = [item['doc_token'] for item in data]

# Dictionary to hold embeddings
dataset = TokenizedTextDataset(input_data, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

embeddings_sum_dict = {}
token_count_dict = {}

for batch in tqdm(loader, desc="Processing Batches"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.cpu().numpy()

    for i in range(embeddings.shape[0]):
        seq_len = sum(batch['attention_mask'][i])

        for j in range(seq_len):
            token_id = input_ids[i][j].item()
            token = tokenizer.convert_ids_to_tokens(token_id)

            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                token_embedding = embeddings[i, j, :].astype(np.float32)

                if token in embeddings_sum_dict:
                    embeddings_sum_dict[token] += token_embedding
                    token_count_dict[token] += 1
                else:
                    embeddings_sum_dict[token] = token_embedding
                    token_count_dict[token] = 1

    del input_ids, attention_mask, outputs, embeddings
    torch.cuda.empty_cache()
    gc.collect()

for token in embeddings_sum_dict:
    embeddings_sum_dict[token] /= token_count_dict[token]

# Save the dictionary using pickle
with open('../NLPClassifierTool/test_embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings_sum_dict, file)

# Diagnostic Logging
print(f"Total unique tokens: {len(embeddings_sum_dict)}")
for token, count in sorted(token_count_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Token: {token}, Count: {count}")
