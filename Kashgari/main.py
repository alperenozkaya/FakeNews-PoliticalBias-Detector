import sys
#print(sys.getrecursionlimit()) default 1000
#sys.setrecursionlimit(1500)

import pandas as pd
import nltk
import json
import numpy as np
import kashgari
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.embeddings import bert_embedding

def load_json_line_by_line(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            # Add the entire JSON object as a row
            data.append(json_data)
    return data

def flatten_labels(labels):
    # Flatten the labels if they are in a list of lists format
    return [label[0] if isinstance(label, list) and len(label) > 0 else label for label in labels]


# Load the data
train_df = pd.DataFrame(load_json_line_by_line('combined_dataset-data_train.json'))
test_df = pd.DataFrame(load_json_line_by_line('combined_dataset-data_test.json'))
validate_df = pd.DataFrame(load_json_line_by_line('combined_dataset-data_val.json'))

# Prepare the data
train_x, train_y = list(train_df['doc_token']), flatten_labels(train_df['doc_label'])
test_x, test_y = list(test_df['doc_token']), flatten_labels(test_df['doc_label'])
validate_x, validate_y = list(validate_df['doc_token']), flatten_labels(validate_df['doc_label'])

# Initialize the model
model = BiLSTM_Model()

# Build & train the model
model.fit(train_x, train_y, validate_x, validate_y, epochs=10)

# Evaluate the model
model.evaluate(test_x, test_y)

# Model data will save to `saved_ner_model` folder
model.save('saved_classification_model')

# Load saved model
'''
loaded_model = BiLSTM_Model.load_model('saved_classification_model')
loaded_model.predict(test_x[:10])

# To continue training, compile the newly loaded model first
loaded_model.compile_model()
model.fit(train_x, train_y, validate_x, validate_y)
'''


model.save('./model')
