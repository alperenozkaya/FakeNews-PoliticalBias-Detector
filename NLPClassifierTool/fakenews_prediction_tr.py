import codecs
import math

import nltk
import json
import os
import numpy as np
from transformers import AutoTokenizer

import predict

from config import Config


# Ensure NLTK tokenizers are downloaded
nltk.download('punkt')
turkish_tokenizer = AutoTokenizer.from_pretrained("dbmdz/electra-base-turkish-mc4-uncased-discriminator")


# Function to tokenize text, convert to lowercase, and filter out non-alphabetic characters
def tokenize(text):

    encoded_input = turkish_tokenizer(text, add_special_tokens=True)
    tokens = encoded_input.tokens()
    # Exclude special tokens
    tokens = [token for token in tokens if token not in turkish_tokenizer.all_special_tokens]
    return tokens


def tokenize_text_to_json_file(text, output_dir, file_name="rcv1_test.json"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tokenize the text
    doc_token = tokenize(text)

    # Prepare the JSON data with only tokens
    json_data = {
        'doc_label': [],
        'doc_token': doc_token,
        'doc_keyword': [],  # Empty list for doc_keyword
        'doc_topic': []  # Empty list for doc_topic
    }

    # Writing JSON data to a file
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w') as file:
        json.dump(json_data, file)

    print(f"Tokenized text saved to {output_path}")


def predict_label(text):

    output_directory = "output"
    json_file_name = "rcv1_test.json"
    tokenize_text_to_json_file(text, output_directory, json_file_name)
    json_file_path = os.path.join(output_directory, json_file_name)
    config = Config(config_file='../NLPClassifierTool/conf/train.json')
    predictor = predict.Predictor(config)
    is_multi = config.task_info.label_type == predict.ClassificationType.MULTI_LABEL

    for line in codecs.open(json_file_path, "r", predictor.dataset.CHARSET):
        input_text = line.strip("\n")

    # Perform prediction
    predict_prob = predictor.predict([input_text])
    # Process the prediction results
    if not is_multi:
        predict_label_id = predict_prob.argmax()
        if predict_label_id == 0:
            predict_label_name = 'real'
        else:
            predict_label_name = 'fake'
        predict_label_prob = "{:.3f}".format(predict_prob[0][predict_label_id])
    else:
        predict_label_idx = np.argsort(-predict_prob)
        top_label_id = predict_label_idx[0]  # Taking the top label
        if top_label_id == 0:
            predict_label_name = 'real'
        else:
            predict_label_name = 'fake'
        predict_label_prob = "{:.3f}".format(predict_prob[0][top_label_id])
    return predict_label_name, predict_label_prob


text = "Ben gittim, sen niye gelmedin!"
label, probability = predict_label(text)
print(f"Predicted Label: {label}")
print(f"Probability: {probability}")