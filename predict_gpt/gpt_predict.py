from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

path_gpt2_en = 'gpt_finetune'
path_gpt2_tr = 'gpt_finetune_tr'

inp_fpath = 'input.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(path_gpt2_en)
model_en = GPT2ForSequenceClassification.from_pretrained(path_gpt2_en, trust_remote_code=True)
model_tr = GPT2ForSequenceClassification.from_pretrained(path_gpt2_tr, trust_remote_code=True)
model_en.to(device)
model_tr.to(device)


def classify_text(text, is_turkish=False):
    if is_turkish:
        model = model_tr
    else:
        model = model_en
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Predict the class
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Convert probabilities to numpy array
        probabilities = probabilities.cpu().numpy()

        # Get the predicted class
        predicted_class = np.argmax(probabilities, axis=-1).item()

    labels_map = {0: 'left', 1: 'center', 2: 'right'}
    prediction = labels_map[predicted_class]
    probabilities = probabilities[0][predicted_class]
    return prediction, probabilities



