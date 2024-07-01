
from gpt_predict import classify_text


inp_fpath = 'input.txt'
# Read the input text
with open(inp_fpath, 'r') as file:
    text = file.read()

prediction, probabilities = classify_text(text, is_turkish=True)
print(f"Predicted class: {prediction}")
print(f"Probabilities: {probabilities}")
print(f'Probability result: {probabilities[0][prediction]}')