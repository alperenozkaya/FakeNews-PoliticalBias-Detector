import matplotlib.pyplot as plt
import numpy as np
import regex

test_metrics_outside = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'loss': []
}
test_metrics_current_model = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'loss': []
}

test_metrics_tf = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'loss': []
}

with open('tranfsformer_tests.txt', 'r') as file:
    data = file.read()


data = data.lower()
raw_data_newline = regex.split(r'\n', data)

for datum in raw_data_newline:
    test_metrics_outside['accuracy'] += regex.findall(r'accuracy: (\d+\.\d+)', datum)
    test_metrics_outside['precision'] += regex.findall(r'precision: (\d+\.\d+)', datum)
    test_metrics_outside['recall'] += regex.findall(r'recall: (\d+\.\d+)', datum)
    test_metrics_outside['f1'] += regex.findall(r'f1 score: (\d+\.\d+)', datum)
    test_metrics_outside['loss'] += regex.findall(r'loss: (\d+\.\d+)', datum)

with open('logs.txt', 'r') as file:
    data_logs = file.read()

data_logs = data_logs.lower()
raw_data_logs = regex.split(r'\n', data_logs)

for datum in raw_data_logs:
    test_metrics_current_model['precision'] += regex.findall(r'precision: (\d+\.\d+)', datum)
    test_metrics_current_model['recall'] += regex.findall(r'recall: (\d+\.\d+)', datum)
    test_metrics_current_model['f1'] += regex.findall(r'fscore: (\d+\.\d+)', datum)
    test_metrics_current_model['loss'] += regex.findall(r'loss: (\d+\.\d+)', datum)




with open('tf_logs.txt', 'r') as file:
    data_tf = file.read()

data_tf = data_tf.lower()
raw_data_tf = regex.split(r'\n', data_tf)

for datum in raw_data_tf:
    test_metrics_tf['precision'] += regex.findall(r'precision: (\d+\.\d+)', datum)
    test_metrics_tf['recall'] += regex.findall(r'recall: (\d+\.\d+)', datum)
    test_metrics_tf['f1'] += regex.findall(r'fscore: (\d+\.\d+)', datum)
    test_metrics_tf['loss'] += regex.findall(r'loss: (\d+\.\d+)', datum)


precision_outside = test_metrics_outside['precision']
precision_current_model = test_metrics_current_model['precision']
precision_tf = test_metrics_tf['precision']

precision_outside_float = [float(precision) for precision in precision_outside]
precision_current_model_float = [float(precision) for precision in precision_current_model]
precision_tf_float = [float(precision) for precision in precision_tf]


# Assuming the lengths of all precision data lists are the same
num_measurements = len(precision_outside_float)

# Creating x-ticks starting from 1
xticks = list(range(1, num_measurements + 1))

# Creating the comparative graph for precision
plt.figure(figsize=(10, 6))
plt.plot(xticks, precision_outside_float, label='Precision of BiLSTM Model', marker='o')
plt.plot(xticks, precision_current_model_float, label='Precision of Current Model', marker='x')
# plt.plot(xticks, precision_tf_float, label='Precision of Transformer Model', marker='*')

plt.title('Performance comparison of BiLSTM and Default wrt precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.xticks(xticks)  # Setting the x-ticks
plt.legend()
plt.grid(True)
plt.show()


