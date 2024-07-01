
import json

data_path ='../JsonParser/formatted_datasets_json/shuffled_dataset_to_analyze.json'

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def calculate_average_word_count(data):
    number_of_words = 0
    for i in range(len(data)):
        number_of_words += len(data[i]['doc_token'])

    return number_of_words / len(data)

def calculate_rates(data_labels):

    reals = 0
    fakes = 0
    for label in data_labels:
        if label == 'real':
            reals += 1
        elif label == 'fake':
            fakes += 1

    return fakes/(fakes + reals) * 100



data_to_analyze = load_data(data_path)

data_labels = []

for data in data_to_analyze:
    data_labels += data['doc_label']


avg_words = calculate_average_word_count(data_to_analyze)
avg_rates = calculate_rates(data_labels)


print(f'Average word count: {avg_words} \nFake/Real rates: {avg_rates}')