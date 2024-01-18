import json
import random


def reduce_json_file(file_path, percentage):

    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]


    number_to_keep = int(len(data) * (percentage / 100))

    # randomly select elements
    reduced_data = random.sample(data, number_to_keep)

    new_file_path = file_path.split('.')[0] + f'_reduced_{percentage}percent.json'
    with open(new_file_path, 'w') as file:
        for item in reduced_data:
            file.write(json.dumps(item))
            file.write("\n")

    return new_file_path

file_path = 'combined_dataset_no_stop_words.json'  # Replace with the path to your JSON file
percentage = 10
new_file_path = reduce_json_file(file_path, percentage)
print(f"Reduced file created at: {new_file_path}")
