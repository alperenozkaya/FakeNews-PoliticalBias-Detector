import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import re


def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {}
    current_embedding = None

    for i, line in enumerate(lines):
        if 'dataset name:' in line and i > 0:
            current_embedding = lines[i - 1].strip()
            data[current_embedding] = {}

        elif 'precision:' in line and current_embedding:
            epoch = int(re.search(r'epoch (\d+)', line).group(1))
            precision = float(re.search(r'precision: ([0-9.]+)', line).group(1))

            precision_type = 'Train' if 'Train performance' in line else 'Test'
            column_name = f'{precision_type} Precision Epoch {epoch}'
            data[current_embedding][column_name] = precision


    return data


# function to create a DataFrame from the extracted data
def create_dataframe(data):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)
    return df


file_path = 'turkish_best_results.txt'
data = extract_data_from_file(file_path)
df = create_dataframe(data)
print(df)
df # colab


def plot_all_models_test_precision(dataframe):
    plt.figure(figsize=(12, 8))

    models = dataframe['Model'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))  # Color map for distinct colors

    for model, color in zip(models, colors):
        # Extract the relevant columns for the model
        model_data = dataframe[dataframe['Model'] == model]
        test_precision_cols = [col for col in model_data if col.startswith('Test Precision')]

        # Extract epochs and corresponding precision values
        epochs = [int(col.split()[-1]) for col in test_precision_cols]
        precision_values = [model_data.iloc[0][col] for col in test_precision_cols]

        # Plotting
        plt.plot(epochs, precision_values, marker='o', color=color, label=model)

    plt.title('Test Precision per Epoch for All Embeddings')
    plt.xlabel('Epoch')
    plt.ylabel('Test Precision')
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function to plot the chart
plot_all_models_test_precision(df)