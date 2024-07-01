import pandas as pd
import json

data = pd.read_csv('formatted_datasets/combined_dataset.csv', encoding='utf-8')

#remove duplicates
data.drop_duplicates(subset=['Title', 'Text'], inplace=True)

label_counts = data['Label'].value_counts()

total_samples = len(data)

label_rates = label_counts / total_samples
print(total_samples)
print(label_rates)