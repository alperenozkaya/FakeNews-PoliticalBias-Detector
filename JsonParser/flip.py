import pandas as pd

file_name = 'formatted_datasets/WELFake_Dataset.csv'

df = pd.read_csv(file_name)

for i in range (len(df)):
    if df['Label'][i] == 0:
        df['Label'][i] = 1
    else:
        df['Label'][i] = 0

df.to_csv('formatted_datasets/WELFake_Dataset.csv', index=False)