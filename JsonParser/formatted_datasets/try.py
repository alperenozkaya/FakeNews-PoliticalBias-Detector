import pandas as pd

df_path = 'cleaned_combined_data_w3.csv'

df = pd.read_csv(df_path)

df = df.dropna()

df = df.drop_duplicates()
df = df.reset_index(drop=True)

df.to_csv('combined_data_w3_dds.csv', index=False)