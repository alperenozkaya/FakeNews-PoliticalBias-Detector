
import pandas as pd

file_name = 'combined_data_w3.csv'

df = pd.read_csv(file_name)

df['Text'] = df['title'] + ' ' + df['content']
df.rename(columns={'alignment': 'Label'}, inplace=True)

clean_df = df[['Text', 'Label']]
clean_df.to_csv('cleaned_' + file_name, index=False)