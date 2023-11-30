""" To flip labels in WELFake database """

import pandas as pd

file_name = 'datasets/WELFake_Dataset.csv'

df = pd.read_csv(file_name)
df_new = pd.DataFrame(columns=['title', 'text', 'label'], index=range(len(df)))
for i in range(len(df)):
    df_new['title'][i] = df['title'][i]
    df_new['text'][i] = df['text'][i]
    if df['label'][i] == 0:
        df_new['label'][i] = 1
    else:
        df_new['label'][i] = 0

df_new.to_csv(file_name, index=False)