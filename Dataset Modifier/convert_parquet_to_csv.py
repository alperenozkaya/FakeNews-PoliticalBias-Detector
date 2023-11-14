import pandas as pd

print('Enter parquet file name to convert to csv:')
parquet_file = pd.read_parquet(input())

print('Enter the name for the converted csv file:')
parquet_file.to_csv(input())