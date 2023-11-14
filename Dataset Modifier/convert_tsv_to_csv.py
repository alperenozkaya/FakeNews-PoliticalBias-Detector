import pandas as pd

print('Enter tsv file name to convert to csv:')
csv_table = pd.read_table(input(), sep='\t')

print('Enter the name for the converted csv file:')
csv_table.to_csv(input(), index=False)

