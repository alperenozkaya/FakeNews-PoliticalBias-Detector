import sys
import time

import pandas as pd
import json
import dataset_sources
import os
from config import Config

import gdown


# function to extract label from prediction specifically in argilla.csv
def extract_label_from_prediction(prediction):
    try:
        prediction_dict = json.loads(prediction.replace("'", "\""))
        if isinstance(prediction_dict, list) and 'label' in prediction_dict[0]:
            return 0 if prediction_dict[0]['label'] == 'fake' else 1
    except json.JSONDecodeError:
        pass
    return None


# function to process csv files
def process_file(file_name, label_type, dataset_urls):
    file_path = f'datasets/{file_name}'

    # read csv file
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except (UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Error reading {file_name}: {e}")
        return pd.DataFrame()

    # if the dataset has only one type of news, set the label accordingly
    if label_type in ['fake', 'real']:
        df['Label'] = 0 if label_type == 'fake' else 1
    elif 'label' in df.columns:
        # the cases where the label is a string 'Real' or 'Fake'
        df['Label'] = df['label'].astype(str).apply(lambda x: 0 if x.strip().lower() in ['0', 'fake'] else 1)
    elif 'prediction' in df.columns:  # for argilla.csv
        df['Label'] = df['prediction'].apply(extract_label_from_prediction)
    else:
        print(f"No label column found in {file_name}")
        df['Label'] = None

    # standardize the 'Title' column, fill empty if it doesn't exist
    df['Title'] = df.get('title', '')

    # standardize the 'Text' column, fill empty if it doesn't exist
    df['Text'] = df.get('text', '')

    # add the 'Resource' column using the dataset urls dictionary
    df['Resource'] = dataset_urls.get(file_name, '')

    # select and rename columns
    required_columns = ['Title', 'Text', 'Label', 'Resource']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    # drop row if no label
    df = df[required_columns].dropna(subset=['Label'])

    return df


# function to get average length of text in a dataset
def get_avg_len(df):
    word_length = 0
    count = 0
    for text in df['Text']:
        if pd.isna(text):
            count += 1
            continue
        word_list = text.split()
        word_length += len(word_list)
        count += 1

    return word_length / count


# main function to combine datasets
def combine_datasets(datasets, dataset_fake_or_real, dataset_urls):
    combined_df = pd.DataFrame(columns=['Title', 'Text', 'Label', 'Resource'])

    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        label_type = dataset_fake_or_real.get(dataset_name, 'both')
        processed_df = process_file(dataset_name, label_type, dataset_urls)
        
        avg_len = get_avg_len(processed_df)
        print(f'Average text length for {processed_df["Resource"][0]}: {avg_len}')
        
        print(f"Finished processing {dataset_name}, shape: {processed_df.shape}")
        combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        print(f"Combined dataframe shape now: {combined_df.shape}")

    return combined_df


# function to format csv files separately
def format_datasets(datasets, dataset_fake_or_real, dataset_urls):
    formatted_df = pd.DataFrame(columns=['Title', 'Text', 'Label', 'Resource'])
    formatted_datasets = []

    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        label_type = dataset_fake_or_real.get(dataset_name, 'both')
        processed_df = process_file(dataset_name, label_type, dataset_urls)

        avg_len = get_avg_len(processed_df)
        print(f'Average text length for {processed_df["Resource"][0]}: {avg_len}')

        print(f"Finished processing {dataset_name}, shape: {processed_df.shape}")
        formatted_datasets.append(processed_df)
        print(f"Formatted dataset shape now: {formatted_df.shape}")

    return formatted_datasets





def download_datasets_from_google_drive():
    datasets_drive_url = 'https://drive.google.com/drive/folders/1ZsHH4J7MaShACaPUOg6O1tlbbVziatS8?usp=drive_link'
    if datasets_drive_url.split('/')[-1] == '?usp=drive_link':
        datasets_drive_url = datasets_drive_url.replace('?usp=drive_link', '')

    gdown.download_folder(datasets_drive_url)
    print('')


def main():


    # download csv datasets from gdrive
    if config.dataset_modifier.download_from_gdrive is True:
        download_datasets_from_google_drive()

    save_dir = config.dataset_modifier.formatted_csv_save_dir
    # whether to combine all datasets or process separately
    if config.dataset_modifier.combine_datasets is True:
        final_dataset = combine_datasets(
            dataset_sources.datasets,
            dataset_sources.dataset_fake_or_real,
            dataset_sources.dataset_urls
        )
        print(f"Combined dataset average wordsize: {get_avg_len(final_dataset)}")
        # save the combined dataset
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, 'combined_dataset.csv')
        final_dataset.to_csv(file_path, index=False)
        print(f"Combined dataset saved as combined_dataset.csv with {len(final_dataset)} rows.")
    else:
        formatted_datasets = format_datasets(
            dataset_sources.datasets,
            dataset_sources.dataset_fake_or_real,
            dataset_sources.dataset_urls
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(len(formatted_datasets)):
            file_path = os.path.join(save_dir, dataset_sources.datasets[i])
            formatted_datasets[i].to_csv(file_path, index=False)


if __name__ == '__main__':
    config = Config(config_file='../config/dataset_modifier.json')
    main()
