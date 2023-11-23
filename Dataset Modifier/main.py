import time

import pandas as pd
import json
import dataset_sources
import keyboard
import os


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


def main():

    # choose to combine all datasets or process separately
    print("Press '1' to combine all datasets or press '2' to process files separately")
    while True:
        if keyboard.is_pressed('1'):
            final_dataset = combine_datasets(
                dataset_sources.datasets,
                dataset_sources.dataset_fake_or_real,
                dataset_sources.dataset_urls
            )
            print(f"Combined dataset average wordsize: {get_avg_len(final_dataset)}")
            # save the combined dataset
            final_dataset.to_csv('combined_dataset.csv', index=False)
            print(f"Combined dataset saved as combined_dataset.csv with {len(final_dataset)} rows.")
            break
        elif keyboard.is_pressed('2'):
            formatted_datasets = format_datasets(
                dataset_sources.datasets,
                dataset_sources.dataset_fake_or_real,
                dataset_sources.dataset_urls
            )

            save_dir = '../JsonParser/formatted_datasets'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for i in range(len(formatted_datasets)):
                file_path = os.path.join(save_dir, dataset_sources.datasets[i])
                formatted_datasets[i].to_csv(file_path, index=False)
            break
        else:
            continue
        # to reduce CPU load
        time.sleep(0.01)


if __name__ == '__main__':
    main()
