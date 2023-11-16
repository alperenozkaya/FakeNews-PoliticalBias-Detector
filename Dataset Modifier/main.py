import pandas as pd
import json
import dataset_sources


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
        df['Label'] = df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'fake' else 1)
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
    df = df[required_columns].dropna(subset=['Label'])

    return df


# main function to combine datasets
def combine_datasets(datasets, dataset_fake_or_real, dataset_urls):
    combined_df = pd.DataFrame(columns=['Title', 'Text', 'Label', 'Resource'])

    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        label_type = dataset_fake_or_real.get(dataset_name, 'both')
        processed_df = process_file(dataset_name, label_type, dataset_urls)
        print(f"Finished processing {dataset_name}, shape: {processed_df.shape}")
        combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        print(f"Combined dataframe shape now: {combined_df.shape}")

    return combined_df


def main():
    final_dataset = combine_datasets(
        dataset_sources.datasets,
        dataset_sources.dataset_fake_or_real,
        dataset_sources.dataset_urls
    )

    # save the combined dataset
    final_dataset.to_csv('combined_dataset.csv', index=False)
    print(f"Combined dataset saved as combined_dataset.csv with {len(final_dataset)} rows.")


if __name__ == '__main__':
    main()
