# FakeNews-PoliticalBias-Detector
 


## PRE-RUN CONFIGURATIONS:
To run the project, first configure the dataset_modifier.json file inside of the config file.
Params for the file:

        download_from_gdrive: whether or not install the dataset files from google drive.
        combine_datasets: whether or not to combine datasets into one csv file
        formatted_csv_save_dir: path to write new data after dataset modifier is completed
        input_dir: path to read for parsing csv dataset to JSON and tokenize
        output_dir: path for tokenized json file to write
        dataset_name: dataset to split into train/test/validate

## RUNNING THE APPLICATION:
after the configuration is done, run main.py inside the dataset modifier directory. here, dataset modifier will chop off unneccessary data from the dataset and write the new data into the path provided in formatted_csv_save_dir

then, you'll need to run tokenize_n_json.py and split_train_test_validate.py files inside the JsonParser directory respectively. (assuming you've put the correct parameter you desire into the config file)

##### lastly, to train the model, you'll need to run the train.py inside the NLPClassfierTool directory. to run it, first we need to configure another config file. to run:
        python train.py /conf/train.json
config file is located @ conf/train.json and this file is needed to be provided as a command line argument to the python terminal. here, you can configure training parameters such as model details, epochs, optimizers & loss funcitons etc.
if all dependencies are correctly installed train.py should run smoothly.

# IMPORTANT NOTICE: 
The classification tool we use is the Tencent's NeuralNLP classifier model which is licensed under MIT license. For more information:
https://github.com/Tencent/NeuralNLP-NeuralClassifier

