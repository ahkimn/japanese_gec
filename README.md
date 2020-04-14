# General Setup

1. Change the variable PROJECT_DIR in src/config.py to match the filepath to the root of your local project directory.
2. Edit the filepath listed under cfg['parser_params']['dictionary_dir'] to match the dictionary directory of Mecab on your local machine.
3. Run *src/config.py* as follows:
	```console
	python src/config.py
	```

# Data Synthesis Setup

To setup and run the data synthesis pipeline the following scripts must be executed:

1. Run the script *compile_languages.py*:
  - This script creates several Language instances into a sub-directory of *data/languages/* folder
  - Requires as input a sub-directory of *data/source_corpora/* containing text files
  - Example:
	  ```console
	  python compile_languages.py --corpus_dir sample --lang_save_dir del --n_files 100
	  ```
2. Run the script *construct_databases.py*
  - This tokenizes the text contained in a sub-directory of *data/source_corpora* and then converts the tokens using the previously constructed Language instances
  - This also saves a Database instance containing the resulting data into a sub-directory of *data/databases/*
  - Example:
  	  ```console
	  python construct_databases.py --corpus_dir sample --db_append False --db_save_dir del --db_n_files 100 --lang_load_dir del
	  ```
3. Run the script *construct_sorted_tag_databases.py* from the root directory
  - This creates token/syntactic tag lookup tables from a saved Database instance and saves results as a SortedTagDatabase instance into a sub-directory of *data/sorted_tag_databases/*
  - Example:
  	  ```console
	  python construct_sorted_tag_databases.py --stdb_save_dir del --db_load_dir del
	  ```
4. Run the script *gen_synthetic_data.py* from the root directory
  - Requires a file containing syntatic rule information to create a RuleList instance
  - This then iterates over the Rule instances of the RuleList
    - For each Rule, the script searches the saved Database instance for sentences matching the correct template phrase of the Rule
  - The script then generates error sentences from each matched correct sentence using saved Language instances and a saved SortedTagDatabase instance
  - Finally, the script saves the results (paired error/correct sentences) as a set of Dataset instances (one per each rule) into a sub-directory of *data/datasets/*
  - Example:
  	  ```console
	  python gen_synthetic_data.py --language_dir del --rule_file full.csv --gen_rule -1 --stdb_load_dir del --db_load_dir del --ds_save_dir del --override True --manual_check False
      ```
5. Run the script *merge_dataset.py* from the root directory
  - This merges the Dataset instances within a given directory and saves the resulting merged Dataset instance into a different sub-directory of *data/datasets/*
  - Example:
  	  ```console
	  python merge_dataset.py --ds_load_dir del --ds_merge_dir del_merged --ds_merge_name ex
      ```
6. Run the script *split_dataset.py* from the root directory
  - This automatically performs a balanced sampling of training, development, and testing data from a Dataset instance
  - The split data is saved into three new Dataset instances in a user-defined directory
    ```console
    python split_dataset.py --ds_load_dir del_merged --ds_name ex --ds_split_dir del_split
    ```

# Model Training and Inference

1. Run the script *train_model.py* from the root directory
  - Requires three Dataset instances (one for each of training/validation/testing) as input
  - The script trains a model in the following fashion:
    - The script first writes each Dataset's paired data to .correct and .error files in a temporary directory
    - It invokes fairseq-preprocess on the .correct and .error files
    - It then invokes fairseq-generate on the preprocessed data and extracts the model hypothesis from the output
    - Deletes all but the best model epoch and saves the model to the *./models/* directory
  - Example:
    ```console
    python train_model.py --ds_load_dir del_split --ds_name_train syn_train --ds_name_dev syn_dev --ds_name_test syn_test --command all --model_arch fconv_jp_mini --model_save_dir del
    ```
2. Run the script *classify_dataset.py* from the root directory
  - Requires a trained FConv model instance (e.g. a model saved in *./models/model/* would require the flag '--model_load_dir model')
  - Also takes as input either a saved Dataset instance (with flag '--command ds_generate') or a saved file instance (with flag '--command file_generate')
    - Input files can be either paired or unpaired and may also be pre-tokenized
    - Parsing options are controlled with various flags include ('--tokenized', and '--sentence_delimiter')
  - Writes model corrections to the *./data/model_output/* directory
  - For example, for an untokenized file with path *./models/test_corpora/test.txt*, and a model in *./models/model/* the following command could be used:
    ```console
    python classify_dataset.py --language_dir del --stdb_load_dir del --ds_load_dir teacher --ds_name test --rule_file full.csv
    ```

# Dataset Testing

1. Run the script *import_dataset.py* from the root directory
  - Creates a new Dataset instance from a set of error/correct sentence pairs or imports a new column of data to an existing Dataset
    - For former option, requires file containing error/correct sentence pairs (with flag and option '--ds_action create')
      - Sentences may have annotations, and can be pre-tokenized
    - For latter option, requires file containing unpaired sentences (with flag and option '--ds_action' import) and an existing Dataset
  - For example, to create a new Dataset instance at *./data/datasets/teacher/test.ds* from the annotated file *./data/test_corpora/teacher.txt*, the following command could be used:
    ```console
    python import_dataset.py --ds_dir teacher --ds_name test --ds_action create --file_dir data/test_corpora --file_name teacher.txt --annotated True --sentence_delimiter "\t"
    ```

2. Run the script *clasify_dataset.py* from the root directory
  - Requires an existing Dataset instance and a rule file
  - This assigns rule labels to the correct/error phrase pairs of the Dataset according to which rules of the rule file each pair is matched by
  - This also displays rule coverage (e.g. the number of sentence pairs/unique error phrases covered by each rule)
  - Example:










# Dataset Manipulation

There are several scripts in the root directory used to manipulated saved Dataset instances. These are as follows:

- *write_dataset.py* - extracts sentence pairs stored in Dataset instances and writes them into readable text files
- *split_dataset.py* - splits a Dataset instance into training/development/test datasets using balanced sampling based on subrules
- *interact_dataset.py* - loads a Dataset instance and allows user to perform simple analytical applications (e.g. data sampling)
- *merge_dataset.py* - merges Dataset instances in a given directory
- *import_dataset.py* - contains two primary functions:
  - creates a new dataset instance from a text file containing paired sentence data
    - text file can contain annotations
  - imports unpaired data from a text file into an existing dataset under a specified column name (e.g. model results)
- *classify_dataset.py* - determines and updates rule coverage on a Dataset instance given a rule file

# Model Manipulation

The following scripts involve the training and evaluation of fairseq FConv models:

- *train_model.py* - trains a new FConv model given a saved Dataset instance and outputs the best checkpoint to the *models/* directory
  - the major operations of the file are split into different 'commands' as each individual operation may take a significant amount of time
- *gen_model_output.py* - takes a trained FConv model and one of two possible inputs:
  - a text file containing error sentences (can be paired or unpaired)
  - a saved Dataset instance
and writes model corrections to a readable file.
