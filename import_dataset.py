import argparse
import os

from src import config
from src import convert
from src import util
from src.datasets import Dataset

from src.util import str_bool, str_list, literal_str

cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import sentences into an existing dataset \
            or create new Dataset instance from sentences')

    # ====================================================
    #      Parameters for Dataset to import data into
    # ====================================================

    # Required
    parser.add_argument(
        '--ds_dir', metavar='DS_LOAD_DIR',
        type=str, help='sub-directory of ./data/datasets \
            where created Dataset instance is to be saved', required=True)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='File extension of saved Dataset instances',
        required=False)

    parser.add_argument(
        '--ds_name', metavar='DS_NAME',
        default='', type=str,
        help='filename of saved dataset instance.', required=True)

    parser.add_argument(
        '--ds_action', metavar='DS_ACTION',
        default='create', type=str, help='one of \'create\' or \'import\'. \
            If \'create\', data from the file is used to create a new Dataset \
            instance. If \'import\', data is added as a new column in an \
            existing Dataset instance', required=True)

    parser.add_argument(
        '--import_name', metavar='IMPORT_NAME', type=str, default=None,
        help='Column name to import data under')

    # ====================================================
    #           Parameters for loaded file
    # ====================================================

    # Required for file_generate
    parser.add_argument(
        '--file_name', metavar='FILE_NAME',
        type=str, help='full filename (including extension) of file \
            within directory specified by --file_dir', required=True)

    parser.add_argument(
        '--file_dir', metavar='FILE_DIR', type=str,
        default=DIRECTORIES['test_corpora'], help='path to directory containing \
        file to parse')

    parser.add_argument(
        '--annotated', metavar='ANNOTATED',
        type=str_bool, help='True if the file to be loaded is annotated \
            (e.g. contains delimiters for error and correct phrases',
        default=False, required=False)

    parser.add_argument(
        '--tokenized', metavar='TOKENIZED',
        type=str_bool, help='True if the file to be loaded is already \
            tokenized', default=False, required=False)

    parser.add_argument(
        '--token_delimiter', metavar='TOKEN_DELIMITER',
        type=literal_str, help='delimiter between tokens in file if the file is \
            tokenized', default=' ', required=False)

    parser.add_argument(
        '--sentence_delimiter', metavar='SENTENCE_DELIMITER',
        type=literal_str, help='delimiter between sentences in file if the file \
            contains sentence pairs', default=',', required=False)

    parser.add_argument(
        '--error_delimiters', metavar='ERROR_DELIMITERS',
        type=str_list, default=DS_PARAMS['error_delimiters'],
        help='list of pair of characters that delimit error phrases in \
            annotated text file', required=False)

    parser.add_argument(
        '--correct_delimiters', metavar='CORRECT_DELIMITERS',
        type=str_list, default=DS_PARAMS['correct_delimiters'],
        help='list of pair of characters that delimit correct phrases in \
            annotated text file', required=False)

    parser.add_argument(
        '--error_first', metavar='ERROR_FIRST',
        type=str_bool, help='for files with error/correct pairs. If True, the error \
            sentence is the first sentence of the pair', default=True,
        required=False)

    args = parser.parse_args()
    ds_file = os.path.join(DIRECTORIES['datasets'],
                           args.ds_dir, '%s.%s' %
                           (args.ds_name, args.ds_suffix))

    load_file = os.path.join(args.file_dir, args.file_name)

    if args.annotated:

        err_sentences, crt_sentences, err_bounds, crt_bounds = \
            convert.process_annotated_file(
                load_file,
                sentence_delimiter=args.sentence_delimiter,
                error_delimiters=args.error_delimiters,
                correct_delimiters=args.correct_delimiters,
                error_first=args.error_first)

    else:

        err_sentences, crt_sentences, err_bounds, crt_bounds = \
            convert.process_file(
                load_file,
                token_delimiter=args.token_delimiter,
                sentence_delimiter=args.sentence_delimiter,
                tokenized=args.tokenized,
                error_first=args.error_first)

    assert(os.path.isfile(load_file))

    if args.ds_action == 'create':

        if not os.path.isfile(ds_file):
            util.mkdir_p(ds_file, file=True)
            print('Creating new dataset instance at: %s' % ds_file)

        else:
            print('WARNING: Dataset at: %s already exists' % ds_file)
            override = input('Override dataset? (y/n): ').lower()
            if override != 'y':
                exit()

        DS = Dataset.import_data(err_sentences, crt_sentences,
                                 err_bounds, crt_bounds)

    elif args.ds_action == 'import':

        DS = Dataset.load(ds_file)

        print('Importing data into existing dataset instance at: %s' % ds_file)

        assert(args.import_name is not None)

        DS.import_columns(err_sentences, args.import_name)

    else:
        raise ValueError('Argument \'--ds_action\' must be one of \
                \'create\' or \'import\'')

    print('Showing first 10 columns of created Dataset:')
    print(DS.df.iloc[:10])

    response = None
    while response != 'y' and response != 'n':
        response = input('Save dataset (y/n): ').lower()

    if response == 'y':
        DS.save(ds_file)

