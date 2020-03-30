import argparse
import os

from src import config
from src.datasets import Dataset

cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Hiragana/Katakana List')

    # Required
    parser.add_argument(
        '--ds_load_dir', metavar='DS_LOAD_DIR',
        type=str, help='sub-directory of ./data/datasets \
            where Dataset instances are saved', required=True)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='File extension of saved Dataset instances',
        required=False)

    parser.add_argument(
        '--ds_prefix', metavar='DS_PREFIX',
        default=SAVE_PARAMS['rule_file_prefix'], type=str,
        help='prefix preceding rule name in filenames of saved Dataset \
            instances. If empty string no prefix is used.', required=False)

    # Required
    parser.add_argument(
        '--write_dir', metavar='WRITE_DIR', type=str,
        help='sub-directory of ./data/datasets to write split dataset to',
        required=True)

    parser.add_argument(
        '--token_delimiter', metavar='TOKEN_DELIMITER', type=str,
        help='delimiter between tokens of output files', default=' ')

    parser.add_argument(
        '--data_delimiter', metavar='DATA_DELIMITER', type=str,
        help='delimiter between data entries of output files', default=',')

    parser.add_argument(
        '--write_separation', metavar='WRITE_SEPARATION', type=str,
        help='granularity of output files. Must be one of \'%s\' \'%s\', \
            or \'%s\''
        % ('rule', 'subrule', 'none'), default='none')

    parser.add_argument(
        '--max_per_rule', metavar='MAX_PER_RULE', type=int,
        help='maximum number of sentences to output per rule. If value is \
            \'-1\' all sentences of rule are outputted.', default=-1)

    args = parser.parse_args()
    ds_load_dir = os.path.join(DIRECTORIES['datasets'],
                               args.ds_load_dir)

    args = parser.parse_args()

    DS = Dataset.merge_directory(ds_load_dir, args.ds_prefix,
                                 args.ds_suffix)

    write_dir = os.path.join(DIRECTORIES['synthesized_data'],
                             args.write_dir)

    DS.write(write_dir, token_delimiter=args.token_delimiter,
             data_delimiter=args.data_delimiter, include_tags=[],
             separation=args.write_separation, max_per_rule=args.max_per_rule)
