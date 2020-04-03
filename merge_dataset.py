import argparse
import os

from src import config
from src import util
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
        '--ds_name_filter', metavar='DS_NAME_FILTER',
        default='', type=str,
        help='identifying string for filenames of saved Dataset \
            instances. If empty string no prefix is used.', required=False)

    # Required
    parser.add_argument(
        '--ds_merge_dir', metavar='DS_MERGE_DIR', type=str,
        help='sub-directory of ./data/datasets to save merged Dataset to',
        required=True)

    # Required
    parser.add_argument(
        '--ds_merge_name', metavar='DS_MERGE_NAME', type=str,
        default='merge', help='filename of saved merge Dataset instance',
        required=True)

    args = parser.parse_args()
    ds_load_dir = os.path.join(DIRECTORIES['datasets'],
                               args.ds_load_dir)

    ds_merge_dir = os.path.join(DIRECTORIES['datasets'],
                                args.ds_merge_dir)

    if not os.path.isdir(ds_merge_dir):
        util.mkdir_p(ds_merge_dir)

    print('\nLoading datasets from: %s' % ds_load_dir)
    print(cfg['BREAK_LINE'])

    DS = Dataset.merge_directory(ds_load_dir, args.ds_name_filter,
                                 args.ds_suffix)

    ds_save_path = os.path.join(ds_merge_dir, '%s.%s' %
                                (args.ds_merge_name, args.ds_suffix))

    print('Saving merged dataset to: %s' % ds_save_path)

    DS.save(ds_save_path)
