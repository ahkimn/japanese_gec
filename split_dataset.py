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

    parser = argparse.ArgumentParser(
        description='Split Dataset instance into training, \
            development, and testing sets')

    # Required
    parser.add_argument(
        '--ds_load_dir', metavar='DS_LOAD_DIR',
        type=str, help='sub-directory of ./data/datasets \
            where Dataset instances are saved', required=True)
    # Required
    parser.add_argument(
        '--ds_name', metavar='DS_NAME',
        default='', type=str,
        help='name of dataset instance to load.', required=True)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='file extension of saved Dataset instances',
        required=False)

    # Required
    parser.add_argument(
        '--ds_split_dir', metavar='DS_SPLIT_DIR', type=str,
        help='sub-directory of ./data/datasets to save split Dataset \
            instances to',
        required=True)

    args = parser.parse_args()

    parser.add_argument(
        '--ds_name_train', metavar='DS_NAME_TRAIN', type=str,
        default='%s_%s' % (args.ds_name, SAVE_PARAMS['train_suffix']),
        help='filename of split Dataset instance containing training data',
        required=False)

    parser.add_argument(
        '--ds_name_dev', metavar='DS_NAME_DEV', type=str,
        default='%s_%s' % (args.ds_name, SAVE_PARAMS['dev_suffix']),
        help='filename of split Dataset instance containing development data',
        required=False)

    parser.add_argument(
        '--ds_name_test', metavar='DS_NAME_TEST', type=str,
        default='%s_%s' % (args.ds_name, SAVE_PARAMS['test_suffix']),
        help='filename of split Dataset instance containing testing data',
        required=False)

    parser.add_argument(
        '--ds_split_train', metavar='DS_SPLIT_TRAIN', type=float,
        default=0.90, help='percentage of sampled data given to Dataset \
            instance containing training data')

    parser.add_argument(
        '--ds_split_dev', metavar='DS_SPLIT_DEV', type=float,
        default=0.05, help='percentage of sampled data given to Dataset \
            instance containing validation data')

    args = parser.parse_args()
    ds_load_path = os.path.join(DIRECTORIES['datasets'],
                                args.ds_load_dir, '%s.%s' %
                                (args.ds_name, args.ds_suffix))

    assert(os.path.isfile(ds_load_path))

    ds_split_dir = os.path.join(DIRECTORIES['datasets'],
                                args.ds_split_dir)

    if not os.path.isdir(ds_split_dir):
        util.mkdir_p(ds_split_dir)

    print('\nLoading dataset Instance at: %s' % ds_load_path)
    print(cfg['BREAK_LINE'])

    ds_save_train = os.path.join(
        ds_split_dir, '%s.%s' % (args.ds_name_train, args.ds_suffix))
    ds_save_dev = os.path.join(
        ds_split_dir, '%s.%s' % (args.ds_name_dev, args.ds_suffix))
    ds_save_test = os.path.join(
        ds_split_dir, '%s.%s' % (args.ds_name_test, args.ds_suffix))

    if os.path.isfile(ds_save_train):
        os.remove(ds_save_train)
    if os.path.isfile(ds_save_dev):
        os.remove(ds_save_dev)
    if os.path.isfile(ds_save_test):
        os.remove(ds_save_test)

    DS = Dataset.load(ds_load_path)

    DS_train, DS_dev, DS_test = \
        DS.split(args.ds_split_train, args.ds_split_dev)

    DS_train.save(ds_save_train)
    DS_dev.save(ds_save_dev)
    DS_test.save(ds_save_test)
