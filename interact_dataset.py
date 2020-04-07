import argparse
import os

from src import config
from src.datasets import Dataset

cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']
VALID_COMMANDS = ['rules',
                  'sample',
                  'subrules',
                  'exit']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Write output from dataset')

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
        '--ds_name', metavar='DS_PREFIX',
        default=SAVE_PARAMS['rule_file_prefix'], type=str,
        help='filename (preceding extension) of saved Dataset instance',
        required=True)

    args = parser.parse_args()
    ds_load_file = os.path.join(DIRECTORIES['datasets'],
                                args.ds_load_dir, '%s.%s' %
                                (args.ds_name, args.ds_suffix))

    assert(os.path.isfile(ds_load_file))

    DS = Dataset.load(ds_load_file)

    status = True

    print('Commands:')
    print('\tsample: Sample a rule')
    print('\trules: Show statistics for all rules')
    print('\tsubrules: Show subrule statistics for a particular rule')
    print('\texit: Exit script')

    print(cfg['BREAK_LINE'])

    while status:

        command = ''

        while command not in VALID_COMMANDS:
            print('Please enter command:\n')
            command = input().lower()
            print()

        if command == 'rules':

            DS.print_rule_stats()

        elif command == 'sample':

            print('Please enter rule to sample from:')
            rule = input()
            print()

            DS.sample_rule_data(rule)

        elif command == 'subrules':

            print('Please enter rule to sample from:')
            rule = input()
            print()

            DS.print_subrule_stats(rule)

        else:

            status = False

        print()
