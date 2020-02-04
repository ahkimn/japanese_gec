import argparse
import os
import shutil

from src import config
from src import databases
from src.sorted_tag_database import SortedTagDatabase

cfg = config.parse()

D_PARAMS = cfg['data_params']
DB_PARAMS = cfg['database_params']
DIRECTORIES = cfg['directories']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sort Array Construction')

    # ====================================================
    #      Parameters for SortedTagDatabase instance
    # ====================================================

    # Required
    parser.add_argument(
        '--save_dir', metavar='SAVE_DIR',
        type=str, help='sub-directory of ./data/sorted_tag_databases \
            to save constructed SortedTagDatabase to', required=True)

    parser.add_argument(
        '--unique_token_prefix', metavar='UNIQUE_TOKEN_PREFIX',
        default=DB_PARAMS['unique_token_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            unique tokens', required=False)

    parser.add_argument(
        '--unique_syntactic_tag_prefix', metavar='UNIQUE_SYNTACTIC_TAG_PREFIX',
        default=DB_PARAMS['unique_syntactic_tag_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            unique syntactic tags (excluding root forms)', required=False)

    parser.add_argument(
        '--unique_form_prefix', metavar='UNIQUE_FORM_PREFIX',
        default=DB_PARAMS['unique_form_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            unique root form information', required=False)

    parser.add_argument(
        '--sort_tag_prefix', metavar='SORT_TAG_PREFIX',
        default=DB_PARAMS['sort_tag_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            sort order of unique syntactic tag matrix ', required=False)

    parser.add_argument(
        '--sort_form_prefix', metavar='SORT_FORM_PREFIX',
        default=DB_PARAMS['sort_form_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            sort order of unique form matrix', required=False)

    # ====================================================
    #       Parameters for loaded Database instance
    # ====================================================

    # Required
    parser.add_argument(
        '--db_load_dir', metavar='DB_LOAD_DIR',
        type=str, help='sub-directory of ./data/databases \
            to load Database instance from', required=True)

    parser.add_argument(
        '--db_length_prefix', metavar='DB_LENGTH_PREFIX',
        default=DB_PARAMS['length_prefix'], type=str,
        help='Save filename prefix of Database matrices containing \
            length of each sentence', required=False)

    parser.add_argument(
        '--db_token_prefix', metavar='DB_TOKEN_PREFIX',
        default=DB_PARAMS['token_prefix'], type=str,
        help='Save filename prefix of Database matrices containing \
            token information', required=False)

    parser.add_argument(
        '--db_syntactic_tag_prefix', metavar='DB_SYNTACTIC_TAG_PREFIX',
        default=DB_PARAMS['syntactic_tag_prefix'], type=str,
        help='Save filename prefix of Database matrices containing \
            syntactic tag information', required=False)

    args = parser.parse_args()

    # Initialize SortedTagDatabase instance
    stdb_save_dir = os.path.join(
        DIRECTORIES['sorted_tag_databases'], args.save_dir)

    unique_token_prefix = args.unique_token_prefix
    unique_syntactic_tag_prefix = args.unique_syntactic_tag_prefix
    unique_form_prefix = args.unique_form_prefix

    sort_tag_prefix = args.sort_tag_prefix
    sort_form_prefix = args.sort_form_prefix

    stdb = SortedTagDatabase(stdb_save_dir,
                             unique_token_prefix,
                             unique_syntactic_tag_prefix,
                             unique_form_prefix,
                             sort_tag_prefix, sort_form_prefix)

    # Load database instance
    database_load_dir = os.path.join(
        DIRECTORIES['databases'], args.db_load_dir)

    token_prefix = args.db_token_prefix
    syntactic_tag_prefix = args.db_syntactic_tag_prefix
    length_prefix = args.db_length_prefix

    db = databases.Database(
        token_prefix,
        syntactic_tag_prefix,
        length_prefix,
        database_load_dir)

    stdb.construct(db)
