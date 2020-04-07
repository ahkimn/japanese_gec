# -*- coding: utf-8 -*-

# Filename: construct_sorted_tag_databases.py
# Date Created: 01/22/2020
# Description: Script to construct instance of SortedTagDatabase class
#   given a Database instance
# Python Version: 3.7

import argparse
import os

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
        '--stdb_save_dir', metavar='STDB_SAVE_DIR',
        type=str, help='sub-directory of ./data/sorted_tag_databases \
            to save constructed SortedTagDatabase to', required=True)

    parser.add_argument(
        '--stdb_unique_token_prefix', metavar='STDB_UNIQUE_TOKEN_PREFIX',
        default=DB_PARAMS['unique_token_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            unique tokens', required=False)

    parser.add_argument(
        '--stdb_unique_syntactic_tag_prefix',
        metavar='STDB_UNIQUE_SYNTACTIC_TAG_PREFIX',
        default=DB_PARAMS['unique_syntactic_tag_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            unique syntactic tags (excluding root forms)', required=False)

    parser.add_argument(
        '--stdb_unique_form_prefix', metavar='STDB_UNIQUE_FORM_PREFIX',
        default=DB_PARAMS['unique_form_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            unique root form information', required=False)

    parser.add_argument(
        '--stdb_sort_tag_prefix', metavar='STDB_SORT_TAG_PREFIX',
        default=DB_PARAMS['sort_tag_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            sort order of unique syntactic tag matrix ', required=False)

    parser.add_argument(
        '--stdb_sort_form_prefix', metavar='STDB_SORT_FORM_PREFIX',
        default=DB_PARAMS['sort_form_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            sort order of unique form matrix', required=False)

    # ====================================================
    #       Parameters for loaded Database instance
    # ====================================================

   # Required
    parser.add_argument(
        '--db_load_dir', metavar='DB_LOAD_DIR',
        type=str, help='sub-directory of ./data/databases \
            to load Database instance from', required=True)

    # ====================================================

    parser.add_argument(
        '--db_form_char_prefix', metavar='DB_FORM_PREFIX',
        default=DB_PARAMS['form_char_prefix'], type=str,
        help='Filename prefix of database matrices containing \
            character information for each base form', required=False)

    parser.add_argument(
        '--db_form_char_len_prefix', metavar='DB_FORM_PREFIX',
        default=DB_PARAMS['form_char_len_prefix'], type=str,
        help='Filename prefix of database matrices containing \
            lengths (in characters) for each base form', required=False)

    parser.add_argument(
        '--db_sentence_len_prefix', metavar='DB_SENTENCE_LENGTH_PREFIX',
        default=DB_PARAMS['sentence_len_prefix'], type=str,
        help='Filename prefix of Database matrices containing \
            length of each sentence', required=False)

    parser.add_argument(
        '--db_syntactic_tag_prefix', metavar='DB_SYNTACTIC_TAG_PREFIX',
        default=DB_PARAMS['syntactic_tag_prefix'], type=str,
        help='Save filename prefix of Database matrices containing \
            syntactic tag information', required=False)

    parser.add_argument(
        '--db_token_char_prefix', metavar='DB_CHARACTER_PREFIX',
        default=DB_PARAMS['token_char_prefix'], type=str,
        help='Filename prefix of database matrices containing \
            character information for each token', required=False)

    parser.add_argument(
        '--db_token_char_len_prefix', metavar='DB_CHARACTER_PREFIX',
        default=DB_PARAMS['token_char_len_prefix'], type=str,
        help='Filename prefix of database matrices containing \
            lengths (in characters) for each token', required=False)

    parser.add_argument(
        '--db_token_prefix', metavar='DB_TOKEN_PREFIX',
        default=DB_PARAMS['token_prefix'], type=str,
        help='Filename prefix of Database matrices containing \
            token information', required=False)

    parser.add_argument(
        '--db_max_sentence_length', metavar='MAX_SENTENCE_LENGTH',
        default=DB_PARAMS['max_sentence_length'], help='Maximum number of \
            tokens (as parsed by MeCab) allowed in each Database sentence',
        required=False)

    parser.add_argument(
        '--db_max_token_length', metavar='MAX_SENTENCE_LENGTH',
        default=DB_PARAMS['max_token_length'], help='Maximum number of \
            characters in each token (as parsed by MeCab) of the Database \
            instance',
        required=False)

    args = parser.parse_args()

    # Initialize SortedTagDatabase instance
    stdb_save_dir = os.path.join(
        DIRECTORIES['sorted_tag_databases'], args.stdb_save_dir)

    stdb = SortedTagDatabase(stdb_save_dir,
                             args.stdb_unique_token_prefix,
                             args.stdb_unique_syntactic_tag_prefix,
                             args.stdb_unique_form_prefix,
                             args.stdb_sort_tag_prefix,
                             args.stdb_sort_form_prefix)

    # Load database instance
    database_load_dir = os.path.join(
        DIRECTORIES['databases'], args.db_load_dir)

    db = databases.Database(
        database_load_dir,
        form_char_prefix=args.db_form_char_prefix,
        form_char_len_prefix=args.db_form_char_len_prefix,
        max_sentence_length=args.db_max_sentence_length,
        max_token_length=args.db_max_token_length,
        sentence_len_prefix=args.db_sentence_len_prefix,
        syntactic_tag_prefix=args.db_syntactic_tag_prefix,
        token_char_prefix=args.db_token_char_prefix,
        token_char_len_prefix=args.db_token_char_len_prefix,
        token_prefix=args.db_token_prefix)

    stdb.construct(db)
