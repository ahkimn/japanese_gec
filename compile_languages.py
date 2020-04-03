# -*- coding: utf-8 -*-

# Filename: compile_languages.py
# Date Created: 20/12/2020
# Description: Script to construct Language class instances for tokens
#   and MeCab syntactic tags from a given directory containing files
#   with correct Japanese sentences
# Python Version: 3.7

import argparse
import os

from src import config
from src import languages

cfg = config.parse()

D_PARAMS = cfg['data_params']
L_PARAMS = cfg['language_params']
P_PARAMS = cfg['parser_params']
DIRECTORIES = cfg['directories']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compile Languages')

    # ====================================================
    #    Parameters for constructed source corpus
    # ====================================================

    # Required
    parser.add_argument(
        '--corpus_dir', metavar='CORPUS_DIR',
        type=str, help='sub-directory of ./data/source_corpora \
            containing source corpus files', required=True)

    parser.add_argument(
        '--filetype', metavar='FILE_TYPE',
        default=D_PARAMS['source_corpus_filetype'],
        type=str, help='filetype of source corpus files',
        required=False)

    # ====================================================
    #    Parameters for constructed Language instances
    # ====================================================

    parser.add_argument(
        '--lang_token_prefix', metavar='LANG_TOKEN_PREFIX',
        default=L_PARAMS['token_prefix'], type=str,
        help='Filename prefix of Language instance containing \
            token information', required=False)

    parser.add_argument(
        '--lang_syntactic_tag_prefix', metavar='LANG_SYNTACTIC_TAG_PREFIX',
        default=L_PARAMS['syntactic_tag_prefix'], type=str,
        help='Filename prefix of Language instances containing \
            syntactic tag information', required=False)

    parser.add_argument(
        '--lang_save_dir', metavar='LANG_SAVE_DIR',
        type=str, help='sub-directory of ./data/languages \
            to save Language instance dictionaries', required=True)

    parser.add_argument(
        '--token_prefix', metavar='TOKEN_PREFIX',
        default=L_PARAMS['token_prefix'], type=str,
        help='Save filename prefix of Language instance containing \
            token information', required=False)

    parser.add_argument(
        '--syntactic_tag_prefix', metavar='SYNTACTIC_TAG_PREFIX',
        default=L_PARAMS['syntactic_tag_prefix'], type=str,
        help='Save filename prefix of Language instances containing \
            syntactic tag information', required=False)

    parser.add_argument(
        '--character_prefix', metavar='CHARACTER_PREFIX',
        default=L_PARAMS['character_prefix'], type=str,
        help='Save filename prefix of Language instances containing \
            character information', required=False)

    parser.add_argument(
        '--n_files', metavar='N_FILES', default=100, type=int,
        help='Number of source corpus files to parse. Use \'-1\'\
            to parse all files', required=False)

    args = parser.parse_args()

    source_corpus_dir = os.path.join(
        DIRECTORIES['source_corpora'], args.corpus_dir)
    source_corpus_filetype = args.filetype

    language_save_dir = os.path.join(
        DIRECTORIES['languages'], args.lang_save_dir)

    token_prefix = args.token_prefix
    syntactic_tag_prefix = args.syntactic_tag_prefix
    character_prefix = args.character_prefix
    n_files = args.n_files

    languages.compile_languages(
        source_corpus_dir, source_corpus_filetype,
        language_save_dir, token_prefix,
        syntactic_tag_prefix, character_prefix,
        n_files)
