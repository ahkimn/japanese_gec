# -*- coding: utf-8 -*-

# Filename: gen_synthetic_data.py
# Date Created: 23/01/2020
# Description: Script to generate synthesized error-correct Japanese sentence
#   pairs given compiled Language class instances, and corresponding
#   Database and SortedTagDatabase instances. Results are stored per-rule in
#   separate Dataset instances in a specified folder
# Python Version: 3.7

import argparse
import os
import shutil

from src import config
from src import databases
from src import generate
from src import kana
from src import languages
from src import match
from src import rules
from src import util

from src.datasets import Dataset
from src.sorted_tag_database import SortedTagDatabase
from src.util import str_bool

import numpy as np

cfg = config.parse()

D_PARAMS = cfg['data_params']
L_PARAMS = cfg['language_params']
M_PARAMS = cfg['morpher_params']
P_PARAMS = cfg['parser_params']
DB_PARAMS = cfg['database_params']
DS_PARAMS = cfg['dataset_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Database Construction')

    # ====================================================
    #      Parameters for loaded Language instances
    # ====================================================

    # Required
    parser.add_argument(
        '--language_dir', metavar='LANGUAGE_DIR', type=str,
        help='sub-directory of ./data/languages where \
        Language dictionaries are saved', required=True)

    parser.add_argument(
        '--lang_token_prefix', metavar='LANG_TOKEN_PREFIX',
        default=L_PARAMS['token_prefix'], type=str,
        help='Filename prefix of saved Language instance containing \
            token information', required=False)

    parser.add_argument(
        '--lang_syntactic_tag_prefix', metavar='LANG_SYNTACTIC_TAG_PREFIX',
        default=L_PARAMS['syntactic_tag_prefix'], type=str,
        help='Filename prefix of Language instances containing \
            syntactic tag information', required=False)

    parser.add_argument(
        '--lang_character_prefix', metavar='LANG_CHARACTER_PREFIX',
        default=L_PARAMS['character_prefix'], type=str,
        help='Filename prefix of Language instance containing \
            character information', required=False)

    # ====================================================
    #           Parameters for loaded rule file
    # ====================================================

    # REQUIRED
    parser.add_argument(
        '--rule_file', metavar='RULE_FILE', type=str,
        help='CSV file within ./data/rules containing rule information',
        required=True)

    # REQUIRED
    parser.add_argument(
        '--gen_rule', metavar='GEN_RULE', type=str, default='-1',
        help='Specific rule to generate data for. Use \'-1\'\
            to generate using all rules', required=True)

    # ====================================================
    #    Parameters for loaded SortedTagDatabase instance
    # ====================================================

    # Required
    parser.add_argument(
        '--stdb_load_dir', metavar='STDB_LOAD_DIR',
        type=str, help='sub-directory of ./data/sorted_tag_databases \
            to load SortedTagDatabase instance from', required=True)

    parser.add_argument(
        '--stdb_unique_token_prefix', metavar='UNIQUE_TOKEN_PREFIX',
        default=DB_PARAMS['unique_token_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            unique tokens', required=False)

    parser.add_argument(
        '--stdb_unique_syntactic_tag_prefix',
        metavar='UNIQUE_SYNTACTIC_TAG_PREFIX',
        default=DB_PARAMS['unique_syntactic_tag_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            unique syntactic tags (excluding root forms)', required=False)

    parser.add_argument(
        '--stdb_unique_form_prefix', metavar='UNIQUE_FORM_PREFIX',
        default=DB_PARAMS['unique_form_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            unique root form information', required=False)

    parser.add_argument(
        '--stdb_sort_tag_prefix', metavar='SORT_TAG_PREFIX',
        default=DB_PARAMS['sort_tag_prefix'], type=str,
        help='Filename prefix of SortedTagDatabase matrix containing \
            sort order of unique syntactic tag matrix ', required=False)

    parser.add_argument(
        '--stdb_sort_form_prefix', metavar='SORT_FORM_PREFIX',
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

    # ====================================================
    #               Parameters for kana file
    # ====================================================

    parser.add_argument('--kana_file', metavar='KANA_FILE',
                        default=M_PARAMS['kana_default'],
                        type=str, help='File within data/const containing organized \
                    list of kana', required=False)

    # ====================================================
    #          Parameters for saved Dataset files
    # ====================================================

    # Required
    parser.add_argument(
        '--ds_save_dir', metavar='SAVE_DIR',
        type=str, help='sub-directory of ./data/datasets \
            to save data to', required=True)

    # Required
    parser.add_argument(
        '--override', metavar='OVERRIDE',
        type=str_bool, help='if True, delete files within save directory prior to \
            data generation', required=True)

    # Required
    parser.add_argument(
        '--manual_check', metavar='MANUAL_CHECK', default=False,
        type=str_bool, help='if True, allow manual checking of synthesized \
        data', required=True)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='File extension of saved Dataset instances',
        required=False)

    parser.add_argument(
        '--ds_prefix', metavar='DS_PREFIX',
        default=SAVE_PARAMS['rule_file_prefix'], type=str,
        help='prefix preceding rule name in filenames of saved Dataset \
            instances. If empty string no prefix is used.', required=False)

    # ====================================================
    #               Other parameters
    # ====================================================

    parser.add_argument(
        '--suppress_errors', metavar='SUPPRESS',
        default=False, type=str_bool,
        help='if True, suppress any exceptions raised during generation',
        required=False)

    args = parser.parse_args()

    rule_file = os.path.join(DIRECTORIES['rules'], args.rule_file)

    lang_load_dir = os.path.join(
        DIRECTORIES['languages'], args.language_dir)

    lang_token_prefix = args.lang_token_prefix
    lang_syntactic_tag_prefix = args.lang_syntactic_tag_prefix
    lang_character_prefix = args.lang_character_prefix

    token_language, tag_languages, character_language = \
        languages.load_languages(lang_load_dir,
                                 lang_token_prefix,
                                 lang_syntactic_tag_prefix,
                                 lang_character_prefix)

    unk_token = token_language.unknown_token

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

    # Load SortedTagDatabase instance
    stdb_load_dir = os.path.join(
        DIRECTORIES['sorted_tag_databases'], args.stdb_load_dir)

    stdb = SortedTagDatabase(stdb_load_dir,
                             args.stdb_unique_token_prefix,
                             args.stdb_unique_syntactic_tag_prefix,
                             args.stdb_unique_form_prefix,
                             args.stdb_sort_tag_prefix,
                             args.stdb_sort_form_prefix)

    kana_file = os.path.join(DIRECTORIES['const'], args.kana_file)
    KL = kana.KanaList(kana_file)

    rl = rules.RuleList(rule_file, character_language, token_language,
                        tag_languages, KL=KL)

    print('\nBeginning data synthesis')
    print(cfg['BREAK_LINE'])

    RS = np.random.RandomState(seed=0)

    save_dir = os.path.join(DIRECTORIES['datasets'], args.ds_save_dir)
    ds_suffix = args.ds_suffix

    if args.override and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    if not os.path.isdir(save_dir):

        util.mkdir_p(save_dir)

    n_rule = -1

    for rule, idx in rl.iterate_rules(args.gen_rule):

        n_rule += 1
        print('\n\n')
        rl.print_rule(idx)
        print(cfg['BREAK_LINE'])

        try:

            matches = match.match_correct(rule, db, stdb, RS=RS)
            gen_error, gen_correct, gen_error_bounds, gen_correct_bounds, \
                gen_rules, gen_subrules, _ = \
                generate.generate_synthetic_pairs(stdb, token_language,
                                                  tag_languages, rule, matches,
                                                  KL=KL)

            DS = Dataset.import_data(gen_error, gen_correct,
                                     gen_error_bounds, gen_correct_bounds,
                                     gen_rules, gen_subrules)

        except ValueError:

            if args.suppress_error:
                continue

            else:
                raise

        if args.manual_check:

            print('\n\tManual data check enabled')
            print(cfg['BREAK_SUBLINE'])

            print('Rule: %s' % str(rule))

            DS.sample_rule_data(rule.name, RS=RS)

            validate = ''

            while validate != 'n' and validate != 'y':

                validate = \
                    input('\n\tWould you like to save this data? (y/n): ')

            if validate == 'n':

                continue

        ds_prefix = args.ds_prefix
        if ds_prefix != '':
            ds_prefix += '_'

        ds_save = os.path.join(save_dir, '%s%s.%s' %
                               (ds_prefix, rule.name, args.ds_suffix))

        DS.save(ds_save)
