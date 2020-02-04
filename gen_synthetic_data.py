import argparse
import os

from src import config
from src import databases
from src import generate
from src import languages
from src import match
from src import rules
from src.sorted_tag_database import SortedTagDatabase

import numpy as np

cfg = config.parse()

D_PARAMS = cfg['data_params']
L_PARAMS = cfg['language_params']
P_PARAMS = cfg['parser_params']
DB_PARAMS = cfg['database_params']
DIRECTORIES = cfg['directories']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Database Construction')

    # ====================================================
    #    Parameters for loaded source corpus files
    # ====================================================

    parser.add_argument(
        '--corpus_dir', metavar='CORPUS_DIR', default='scrape',
        type=str, help='sub-directory of ./data/source_corpora \
            containing source corpus files', required=False)

    parser.add_argument(
        '--filetype', metavar='FILE_TYPE',
        default=D_PARAMS['source_corpus_filetype'],
        type=str, help='filetype of source corpus files',
        required=False)

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

    # ====================================================
    #           Parameters for loaded rule file
    # ====================================================

    # REQUIRED
    parser.add_argument(
        '--rule_file', metavar='RULE_FILE', type=str,
        help='CSV file within ./data/rules containing rule information',
        required=True)

    parser.add_argument(
        '--gen_rule', metavar='GEN_RULE', type=int, default=-1,
        help='Specific rule to generate data for. Use \'-1\'\
            to generate using all rules', required=False)

    # ====================================================
    #    Parameters for loaded SortedTagDatabase instance
    # ====================================================

    # Required
    parser.add_argument(
        '--stdb_load_dir', metavar='STDB_LOAD_DIR',
        type=str, help='sub-directory of ./data/sorted_tag_databases \
            to load SortedTagDatabase instance from', required=True)

    parser.add_argument(
        '--stdb_unique_token_prefix', metavar='STDB_UNIQUE_TOKEN_PREFIX',
        default=DB_PARAMS['unique_token_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            unique tokens', required=False)

    parser.add_argument(
        '--stdb_unique_syntactic_tag_prefix',
        metavar='STDB_UNIQUE_SYNTACTIC_TAG_PREFIX',
        default=DB_PARAMS['unique_syntactic_tag_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            unique syntactic tags (excluding root forms)', required=False)

    parser.add_argument(
        '--stdb_unique_form_prefix', metavar='STDB_UNIQUE_FORM_PREFIX',
        default=DB_PARAMS['unique_form_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            unique root form information', required=False)

    parser.add_argument(
        '--stdb_sort_tag_prefix', metavar='STDB_SORT_TAG_PREFIX',
        default=DB_PARAMS['sort_tag_prefix'], type=str,
        help='Save filename prefix of SortedTagDatabase matrix containing \
            sort order of unique syntactic tag matrix ', required=False)

    parser.add_argument(
        '--stdb_sort_form_prefix', metavar='STDB_SORT_FORM_PREFIX',
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

    # ====================================================
    #               Parameters for saved files
    # ====================================================

    # Required
    parser.add_argument(
        '--save_dir', metavar='SAVE_DIR',
        type=str, help='sub-directory of ./data/synthesized \
            to save data to', required=True)

    parser.add_argument(
        '--manual_check', metavar='MANUAL_CHECK', default=False,
        type=bool, help='if True, allow manual checking of synthesized \
        data')

    args = parser.parse_args()

    rule_file = os.path.join(DIRECTORIES['rules'], args.rule_file)

    lang_load_dir = os.path.join(
        DIRECTORIES['languages'], args.language_dir)

    lang_token_prefix = args.lang_token_prefix
    lang_syntactic_tag_prefix = args.lang_syntactic_tag_prefix

    token_language, tag_languages = \
        languages.load_languages(lang_load_dir,
                                 lang_token_prefix,
                                 lang_syntactic_tag_prefix)
    unk_token = token_language.unknown_token

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

    # Load SortedTagDatabase instance
    stdb_load_dir = os.path.join(
        DIRECTORIES['sorted_tag_databases'], args.stdb_load_dir)

    unique_token_prefix = args.stdb_unique_token_prefix
    unique_syntactic_tag_prefix = args.stdb_unique_syntactic_tag_prefix
    unique_form_prefix = args.stdb_unique_form_prefix

    sort_tag_prefix = args.stdb_sort_tag_prefix
    sort_form_prefix = args.stdb_sort_form_prefix

    stdb = SortedTagDatabase(stdb_load_dir,
                             unique_token_prefix,
                             unique_syntactic_tag_prefix,
                             unique_form_prefix,
                             sort_tag_prefix, sort_form_prefix)

    rl = rules.RuleList(rule_file, tag_languages)

    print('\nBeginning data synthesis')
    print(cfg['BREAK_LINE'])

    RS = np.random.RandomState(seed=0)

    for rule, idx in rl.iterate_rules(args.gen_rule):

        rl.print_rule(idx)
        print(cfg['BREAK_LINE'])

        matches = match.match_correct(rule, db, stdb, RS=RS)
        paired_sentences, paired_starts = \
            generate.generate_synthetic_pairs(stdb, token_language,
                                              tag_languages, rule, matches)

        if args.manual_check:

            print('\n\tManual data check enabled')
            print(cfg['BREAK_SUBLINE'])

            generate.sample_data(rule, paired_sentences, paired_starts, RS=RS)

            validate = ''

            while validate != 'n' and validate != 'y':

                validate = \
                    input('\n\tWould you like to save this data? (y/n): ')

            if validate == 'n':

                continue

        save_dir = os.path.join(DIRECTORIES['synthesized_data'], args.save_dir,
                                str(rule.number))

        generate.save_synthetic_sentences(
            paired_sentences, paired_starts, save_dir, unknown=unk_token)
