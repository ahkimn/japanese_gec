import argparse
import os
import shutil

from src import config
from src import kana
from src import languages
from src import rules
from src.sorted_tag_database import SortedTagDatabase
from src.datasets import Dataset

cfg = config.parse()
L_PARAMS = cfg['language_params']
M_PARAMS = cfg['morpher_params']
DB_PARAMS = cfg['database_params']
DS_PARAMS = cfg['dataset_params']
DIRECTORIES = cfg['directories']
SAVE_PARAMS = DS_PARAMS['save_names']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Write output from dataset')

    # ====================================================
    #     Parameters for loaded Language instances
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
    #     Parameters for loaded Dataset instance
    # ====================================================

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

    # ====================================================
    #               Parameters for kana file
    # ====================================================

    parser.add_argument('--kana_file', metavar='KANA_FILE',
                        default=M_PARAMS['kana_default'],
                        type=str, help='File within data/const containing organized \
                    list of kana', required=False)

    # ====================================================
    #           Parameters for loaded rule file
    # ====================================================

    # REQUIRED
    parser.add_argument(
        '--rule_file', metavar='RULE_FILE', type=str,
        help='CSV file within ./data/rules containing rule information',
        required=True)

    # ====================================================
    #                   Other parameters
    # ====================================================

    parser.add_argument(
        '--tmp_dir', metavar='TMP_DIR', type=str, default=DIRECTORIES['tmp'],
        help='sub-directory of (./data/tmp/) to write temporary files to',
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

    # Load SortedTagDatabase instance
    stdb_load_dir = os.path.join(
        DIRECTORIES['sorted_tag_databases'], args.stdb_load_dir)

    STDB = SortedTagDatabase(stdb_load_dir,
                             args.stdb_unique_token_prefix,
                             args.stdb_unique_syntactic_tag_prefix,
                             args.stdb_unique_form_prefix,
                             args.stdb_sort_tag_prefix,
                             args.stdb_sort_form_prefix)

    kana_file = os.path.join(DIRECTORIES['const'], args.kana_file)
    KL = kana.KanaList(kana_file)

    RL = rules.RuleList(rule_file, character_language, token_language,
                        tag_languages, KL=KL)

    ds_load_file = os.path.join(DIRECTORIES['datasets'],
                                args.ds_load_dir, '%s.%s' %
                                (args.ds_name, args.ds_suffix))

    assert(os.path.isfile(ds_load_file))

    DS = Dataset.load(ds_load_file)

    tmp_db_dir = os.path.join(args.tmp_dir, 'db')

    if os.path.isdir(tmp_db_dir):
        shutil.rmtree(tmp_db_dir)

    DS.classify(character_language, token_language, tag_languages,
                RL=RL, KL=KL, STDB=STDB, tmp_db_dir=tmp_db_dir)
