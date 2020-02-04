import argparse
import os
import shutil

from src import config
from src import databases
from src import languages
from src.util import str_bool

cfg = config.parse()

D_PARAMS = cfg['data_params']
L_PARAMS = cfg['language_params']
P_PARAMS = cfg['parser_params']
DB_PARAMS = cfg['database_params']
DIRECTORIES = cfg['directories']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Database Construction')

    # ====================================================
    #    Parameters for constructed source corpus files
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
    #         Parameters for constructed Database
    # ====================================================

    # Required
    parser.add_argument(
        '--append', metavar='APPEND', type=str_bool,
        help='If True, appends to existing database. If False, \
            constructs new database (deletes save directory',
        required=True)

    # Required
    parser.add_argument(
        '--save_dir', metavar='SAVE_DIR',
        type=str, help='sub-directory of ./data/databases \
            to save Database instance files', required=True)

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

    parser.add_argument(
        '--n_files', metavar='N_FILES', default=100, type=int,
        help='Number of source corpus files to parse. Use \'-1\'\
            to parse all files', required=False)

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

    args = parser.parse_args()

    source_corpus_dir = os.path.join(
        DIRECTORIES['source_corpora'], args.corpus_dir)
    source_corpus_filetype = args.filetype

    database_save_dir = os.path.join(
        DIRECTORIES['databases'], args.save_dir)

    print(args.append)

    if not args.append and os.path.isdir(database_save_dir):
        shutil.rmtree(database_save_dir)

    token_prefix = args.db_token_prefix
    syntactic_tag_prefix = args.db_syntactic_tag_prefix
    length_prefix = args.db_length_prefix
    n_files = args.n_files

    db = databases.Database(
        token_prefix,
        syntactic_tag_prefix,
        length_prefix,
        database_save_dir)

    lang_load_dir = os.path.join(
        DIRECTORIES['languages'], args.language_dir)

    lang_token_prefix = args.lang_token_prefix
    lang_syntactic_tag_prefix = args.lang_syntactic_tag_prefix

    token_language, tag_languages = \
        languages.load_languages(lang_load_dir,
                                 lang_token_prefix,
                                 lang_syntactic_tag_prefix)

    db.construct(token_language, tag_languages, source_corpus_dir,
                 source_corpus_filetype, n_files)
