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
        '--db_append', metavar='DB_APPEND', type=str_bool,
        help='If True, appends to existing database. If False, \
            constructs new database (deletes save directory',
        required=True)

    # Required
    parser.add_argument(
        '--db_save_dir', metavar='DB_SAVE_DIR',
        type=str, help='sub-directory of ./data/databases \
            to save Database instance files', required=True)

    # Required
    parser.add_argument(
        '--db_n_files', metavar='DB_N_FILES', default=100, type=int,
        help='Number of source corpus files to parse. Use \'-1\'\
            to parse all files', required=True)

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

    parser.add_argument(
        '--db_partition_size', metavar='PARTITION_SIZE',
        default=DB_PARAMS['partition_size'], help='Number of sentences \
            to save per partition of the Database instance',
        required=False)

    # ====================================================
    #      Parameters for loaded Language instances
    # ====================================================

    # Required
    parser.add_argument(
        '--lang_load_dir', metavar='LANGUAGE_DIR', type=str,
        help='sub-directory of ./data/languages where \
        Language dictionaries are saved', required=True)

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
        '--lang_character_prefix', metavar='LANG_CHARACTER_PREFIX',
        default=L_PARAMS['character_prefix'], type=str,
        help='Filename prefix of Language instance containing \
            character information', required=False)

    args = parser.parse_args()

    source_corpus_dir = os.path.join(
        DIRECTORIES['source_corpora'], args.corpus_dir)
    source_corpus_filetype = args.filetype

    database_save_dir = os.path.join(
        DIRECTORIES['databases'], args.db_save_dir)

    if not args.db_append and os.path.isdir(database_save_dir):
        shutil.rmtree(database_save_dir)

    db = databases.Database(
        args.db_form_char_prefix,
        args.db_form_char_len_prefix,
        args.db_max_sentence_length,
        args.db_max_token_length,
        args.db_sentence_len_prefix,
        args.db_syntactic_tag_prefix,
        args.db_token_char_prefix,
        args.db_token_char_len_prefix,
        args.db_token_prefix,
        partition_dir=database_save_dir)

    lang_load_dir = os.path.join(
        DIRECTORIES['languages'], args.lang_load_dir)

    lang_character_prefix = args.lang_character_prefix
    lang_token_prefix = args.lang_token_prefix
    lang_syntactic_tag_prefix = args.lang_syntactic_tag_prefix

    token_language, tag_languages, character_language = \
        languages.load_languages(lang_load_dir,
                                 lang_token_prefix,
                                 lang_syntactic_tag_prefix,
                                 lang_character_prefix)

    db.construct(character_language, token_language, tag_languages,
                 source_corpus_dir, source_corpus_filetype,
                 n_files=args.db_n_files,
                 partition_size=args.db_partition_size)
