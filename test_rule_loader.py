import argparse
import os

from src import config
from src import kana
from src import languages
from src import rules

cfg = config.parse()

L_PARAMS = cfg['language_params']
M_PARAMS = cfg['morpher_params']
DIRECTORIES = cfg['directories']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Database Construction')

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

    parser.add_argument(
        '--rule_index', metavar='RULE_INDEX', type=str,
        help='Number/name of rule within rule file', required=True)

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

    kana_file = os.path.join(DIRECTORIES['const'], args.kana_file)
    KL = kana.KanaList(kana_file)

    rl = rules.RuleList(rule_file, character_language, token_language,
                        tag_languages, KL=KL)

    # Example of how to get specific rule from RuleList instance
    rl.print_rule(args.rule_index)
