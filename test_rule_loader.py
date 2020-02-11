import argparse
import os

from src import config
from src import languages
from src import rules

cfg = config.parse()

L_PARAMS = cfg['language_params']
DIRECTORIES = cfg['directories']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Database Construction')

    # REQUIRED
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

    token_language, tag_languages = \
        languages.load_languages(lang_load_dir,
                                 lang_token_prefix,
                                 lang_syntactic_tag_prefix)

    rl = rules.RuleList(rule_file, token_language, tag_languages)

    # Example of how to get specific rule from RuleList instance
    rl.print_rule(args.rule_index)
