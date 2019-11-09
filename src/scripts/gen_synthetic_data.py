from .. import configx
from .. import convert
from .. import database
from .. import generate
from .. import languages
from .. import rules
from .. import save

import os

RULE_FILE_DIRECTORY = 'rules'


def gen_synthetic_data(database_dir, save_name, rule_file_name, rule_index=-1,
                       max_search=-1, max_token=50000, max_per_rule=50000,
                       min_per_rule=5000, rule_out_ratio=0.1,
                       corpus_save_name='tmp', pause_gen=False):

    token_tagger, pos_taggers = languages.load_languages(database_dir)

    print("\nLoading token database...")
    print(configx.BREAK_LINE)

    # Load matrices necessary for sentence generation
    search_matrices = database.load_search_matrices(database_dir, pos_taggers)
    unique_matrices = database.load_unique_matrices(database_dir, pos_taggers)

    print("\nFinished loading token databases...")
    print(configx.BREAK_LINE)

    # Load rule file
    rule_file = os.path.join(RULE_FILE_DIRECTORY, '%s.csv' % rule_file_name)

    idx = 1

    for rule_dict in rules.parse_rule_file(rule_file, pos_taggers, rule_index):

        corrected_sentence = rule_dict['correct']
        error_sentence = rule_dict['error']
        pos_tags = rule_dict['pos']

        print('\nReading Rule %d: %s' % (idx, rule_dict['str']))
        print(configx.BREAK_LINE)

        selections = rule_dict['selections']

        print("\n\tFinding potential substitute tokens...")
        print(configx.BREAK_SUBLINE)

        # List of possible substitute token classes
        #   (part-of-speech combinations) per each index of correct sentence
        # as defined by the selections matrix
        possible_classes = convert.match_rule_templates(
            rule_dict, unique_matrices, max_token)

        print("\n\tSearching for sentences matching pair template...")
        print(configx.BREAK_SUBLINE)
        matched = convert.match_template_sentence(
            search_matrices, pos_tags, selections, possible_classes,
            token_tagger, pos_taggers,
            n_search=max_search, n_token=max_token, n_max_out=max_per_rule,
            n_min_out=min_per_rule, out_ratio=rule_out_ratio)

        print("\n\tGenerating new sentence pairs...")
        print(configx.BREAK_SUBLINE)
        paired_data, starts = generate.create_errored_sentences(
            unique_matrices, matched, token_tagger, pos_taggers,
            rule_dict)

        # TODO: Insert code that will make (correct, correct) sentence pairings
        if pause_gen:

            print('\n\tManual data check enabled')
            print(configx.BREAK_SUBLINE)

            print('')
            validate = ''

            while validate != 'n' and validate != 'y':

                validate = input('\tWould you like to save this data? (y/n): ')

            if validate == 'y':

                print("\n\tSaving new data...")
                save.save_rule(corrected_sentence, error_sentence,
                               paired_data, starts, idx, save_prefix=save_name)

                print("\n\tPaired data saved successfully...\n")

        else:

            print("\n\tSaving new data...")
            save.save_rule(corrected_sentence, error_sentence,
                           paired_data, starts, idx, save_prefix=save_name)

            print("\n\tPaired data saved successfully...\n")

        idx += 1


def gen_synthetic_data_default():

    database_dir = 'database/full'
    save_name = 'updated_final'
    rule_file_name = 'new'
    gen_synthetic_data(database_dir=database_dir, save_name=save_name,
                       rule_file_name=rule_file_name, pause_gen=False,
                       rule_index=145)
