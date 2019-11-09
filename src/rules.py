from . import configx
from . import languages

import ast
import csv
import numpy as np


def _get_rule_info(rule_text, pos_taggers):

    n_pos = len(pos_taggers)
    rule_dict = dict()

    # Paired sentence data
    corrected_sentence = rule_text[0]
    error_sentence = rule_text[1]

    rule_string = "%s --> %s" % \
        (corrected_sentence, error_sentence)

    # Retrieve unencoded part-of-speech tags of the correct sentence
    pos_tags = rule_text[2]
    pos_tags = pos_tags.split(',')

    # Convert part-of-speech tags to index form
    n_tokens = int(len(pos_tags) / n_pos)
    pos_tags = np.array(list(languages.parse_node_matrix(
        pos_tags[i * n_pos: i * n_pos + n_pos], pos_taggers) for
        i in range(n_tokens)))

    # Array of arrays denoting hows part-of-speech tags have been selected
    # This is marked as -1 = null, 0 = no match, 1 = match
    selections = rule_text[3]
    selections = np.array(list(int(j) for j in selections.split(',')))
    selections = selections.reshape(-1, n_pos)

    # Arrays of tuples denoting token mappings between errored and correct
    #   sentence
    created = rule_text[4]
    altered = rule_text[5]
    preserved = rule_text[6]

    # Convert string representations to lists
    created = ast.literal_eval(created)
    altered = ast.literal_eval(altered)
    preserved = ast.literal_eval(preserved)

    # Aggregate mapping into single tuple
    mapping = (created, altered, preserved)

    rule_dict['correct'] = corrected_sentence
    rule_dict['error'] = error_sentence
    rule_dict['pos'] = pos_tags

    rule_dict['str'] = rule_string

    rule_dict['selections'] = selections
    rule_dict['mapping'] = mapping
    rule_dict['n_tokens'] = n_tokens

    return rule_dict


def parse_rule_file(rule_file, pos_taggers, rule_index=-1, ignore_first=True):

    rules = list()

    line_count = -1
    rule_count = 0

    # Process rule file
    with open(rule_file, 'r') as f:

        csv_reader = csv.reader(f, delimiter=',')
        # Read each line (rule) of CSV
        for rule_text in csv_reader:

            line_count += 1

            if line_count == 0 and ignore_first:

                continue

            elif len(rule_text) > 2 and rule_text[0] != '#':

                rule_count += 1

                if rule_count == rule_index or rule_index == -1:

                    rule_dict = _get_rule_info(rule_text, pos_taggers)
                    rules.append(rule_dict)

    f.close()

    return rules
