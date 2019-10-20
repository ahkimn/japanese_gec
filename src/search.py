# Filename: convert.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 10/10/2019
# Date Last Modified: 10/10/2019
# Python Version: 3.7

import csv
import numpy as np

from . import convert
from . import configx
from . import languages
from . import process


def _matrices(text, token_tagger, pos_taggers, n_max=50):

    matrices = dict()

    n_pos_taggers = len(pos_taggers)
    n_sentences = len(text)

    token_matrix = np.zeros((n_sentences, n_max), dtype='uint32')
    form_matrix = np.zeros((n_sentences, n_max), dtype='uint32')
    pos_matrix = np.zeros(
        (n_sentences, n_max, n_pos_taggers - 1), dtype='uint8')
    lengths = np.zeros(n_sentences, dtype='uint8')

    n_processed = 0

    for i in range(n_sentences):

        sentence = text[i]

        nodes, pos = languages.parse_full(sentence, configx.CONST_PARSER, None)

        indices = token_tagger.parse_sentence(nodes)
        n_nodes = len(indices)

        if n_nodes > n_max:
            raise

        lengths[n_processed] = 0

        token_matrix[n_processed, :n_nodes] = indices[:]
        form_indices = pos_taggers[-1].parse_sentence(pos[-1])
        form_matrix[n_processed, :n_nodes] = form_indices[:]

        for i in range(n_pos_taggers - 1):

            pos_matrix[n_processed, :n_nodes,
                       i] = pos_taggers[i].parse_sentence(pos[i])

        lengths[n_processed] = n_nodes

        n_processed += 1

    token_matrix = token_matrix[:n_processed]
    form_matrix = form_matrix[:n_processed]
    pos_matrix = pos_matrix[:n_processed]
    lengths = lengths[:n_processed]

    pos_matrix = np.moveaxis(pos_matrix, -1, 0)

    matrices['token'] = token_matrix
    matrices['form'] = form_matrix
    matrices['pos'] = pos_matrix
    matrices['lengths'] = lengths

    return matrices


def _confirm_error(match_data, rule_data, correct_matrices, error_matrices,
                   orrect_phrases, error_phrases):

    raise

    print("SHIT SHIT SHI SDFSSDFPIU O UIOSIUDFOISUDFIO IO")


def match_parallel_text_rules(input_source, input_target, input_start, rules_file):

    token_tagger, pos_taggers = languages.load_default_languages()
    attribute_indices = [0, 1, 4, 5, 6]

    print("\nLoading token database...")
    print(configx.BREAK_LINE)

    f = open(rules_file, 'r')
    csv_reader = csv.reader(f, delimiter=',')

    iter_count = 0

    rules = list()
    unique_matrices = convert.load_unique_matrices(
        configx.CONST_DEFAULT_DATABASE_DIRECTORY, pos_taggers)

    error_phrases, correct_phrases = process.get_paired_phrases(
        token_tagger, input_source, input_target, input_start)

    print(correct_phrases[20])
    print(error_phrases[20])

    correct_matrices = _matrices(correct_phrases, token_tagger, pos_taggers)
    error_matrices = _matrices(error_phrases, token_tagger, pos_taggers)

    for rule in csv_reader:

        iter_count += 1

        if iter_count == 1:

            continue

        rule_dict = convert.get_rule_info(rule, pos_taggers)
        possible_classes = convert.match_rule_templates(
            rule_dict, unique_matrices)

        pos_tags = rule_dict['pos']
        selections = rule_dict['selections']

        matched \
            = convert.match_template_sentence(correct_matrices, pos_tags, selections, possible_classes,
                                              token_tagger, pos_taggers, 10000, -1, randomize=False)

        data = _confirm_error(matched, rule_dict,
                              correct_matrices, error_matrices,
                              correct_phrases, error_phrases)

        rules.append(rule_dict)

    print("\nFinished loading token databases...")
    print(configx.BREAK_LINE)
