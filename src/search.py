# Filename: convert.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 10/10/2019
# Date Last Modified: 10/10/2019
# Python Version: 3.7

import csv
import numpy as np
import os
import re

from . import convert
from . import configx
from . import languages
from . import process
from . import replace
from . import util


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

    # for i in form_matrix:
    #     for j in i:

    #         if j != 0:
    #             print(pos_taggers[-1].parse_index(j))

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


# TODO CREATED
# Maybe do string-based in future
#   parser-agnostic (but much much slower)
def _confirm_error(token_tagger, pos_taggers,
                   match_data, rule_data, unique_matrices,
                   correct_matrices, error_matrices,
                   correct_phrases, error_phrases,
                   possible_classes):

    form_pos = unique_matrices['form_dict']

    correct_phrase = rule_data['correct']
    error_phrase = rule_data['error']

    nodes_correct, pos_correct = languages.parse_full(
        correct_phrase, configx.CONST_PARSER, None)
    nodes_error, pos_error = languages.parse_full(
        error_phrase, configx.CONST_PARSER, None)

    pos_correct = np.array(list(
        languages.parse_node_matrix(pos_token, pos_taggers)
        for pos_token in np.array(pos_correct).T))

    pos_error = np.array(list(
        languages.parse_node_matrix(pos_token, pos_taggers)
        for pos_token in np.array(pos_error).T))

    indices_error = token_tagger.parse_sentence(nodes_error)

    mapping = rule_data['mapping']
    selections = rule_data['selections']

    created, altered, preserved = mapping
    n_correct = len(selections)
    n_error = len(created) + len(altered) + len(preserved)

    indices = match_data['indices']
    forms = match_data['forms']
    pos = np.moveaxis(match_data['pos'], 0, -1)
    starts = match_data['starts']

    correct_token = correct_matrices['token'][indices]
    error_token = error_matrices['token'][indices]
    error_form = error_matrices['form'][indices]
    error_pos = np.moveaxis(error_matrices['pos'], 0, -1)[indices]
    error_len = error_matrices['lengths'][indices]

    n_pos = pos.shape[2]
    n_matches = len(forms)

    assert(n_pos + 1 == len(selections[0]))

    template_correct = np.zeros((n_matches, n_correct, n_pos + 1),
                                dtype=error_form.dtype)
    template_error = np.zeros(
        (n_matches, n_error, n_pos + 1), dtype=error_form.dtype)

    template_correct_token = np.zeros((n_matches, n_correct),
                                      dtype=error_form.dtype)
    template_error_token = np.zeros((n_matches, n_error),
                                    dtype=error_form.dtype)
    token_selections = np.zeros(n_error, dtype=np.bool)
    token_ignore = list()

    bin_selections = selections > 0
    error_selections = np.zeros((n_error, n_pos + 1), dtype=np.bool)

    error_search = np.zeros(
        (error_pos.shape[0], error_pos.shape[1], error_pos.shape[2] + 1),
        dtype=error_form.dtype)

    error_search[:, :, :-1] = error_pos
    error_search[:, :, -1] = error_form

    for i in range(n_matches):

        s = starts[i]
        template_correct[i, :, :-1] = pos[i, s:s + n_correct]
        template_correct[i, :, -1] = forms[i, s:s + n_correct]
        template_correct_token[i, :] = correct_token[i, s:s + n_correct]

    for i in range(len(altered)):

        m = altered[i]
        template_error[:, m[0], :] = template_correct[:, m[1], :]

        error_selections[m[0], :] = bin_selections[m[1], :]
        error_selections[m[0], -1] = True

        epos = pos_error[m[0]]
        cpos = pos_correct[m[1]]

        not_ = (epos == cpos)

        for i in range(len(not_)):

            if not_[i]:

                continue

            if error_selections[m[0]][i]:

                template_error[:, m[0], i] = epos[i]

        match_sel = np.argwhere(bin_selections[m[1], :-1]).reshape(-1)

        alterer = replace.Morpher((nodes_correct[m[1]], nodes_error[m[0]]))

        print('Alteration of token: %s' % alterer.get_rule())

        n_print = 10
        print_perm = np.random.permutation(n_matches)[:n_print].tolist()

        for j in range(n_matches):

            valid_tokens = set()
            final_index = -1
            final_token = ''
            base_token = token_tagger.parse_index(
                correct_token[j, starts[j] + m[0]])

            base_form = template_correct[j, m[1], -1]
            form_info = form_pos.get(base_form, None)

            if form_info is not None:

                for _pos in form_info.keys():

                    valid = True

                    for x in match_sel:

                        if _pos[x] != template_error[j, m[0], x]:

                            valid = False

                    if valid:

                        for t in form_info[_pos]:

                            valid_tokens.add(t)

                if len(valid_tokens) > 0:

                    # Take most frequent substitution
                    t = min(valid_tokens)
                    final_token = token_tagger.parse_index(t)

                    # if len(base_token) - len(new_token) \
                    #         == alterer.del_length():

                    final_index = t

            if final_index == -1:

                if alterer.is_deletion() and alterer.can_morph():

                    final_token = alterer.morph(base_token)
                    final_index = token_tagger.add_node(final_token)

                elif alterer.is_substitution() and alterer.can_morph():

                    sub_token = alterer.morph(base_token)
                    sub_node, sub_pos = \
                        languages.parse_full(sub_token, configx.CONST_PARSER,
                                             None)

                    if len(sub_node) > 1:
                        continue

                    sub_node = token_tagger.parse_node(sub_node[0])
                    sub_pos = list(pos_taggers[q].parse_node(
                        sub_pos[q][0]) for q in range(len(pos_taggers)))

                    valid = (sub_pos[-1] == base_form)

                    for x in match_sel:

                        if sub_pos[x] != template_error[j, m[0], x]:

                            valid = False

                    if valid:

                        final_token = sub_token
                        final_index = token_tagger.add_node(final_token)

                        print_perm.append(j)

                if final_token == '':

                    token_ignore.append(j)
                    continue

            assert(final_token != '')
            assert(final_index != -1)

            if j in print_perm:

                print('\tMatch %d: %s -> %s' %
                      (j + 1, base_token, final_token))

            template_error_token[j, m[0]] = final_index

        token_selections[m[0]] = True

    for i in range(len(preserved)):

        m = preserved[i]
        template_error[:, m[0], :] = template_correct[:, m[1], :]
        error_selections[m[0], :] = bin_selections[m[1], :]
        # error_selections[m[0], -1] = True
        template_error_token[:, m[0]] = template_correct_token[:, m[1]]
        token_selections[m[0]] = True

    match_indices = np.where(error_selections.reshape(-1))[0]
    matches_correct, matches_error, ret_indices, ret_starts = [], [], [], []
    token_indices = np.where(token_selections.reshape(-1))[0]

    for i in range(n_matches):

        if error_len[i] < n_error:
            continue

        matches_pos = util.search_1d(
            error_search[i], match_indices,
            template_error[i], n_error, error_len[i])

        run_string = (len(created) != 0)

        if i in token_ignore:

            matches = matches_pos

        elif not run_string:

            matches_token = util.search_1d(
                error_token[i], token_indices,
                template_error_token[i], n_error, error_len[i])

            matches = np.logical_or(matches_token, matches_pos)

            if np.sum(matches) == 0:

                run_string = True

        special_lengths = dict()

        # String matching
        if run_string:

            matches = np.zeros(error_len[i], dtype=np.bool)

            for j in range(len(created)):

                m = created[j]
                template_error_token[i, m] = indices_error[m]

            template_error_string = token_tagger.parse_indices(
                template_error_token[i], delimiter='')

            # template_correct_string = token_tagger.parse_indices(
            #     template_correct_token[i], delimiter='')

            c = token_tagger.parse_indices(
                error_token[i, :error_len[i]]).split(',')

            combined = ''.join(c)

            char_len = [0]

            for t in c:

                char_len.append(char_len[-1] + len(t))

            try:

                # unchanged = list(m.start() for m
                #                  in re.finditer(template_correct_string,
                #                                 combined))

                for m in re.finditer(template_error_string, combined):

                    c_start_idx = m.start()
                    # match_start_idx = c_start_idx - alterer.del_length()

                    # if match_start_idx in unchanged:
                    # continue

                    c_end_idx = c_start_idx + len(template_error_string)
                    start_idx = char_len.index(c_start_idx)

                    if c_end_idx in char_len:

                        end_idx = char_len.index(c_end_idx)
                        special_lengths[start_idx] = end_idx - start_idx
                        matches[start_idx] = True

            except ValueError:

                continue

        # print(error_search[i])
        # print(template_error[i])

        # TODO:
        # When token can be parsed in two different ways
        #   pos-matching fails (i.e. in 使うて,　使う gets marked as
        #   連用タ接続, which leads to failure of first rule)

        correct_tokens = correct_token[i][starts[i]:starts[i] + n_correct]

        for j in range(len(matches)):

            if matches[j]:

                if len(created) == 0:

                    valid = False

                    for k in range(n_correct):

                        if correct_tokens[k] != error_token[i, j + k]:

                            valid = True

                    if not valid:

                        continue

                match_length = special_lengths.get(j, n_error)

                err = error_token[i][j:j + match_length]
                crt = correct_tokens

                matches_correct.append(err)
                matches_error.append(crt)
                ret_indices.append(indices[i])
                ret_starts.append([starts[i], starts[i] + n_correct,
                                   j, j + match_length])

    n_matches = len(matches_correct)
    print('Number of matches found: %d' % n_matches)

    for i in range(n_matches):

        print('Sentence %d: %s -> %s' %
              (ret_indices[i],
               token_tagger.sentence_from_indices(matches_correct[i]),
               token_tagger.sentence_from_indices(matches_error[i])))

    ret = dict()
    ret['indices'] = ret_indices
    ret['starts'] = ret_starts
    ret['correct'] = matches_correct
    ret['error'] = matches_error
    ret['count'] = n_matches
    ret['nodes_correct'] = nodes_correct
    ret['nodes_error'] = nodes_error

    return ret


def match_parallel_text_rules(input_source, input_target, input_start,
                              rules_file, rule_index=-1, language_dir=None,
                              unique_dir=None, output_dir=None,
                              output_prefix='test', print_unmatched=True):

    if language_dir is None:

        token_tagger, pos_taggers = languages.load_default_languages()

    else:

        token_tagger, pos_taggers = languages.load_languages(language_dir)

    print("\nLoading token database...")
    print(configx.BREAK_LINE)

    f = open(rules_file, 'r')
    csv_reader = csv.reader(f, delimiter=',')

    iter_count = -1
    current_rule_index = 0

    if unique_dir is None:
        unique_dir = configx.CONST_DEFAULT_DATABASE_DIRECTORY
    unique_matrices = convert.load_unique_matrices(
        unique_dir, pos_taggers)

    error_phrases, correct_phrases = process.get_paired_phrases(
        token_tagger, input_source, input_target, None)
    n_sentences = len(error_phrases)

    original_data = dict()

    original_data['correct'] = open(input_target, 'r').readlines()
    original_data['error'] = open(input_source, 'r').readlines()
    original_data['start'] = open(input_start, 'r').readlines()

    correct_matrices = _matrices(correct_phrases, token_tagger, pos_taggers)
    error_matrices = _matrices(error_phrases, token_tagger, pos_taggers)

    matched_count = 0

    empty_rules = set()

    full_matched_indices = set()
    matched_indices = dict()

    match_data = dict()
    rule_data = dict()

    unique_error_count = 0

    unique_errors = dict()
    unique_index = dict()
    index_unique = dict()

    indices_unique = dict()
    removed_unique = set()

    matched_unique_indices = set()

    for i in range(len(error_phrases)):

        ep = error_phrases[i]
        cp = correct_phrases[i]

        if (ep == cp):

            removed_unique.add(ep)
            continue

        if ep in unique_errors:

            unique_errors[ep].add(i)
            indices_unique[i] = index_unique[ep]

        else:

            unique_errors[ep] = set([i])
            unique_index[unique_error_count] = ep
            index_unique[ep] = unique_error_count
            indices_unique[i] = unique_error_count

            unique_error_count += 1

    for x in removed_unique:

        if x in unique_errors:

            set_idx = unique_errors.pop(x)

            for idx in set_idx:

                unique_idx = indices_unique.pop(idx)

                if unique_idx in unique_index:

                    unique_index.pop(unique_idx)

    valid_indices = set(indices_unique.keys())

    for rule in csv_reader:

        iter_count += 1

        if len(rule) < 2 or rule[0] == '#':

            continue

        elif iter_count == 0:

            continue

        current_rule_index += 1

        if rule_index != -1 and rule_index != current_rule_index:

            continue

        rule_dict = convert.get_rule_info(rule, pos_taggers)

        rule_data[current_rule_index] = rule_dict

        possible_classes = convert.match_rule_templates(
            rule_dict, unique_matrices)

        pos_tags = rule_dict['pos']
        selections = rule_dict['selections']

        matched \
            = convert.match_template_sentence(
                correct_matrices, pos_tags, selections, possible_classes,
                token_tagger, pos_taggers, 250000, -1, randomize=False)

        confirmed = _confirm_error(
            token_tagger, pos_taggers, matched,
            rule_dict, unique_matrices,
            correct_matrices, error_matrices,
            correct_phrases, error_phrases,
            possible_classes)

        rule_count = confirmed['count']
        rule_indices = confirmed['indices']

        matched_count += rule_count
        matched_indices[current_rule_index] = set(rule_indices)

        match_data[current_rule_index] = confirmed

        full_matched_indices.update(rule_indices)
        print('Total matches so far: %d' % matched_count)
        print('Total unique matches so far: %d' % len(full_matched_indices))

        if rule_count == 0:

            empty_rules.add(current_rule_index)

        if rule_count != 0 and output_dir is not None:

            rule_output_dir = os.path.join(output_dir, str(current_rule_index))

            _write_outputs(confirmed, original_data, rule_output_dir,
                           output_prefix, rule_starts=True)

    full_unique_indices = set()
    full_valid_indices = full_matched_indices.intersection(valid_indices)

    for i in full_valid_indices:
        full_unique_indices.add(indices_unique[i])

    print('Number of unique pairs matched: %d/%d'
          % (len(full_valid_indices), len(valid_indices)))

    print('Number of unique errors matched: %d/%d'
          % (len(full_unique_indices), len(unique_index)))

    missing_indices = valid_indices.difference(full_valid_indices)

    if output_dir is not None:

        rule_output_dir = os.path.join(output_dir, 'NONE')

        output = {'indices': missing_indices}

        _write_outputs(output, original_data, rule_output_dir, output_prefix,
                       rule_starts=False)

    print('Finding supersets')

    for r_1 in matched_indices.keys():

        s_1 = rule_data[r_1]['str']

        if r_1 in empty_rules:

            print('WARNING: Rule %d (%s) is empty' % (r_1, s_1))

            continue

        for r_2 in matched_indices.keys():

            s_2 = rule_data[r_2]['str']

            if r_2 in empty_rules:

                continue

            elif r_2 >= r_1:

                continue

            set_1 = matched_indices[r_1]
            set_2 = matched_indices[r_2]

            if set_1 == set_2:

                print('ERROR: Rule %d (%s) = %d (%s)' % (r_1, s_1, r_2, s_2))
                continue

            diff = set_1.union(set_2)

            if diff == set_1:

                print('Rule %d (%s) ⊆ %d (%s)' % (r_2, s_2, r_1, s_1))

            elif diff == set_2:

                print('Rule %d (%s) ⊆ %d (%s)' % (r_1, s_1, r_2, s_2))

    perm = np.random.permutation(len(missing_indices))
    missing_indices = list(missing_indices)
    perm = list(missing_indices[y] for y in perm)

    if print_unmatched:

        for idx in perm[:100]:

            print('%s -> %s' % (error_phrases[idx], correct_phrases[idx]))


def _write_outputs(match_info, original_data, rule_output_dir,
                   output_prefix, rule_starts=False):

    rule_indices = match_info['indices']

    correct_lines = original_data['correct']
    error_lines = original_data['error']
    start_lines = original_data['start']

    util.mkdir_p(rule_output_dir)

    output_target = os.path.join(rule_output_dir,
                                 '%s.target' % output_prefix)
    output_source = os.path.join(rule_output_dir,
                                 '%s.source' % output_prefix)
    output_start = os.path.join(rule_output_dir,
                                '%s.start' % output_prefix)
    output_indices = os.path.join(rule_output_dir,
                                  '%s.indices' % output_prefix)
    output_rule = os.path.join(rule_output_dir,
                               '%s.rule' % output_prefix)

    output_target = open(output_target, 'w+')
    output_source = open(output_source, 'w+')
    output_indices = open(output_indices, 'w+')
    output_start = open(output_start, 'w+')

    if rule_starts:

        rule_starts = match_info['starts']

        for i in rule_starts:

            output_start.write(','.join(list(str(x)
                                             for x in i)) + os.linesep)

        output_rule = open(output_rule, 'w+')

        nodes_correct, nodes_error = match_info['nodes_correct'], \
            match_info['nodes_error']

        rule_text = ','.join(
            [' '.join(nodes_correct), ' '.join(nodes_error)])
        rule_lengths = ','.join(
            [str(len(nodes_correct)), str(len(nodes_error))])

        output_rule.write(rule_text + os.linesep)
        output_rule.write(rule_lengths + os.linesep)

    for j in rule_indices:

        if not rule_starts:

            output_start.write(start_lines[j])

        output_target.write(correct_lines[j])
        output_source.write(error_lines[j])
        output_indices.write(str(j) + os.linesep)
