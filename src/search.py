# Filename: convert.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 10/10/2019
# Date Last Modified: 09/11/2019
# Python Version: 3.7

import numpy as np
import os
import re

from . import convert
from . import database
from . import configx
from . import languages
from . import process
from . import replace
from . import rules
from . import util

RULE_FILE_DIRECTORY = 'rules'


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
                   coverage_data, rule_data, unique_matrices,
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

    print("\tGenerating template phrases...")

    print("\n\tCorrect: " + ' | '.join(nodes_correct))
    print("\tError: " + ' | '.join(nodes_error))

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

    indices = coverage_data['indices']
    forms = coverage_data['forms']
    pos = np.moveaxis(coverage_data['pos'], 0, -1)
    starts = coverage_data['starts']

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

        print('\n\tAlteration of token: %s' % alterer.get_rule())

        n_print = 10
        print_perm = np.random.permutation(n_matches)[:n_print].tolist()

        for j in range(n_matches):

            valid_tokens = set()
            final_index = -1
            final_token = ''
            base_token = token_tagger.parse_index(
                correct_token[j, starts[j] + m[1]])

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
                    final_index = t

                    if alterer.is_deletion() and \
                        (len(base_token) - len(final_token) !=
                            alterer.del_length()):

                        final_index = -1

            if final_index == -1:

                if alterer.is_deletion() and alterer.can_morph():

                    final_token = alterer.morph(base_token)
                    final_index = token_tagger.add_node(final_token)

                elif (alterer.is_substitution() or alterer.is_addition()) \
                        and alterer.can_morph():

                    sub_token = \
                        alterer.morph_pos(base_token, base_form, token_tagger,
                                          pos_taggers, configx.CONST_PARSER,
                                          template_error[j, m[0]], match_sel)

                    if sub_token is not None:

                        final_token = sub_token
                        final_index = token_tagger.add_node(final_token)

                        print_perm.append(j)

                if final_token == '':

                    token_ignore.append(j)
                    continue

            assert(final_token != '')
            assert(final_index != -1)

            if j in print_perm:

                print('\t\tMatch %d: %s -> %s' %
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

            template_correct_string = token_tagger.parse_indices(
                template_correct_token[i], delimiter='')

            c = token_tagger.parse_indices(
                error_token[i, :error_len[i]]).split(',')

            combined = ''.join(c)

            char_len = [0]

            for t in c:

                char_len.append(char_len[-1] + len(t))

            try:
                if template_error_string == template_correct_string:
                    raise ValueError

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

                # print(token_tagger.sentence_from_indices(error_token[i]))

                matches_correct.append(err)
                matches_error.append(crt)
                ret_indices.append(indices[i])
                ret_starts.append([starts[i], starts[i] + n_correct,
                                   j, j + match_length])

    n_matches = len(matches_correct)

    print('\n' + configx.BREAK_SUBLINE + '\n')
    print('\tNumber of matches found: %d\n' % n_matches)

    print('\tMatched pairs:\n')

    for i in range(n_matches):

        print('\t\tSentence %d: %s -> %s' %
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


def _unique_pairs(error_phrases, correct_phrases):

    print("\nFinding error phrases with null corrections...")
    print(configx.BREAK_LINE)

    n_unq_pred = 0

    unq_pred = dict()
    unq_idx_unq_pred = dict()
    unq_pred_unq_idx = dict()

    s_idx_unq_idx = dict()
    removed_unique = set()
    removed_indices = list()

    ret = dict()

    for i in range(len(error_phrases)):

        pred = error_phrases[i]
        tgt = correct_phrases[i]

        if (pred == tgt):

            print('\tLine %d: %s == %s' % (i + 1, pred, tgt))

            removed_unique.add(pred)
            removed_indices.append(i)
            continue

        if pred in unq_pred:

            unq_pred[pred].add(i)
            s_idx_unq_idx[i] = unq_pred_unq_idx[pred]

        else:

            unq_pred[pred] = set([i])
            unq_idx_unq_pred[n_unq_pred] = pred
            unq_pred_unq_idx[pred] = n_unq_pred
            s_idx_unq_idx[i] = n_unq_pred

            n_unq_pred += 1

    for x in removed_unique:

        if x in unq_pred:

            set_idx = unq_pred.pop(x)

            for idx in set_idx:

                unique_idx = s_idx_unq_idx.pop(idx)

                if unique_idx in unq_idx_unq_pred:

                    unq_idx_unq_pred.pop(unique_idx)

    valid_s_idx = set(s_idx_unq_idx.keys())

    ret['valid_sentences'] = valid_s_idx
    ret['unique_predicate'] = unq_idx_unq_pred
    ret['pair_unique_predicate'] = s_idx_unq_idx
    ret['pred_pair'] = unq_pred
    ret['removed_indices'] = removed_indices

    return ret


def _write_mapping(unique_mapping, output_dir, f_prefix='mapping'):

    idx_unique = unique_mapping['pair_unique_predicate']
    idx_file = os.path.join(output_dir, '%s_indices.txt' % f_prefix)
    idx_file = open(idx_file, 'w+')

    for i in idx_unique.keys():

        j = idx_unique[i]
        idx_file.write('%d,%d\n' % (i, j))


def _write_outputs(match_info, unique_mapping, original_data, rule_output_dir,
                   output_prefix, rule_starts=False):

    print('\tSaving rule output to directory: %s' % rule_output_dir)

    confirmed_sentences = match_info['indices']
    valid_indices = unique_mapping['valid_sentences']

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

    written_indices = list()

    for j in range(len(confirmed_sentences)):

        idx = confirmed_sentences[j]

        if idx not in valid_indices:
            continue

        written_indices.append(idx)

        if not rule_starts:

            output_start.write(start_lines[idx])

        else:

            output_start.write(
                ','.join(list(str(x)
                              for x in match_info['starts'][j])) + os.linesep)

        output_target.write(correct_lines[idx])
        output_source.write(error_lines[idx])
        output_indices.write(str(idx) + os.linesep)

    if rule_starts:

        output_rule = open(output_rule, 'w+')

        nodes_correct, nodes_error = match_info['nodes_correct'], \
            match_info['nodes_error']

        rule_text = ','.join(
            [' '.join(nodes_correct), ' '.join(nodes_error)])
        rule_lengths = ','.join(
            [str(len(nodes_correct)), str(len(nodes_error))])

        output_rule.write(rule_text + os.linesep)
        output_rule.write(rule_lengths + os.linesep)

    print('\tNumber of sentences written: %d' % len(written_indices))


def _display_multiple_coverage(coverage, predicates):

    print('\nDisplaying predicates matching multiple rules...')
    print(configx.BREAK_LINE + '\n')

    for i in coverage.keys():

        matched_rules = coverage[i]

        if len(matched_rules) > 1:

            predicate = predicates[i]

            print('\tPredicate %d: %s' % (i, predicate))
            print('\t\tMatched by rules: %s' %
                  ', '.join(str(x) for x in list(matched_rules)))


def _display_coverage_supersets(rule_coverage, rule_data, empty_rules):

    print('\nFinding coverage supersets...')
    print(configx.BREAK_LINE + '\n')

    for r_1 in rule_coverage.keys():

        s_1 = rule_data[r_1]['str']

        if r_1 in empty_rules:

            print('\tWARNING: Rule %d (%s) is empty' % (r_1, s_1))

            continue

        for r_2 in rule_coverage.keys():

            s_2 = rule_data[r_2]['str']

            if r_2 in empty_rules:

                continue

            elif r_2 >= r_1:

                continue

            set_1 = rule_coverage[r_1]
            set_2 = rule_coverage[r_2]

            if set_1 == set_2:

                print('\tERROR: Rule %d (%s) = %d (%s)' %
                      (r_1, s_1, r_2, s_2))
                continue

            diff = set_1.union(set_2)

            if diff == set_1:

                print('\tRule %d (%s) ⊆ %d (%s)' % (r_2, s_2, r_1, s_1))

            elif diff == set_2:

                print('\tRule %d (%s) ⊆ %d (%s)' % (r_1, s_1, r_2, s_2))


def _sample_unmatched_sentences(missing_indices, error_phrases,
                                correct_phrases):

    print('\nDisplaying sample of unmatched sentences..')
    print(configx.BREAK_LINE + '\n')

    perm = np.random.permutation(len(missing_indices))
    missing_indices = list(missing_indices)
    perm = list(missing_indices[y] for y in perm)

    for idx in perm[:200]:

        print('Match %d: %s -> %s' %
              (idx, error_phrases[idx], correct_phrases[idx]))


def match_parallel_text_rules(input_source, input_target, input_start,
                              rule_file, rule_index=-1, language_dir=None,
                              unique_dir=None, output_dir=None,
                              output_prefix='test', print_unmatched=True,
                              raise_on_error=True, display_coverage=False):

    if language_dir is None:

        token_tagger, pos_taggers = languages.load_default_languages()

    else:

        token_tagger, pos_taggers = languages.load_languages(language_dir)

    current_rule_index = 0

    error_phrases, correct_phrases = process.get_paired_phrases(
        token_tagger, input_source, input_target, None)

    unique_mapping = _unique_pairs(error_phrases, correct_phrases)
    indices_predicates = unique_mapping['pair_unique_predicate']
    valid_indices = unique_mapping['valid_sentences']
    unique_predicates = unique_mapping['unique_predicate']

    if raise_on_error:

        if len(unique_mapping['removed_indices']) != 0:

            raise ValueError("ERROR: Some error phrases have null corrections")

    original_data = dict()

    original_data['correct'] = open(input_target, 'r').readlines()
    original_data['error'] = open(input_source, 'r').readlines()
    original_data['start'] = open(input_start, 'r').readlines()

    correct_matrices = _matrices(correct_phrases, token_tagger, pos_taggers)
    error_matrices = _matrices(error_phrases, token_tagger, pos_taggers)

    print("\nLoading token database...")
    print(configx.BREAK_LINE)

    if unique_dir is None:
        unique_dir = configx.CONST_DEFAULT_DATABASE_DIRECTORY
    unique_matrices = database.load_unique_matrices(
        unique_dir, pos_taggers)

    total_confirmed_count = 0

    empty_rules = set()

    full_covered_indices = set()
    rule_coverage = dict()

    unique_coverage_data = dict()
    coverage_data = dict()
    rule_data = dict()

    # Load rule file
    rule_file = os.path.join(RULE_FILE_DIRECTORY, '%s.csv' % rule_file)

    for rule_dict in rules.parse_rule_file(rule_file, pos_taggers, rule_index):

        if rule_index != -1:
            current_rule_index = rule_index

        else:
            current_rule_index += 1

        print('\nReading Rule %d: %s' % (current_rule_index, rule_dict['str']))
        print(configx.BREAK_LINE)

        rule_data[current_rule_index] = rule_dict

        possible_classes = convert.match_rule_templates(
            rule_dict, unique_matrices)

        pos_tags = rule_dict['pos']
        selections = rule_dict['selections']

        matched \
            = convert.match_template_sentence(
                correct_matrices, pos_tags, selections, possible_classes,
                token_tagger, pos_taggers, 250000, -1, randomize=False)

        print('\n' + configx.BREAK_SUBLINE + '\n')

        confirmed = _confirm_error(
            token_tagger, pos_taggers, matched,
            rule_dict, unique_matrices,
            correct_matrices, error_matrices,
            correct_phrases, error_phrases,
            possible_classes)

        confirmed_count = confirmed['count']
        confirmed_sentences = confirmed['indices']

        for i in confirmed_sentences:

            unq_idx = indices_predicates[i]

            if unq_idx in unique_coverage_data:

                unique_coverage_data[unq_idx].add(current_rule_index)

            else:

                unique_coverage_data[unq_idx] = set([current_rule_index])

        total_confirmed_count += confirmed_count
        rule_coverage[current_rule_index] = set(confirmed_sentences[:])

        coverage_data[current_rule_index] = confirmed

        full_covered_indices.update(confirmed_sentences)

        print('\n' + configx.BREAK_SUBLINE + '\n')

        print('\tTotal matches so far: %d' % total_confirmed_count)
        print('\tTotal unique matches so far: %d' % len(full_covered_indices))

        if confirmed_count == 0:

            empty_rules.add(current_rule_index)

        if confirmed_count != 0 and output_dir is not None:

            rule_output_dir = os.path.join(output_dir, str(current_rule_index))

            _write_outputs(confirmed, unique_mapping, original_data,
                           rule_output_dir, output_prefix, rule_starts=True)

    full_unique_indices = set()
    full_valid_indices = full_covered_indices.intersection(valid_indices)

    for i in full_valid_indices:
        full_unique_indices.add(indices_predicates[i])

    missing_indices = valid_indices.difference(full_valid_indices)

    if output_dir is not None:

        print('\nSaving unmatched sentences...')
        print(configx.BREAK_LINE + '\n')

        rule_output_dir = os.path.join(output_dir, 'NONE')

        output = {'indices': sorted(missing_indices)}

        _write_outputs(output, unique_mapping, original_data, rule_output_dir,
                       output_prefix, rule_starts=False)
        _write_mapping(unique_mapping, output_dir)

    if display_coverage:

        _display_multiple_coverage(unique_coverage_data, unique_predicates)
        _display_coverage_supersets(rule_coverage, rule_data, empty_rules)

    if print_unmatched:

        _sample_unmatched_sentences(
            missing_indices, error_phrases, correct_phrases)

    print('\nResults...')
    print(configx.BREAK_LINE + '\n')

    print('Number of unique pairs matched: %d/%d'
          % (len(full_valid_indices), len(valid_indices)))

    print('Number of unique errors matched: %d/%d'
          % (len(full_unique_indices), len(unique_predicates)))
