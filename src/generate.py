# -*- coding: utf-8 -*-

# Filename: generate.py
# Date Created: 23/08/2018
# Description: Functions to generate parallel synthetic data
# Python Version: 3.7

import csv
import os

import numpy as np

from . import config
from . import languages
from . import morph
from . import parse
from . import util

from .match import TemplateMatch
from .rules import Rule
from .sorted_tag_database import SortedTagDatabase

from termcolor import colored
from numpy.random import RandomState

cfg = config.parse()

D_PARAMS = cfg['data_params']


def generate_synthetic_pairs(
        stdb: SortedTagDatabase, token_language: languages.Language,
        tag_languages: list, rule: Rule, matches: TemplateMatch,
        n_sample: int=30, verbose: bool=True):
    """
    Function to generate errored sentences from matched sentences using the
        mapping and template phrases of a given rule
    """

    # Return array containing newly generated sentence pairs
    gen_sentences = list()
    # Return array containing start index of error in each sentence
    gen_starts = list()

    indices_error = token_language.parse_nodes(rule.tokens_error)

    if verbose:
        print("\n\tCorrect: " + ' | '.join(rule.tokens_correct))
        print("\tError: " + ' | '.join(rule.tokens_error) + '\n')

    n_sentences = matches.n_sentences

    template_correct = np.zeros((n_sentences, rule.n_correct_tokens,
                                 matches.tags.shape[2]),
                                dtype=matches.tags.dtype)
    template_error = np.zeros((n_sentences, rule.n_error_tokens,
                               matches.tags.shape[2]),
                              dtype=matches.tags.dtype)
    template_correct_token = np.zeros((n_sentences, rule.n_correct_tokens),
                                      dtype=matches.tags.dtype)
    template_error_token = np.zeros((n_sentences, rule.n_error_tokens),
                                    dtype=matches.tags.dtype)

    bin_mask = rule.tag_mask > 0

    for i in range(n_sentences):

        s = matches.starts[i]
        template_correct[i, :, :] = \
            matches.tags[i, s:s + rule.n_correct_tokens]
        template_correct_token[i, :] = \
            matches.tokens[i, s:s + rule.n_correct_tokens]

    inserted, modified, preserved = rule.get_mapping()

    for i in range(len(preserved)):

        m = preserved[i]
        template_error_token[:, m[0]] = template_correct_token[:, m[1]]

    for i in range(len(inserted)):

        m = inserted[i]
        template_error_token[:, m] = indices_error[m]

    form_token = stdb.get_form_to_token()

    valid_indices = list(range(n_sentences))
    morph_print = 0

    for i in range(len(modified)):

        m = modified[i]
        template_error[:, m[0], :] = template_correct[:, m[1], :]

        epos = tags_error[m[0]]
        cpos = tags_correct[m[1]]

        not_ = (epos == cpos)

        for i in range(len(not_)):

            if not_[i]:

                continue

            if bin_mask[m[1], i]:

                template_error[:, m[0], i] = epos[i]

        match_sel = np.argwhere(bin_mask[m[1], :-1]).reshape(-1)

        morpher = morph.Morpher((tokens_correct[m[1]], tokens_error[m[0]]))

        print('\tAlteration of token: %s' % morpher.get_rule())

        n_print = n_sample
        print_perm = np.random.permutation(n_sentences)[:n_print].tolist()

        for j in range(n_sentences):

            printed = False

            valid_tokens = set()
            final_index = -1
            final_token = ''
            base_token = token_language.parse_index(
                template_correct_token[j, m[1]])

            base_form = template_correct[j, m[1], -1]
            form_info = form_token.get(base_form, None)

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
                    final_token = token_language.parse_index(t)

                    final_index = t

                    if (morpher.is_deletion() and morpher.can_morph()) and \
                            (len(base_token) - len(final_token) !=
                                morpher.del_length()):

                        final_index = -1

            if final_index == -1:

                if morpher.is_deletion() and morpher.can_morph():

                    final_token = morpher.morph(base_token)
                    final_index = token_language.add_node(final_token)

                elif (morpher.is_substitution() or morpher.is_addition()) \
                        and morpher.can_morph():

                    # sub_token = morpher.morph(base_token)
                    sub_token = \
                        morpher.morph_pos(
                            base_token, base_form, token_language,
                            tag_languages, parse.default_parser(),
                            template_error[j, m[0]], match_sel)

                    if sub_token is not None:

                        final_token = sub_token

                        if morph_print < n_sample:

                            print('\t\tMorph gen %d: %s -> %s' %
                                  (j + 1, base_token, final_token))

                            morph_print += 1
                            printed = True

                        final_index = token_language.add_node(final_token)

                # If generation fails, sentence is no longer valid
                if final_index == -1:

                    if j in valid_indices:

                        valid_indices.remove(j)

                    continue

            assert(final_index != -1)

            if j in print_perm and not printed:

                print('\t\tMatch %d: %s -> %s' %
                      (j + 1, base_token, final_token))

            template_error_token[j, m[0]] = final_index

    n_valid = len(valid_indices)
    print_perm = np.random.permutation(n_valid)[:n_sample]

    print('\n\tTokens succesfully generated for %d / %d sentences' %
          (n_valid, n_sentences))
    print(cfg['BREAK_SUBLINE'])

    valid_indices = set(valid_indices)
    not_seen = valid_indices.copy()
    n_generated = 0

    # Iterate over each sub-rule
    for i in range(matches.n_subrules):

        if verbose:

            print("\n\t\tSub-rule %d of %d..." %
                  (i + 1, matches.n_subrules))
            print(cfg['BREAK_HALFLINE'])

        sub_indices = matches.subrule_indices[i]

        subrule_sentences = list()
        subrule_starts = list()

        printed_in_rule = 0

        # Iterate over each sentence in each subrule
        for j in range(len(sub_indices)):

            try:

                valid = True

                idx = sub_indices[j]

                if idx not in valid_indices:

                    continue

                s = matches.starts[idx]

                not_seen.remove(idx)

                original_phrase = token_language.parse_indices(
                    template_correct_token[idx], delimiter='')

                generated_phrase = list(token_language.parse_index(k)
                                        for k in template_error_token[idx])

                template_sentence = \
                    token_language.parse_indices(
                        matches.tokens[idx]).split(',')

                # Don't want error phrase equal to correct
                if original_phrase == ''.join(generated_phrase):
                    raise ValueError

                if verbose and (idx in print_perm or
                                (j == len(sub_indices) - 1 and
                                 printed_in_rule == 0)):

                    print("\t\tSentence %d: %s -> %s" %
                          (idx, original_phrase, ''.join(generated_phrase)))

                    printed_in_rule += 1

                t_start, t_stop = 0, matches.subrule_lengths[i][j]

                if template_sentence[0] == token_language.start_token:

                    t_start = 1

                if template_sentence[t_stop] != token_language.stop_token:

                    t_stop += 1

                # Finish constructing generated sentence by placing
                #   generated phrase within non-modified portions of
                #   template sentence
                generated_sentence = template_sentence[t_start:s] + \
                    generated_phrase + \
                    template_sentence[s + rule.n_correct_tokens:t_stop]

                s = s - t_start

                subrule_starts.append([s, s + rule.n_error_tokens,
                                       s, s + rule.n_correct_tokens])
                subrule_sentences.append(
                    (generated_sentence, template_sentence[t_start:t_stop]))

                n_generated += 1

            except Exception:

                print('\t\tWARNING: Original and error phrase are identical')
                print('\t\t%s == %s' % (original_phrase,
                                        ''.join(generated_phrase)))

                continue

        gen_sentences.append(subrule_sentences)
        gen_starts.append(subrule_starts)

    if len(not_seen) > 0:

        print('\n\tWARNING: %d sentences not classified into sub-rule'
              % len(not_seen))

    print('\n\tErrors succesfully generated for %d / %d sentences' %
          (n_generated, n_sentences))
    print(cfg['BREAK_SUBLINE'])

    return gen_sentences, gen_starts


def sample_data(rule: Rule, paired_sentences: list, paired_starts: int, 
                n_per_subrule: int=1, RS: RandomState=None):

    n_subrules = len(paired_sentences)

    for i in range(n_subrules):

        print('\n\tSample sentences for sub-rule %d of %d\n'
              % (i + 1, n_subrules))

        subrule_sentences = paired_sentences[i]
        n_subrule = len(subrule_sentences)
        perm = np.arange(n_subrule) if RS is None \
            else RS.permutation(n_subrule)

        for j in perm[:n_per_subrule]:

            pair = subrule_sentences[j]
            starts = paired_starts[i][j]

            highlighted_error = ''.join(pair[0][:starts[0]]) \
                + colored(''.join(pair[0][starts[0]:starts[1]]), 'red') \
                + ''.join(pair[0][starts[1]:])

            highlighted_correct = ''.join(pair[1][:starts[2]]) \
                + colored(''.join(pair[1][starts[2]:starts[3]]) , 'green') \
                + ''.join(pair[1][starts[3]:])

            print('\tE: %s\n\tC: %s' % (highlighted_error, highlighted_correct))


def save_synthetic_sentences(paired_sentences: list, paired_starts: list,
                             save_dir: str, unknown=None):

    if not os.path.isdir(save_dir):
        util.mkdir_p(save_dir)

    print("\t\tSave directory: %s" % save_dir)

    # Iterate over each subrule
    for i in range(len(paired_sentences)):

        f_name = '%s%d%s' % (D_PARAMS['synthesized_data_prefix'], i,
                             D_PARAMS['paired_data_filetype'])

        with open(os.path.join(save_dir, f_name), "w+") as f:

            csv_writer = csv.writer(f, delimiter=',')

            n_subrule = len(paired_sentences[i])

            for j in range(n_subrule):

                error_sentence = paired_sentences[i][j][0]
                correct_sentence = paired_sentences[i][j][1]

                if unknown is not None and (unknown in error_sentence or
                                            unknown in correct_sentence):

                    print('WARNING: Pair %s -> %s contains unknown tokens'
                          % (error_sentence, correct_sentence))

                    continue

                csv_writer.writerow(
                    [' '.join(error_sentence), ' '.join(correct_sentence)] + paired_starts[i][j])

        f.close()
