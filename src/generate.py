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
from . import util

from .match import TemplateMatch
from .rules import Rule, CharacterRule
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

    if verbose:
        print("\n\tCorrect: " + ' | '.join(rule.tokens_correct))
        print("\tError: " + ' | '.join(rule.tokens_error) + '\n')

    n_sentences = matches.n_sentences

    correct_tags = np.zeros((n_sentences, rule.n_correct_tokens,
                             matches.tags.shape[2]),
                            dtype=matches.tags.dtype)
    correct_token = np.zeros((n_sentences, rule.n_correct_tokens),
                             dtype=matches.tags.dtype)
    error_token = np.zeros((n_sentences, rule.n_error_tokens),
                           dtype=matches.tags.dtype)

    for i in range(n_sentences):

        s = matches.starts[i]
        correct_tags[i, :, :] = \
            matches.tags[i, s:s + rule.n_correct_tokens]
        correct_token[i, :] = \
            matches.tokens[i, s:s + rule.n_correct_tokens]

    if isinstance(rule, CharacterRule):

        valid_indices = rule.convert_phrases(
            correct_token, error_token, token_language)

    else:

        error_tags = np.zeros((n_sentences, rule.n_error_tokens,
                               matches.tags.shape[2]),
                              dtype=matches.tags.dtype)

        valid_indices = rule.convert_phrases(
            correct_token, correct_tags, error_token, error_tags,
            token_language, tag_languages, stdb)

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

                idx = sub_indices[j]

                if idx not in valid_indices:

                    continue

                s = matches.starts[idx]

                not_seen.remove(idx)

                original_phrase = token_language.parse_indices(
                    correct_token[idx], delimiter='')

                generated_phrase = list(token_language.parse_index(k)
                                        for k in error_token[idx])

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

                t_start, t_stop = 0, matches.subrule_sentence_lengths[i][j]

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

            except ValueError:

                print('\t\tWARNING: Original and error phrase are identical')
                print('\t\t%s == %s' % (original_phrase,
                                        ''.join(generated_phrase)))

                continue

            except Exception:

                raise

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
                n_per_subrule: int=5, RS: RandomState=None):

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
                + colored(''.join(pair[1][starts[2]:starts[3]]), 'green') \
                + ''.join(pair[1][starts[3]:])

            print('\tE: %s\n\tC: %s' %
                  (highlighted_error, highlighted_correct))


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

                    # print('WARNING: Pair %s -> %s contains unknown tokens'
                    #       % (error_sentence, correct_sentence))
                    print('WARNING: Unknown Tokens')
                    continue

                csv_writer.writerow(
                    [' '.join(error_sentence), ' '.join(correct_sentence)] +
                    paired_starts[i][j])

        f.close()
