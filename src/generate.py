# -*- coding: utf-8 -*-

# Filename: generate.py
# Date Created: 19/12/2019
# Description: Functions to generate parallel synthetic data
# Python Version: 3.7

import numpy as np

from . import config
from . import languages

from .datasets import Dataset
from .kana import KanaList
from .match import TemplateMatch
from .rules import Rule, CharacterRule
from .sorted_tag_database import SortedTagDatabase

cfg = config.parse()

D_PARAMS = cfg['data_params']
DS_PARAMS = cfg['dataset_params']


def generate_synthetic_pairs(
        stdb: SortedTagDatabase, token_language: languages.Language,
        tag_languages: list, rule: Rule, matches: TemplateMatch,
        KL: KanaList, n_sample: int=30, verbose: bool=True):
    """
    Function to generate errored sentences from matched sentences using the
        mapping and template phrases of a given rule
    """

    # Return arrays containing newly generated sentence pairs
    gen_correct = list()
    gen_error = list()
    # Return array containing start/end index of correct/error phrases
    #    in each sentence
    gen_correct_bounds = list()
    gen_error_bounds = list()
    # Return array containing sub-rule of each sentence
    gen_subrules = list()

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
            correct_token, error_token, correct_tags[:, :, -1],
            token_language, tag_languages[-1], KL=KL)

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

                gen_error.append(generated_sentence)
                gen_correct.append(template_sentence[t_start:t_stop])

                gen_error_bounds.append([s, s + rule.n_error_tokens])
                gen_correct_bounds.append([s, s + rule.n_correct_tokens])

                gen_subrules.append(i)
                n_generated += 1

            except ValueError:

                print('\t\tWARNING: Original and error phrase are identical')
                print('\t\t%s == %s' % (original_phrase,
                                        ''.join(generated_phrase)))

                continue

            except Exception:

                raise

    if len(not_seen) > 0:

        print('\n\tWARNING: %d sentences not classified into sub-rule'
              % len(not_seen))

    print('\n\tErrors succesfully generated for %d / %d sentences' %
          (n_generated, n_sentences))
    print(cfg['BREAK_SUBLINE'])

    gen_rules = [rule.name] * n_generated

    return Dataset.import_data(gen_error, gen_correct,
                               gen_error_bounds, gen_correct_bounds,
                               gen_rules, gen_subrules)
