# Filename: generate.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 23/06/2018
# Date Last Modified: 04/03/2019
# Python Version: 3.7

'''
Functions to generate new errored sentences from template sentences
'''

import numpy as np

from termcolor import colored

from . import configx
from . import languages
from . import replace


def create_errored_sentences(unique_matrices, matched, token_tagger, pos_taggers,
                             rule_data, verbose=True, n_sample=30):
    """
    Function to generate errored sentences from template sentences using a given mapping on the 
        correct -> errored rule designated by corrected and errored inputs

    Args:
        unique_arrays (arr): Arrays containing information on unique tokens within the corpus and their 
                              corresponding part-of-speech information
        token_tagger (Language): Language class instance used to tag tokens
        pos_taggers (arr): List of Language class instances used to tag each part-of-speech index
        mapping (TYPE): List of three arrays determining which tokens from the template sentence
        selections (arr): Array determining which part-of-speech indices need to be matched
        matched_sentences (TYPE): Template sentences to generate from, grouped by sub-rule
        start_indices (TYPE): Start indices of template phrases within each template sentence
        errored (str): Errored phrase of rule
        corrected (str): Corrected phrase of rule
        verbose (bool, optional): Determines whether debugging string output is printed to terminal or not
        max_per_colored (int, optional): Determines maximum number of colored output sentences per subrule outputted

    Returns:
        ret (arr): A list of list of pairs of sentences (template, generated), grouped by sub-rule
    """
    # Separate individual matrices for use

    form_pos = unique_matrices['form_dict']

    tokens = matched['tokens']
    sentences = matched['sentences']
    forms = matched['forms']
    pos = np.moveaxis(matched['pos'], 0, -1)
    starts = matched['starts']
    subrule_indices = matched['subrule_indices']

    n_pos = pos.shape[2]
    n_matches = len(forms)

    correct_phrase = rule_data['correct']
    error_phrase = rule_data['error']

    # Return array containing newly generated sentence pairs
    ret = list()
    # Return coloured variant of generated pairs
    valid_rule_starts = list()

    # Parse template phrases
    nodes_correct, pos_correct = languages.parse_full(
        correct_phrase, configx.CONST_PARSER, None)
    nodes_error, pos_error = languages.parse_full(
        error_phrase, configx.CONST_PARSER, None)

    # Obtain 2D part-of-speech matrix for errored phrase
    # Of form (n, k), where n is the length of the phrase,
    #   and k is the number of part-of-speech tags per token
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

    if verbose:
        print("\n\tCorrect: " + ' | '.join(nodes_correct))
        print("\tError: " + ' | '.join(nodes_error) + '\n')

    template_correct = np.zeros((n_matches, n_correct, n_pos + 1),
                                dtype=forms.dtype)
    template_error = np.zeros((n_matches, n_error, n_pos + 1),
                              dtype=forms.dtype)
    template_correct_token = np.zeros((n_matches, n_correct),
                                      dtype=forms.dtype)
    template_error_token = np.zeros((n_matches, n_error),
                                    dtype=forms.dtype)
    bin_selections = selections > 0

    for i in range(n_matches):

        s = starts[i]
        template_correct[i, :, :-1] = pos[i, s:s + n_correct]
        template_correct[i, :, -1] = forms[i, s:s + n_correct]
        template_correct_token[i, :] = sentences[i, s:s + n_correct]

    for i in range(len(preserved)):

        m = preserved[i]
        template_error_token[:, m[0]] = template_correct_token[:, m[1]]

    for i in range(len(created)):

        m = created[i]
        template_error_token[:, m] = indices_error[m]

    valid_indices = list(range(n_matches))
    morph_print = 0

    for i in range(len(altered)):

        m = altered[i]
        template_error[:, m[0], :] = template_correct[:, m[1], :]

        epos = pos_error[m[0]]
        cpos = pos_correct[m[1]]

        not_ = (epos == cpos)

        for i in range(len(not_)):

            if not_[i]:

                continue

            if bin_selections[m[1], i]:

                template_error[:, m[0], i] = epos[i]

        match_sel = np.argwhere(bin_selections[m[1], :-1]).reshape(-1)

        alterer = replace.Morpher((nodes_correct[m[1]], nodes_error[m[0]]))

        print('\tAlteration of token: %s' % alterer.get_rule())

        n_print = n_sample
        print_perm = np.random.permutation(n_matches)[:n_print].tolist()

        for j in range(n_matches):

            printed = False

            valid_tokens = set()
            final_index = -1
            final_token = ''
            base_token = token_tagger.parse_index(
                template_correct_token[j, m[1]])

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

                    if (alterer.is_deletion() and alterer.can_morph()) and \
                        (len(base_token) - len(final_token)
                            != alterer.del_length()):

                        final_index = -1

            if final_index == -1:

                if alterer.is_deletion() and alterer.can_morph():

                    final_token = alterer.morph(base_token)
                    final_index = token_tagger.add_node(final_token)

                elif (alterer.is_substitution() or alterer.is_addition()) \
                        and alterer.can_morph():

                    # sub_token = alterer.morph(base_token)
                    sub_token = \
                        alterer.morph_pos(base_token, base_form, token_tagger,
                                          pos_taggers, configx.CONST_PARSER,
                                          template_error[j, m[0]], match_sel)

                    if sub_token is not None:

                        final_token = sub_token

                        if morph_print < n_sample:

                            print('\t\tMorph gen %d: %s -> %s' %
                                  (j + 1, base_token, final_token))

                            morph_print += 1
                            printed = True

                        final_index = token_tagger.add_node(final_token)

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
          (n_valid, n_matches))
    print(configx.BREAK_SUBLINE)

    valid_indices = set(valid_indices)
    n_generated = 0

    not_seen = set(list(valid_indices))

    # Iterate over each sub-rule
    for i in range(len(subrule_indices) + 1):

        if verbose:

            print("\n\t\tSub-rule %d of %d..." %
                  (i + 1, len(subrule_indices)))
            print(configx.BREAK_HALFLINE)

        if i == len(subrule_indices):
            sub_indices = list(not_seen)
            print(len(not_seen))

        else:
            sub_indices = subrule_indices[i]
        ret_sub_rule = []
        sub_rule_starts = list()

        printed_in_rule = 0

        # Iterate over each sentence in each subrule
        for j in range(len(sub_indices)):

            try:

                valid = True

                idx = sub_indices[j]

                if idx not in valid_indices:

                    continue

                s = starts[idx]

                not_seen.remove(idx)

                original_phrase = token_tagger.sentence_from_indices(
                    template_correct_token[idx])
                generated_phrase = list(token_tagger.parse_index(k)
                                        for k in template_error_token[idx])
                template_sentence = tokens[idx].split(',')

                # Don't want error phrase equal to correct
                if original_phrase == ''.join(generated_phrase):
                    raise ValueError

                if verbose and (idx in print_perm
                                or (j == len(sub_indices) - 1
                                    and printed_in_rule == 0)):

                    print("\t\tSentence %d: %s -> %s" %
                          (idx, original_phrase, ''.join(generated_phrase)))

                    printed_in_rule += 1

                t_s = 0

                if template_sentence[0] == token_tagger.start_token:

                    t_s = 1

                # Finish constructing generated sentence by placing generated phrase
                #   within non-altered portions of tempalte sentence
                generated_sentence = template_sentence[t_s:s] + generated_phrase + \
                    template_sentence[s + n_correct:]

                sub_rule_starts.append(s - t_s)
                ret_sub_rule.append(
                    (''.join(generated_sentence).strip(), ''.join(template_sentence[t_s:]).strip()))

                n_generated += 1

            except:

                raise

                # print('WARNING: Original and error phrase are identical')
                print('%s == %s' % (original_phrase, ''.join(generated_phrase)))

                continue

        ret.append(ret_sub_rule)
        valid_rule_starts.append(sub_rule_starts)

    print('\n\tErrors succesfully generated for %d / %d sentences' %
          (n_generated, n_matches))
    print(configx.BREAK_SUBLINE)

    return ret, valid_rule_starts
