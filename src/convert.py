# Filename: convert.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 19/06/2018
# Date Last Modified: 03/11/2019
# Python Version: 3.7

'''
Functions to convert CSV rules into text data via corpus manipulation
'''

import os
import csv
import ast

import numpy as np

from . import configx
from . import generate
from . import languages
from . import save
from . import util

RS = np.random.RandomState(seed=0)


def match_rule_templates(rule_dict, unique_matrices, n_max=-1):

    possible_classes = list()
    n_tokens = rule_dict['n_tokens']
    pos_tags = rule_dict['pos']
    selections = rule_dict['selections']

    for index in range(n_tokens):

        _, all_classes = match_template_tokens(unique_matrices, pos_tags[index],
                                               selections[index], n_max)
        possible_classes.append(all_classes)

    return possible_classes


def match_template_tokens(unique_matrices, search_numbers, selected_cells, n_max):
    '''
    Function to generate possible substitute tokens given restrictions on which part-of-speech tags must be preserved

    Args:
        unique_matrices (arr): List of np.ndarrays containing information on unique tokens and part-of-speech tags
        search_numbers (arr): Indices of part-of-speech tags of the token
        selected_cells (arr): Array determining which part-of-speech indices need to be matched
        n_max (int): Determines maximal token index that is outputted. If this value is -1, any token
                               is allowed to be outputted
    Returns:
        (tuple): Tuple containing the following:
            (np.ndarray): Array containing all possible substitute part-of-speech combinations (including form)
            (np.ndarray): Array containing all possible substitute part-of-speech combinations (excluding form)
    '''

    if n_max == -1:
        n_max = len(unique_matrices['pos'])

    else:
        n_max = min(len(unique_matrices['pos']), n_max)

    # n_max = len(unique_matrices[2])

    # Part-of-speech indices that need to be matched exactly
    match_indices = np.where(selected_cells == 1)[0]
    # Part-of-speech indices that do not need to be matched
    randomize_indices = np.where(selected_cells == 0)[0]

    # Wildcard tokens
    if len(randomize_indices) == len(selected_cells):
        return None, None

    # If any indices must be
    if len(match_indices) > 0:

        search_numbers_new = search_numbers[match_indices]
        possible_matches = search(
            unique_matrices, match_indices, search_numbers_new, n_max)

    else:

        # Any token class may substitute
        possible_matches = np.arange(n_max)

    classes = get_pos_classes(
        unique_matrices, randomize_indices, search_numbers, possible_matches)

    return classes


def search(unique_matrices, match_indices, search_numbers, n_max):
    """
    Function to obtain a list of possible substitute token indices given
        an input token and restrictions
    on which part-of-speech tags must be preserved

    Args:
        unique_matrices (arr): List of np.ndarrays containing information
            on unique tokens and part-of-speech tags
        match_indices (arr): Array of part-of-speech indices that require
            exact matching
        search_numbers (arr): Array of part-of-speech values corresponding
            to the match_indices
        n_max (int): Determines maximal token index that is outputted

    Returns:
        (tuple): Tuple containing the following:
            (np.ndarray): Array containing all possible substitute part-of-speech
                combinations (including form)
            (np.ndarray): Array containing all possible substitute part-of-speech
                combinations (excluding form)
    """
    pos = unique_matrices['pos']
    sort = unique_matrices['sort']
    search_matrix = unique_matrices['search']

    # Indices of token classes that may be used
    ret_indices = None

    # Iterate over each part-of-speech tag that needs matching
    for i in range(len(match_indices)):

        # Tag value of original token at index
        search_number = search_numbers[i]

        # Maximum index to search
        max_search = len(search_matrix[match_indices[i]])

        if (search_number >= max_search):
            raise ("Illegal search number")

        elif (search_number == max_search - 1):

            start_index = search_matrix[match_indices[i]][search_number] + 1
            end_index = len(sort)

        else:

            start_index = search_matrix[match_indices[i]][search_number] + 1
            end_index = search_matrix[match_indices[i]][search_number + 1] + 1

        # From sorted array, determine all possible classes with part-of-speech at given index matching that of original token
        possible_indices = sort[:, match_indices[i]][start_index:end_index]

        # If not first index matched, intersect along indices to determine which combinations match both part-of-speech indices
        if ret_indices is not None:

            ret_indices = np.intersect1d(ret_indices, possible_indices)

        else:

            ret_indices = possible_indices

    # Restrict output to valid index values
    ret_indices = ret_indices[ret_indices < n_max]

    return ret_indices


def get_pos_classes(
        unique_matrices, randomize_indices, original, possible_matches):
    """
    Function to determine unique part-of-speech combinations from possible
        substitute tokens

    Args:
        unique_matrices (arr): List of np.ndarrays containing information on
            unique tokens and part-of-speech tags
        randomize_indices (arr): Array of part-of-speech indices that require
            no matching whatsoever
        original (arr): Part-of-speech indices of original values
        possible_matches (np.ndarray): Array containing the indices of tokens
            that satisfy the criterion determined by indices and original

    Returns:
        (np.ndarray): Array containing all possible substitute part-of-speech
            combinations (including form)
        (np.ndarray): Array containing all possible substitute part-of-speech
            combinations (excluding form)
    """
    # Separate individual matrices for use
    tokens = unique_matrices['token']
    pos = unique_matrices['pos']
    sort = unique_matrices['sort']
    tags = unique_matrices['complete']
    classes = unique_matrices['classes']

    # Lists containing possible part-of-speech combinations that can be
    #   substituted in
    all_tags = list()
    all_classes = list()

    # Add part-of-speech combination of original token
    all_tags.append(tuple(original))
    all_classes.append(tuple(original[:-1]))

    ret = list()

    if len(randomize_indices) > 0:

        # If the rule is fully lenient, placing no bounds on possible
        #   substitute tokens
        if len(randomize_indices) == sort.shape[1]:

            all_dict = tags
            all_tags = list(all_dict.keys())

            all_class = classes
            all_class_tags = list(classes.keys())

        # Otherwise iterate through possible substitute tokens to determine
        #   all possible substitute part-of-speech combinations
        else:

            all_tags = dict()
            all_classes = dict()

            matched_nodes = pos[possible_matches, :]
            matched_nodes_form = tokens[possible_matches, 1:]

            # Determine possible substitute part-of-speech combinations
            #   (including form, stored on disk with tokens)
            for j in range(len(possible_matches)):

                uc = tuple(matched_nodes[j]) + tuple(matched_nodes_form[j])

                if uc in all_tags:

                    all_tags[uc].append(possible_matches[j])

                else:

                    all_tags[uc] = [possible_matches[j]]

            # Determine possible substitute part-of-speech combinations
            #   excluding form
            for j in range(len(possible_matches)):

                uc = tuple(matched_nodes[j])

                if uc in all_classes:

                    all_classes[uc].append(possible_matches[j])

                else:

                    all_classes[uc] = [possible_matches[j]]

            # Retrieve keys from generated dictionaries
            all_tags = list(all_tags.keys())
            all_classes = list(all_classes.keys())

    return np.array(all_tags), np.array(all_classes)


def match_template_sentence(search_matrices, search_numbers, selections,
                            possible_classes, token_tagger, pos_taggers,
                            n_search=-1, n_max_out=-1, n_min_out=-1,
                            out_ratio=0.10, n_token=100000,
                            randomize=True):
    """
    Given a template phrase, and matching leniency for part-of-speech tags,
        scan a corpus of text (search_matrices) for sentences that contain
        phrases that match with the template phrase

    Args:
        search_matrices (arr): List of np.ndarrays containing the token and
            part-of-speech information for each sentence
        search_numbers (arr): Array of part-of-speech values corresponding to
            the phrase to match too
        selections (arr): Array determining which part-of-speech indices need
            to be matched
        possible_classes (arr): Array containing all possible substitute
            part-of-speech combinations to the template phrase (excluding form)
        token_tagger (Language): Language class instance used to tag tokens
        pos_taggers (arr): List of Language class instances used to tag each
            part-of-speech index
        n_token (int): Maximum index value of tokens matched
        n_search (int): Maximum number of sentences to search through
            (for testing efficiency mostly)

    Returns:
        (tuple): Tuple containing the following arrays:
            ret_sentences (arr): Three-dimensional list containing string/token
                representations of the matched sentences, separated by sub-rule
            ret_indices (arr): Three-dimensional list (lowest level np.ndarray)
                containing the token indices of each of the matched sentences,
                separated by sub-rule
            starts (arr); List of lists denoting the start positions of each
                matched phrase within each matched sentence

    """
    # Separate individual matrices for use
    forms = search_matrices['form']
    lengths = search_matrices['lengths']
    pos = search_matrices['pos']
    tokens = search_matrices['token']

    if randomize:

        # Randomize order of sentences
        perm = RS.permutation(len(forms))[:n_search]

        forms = forms[perm]
        lengths = lengths[perm]
        pos = pos[:, perm]
        tokens = tokens[perm]

    # Number of tokens in correct_sentence
    n_tokens = len(search_numbers)

    # Restrict number of sentences to search through
    if n_search == -1:

        n_search = len(forms)

    else:

        n_search = min(n_search, len(tokens))

    # Permutation to randomly select the sentences from the array
    match_array = np.ones((forms.shape[0],
                           forms.shape[1] - selections.shape[0] + 1),
                          dtype=np.bool)

    # Iterate over each part-of-speech index, restricting possible matches
    #   by part-of-speech indices demanding exact matching
    for i in range(len(pos_taggers)):

        # Part-of-speech matching leniency and tags of index i for each token
        s_column = selections[:, i]
        n_column = search_numbers[:, i]

        # If there is leniency in the part-of-speech for all tokens, continue
        if np.all(s_column != 1):

            pass

        else:

            # Perform array intersection on each part-of-speech index
            if i != len(pos_taggers) - 1:

                match_array = np.logical_and(match_array, util.search_template(
                    pos[i], np.argwhere(s_column == 1), n_column, n_tokens))

            # Form of token is on different data array
            else:

                match_array = np.logical_and(match_array, util.search_template(
                    forms, np.argwhere(s_column == 1), n_column,
                    n_tokens))

    # Get indices of sentences that contain at least one match
    successes = np.any(match_array, axis=1)
    n_matched_sentences = np.sum(successes)
    matched_indices = np.nonzero(successes)[0]

    # Extract the contents and length of sentences that contain matches
    sentences = tokens[successes]
    lengths = lengths[successes]
    pos = pos[:, successes]
    forms = forms[successes]

    match_array = match_array[successes]

    # Get total number of matches (individual sentences may have more than one)
    n_per = np.sum(match_array, axis=1)
    n_matches = np.sum(match_array)

    print("\n\tNumber of sentences with a match: %s" % n_matched_sentences)
    print("\tTotal number of matches: %s" % n_matches)

    # Create new arrays to copy data for sentences with more than one match
    temp_match_array = np.ndarray(
        (n_matches, match_array.shape[1]), match_array.dtype)
    temp_sentences = np.ndarray(
        (n_matches, sentences.shape[1]), sentences.dtype)
    temp_lengths = np.ndarray((n_matches), lengths.dtype)

    temp_pos = np.ndarray((pos.shape[0], n_matches, pos.shape[2]), pos.dtype)
    temp_forms = np.ndarray((n_matches, forms.shape[1]), forms.dtype)

    temp_matched_indices = np.ndarray(n_matches, matched_indices.dtype)

    # Copy data to new array
    temp_match_array[:n_matched_sentences] = match_array[:]
    temp_sentences[:n_matched_sentences] = sentences[:]
    temp_lengths[:n_matched_sentences] = lengths[:]

    temp_forms[:n_matched_sentences] = forms[:]
    temp_pos[:, :n_matched_sentences] = pos[:, :]

    temp_matched_indices[:n_matched_sentences] = matched_indices[:]

    insert_index = n_matched_sentences

    print("\n\tProcessing sentences with more than one match")

    for j in range(len(match_array)):

        # Iterate over sentences with more than one match
        if n_per[j] > 1:

            copy = np.copy(match_array[j])

            # For each extra match
            while n_per[j] > 1:

                # Remove the first instance of the match in the match_array
                first_index = np.argmax(copy)
                copy[first_index] = 0

                # Copy the data for the sentence into each of the new arrays
                temp_match_array[insert_index][:] = copy[:]
                temp_sentences[insert_index][:] = sentences[j][:]
                temp_lengths[insert_index] = lengths[j]

                temp_pos[:, insert_index] = pos[:, j]
                temp_forms[insert_index][:] = forms[j][:]

                temp_matched_indices[insert_index] = matched_indices[j]

                n_per[j] -= 1
                insert_index += 1

                assert(n_per[j] == np.sum(copy))

    print("\tCompleted...")

    assert(insert_index == n_matches)

    # Reset references
    match_array = temp_match_array
    sentences = temp_sentences
    lengths = temp_lengths
    pos = temp_pos
    forms = temp_forms
    matched_indices = temp_matched_indices

    # Get index to start search from per each sentence (note that argmax
    #   selects first index equivalent to max)
    # Copied sentences at end of match_array have had their first (k)
    #   instances of 1 removed
    start = (match_array).argmax(axis=1)

    print("\n\tLimiting matches to those within first %d tokens" % (n_token))

    valid = None

    # Limit valid sentences to those in which all tokens within the matched 
    #   phrase are limited to the first n_max indices
    for index in range(n_tokens):

        # Determine where in each sentence to check for classes
        check = start + index

        # Obtain the token values for each matched phrase
        values = np.array(list(sentences[j][check[j]]
                               for j in range(len(check))))
        # Boolean array mask
        values = (values < n_token)

        if valid is None:

            valid = values

        else:

            valid = np.logical_and(valid, values)

    n_valid = np.sum(valid)

    print("\tFinal number of valid sentences: %d" % (n_valid))

    if n_max_out == -1:
        n_max_out = n_valid

        if n_min_out != -1:

            ratio_out = out_ratio * n_max_out

            if ratio_out < n_min_out:

                n_max_out = n_min_out

            elif ratio_out < n_max_out:

                n_max_out = n_max_out

    else:
        n_max_out = min(n_valid, n_max_out)

    match_array = match_array[valid][:n_max_out]
    sentences = sentences[valid][:n_max_out]
    lengths = lengths[valid][:n_max_out]
    forms = forms[valid][:n_max_out]
    pos = pos[:, valid]
    pos = pos[:, :n_max_out]
    matched_indices = matched_indices[valid][:n_max_out]

    tokens = list(token_tagger.parse_indices(
        sentences[j][:lengths[j]]) for j in range(n_max_out))

    # Get index to start search from per each sentence (note that argmax
    #   selects first index equivalent to max)
    # Copied sentences at end of match_array have had their first (k)
    #   instances of 1 removed
    starts = (match_array).argmax(axis=1)
    del match_array

    ret = dict()

    ret['forms'] = forms
    ret['indices'] = matched_indices
    ret['pos'] = pos
    ret['sentences'] = sentences
    ret['starts'] = starts
    ret['tokens'] = tokens

    all_matches = list()

    # Iterate over each token, checking which sentences match each sub-rule
    #   (as defined by tuples of possible_classes)
    constrained_tokens = set(range(n_tokens))

    for index in range(n_tokens):

        if possible_classes[index] is None:

            constrained_tokens.remove(index)
            all_matches.append(None)
            continue

        assert(len(possible_classes[index]) != 0)

        # Determine where in each sentence to check for classes
        check = starts + index

        matches, counts = util.check_matched_indices(
            pos, check, possible_classes[index])

        all_matches.append(matches)

    rule_types = dict()

    # Iterate over each matched sentence
    for sentence_number in range(n_max_out):

        subrules = list()

        for index in constrained_tokens:

            matches = all_matches[index]

            for k in range(len(matches)):

                # Extract the class of token per each index of 
                #   each matched sentence
                if matches[k][sentence_number]:

                    subrules.append(k)

        # Each sub-rule represents unique combination of classes 
        #   among matched sentences
        subrules = tuple(subrules)

        if len(subrules) == len(constrained_tokens):

            # If the sub-rule has already been seen
            if subrules in rule_types.keys():

                # Place indices corresponding to sub-rule
                rule_types[subrules].append(sentence_number)

            # If sub-rule has not already been seen
            else:

                rule_types[subrules] = [sentence_number]

        else:
            pass

    # All possible sub-rules within this rule
    subrules = list(rule_types.keys())

    print("\n\tNumber of possible sub-rules: %d" % (len(subrules)))

    subrule_sentences = list()
    subrule_starts = list()
    subrule_tokens = list()
    subrule_indices = list()

    # Iterate through each sub-rule
    for sub in range(len(subrules)):

        # Indices associated with each sub-rule
        selected_indices = np.array(rule_types[subrules[sub]])

        # Sentences associated with each sub-rule
        under = sentences[selected_indices]
        chosen_lengths = lengths[selected_indices]

        # Total number of sentences associated with each sub-rule
        n_under = len(under)

        # Only consider sub-rules with sentences matching
        if n_under != 0:

            print("\t\tNumber of sentences under sub-rule %d: %d" %
                  (sub + 1, n_under))

            under = under
            chosen_lengths = chosen_lengths

            subrule_indices.append(selected_indices)
            subrule_sentences.append(
                list(under[i][1:chosen_lengths[i]] for i in range(n_under)))
            subrule_starts.append(starts[selected_indices].tolist())
            subrule_tokens.append(list(token_tagger.parse_indices(
                under[i][1:chosen_lengths[i]]) for i in range(n_under)))

    ret['subrule_indices'] = subrule_indices
    ret['subrule_sentences'] = subrule_sentences
    ret['subrule_starts'] = subrule_starts
    ret['subrule_tokens'] = subrule_tokens

    return ret
