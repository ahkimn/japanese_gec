# -*- coding: utf-8 -*-

# Filename: match.py
# Date Created: 01/02/2020
# Description: TemplateMatch class and associated functions
# Python Version: 3.7

import numpy as np
from . import config
from . import util

from .databases import Database
from .kana import KanaList
from .rules import Rule, CharacterRule
from .sorted_tag_database import SortedTagDatabase

from numpy.random import RandomState

cfg = config.parse()


class TemplateMatch:

    def __init__(self, tokens: np.ndarray, tags: np.ndarray,
                 sentence_lengths: np.ndarray, starts: np.ndarray,
                 match_array=None):

        if match_array is not None:

            # Get indices of sentences that contain at least one match
            successes = np.any(match_array, axis=1)

            match_array = match_array[successes]
            self.tokens = tokens[successes]
            self.tags = tags[successes]
            self.sentence_lengths = sentence_lengths[successes]
            self.n_sentences = self.tokens.shape[0]

            self._resolve_multiple_matches(match_array)

        else:

            assert(starts is not None)

            self.tokens = tokens
            self.tags = tags
            self.sentence_lengths = sentence_lengths
            self.starts = starts

            self.check_sentence_count()

    def _resolve_multiple_matches(self, match_array: np.ndarray):

        print("\tTotal number of sentences with at least one match: %d"
              % self.n_sentences)

        # Get total number of matches (individual sentences
        #   may have more than one)
        n_per = np.sum(match_array, axis=1)
        n_matches = np.sum(match_array)
        print("\tTotal number of matches: %d" % n_matches)

        # Create new arrays to copy data for sentences
        #   with more than one match
        temp_match_array = np.ndarray(
            (n_matches, match_array.shape[1]), match_array.dtype)
        temp_tokens = np.ndarray(
            (n_matches, self.tokens.shape[1]), self.tokens.dtype)
        temp_tags = np.ndarray(
            (n_matches, self.tags.shape[1], self.tags.shape[2]),
            self.tags.dtype)
        temp_sentence_lengths = np.ndarray(
            (n_matches), self.sentence_lengths.dtype)

        # Copy data to new array
        temp_match_array[:self.n_sentences] = match_array[:]
        temp_tokens[:self.n_sentences] = self.tokens[:]
        temp_tags[:self.n_sentences] = self.tags[:]

        temp_sentence_lengths[:self.n_sentences] = self.sentence_lengths[:]

        insert_index = self.n_sentences

        print("\n\tProcessing sentences with more than one match")

        for j in range(self.n_sentences):

            # Iterate over sentences with more than one match
            if n_per[j] > 1:

                copy = np.copy(match_array[j])

                # For each extra match
                while n_per[j] > 1:

                    # Remove the first instance of the match
                    # in the match_array
                    first_index = np.argmax(copy)
                    copy[first_index] = 0

                    # Copy the data for the sentence into
                    #   each of the new arrays
                    temp_match_array[insert_index][:] = copy[:]
                    temp_tokens[insert_index][:] = self.tokens[j][:]
                    temp_tags[insert_index][:] = self.tags[j][:]
                    temp_sentence_lengths[insert_index] = \
                        self.sentence_lengths[j]

                    n_per[j] -= 1
                    insert_index += 1

                    assert(n_per[j] == np.sum(copy))

        print("\tCompleted...")

        assert(insert_index == n_matches)

        # Get index to start search from per each sentence
        #   (note that argmax selects first index equivalent to max)
        # Copied sentences at end of match_array have had their first (k)
        #   instances of 1 removed
        starts = (temp_match_array).argmax(axis=1)
        self.starts = starts

        del temp_match_array
        del match_array

        self.tokens = temp_tokens
        self.tags = temp_tags
        self.sentence_lengths = temp_sentence_lengths

        self.check_sentence_count()

    def filter_valid_matches(self, n_template_tokens: int,
                             max_token: int=5000):

        print("\n\tLimiting matches with all tokens within %d most frequent"
              % (max_token))

        valid = None

        # Limit valid sentences to those in which all tokens within the
        #   matched phrase are limited to the first n_max indices
        for index in range(n_template_tokens):

            # Determine where in each sentence to check for classes
            check = self.starts + index

            # Obtain the token values for each matched phrase
            values = np.array(list(self.tokens[j][check[j]]
                                   for j in range(len(check))))
            # Boolean array mask
            values = (values < max_token)

            if valid is None:

                valid = values

            else:

                valid = np.logical_and(valid, values)

        n_valid = np.sum(valid)

        print("\tNumber of valid matches: %d" % (n_valid))

        self.tokens = self.tokens[valid]
        self.tags = self.tags[valid]
        self.sentence_lengths = self.sentence_lengths[valid]
        self.starts = self.starts[valid]

        self.check_sentence_count()

    def determine_subclasses(self, possible_classes: list,
                             n_template_tokens: int):

        merged_matches = list()

        # Class matching

        # Iterate over each token, checking which sentences match each sub-rule
        #   (as defined by tuples of possible_classes)
        constrained_tokens = set(range(n_template_tokens))

        for index in range(n_template_tokens):

            if possible_classes[index] is None:

                constrained_tokens.remove(index)
                merged_matches.append(None)
                continue

            assert(len(possible_classes[index]) != 0)

            # Determine where in each sentence to check for classes
            check = self.starts + index

            matches, counts = util.check_matched_indices(
                self.tags, check, possible_classes[index])

            merged_matches.append(matches)

        rule_types = dict()

        # Iterate over each matched sentence
        for sentence_number in range(self.n_sentences):

            subrules = list()

            for index in constrained_tokens:

                matches = merged_matches[index]

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

        self.subrule_indices = list()
        self.subrule_tokens = list()
        self.subrule_sentence_lengths = list()
        self.subrule_starts = list()

        # Iterate through each sub-rule
        for sub in range(len(subrules)):

            # Indices associated with each sub-rule
            selected_indices = np.array(rule_types[subrules[sub]])

            # Sentences associated with each sub-rule
            sub_rule_tokens = self.tokens[selected_indices]
            sub_rule_sentence_lengths = self.sentence_lengths[selected_indices]

            # Total number of sentences associated with each sub-rule
            n_subrule = len(sub_rule_tokens)

            # Only consider sub-rules with sentences matching
            if n_subrule != 0:

                print("\t\tNumber of sentences under sub-rule %d: %d" %
                      (sub + 1, n_subrule))

                self.subrule_indices.append(selected_indices)
                self.subrule_tokens.append(
                    list(sub_rule_tokens[i][1:sub_rule_sentence_lengths[i]]
                         for i in range(n_subrule)))
                self.subrule_sentence_lengths.append(
                    self.sentence_lengths[selected_indices].tolist())
                self.subrule_starts.append(
                    self.starts[selected_indices].tolist())

        self.n_subrules = len(self.subrule_indices)

    def check_sentence_count(self):

        self.n_sentences = self.tokens.shape[0]

        assert(self.tags.shape[0] == self.n_sentences)
        assert(self.sentence_lengths.shape[0] == self.n_sentences)
        assert(self.starts.shape[0] == self.n_sentences)

    @classmethod
    def merge(cls, matches: list):

        merged_tokens = np.vstack(list(m.tokens for m in matches))
        merged_tags = np.vstack(list(m.tags for m in matches))
        merged_sentence_lengths = \
            np.hstack(list(m.sentence_lengths for m in matches))
        merged_starts = np.hstack(list(m.starts for m in matches))

        merged = cls(merged_tokens, merged_tags,
                     merged_sentence_lengths, merged_starts)
        merged.check_sentence_count()

        return merged

    def permute(self, permutation: np.ndarray, n_out: int):

        self.tokens = self.tokens[permutation][:n_out]
        self.tags = self.tags[permutation][:n_out]
        self.sentence_lengths = self.sentence_lengths[permutation][:n_out]
        self.starts = self.starts[permutation][:n_out]

        self.check_sentence_count()


def match_correct(rule: Rule,
                  db: Database, stdb: SortedTagDatabase,
                  max_token: int=50000, n_search: int=-1,
                  n_max_out: int=50000, n_min_out: int=5000,
                  out_ratio: float=0.1, RS: RandomState=None,
                  ):

    print("\tFinding potential substitute tokens...")
    print(cfg['BREAK_SUBLINE'])
    substitute_tags = _find_substitute_tags(rule, stdb, max_token)

    # Find sentences matching template
    matches = \
        _find_template_sentences(rule, db, n_search, max_token,
                                 n_min_out, n_max_out, out_ratio, RS)

    print("\n\tCategorizing matches into sub-rules...")
    print(cfg['BREAK_SUBLINE'])

    matches.determine_subclasses(substitute_tags, rule.n_correct_tokens)

    return matches


def _find_substitute_tags(rule: Rule, stdb: SortedTagDatabase,
                          max_token: int):

    possible_tags = list()

    max_token = stdb.size if max_token == -1 else min(max_token, stdb.size)

    for index in range(rule.n_correct_tokens):

        index_tags = rule.correct_tags[index]

        # Syntactic tag indices that need to be matched exactly
        requisite_indices = np.where(rule.tag_mask[index] == 1)[0]
        # Syntactic tag indices that do not need to be matched
        lenient_indices = np.where(rule.tag_mask[index] == 0)[0]

        # Wildcard tokens (no requisite tags)
        if len(lenient_indices) == len(rule.tag_mask[index]):

            possible_tags.append(None)
            continue

        if len(requisite_indices) > 0:

            requisite_index_tags = index_tags[requisite_indices]

            possible_tokens = stdb.find_tokens(
                requisite_index_tags, requisite_indices, max_token)

        else:

            # Any token may substitute
            possible_tokens = np.arange(max_token)

        possible_tags.append(
            stdb.get_possible_tags(possible_tokens, index_tags,
                                   lenient_indices))

    return possible_tags


def _find_template_sentences(
        rule: Rule, db: Database, n_search: int, max_token: int,
        n_min_out: int, n_max_out: int, out_ratio: float,
        RS: RandomState):

    # Restrict number of sentences to search through
    if n_search == -1:
        n_search = db.n_sentences

    else:
        n_search = min(n_search, db.n_sentences)

    n_remaining = n_search
    n_partition = 0

    all_matches = list()

    for tokens, tags, sentence_lengths \
            in db.iterate_partitions(['f_token', 'f_tag', 'f_s_len']):

        n_sentences = len(tokens)

        print("\tProcessing partition: %d\n" % (n_partition + 1))

        if n_remaining == 0:

            break

        elif n_sentences > n_remaining:

            n_sentences = n_remaining

            tokens = tokens[:n_sentences]
            tags = tags[:n_sentences]
            sentence_lengths = sentence_lengths[:n_sentences]

        n_remaining -= n_sentences

        match_array = np.ones((
            tokens.shape[0], tokens.shape[1] - rule.n_correct_tokens + 1),
            dtype=np.bool)

        # Iterate over each syntactic tag index, restricting
        #   possible matches by indices with exact matching
        for i in range(rule.n_tags):

            # Syntactic tag matching leniency and tags of index i for
            #   each token
            index_mask = rule.tag_mask[:, i]
            index_tags = rule.correct_tags[:, i]

            # If there is leniency in the syntactic tag
            #    for all tokens, continue
            if np.all(index_mask != 1):

                pass

            else:

                match_array = np.logical_and(
                    match_array, util.search_template(
                        tags[:, :, i], np.argwhere(index_mask == 1),
                        index_tags, rule.n_correct_tokens))

        if isinstance(rule, CharacterRule):

            if rule.match_form:

                characters = np.load(db.get_file(n_partition, 'f_f_char'))
                lengths = np.load(db.get_file(n_partition, 'f_f_len'))

            else:

                characters = np.load(db.get_file(n_partition, 'f_t_char'))
                lengths = np.load(db.get_file(n_partition, 'f_t_len'))

            match_array = np.logical_and(
                match_array, rule.match_characters(characters, lengths))

        matches = TemplateMatch(
            tokens, tags, sentence_lengths, None, match_array)
        matches.filter_valid_matches(rule.n_correct_tokens, max_token)

        print('\n' + cfg['BREAK_SUBLINE'])

        all_matches.append(matches)
        n_partition += 1

    merged = TemplateMatch.merge(all_matches)

    print('\tTotal number of valid matches: %d' % merged.n_sentences)

    n_out = _get_final_output_count(
        merged.n_sentences, n_max_out=n_max_out, n_min_out=n_min_out,
        out_ratio=out_ratio)

    print('\tTotal number of outputted matches: %d' % n_out)

    perm = np.arange(merged.n_sentences) if RS is None \
        else RS.permutation(merged.n_sentences)

    merged.permute(perm, n_out)

    return merged


def _get_final_output_count(
        n_valid: int, n_max_out: int=-1, n_min_out: int=-1,
        out_ratio: float=0.1):
    """
    Determine number of sentences to output during sentence generation

    Args:
        n_valid (int): Number of valid sentences discovered
        n_max_out (int): Maximum number of sentences to output (-1 if no limit)
        n_min_out (int): Minimum number of sentences to output (-1 if no limit)
        out_ratio (float): Default percentage of valid sentences to output

    Returns:
        (int): Count of output sentences to use
    """
    n_out = int(n_valid * out_ratio)

    n_out = min(n_max_out, n_out) if n_max_out != -1 else n_out
    n_out = max(n_min_out, n_out) if n_min_out != -1 else n_out

    return min(n_out, n_valid)
