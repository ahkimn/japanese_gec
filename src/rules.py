# -*- coding: utf-8 -*-

# Filename: rules.py
# Date Created: 24/12/2019
# Description: Rule and RuleList classes; act as representations of
#   Japanese grammatical errors and their corrections
# Python Version: 3.7

# TODO:
#   Improve efficiency of character rule generation
#       - FOR loop over individual kana is time-consuming
#   Fix filtering process for valid matches
#       - Substring checks for cases where forms are used to match
#           may remove valid entries

import ast
import csv
import numpy as np

from . import config
from . import languages
from . import morph
from . import parse
from . import util

from .kana import CharacterShift, KanaList
from .sorted_tag_database import SortedTagDatabase
from .languages import Language
from enum import Enum

cfg = config.parse()

R_PARAMS = cfg['rule_params']
P_PARAMS = cfg['parser_params']


class MapOperation(Enum):

    INSERTION = 0
    DELETION = 1
    MODIFICATION = 2
    PRESERVATION = 3
    SUBSTITUTION = 4
    NONE = 5


class MatchType(Enum):

    FULL_MATCH = 0
    RIGHT_MATCH = 1
    LEFT_MATCH = 2
    ANY_MATCH = 3
    NO_MATCH = 4


class TemplateMapping:

    def __init__(self, rule_text: list, header_text: list):

        # Arrays of tuples denoting token mappings between errored
        #   and correct sentence
        inserted = rule_text[
            header_text.index(R_PARAMS['mapping_inserted'])]
        modified = rule_text[
            header_text.index(R_PARAMS['mapping_modified'])]
        preserved = rule_text[
            header_text.index(R_PARAMS['mapping_preserved'])]
        substituted = rule_text[
            header_text.index(R_PARAMS['mapping_substituted'])]
        deleted = rule_text[
            header_text.index(R_PARAMS['mapping_deleted'])]

        # Convert string representations to lists
        self.inserted = ast.literal_eval(inserted)
        self.modified = ast.literal_eval(modified)
        self.preserved = ast.literal_eval(preserved)
        self.substituted = ast.literal_eval(substituted)
        self.deleted = ast.literal_eval(deleted)

        self.output_indices = dict()
        self.input_indices = dict()

        for i in self.inserted:
            self._update_output(MapOperation.INSERTION, i, -1)

        for i in self.deleted:
            self._update_input(MapOperation.DELETION, i, -1)

        for i in self.modified:
            self._update_output(MapOperation.MODIFICATION, i[0], i[1])
            self._update_input(MapOperation.MODIFICATION, i[1], i[0])

        for i in self.preserved:
            self._update_output(MapOperation.PRESERVATION, i[0], i[1])
            self._update_input(MapOperation.PRESERVATION, i[1], i[0])

        for i in self.substituted:
            self._update_output(MapOperation.SUBSTITUTION, i[0], i[1])
            self._update_input(MapOperation.SUBSTITUTION, i[1], i[0])

    def get_output_length(self):

        return max(self.output_indices.keys()) if self.output_indices else 0

    def get_input_length(self):

        return max(self.input_indices.keys()) if self.output_indices else 0

    def _update_input(self, operation, input_index, output_index):

        if input_index != -1 and input_index in self.input_indices.keys():
            print('WARNING: Input index: %d doubly specified' % input_index)

        self.input_indices[input_index] = (operation, output_index)

    def _update_output(self, operation, output_index, input_index):

        if output_index != -1 and output_index in self.output_indices.keys():
            raise ValueError('Output index: %d doubly specified' %
                             output_index)

        else:
            self.output_indices[output_index] = (operation, input_index)

    def iterate_output(self, n: int, start: int=0):

        for i in range(start, n):

            yield i, self.output_indices.get(i, (MapOperation.NONE, -1))

    def iterate_input(self, n: int, start: int=0):

        for i in range(start, n):

            yield i, self.input_indices.get(i, (MapOperation.NONE, -1))


class Rule:

    def __init__(self, rule_text: list, header_text: list,
                 token_language: Language, tag_languages: list,
                 token_delimiter: str=',', assert_fully_mapped: bool=False):

        self.n_tags = len(tag_languages)
        self.name = rule_text[header_text.index(R_PARAMS['name'])]
        self.token_delimiter = token_delimiter

        # Template phrases
        self.template_correct = rule_text[
            header_text.index(R_PARAMS['template_correct_phrase'])].replace(
                self.token_delimiter, '')
        self.template_error = rule_text[
            header_text.index(R_PARAMS['template_error_phrase'])].replace(
                self.token_delimiter, '')

        self.rule_string = '%s ~ %s' % \
            (self.template_error, self.template_correct)

        print('Rule: %s' % self.rule_string)

        # Parse template phrases
        tokens_correct, tags_correct = parse.parse_full(
            self.template_correct, parse.default_parser(), None)
        tokens_error, tags_error = parse.parse_full(
            self.template_error, parse.default_parser(), None)

        tags_correct = np.array(list(
            languages.parse_node_matrix(tags, tag_languages)
            for tags in np.array(tags_correct).T))

        tags_error = np.array(list(
            languages.parse_node_matrix(tags, tag_languages)
            for tags in np.array(tags_error).T))

        self.tokens_correct, self.correct_tags = \
            self._check_tokens_and_tags(tokens_correct, tags_correct,
                                        rule_text, header_text,
                                        'correct', tag_languages)
        self.tokens_error, self.error_tags = \
            self._check_tokens_and_tags(tokens_error, tags_error,
                                        rule_text, header_text,
                                        'error', tag_languages)

        # Validate Python MeCab tags with text reference
        # assert(np.array_equal(syntactic_tags, self.correct_tags))

        self.n_correct_characters = len(self.template_correct)
        self.n_error_characters = len(self.template_error)

        self.n_correct_tokens = len(self.tokens_correct)
        self.n_error_tokens = len(self.tokens_error)

        self.len_correct_tokens = [len(t) for t in self.tokens_correct]
        self.len_error_tokens = [len(t) for t in self.tokens_error]

        assert(self.correct_tags.shape[0] == self.n_correct_tokens)
        assert(self.error_tags.shape[0] == self.n_error_tokens)

        # Array of arrays denoting hows part-of-speech tags have been selected
        # This is marked as -1 = null, 0 = no match, 1 = match
        tag_mask = rule_text[header_text.index(R_PARAMS['syntactic_tag_mask'])]
        tag_mask = np.array(list(int(j) for j in tag_mask.split(',')))
        self.tag_mask = tag_mask.reshape(-1, self.n_tags)

        assert(self.tag_mask.shape[0] == self.n_correct_tokens)

        self.mapping = TemplateMapping(rule_text, header_text)

        if assert_fully_mapped:

            fully_mapped = self._check_full_mapping()
            if not fully_mapped:

                print(self.mapping.input_indices.keys())
                print(self.mapping.output_indices.keys())

                raise ValueError('Rule tokens not fully mapped')

        self.n_error_mapping = self.n_error_tokens
        self.n_correct_mapping = self.n_correct_tokens

    def _check_full_mapping(self):

        mapping_correct = self.mapping.input_indices
        mapping_error = self.mapping.output_indices

        if set(mapping_correct.keys()) != \
            set(range(self.n_correct_tokens)) or \
                set(mapping_error.keys()) != \
                set(range(self.n_error_tokens)):

            return False

        return True

    def _check_tokens_and_tags(self, tokens: list, syntactic_tags: np.ndarray,
                               rule_text: list, header_text: list,
                               phrase: str, tag_languages: list):

        assert(phrase in ['correct', 'error'])

        loaded_tokens = rule_text[
            header_text.index(R_PARAMS['template_%s_phrase' % phrase])].split(
                self.token_delimiter)

        if not tokens == loaded_tokens:

            tokens = loaded_tokens

            print('WARNING: Tokens of ' + phrase + ' phrase' +
                  ' inconsistent with MeCab output')

        # Retrieve unencoded part-of-speech tags of the template correct phrase
        loaded_tags = rule_text[
            header_text.index(R_PARAMS['syntactic_tags_%s' % phrase])]
        loaded_tags = loaded_tags.split(',')

        loaded_tags = np.array(list(languages.parse_node_matrix(
            loaded_tags[i * self.n_tags: i * self.n_tags + self.n_tags],
            tag_languages) for i in range(int(
                len(loaded_tags) / self.n_tags))))

        if not np.array_equal(syntactic_tags, loaded_tags):

            syntactic_tags = loaded_tags

            print('WARNING: Syntactic tags of ' + phrase + ' phrase' +
                  ' inconsistent with MeCab output')

        return tokens, syntactic_tags

    def __str__(self):

        return self.rule_string

    def get_mapping(self):

        return self.mapping.get_mapping()

    def print_mapping(self):

        to_print = list()
        to_print.append('Mapping: Error -> Correct')

        for i, t in self.mapping.iterate_output(self.n_error_mapping):

            operation = t[0]
            error = self.template_error[i]
            correct = self.template_correct[t[1]] \
                if (operation != MapOperation.NONE and
                    operation != MapOperation.INSERTION) else '*'
            to_print.append('%s ~ %s, %s' % (error, correct, operation.name))

        to_print.append('Mapping: Correct -> Error')

        for i, t in self.mapping.iterate_input(self.n_correct_mapping):

            operation = t[0]
            error = self.template_error[t[1]] \
                if (operation != MapOperation.NONE and
                    operation != MapOperation.DELETION) else '*'
            correct = self.template_correct[i]
            to_print.append('%s ~ %s, %s' % (correct, error, operation.name))

        print('\n'.join(to_print))

    def convert_phrases(
            self, correct_tokens: np.ndarray, correct_tags: np.ndarray,
            error_tokens: np.ndarray, error_tags: np.ndarray,
            token_language: Language, tag_languages: Language,
            stdb: SortedTagDatabase, n_sample: int=50):

        n_sentences = correct_tokens.shape[0]
        bin_mask = (self.tag_mask > 0)
        indices_error = token_language.parse_nodes(self.tokens_error)

        valid_indices = set(range(n_sentences))

        for e_idx, t in self.mapping.iterate_output(self.n_error_tokens):

            try:

                operation = t[0]
                c_idx = t[1]

                if operation == MapOperation.INSERTION:

                    error_tokens[:, e_idx] = indices_error[e_idx]

                elif operation == MapOperation.MODIFICATION or \
                        operation == MapOperation.SUBSTITUTION:

                    is_sub = (operation == MapOperation.SUBSTITUTION)

                    error_tags[:, e_idx, :] = correct_tags[:, c_idx, :]
                    e_tags = self.error_tags[e_idx]
                    c_tags = self.correct_tags[c_idx]

                    diff = (e_tags != c_tags)

                    for i in range(self.n_tags):

                        if not diff[i]:
                            continue

                        error_tags[:, e_idx, i] = e_tags[i]

                    if is_sub:
                        base_forms = error_tags[:, e_idx, -1]
                        match_indices = [3]
                        assert(e_tags[-1] != c_tags[-1])

                    else:
                        base_forms = correct_tags[:, c_idx, -1]
                        match_indices = \
                            np.argwhere(bin_mask[c_idx, :-1]).reshape(-1)
                        # assert(e_tags[-1] == c_tags[-1])

                    morpher = morph.Morpher((self.tokens_correct[c_idx],
                                             self.tokens_error[e_idx]))

                    print('\tAlteration of token: %s' % morpher.get_rule())

                    n_print = 0
                    print_perm = np.random.permutation(
                        n_sentences)[:n_sample].tolist()

                    for j in range(n_sentences):

                        printed = False

                        final_index = -1
                        final_token = ''
                        base_token = token_language.parse_index(
                            correct_tokens[j, c_idx])

                        sub = stdb.find_tokens_from_form(
                            base_forms[j], error_tags[j, e_idx], match_indices)

                        if sub is not None:

                            final_token = token_language.parse_index(sub)

                            if morpher.verify(base_token, final_token):

                                final_index = sub

                        if final_index == -1:

                            # sub_token = morpher.morph(base_token)
                            final_token = \
                                morpher.morph_pos(
                                    base_token, base_forms[j], token_language,
                                    tag_languages, parse.default_parser(),
                                    error_tags[j, e_idx], match_indices)

                            # If a valid modified token is found
                            if final_token is not None:

                                if n_print < n_sample:

                                    print('\t\tMorph gen %d: %s -> %s' %
                                          (j + 1, base_token, final_token))

                                    n_print += 1
                                    printed = True

                                final_index = token_language.add_node(
                                    final_token)

                        # If no valid modified token is found
                        if final_index == -1:

                            if j in valid_indices:

                                valid_indices.remove(j)

                            continue

                        assert(final_index != -1)

                        if j in print_perm and not printed:

                            print('\t\tMatch %d: %s -> %s' %
                                  (j + 1, base_token, final_token))

                        error_tokens[j, e_idx] = final_index

                elif operation == MapOperation.PRESERVATION:

                    error_tokens[:, e_idx] = correct_tokens[:, c_idx]

                # No operation (Wildcard characters)
                else:

                    raise ValueError

            except Exception:

                raise

        return valid_indices


class CharacterRule(Rule):

    def __init__(self, rule_text: list, header_text: list,
                 character_language: list, token_language: Language,
                 tag_languages: list, KL: KanaList):

        super().__init__(rule_text, header_text, token_language, tag_languages)

        self.characters_correct = character_language.parse_nodes(
            self.template_correct)
        self.characters_error = character_language.parse_nodes(
            self.template_error)

        # Rule generates on per-token basis
        self.n_error_tokens = self.n_correct_tokens

        # If rule is conjugation-specific, match with characters
        # Otherwise, match with base_form
        self.match_forms = [self.tag_mask[j][-2] != 1
                            for j in range(self.n_correct_tokens)]

        self._get_correct_token_starts()
        self._verify_wildcard_indices()

        self._get_sub_characters(character_language, KL)

        self.n_error_mapping = self.n_error_characters
        self.n_correct_mapping = self.n_correct_characters

    def get_wildcard_indices(self):

        return set(range(self.n_correct_characters)).difference(
            self.mapping.input_indices.keys())

    def _get_correct_token_starts(self):

        self.correct_token_starts = [0]
        self.correct_token_ends = []
        for i in range(self.n_correct_tokens - 1):
            self.correct_token_starts.append(
                self.correct_token_starts[-1] + self.len_correct_tokens[i])
            self.correct_token_ends.append(self.correct_token_starts[-1])
        self.correct_token_ends.append(self.n_correct_characters)

        self.char_to_token_index = {-1: -1}
        t_idx = 0
        for i in range(self.n_correct_characters):
            if i >= self.correct_token_ends[t_idx]:
                t_idx += 1
            self.char_to_token_index[i] = t_idx

    def _verify_wildcard_indices(self):

        self.left_offsets = []
        self.right_offsets = []
        self.n_matched_indices = []
        self.match_types = []

        w_i = self.get_wildcard_indices()

        for i in range(self.n_correct_tokens):

            n_characters = self.len_correct_tokens[i]
            token_start = self.correct_token_starts[i]
            token_end = self.correct_token_ends[i]

            left_offset = 0
            right_offset = n_characters

            token_chars = set(range(token_start, token_end))
            matched_indices = token_chars.intersection(w_i)

            self.n_matched_indices.append(n_characters - len(matched_indices))

            for j in range(n_characters):

                if j + token_start in matched_indices:
                    left_offset += 1
                else:
                    break

            for j in range(n_characters)[::-1]:

                if j + token_start in matched_indices:
                    right_offset -= 1
                else:
                    break

            # Make sure no wildcard indices exist in middle of token
            assert(all(i - token_start < left_offset or
                       i - token_start >= right_offset
                       for i in matched_indices))

            if left_offset == 0 and right_offset == n_characters:
                self.match_types.append(MatchType.FULL_MATCH)

            elif left_offset != 0 and right_offset == n_characters:
                self.match_types.append(MatchType.RIGHT_MATCH)

            elif left_offset == 0 and right_offset != n_characters:
                self.match_types.append(MatchType.LEFT_MATCH)

            elif matched_indices == token_chars:
                self.match_types.append(MatchType.NO_MATCH)
                # Make sure that unmatched tokens are present
                #   completely in error
                assert(self.tokens_correct[i] in self.template_error)

            else:
                self.match_types.append(MatchType.ANY_MATCH)

            self.left_offsets.append(left_offset)
            self.right_offsets.append(right_offset)

    def _get_sub_characters(self, character_language: Language, KL: KanaList):

        self.sub_characters = []
        self.search_characters = []
        self.search_masks = []
        self.search_indices = []

        for i in range(self.n_correct_tokens):

            token_start = self.correct_token_starts[i]
            token_end = self.correct_token_ends[i]

            token_sub_characters = []
            token_search_characters = []
            token_search_mask = []
            token_search_indices = []

            for c_idx, t in self.mapping.iterate_input(
                    self.n_correct_characters):

                operation = t[0]

                if c_idx < token_start or c_idx >= token_end \
                        or operation == MapOperation.NONE:
                    continue

                idx_start = self.characters_correct[c_idx]
                char_start = self.template_correct[c_idx]
                char_edit = self.template_error[t[1]]

                index_sub_characters = set()
                shift_type = KL.get_character_shift(char_start, char_edit)
                token_search_characters.append(idx_start)
                token_search_indices.append(c_idx - token_start)

                # TODO: Force substitutiosn to be CROSS_ROW/CROSS_COLUMN
                #   Arbitrary subtitutions should be INSERT/DELETE pairs
                if operation == MapOperation.SUBSTITUTION \
                        and shift_type == CharacterShift.CROSS_ROW:

                    token_search_mask.append(False)

                    for k in KL.get_same_col(char_start,
                                             include_original=True):

                        # Only do single character kana
                        if len(k) > 1:

                            pass

                        index_sub_characters.add(
                            character_language.add_node(k))

                else:

                    token_search_mask.append(True)

                token_sub_characters.append(index_sub_characters)

            assert(self.n_matched_indices[i] == len(token_search_indices))

            self.sub_characters.append(token_sub_characters)
            self.search_characters.append(
                np.array(token_search_characters, dtype=np.uint32))
            self.search_masks.append(
                np.array(token_search_mask, dtype=np.bool))
            self.search_indices.append(token_search_indices)

    def match_characters(
            self, form_characters: np.ndarray, form_lengths: np.ndarray,
            token_characters: np.ndarray, token_lengths: np.ndarray):

        _, max_len, max_char = form_characters.shape

        # Place token axis first for easier readability
        form_characters = form_characters.swapaxes(0, 1)
        form_lengths = form_lengths.swapaxes(0, 1)

        token_characters = token_characters.swapaxes(0, 1)
        token_lengths = token_lengths.swapaxes(0, 1)

        ret_len = max_len - self.n_correct_tokens + 1
        ret = np.ones((ret_len, form_characters.shape[1]), dtype=np.bool)

        # Match characters for each token
        for i in range(self.n_correct_tokens):

            search_start = i
            search_end = max_len - (self.n_correct_tokens - i - 1)

            match_mask = self.search_masks[i]
            match_characters = self.search_characters[i]

            n_indices = len(match_characters)

            # Skip character filtering if token has no characters to match
            if n_indices == 0:
                continue

            if self.match_forms[i]:

                # Limit search using phrase length
                search_characters = \
                    form_characters[search_start:search_end,
                                    :, :].reshape(-1, max_char)
                search_lengths = \
                    form_lengths[search_start:search_end, :]

            else:

                # Limit search using phrase length
                search_characters = \
                    token_characters[search_start:search_end,
                                     :, :].reshape(-1, max_char)
                search_lengths = \
                    token_lengths[search_start:search_end, :]

            match_array = util.search_2d_masked(
                search_characters, match_characters, match_mask)

            n_matches = np.sum(match_array, axis=1)

            # Ignore cases with multiple matches (for simplicity)
            if self.match_types[i] == MatchType.ANY_MATCH:

                valid = np.where(n_matches == 1)[0]

            # TODO: Do last case/first case depending on left/right offset
            else:

                valid = np.where(n_matches > 0)[0]

            valid_matches = match_array[valid]
            valid_lengths = search_lengths.reshape(-1)[valid]

            if self.match_types[i] == MatchType.RIGHT_MATCH:

                # Match last match
                match_index = max_char - \
                    np.argmax(np.flip(valid_matches, axis=1), axis=1) \
                    - n_indices

            else:

                # Match first match
                match_index = np.argmax(valid_matches, axis=1)

            match_end = match_index + n_indices

            f1 = (match_end == valid_lengths)
            f2 = (match_index == 0)

            # Match full token
            if self.match_types[i] == MatchType.FULL_MATCH:

                f3 = np.logical_and(f1, f2)

                valid = valid[np.where(f3)]
                match_index = match_index[np.where(f3)]

            # Match on right side of token
            elif self.match_types[i] == MatchType.RIGHT_MATCH:

                valid = valid[np.where(f1)]
                match_index = match_index[np.where(f1)]

            # Match on left side of token
            elif self.match_types[i] == MatchType.LEFT_MATCH:

                valid = valid[np.where(f2)]
                match_index = match_index[np.where(f2)]

            n_valid = len(valid)
            sub_valid = np.ones(n_valid, dtype=np.bool)

            # Check if characters in substitutions are valid
            #   Outside of main search loop for efficiency
            for j in range(n_indices):

                sub_characters = self.sub_characters[i][j]

                if not sub_characters:
                    continue
                check_chars = search_characters[valid, :]
                check_indices = match_index + j

                for k in range(n_valid):
                    if check_chars[k][check_indices[k]] not in sub_characters:
                        sub_valid[k] = False

            valid = valid[np.where(sub_valid)]
            match = np.zeros(search_lengths.size, dtype='bool')
            match[valid] = True

            ret = np.logical_and(ret, match.reshape(search_lengths.shape))

        return ret.swapaxes(0, 1)

    def convert_phrases(self, correct_tokens: np.ndarray,
                        error_tokens: np.ndarray,
                        correct_forms: np.ndarray,
                        token_language: Language,
                        form_language: Language,
                        KL: KanaList):

        n_sentences = correct_tokens.shape[0]
        diff = self.n_error_characters - self.n_correct_characters

        valid_indices = set(range(n_sentences))

        for i in range(n_sentences):

            phrase_diff = 0
            parsed_e_idx = 0

            try:

                for j in range(self.n_correct_tokens):

                    n_wildcard = self.len_correct_tokens[j] - \
                        len(self.search_indices[j])
                    token_start = self.correct_token_starts[j]

                    # # Make sure matched phrase contains enough characters
                    # assert(len(correct_token) >= self.n_matched_indices)

                    correct_token = token_language.parse_index(
                        correct_tokens[i][j])
                    correct_form = form_language.parse_index(
                        correct_forms[i][j])

                    if self.match_forms[j] and \
                            self.match_types[j] != MatchType.NO_MATCH:

                        correct = correct_form

                        # Poor way of filtering
                        # TODO: Make this not fail on edge cases
                        #  (e.g. volitional form when matching verb tokens)
                        if correct_token not in correct:
                            raise ValueError('Form matching failed')
                        c_off = correct.index(correct_token)

                        if c_off != 0:

                            assert (c_off + len(correct_token) == len(correct))
                            e_off = 0

                        else:

                            e_off = len(correct) - len(correct_token)

                    else:

                        correct = correct_token
                        c_off = 0
                        e_off = 0

                    align_offset = self.left_offsets[j]

                    if self.match_types[j] == MatchType.FULL_MATCH:

                        edit = correct[:]
                        error = ''

                    elif self.match_types[j] == MatchType.RIGHT_MATCH:

                        edit = correct[-self.n_matched_indices[j]:]
                        error = correct[:-self.n_matched_indices[j]]

                    elif self.match_types[j] == MatchType.LEFT_MATCH:

                        edit = correct[:self.n_matched_indices[j]]
                        error = ''

                    elif self.match_types[j] == MatchType.NO_MATCH:

                        error = correct[:]
                        edit = ''

                    # When token matching does not occur at token boundaries,
                    #   align matched token with correct token by searching
                    #   for common characters
                    else:

                        valid_align = False

                        for k in range(self.n_matched_indices[j]):

                            # Operation must be non-morphing substitution
                            if not self.search_masks[j][k]:
                                continue

                            # Offset of the alignment character in correct token
                            t_offset = self.search_indices[j][k]
                            align_char = self.tokens_correct[j][t_offset]

                            if align_char not in correct:
                                continue
                            # Cause exception here
                            align_index = correct.index(align_char)
                            align_offset = align_index - \
                                (t_offset - self.left_offsets[j])

                            edit = correct[align_offset:align_offset +
                                           self.n_matched_indices[j]]
                            error = correct[:align_offset]
                            valid_align = True

                        if not valid_align:
                            raise ValueError('No alignment for token %s found'
                                             % correct)

                    parsed_wildcard = 0

                    # print('CORRECT: %s' % correct)

                    for e_idx, t in self.mapping.iterate_output(
                            self.n_error_characters):

                        operation = t[0]
                        c_idx = t[1]
                        edit_index = c_idx - (token_start + align_offset)

                        # print(n_wildcard, parsed_wildcard)
                        # print(e_idx, parsed_e_idx, operation)

                        if e_idx < parsed_e_idx:

                            continue

                        elif (operation != MapOperation.INSERTION and
                              operation != MapOperation.NONE) and \
                                self.char_to_token_index[c_idx] != j:

                            continue

                        elif operation == MapOperation.INSERTION:

                            if parsed_e_idx != e_idx:

                                continue

                            error += self.template_error[e_idx]

                        elif operation == MapOperation.PRESERVATION:

                            error += edit[edit_index]

                        elif operation == MapOperation.SUBSTITUTION:

                            # TODO: Make this more comprehensive
                            #   (i.e. same row/column tokens)
                            error += KL.convert_kana(
                                edit[edit_index],
                                self.template_correct[c_idx],
                                self.template_error[e_idx])

                        elif operation == MapOperation.NONE:

                            if parsed_wildcard >= n_wildcard:

                                break

                            parsed_wildcard += 1

                        else:

                            raise ValueError("Invalid operation type present")

                        parsed_e_idx += 1
                        # print("INCREMENT")

                    if self.match_types[j] == MatchType.LEFT_MATCH:

                        error += correct[self.n_matched_indices[j]:]

                    elif self.match_types[j] == MatchType.ANY_MATCH:

                        error += \
                            correct[align_offset + self.n_matched_indices[j]:]

                    e_off = len(error) - e_off
                    error = error[c_off:e_off]

                    error_tokens[i][j] = token_language.add_node(error)
                    # print('ERROR: %s' % error)

                    phrase_diff += len(error) - len(correct_token)

            except Exception:

                print('Warning: Failed to synthesize token for: %s' %
                      ''.join(correct_token))

                valid_indices.remove(i)
                continue

            assert(phrase_diff == diff)

        return valid_indices


class RuleList:

    def __init__(self, rule_file: str, character_language: Language,
                 token_language: Language, tag_languages: list,
                 KL: KanaList, ignore_first: bool=True):

        self.rule_dict = dict()

        line_count = 0
        rule_count = 0

        f = open(rule_file, 'r')

        csv_reader = csv.reader(f, delimiter=',')

        header = next(csv_reader)

        # Read each line (rule) of CSV
        for line in csv_reader:

            line_count += 1

            # Ignore first line
            if line_count == 0 and ignore_first:

                continue

            # Ignore comments
            elif len(line) > 2 and line[0] != '#':

                rule_count += 1
                rule_type = line[header.index(R_PARAMS['rule_type'])]

                if rule_type == R_PARAMS['type_token']:
                    rule = Rule(line, header, token_language, tag_languages,
                                assert_fully_mapped=True)

                elif rule_type == R_PARAMS['type_character']:
                    rule = CharacterRule(line, header, character_language,
                                         token_language, tag_languages, KL=KL)

                self.rule_dict[rule.name] = rule

    def get_rule(self, name):

        return self.rule_dict[name]

    def print_rule(self, name):

        assert(name in self.rule_dict.keys())

        rule = self.rule_dict[name]

        print('Rule %s: %s' % (name, str(rule)))
        print(cfg['BREAK_LINE'])
        rule.print_mapping()

    def iterate_rules(self, rule_index):

        if rule_index == '-1':

            indices = sorted(i for i in self.rule_dict.keys())

        else:

            indices = [rule_index]

        for i in indices:

            yield self.rule_dict[i], i
