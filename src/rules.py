# -*- coding: utf-8 -*-

# Filename: rules.py
# Date Created: 24/12/2019
# Description: Rule and RuleList classes
# Python Version: 3.7

import ast
import csv
import numpy as np

from . import config
from . import languages
from . import morph
from . import parse

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
        input_indices = list()

        for i in self.inserted:
            self.output_indices[i] = (MapOperation.INSERTION, -1)

        for i in self.deleted:
            input_indices.append(i)

        for i in self.modified:
            self.output_indices[i[0]] = (MapOperation.MODIFICATION, i[1])
            input_indices.append(i[1])

        for i in self.preserved:
            self.output_indices[i[0]] = (MapOperation.PRESERVATION, i[1])
            input_indices.append(i[1])

        for i in self.substituted:
            self.output_indices[i[0]] = (MapOperation.SUBSTITUTION, i[1])
            input_indices.append(i[1])

        self.input_indices = set(input_indices)

    def get_output_length(self):

        return max(self.output_indices.keys())

    def iterate(self):

        for i in range(self.get_output_length() + 1):

            yield i, self.output_indices.get(i, (MapOperation.NONE, -1))


class Rule:

    def __init__(self, rule_text: list, header_text: list,
                 token_language: Language, tag_languages: list):

        self.n_tags = len(tag_languages)

        self.name = rule_text[header_text.index(R_PARAMS['name'])]

        # Template phrases
        self.template_correct = rule_text[
            header_text.index(R_PARAMS['template_correct_phrase'])]
        self.template_error = rule_text[
            header_text.index(R_PARAMS['template_error_phrase'])]

        self.rule_string = '%s ~ %s' % \
            (self.template_error, self.template_correct)

        # Retrieve unencoded part-of-speech tags of the template correct phrase
        syntactic_tags = rule_text[
            header_text.index(R_PARAMS['syntactic_tags'])]
        syntactic_tags = syntactic_tags.split(',')

        syntactic_tags = np.array(list(languages.parse_node_matrix(
            syntactic_tags[i * self.n_tags: i * self.n_tags + self.n_tags],
            tag_languages) for i in range(int(
                len(syntactic_tags) / self.n_tags))))

        # Parse template phrases
        self.tokens_correct, self.tags_correct = parse.parse_full(
            self.template_correct, parse.default_parser(), None)
        self.tokens_error, self.tags_error = parse.parse_full(
            self.template_error, parse.default_parser(), None)

        # Arrays to index over (tokens for base Rule class,
        #   characters for CharacterRule subclass)
        self.correct = self.tokens_correct.copy()
        self.error = self.tokens_error.copy()

        self.correct_tags = np.array(list(
            languages.parse_node_matrix(tags, tag_languages)
            for tags in np.array(self.tags_correct).T))

        self.error_tags = np.array(list(
            languages.parse_node_matrix(tags, tag_languages)
            for tags in np.array(self.tags_error).T))

        if not np.array_equal(syntactic_tags, self.correct_tags):
            self.correct_tags = syntactic_tags

            print('WARNING: Syntactic tags inconsistent with MeCab output')
            print('\tRule: %s' % self.rule_string)

        # Validate Python MeCab tags with text reference
        # assert(np.array_equal(syntactic_tags, self.correct_tags))

        self.n_correct_characters = len(self.template_correct)
        self.n_error_characters = len(self.template_error)

        self.n_correct_tokens = len(self.tokens_correct)
        self.n_error_tokens = len(self.tokens_error)

        self.n_correct = self.n_correct_tokens
        self.n_error = self.n_error_tokens

        # Array of arrays denoting hows part-of-speech tags have been selected
        # This is marked as -1 = null, 0 = no match, 1 = match
        tag_mask = rule_text[header_text.index(R_PARAMS['syntactic_tag_mask'])]
        tag_mask = np.array(list(int(j) for j in tag_mask.split(',')))
        self.tag_mask = tag_mask.reshape(-1, self.n_tags)

        self.mapping = TemplateMapping(rule_text, header_text)

    def __str__(self):

        return self.rule_string

    def get_mapping(self):

        return self.mapping.get_mapping()

    def print_mapping(self):

        to_print = list()

        for i, t in self.mapping.iterate():

            operation = t[0]
            error = self.error[i]
            correct = self.correct[t[1]] \
                if operation != MapOperation.NONE else ''
            to_print.append('%s ~ %s, %s' % (error, correct, operation.name))

        print('\n'.join(to_print))

    def get_deleted_indices(self):

        return set(range(self.n_correct)).difference(
            self.mapping.input_indices)

    def get_wildcard_indices(self):

        return set(range(self.n_error)).difference(
            self.mapping.output_indices.keys())

    def convert_phrases(
            self, correct_token: np.ndarray, correct_tags: np.ndarray,
            error_token: np.ndarray, error_tags: np.ndarray,
            token_language: Language, tag_languages: Language,
            stdb: SortedTagDatabase, n_sample: int=50):

        n_sentences = correct_token.shape[0]
        bin_mask = (self.tag_mask > 0)
        indices_error = token_language.parse_nodes(self.error)

        valid_indices = set(range(n_sentences))

        for e_idx, t in self.mapping.iterate():

            operation = t[0]
            c_idx = t[1]

            if operation == MapOperation.INSERTION:

                error_token[:, e_idx] = indices_error[e_idx]

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

                    if bin_mask[c_idx, i] or is_sub:

                        error_tags[:, e_idx, i] = e_tags[i]

                if is_sub:
                    base_forms = error_tags[:, e_idx, -1]
                    match_indices = [3]
                    assert(e_tags[-1] != c_tags[-1])

                else:
                    base_forms = correct_tags[:, c_idx, -1]
                    match_indices = \
                        np.argwhere(bin_mask[c_idx, :-1]).reshape(-1)
                    assert(e_tags[-1] == c_tags[-1])

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
                        correct_token[j, c_idx])

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

                    error_token[j, e_idx] = final_index

            elif operation == MapOperation.PRESERVATION:

                error_token[:, e_idx] = correct_token[:, c_idx]

            # No operation (Wildcard characters)
            else:

                raise ValueError

        return valid_indices


class CharacterRule(Rule):

    def __init__(self, rule_text: list, header_text: list,
                 token_language: Language, tag_languages: list):

        super().__init__(rule_text, header_text, token_language, tag_languages)

        self.n_correct = self.n_correct_characters
        self.n_error = self.n_error_characters

        self.correct = list(self.template_correct)
        self.error = list(self.template_error)

        self.n_error_tokens = 1

        self._verify_wildcard_indices()

    def _verify_wildcard_indices(self):

        self.left_offset = 0
        self.right_offset = 0

        w_i = self.get_wildcard_indices()

        for i in range(self.n_correct):

            if i in w_i:
                self.left_offset += 1
            else:
                break

        for i in range(self.n_correct)[::-1]:

            if i in w_i:
                self.right_offset += 1
            else:
                break

        # Make sure no wildcard indices exist in middle of template phrase
        assert(all(i < self.left_offset or i >= self.right_offset
                   for i in w_i))

        print(self.right_offset)
        print(self.left_offset)

    def convert_phrases(self, correct_token: np.ndarray,
                        error_token: np.ndarray, token_language: Language):

        n_sentences = correct_token.shape[0]
        diff = self.n_error - self.n_correct

        valid_indices = set(range(n_sentences))

        for j in range(n_sentences):

            correct_phrase = token_language.parse_indices(
                correct_token[j], delimiter='')
            error_phrase = correct_phrase[:self.left_offset]
            edit_phrase = correct_phrase[self.left_offset:self.right_offset]

            # Make sure matched phrase contains enough characters
            assert(len(correct_phrase) >= max(self.mapping.input_indices) + 1)

            for e_idx, t in self.mapping.iterate():

                operation = t[0]
                c_idx = t[1]

                if operation == MapOperation.INSERTION:

                    error_phrase += self.error[e_idx]

                # TODO: Clarify if this is necessary for character rules
                elif operation == MapOperation.MODIFICATION:

                    raise ValueError

                elif operation == MapOperation.PRESERVATION:

                    error_phrase += edit_phrase[c_idx]

                elif operation == MapOperation.SUBSTITUTION:

                    # TODO: Make this more comprehensive
                    #   (i.e. same row/column tokens)
                    error_phrase += self.error[e_idx]

                else:

                    raise ValueError

            error_phrase += correct_phrase[self.right_offset:]
            assert(len(error_phrase) == len(correct_phrase) + diff)
            error_token[j][0] = token_language.add_node(error_phrase)

        return valid_indices


class RuleList:

    def __init__(self, rule_file: str, token_language: Language,
                 tag_languages: list, ignore_first: bool=True):

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
                    rule = Rule(line, header, token_language, tag_languages)

                elif rule_type == R_PARAMS['type_character']:
                    rule = CharacterRule(line, header, token_language,
                                         tag_languages)

                self.rule_dict[rule.name] = rule

    def print_rule(self, name):

        assert(name in self.rule_dict.keys())

        rule = self.rule_dict[name]

        print('Rule %s: %s' % (name, str(rule)))
        print('Mapping:')
        rule.print_mapping()

    def iterate_rules(self, rule_index):

        if rule_index == -1:

            indices = sorted(i for i in self.rule_dict.keys())

        else:

            indices = [rule_index]

        for i in indices:

            yield self.rule_dict[i], i
