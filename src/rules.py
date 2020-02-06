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
from . import parse

from .languages import Language

cfg = config.parse()

R_PARAMS = cfg['rule_params']


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

        # Convert string representations to lists
        self.inserted = ast.literal_eval(inserted)
        self.modified = ast.literal_eval(modified)
        self.preserved = ast.literal_eval(preserved)
        self.substituted = ast.literal_eval(substituted)

        self.output_indices = dict()

        for i in self.inserted:
            self.output_indices[i] = ('Insert')

        for i in self.modified:
            self.output_indices[i[0]] = ('Modify', i[1])

        for i in self.preserved:
            self.output_indices[i[0]] = ('Preserve', i[1])

        for i in self.substituted:
            self.output_indices[i[0]] = ('Substitute', i[1])

    def get_output_length(self):

        return max(self.output_indices.keys())

    def iterate(self):

        for i in range(self.get_output_length() + 1 ):

            yield i, self.output_indices.get(i, ('*'))


class Rule:

    def __init__(self, rule_text: list, header_text: list,
                 token_language: Language, tag_languages: list):

        self.n_tags = len(tag_languages)

        self.number = int(rule_text[header_text.index(R_PARAMS['number'])])

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

        self.correct_tags = np.array(list(
            languages.parse_node_matrix(tags, tag_languages)
            for tags in np.array(self.tags_correct).T))

        self.error_tags = np.array(list(
            languages.parse_node_matrix(tags, tag_languages)
            for tags in np.array(self.tags_error).T))

        # Validate Python MeCab tags with text reference
        assert(np.array_equal(syntactic_tags, self.correct_tags))

        self.n_correct_tokens = len(self.tokens_correct)
        self.n_error_tokens = len(self.tokens_error)

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

            error = self.tokens_error[i]
            correct = self.tokens_correct[t[1]] if len(t) == 2 else ''
            operation = t[0]
            to_print.append('%s ~ %s, %s' % (error, correct, operation))

        print('\n'.join(to_print))


class CharacterRule(Rule):

    def __init__(self, rule_text: list, header_text: list,
                 token_language: Language, tag_languages: list):

        super().__init__(rule_text, header_text, token_language, tag_languages)

        self.n_correct_characters = len(self.template_correct)
        self.n_error_characters = len(self.template_error)

    def __str__(self):

        return super().__str__()

    def print_mapping(self):

        to_print = list()

        for i, t in self.mapping.iterate():

            error = self.template_error[i]
            correct = self.template_correct[t[1]] if len(t) == 2 else ''
            operation = t[0]
            to_print.append('%s ~ %s, %s' % (error, correct, operation))

        print('\n'.join(to_print))


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

                self.rule_dict[rule.number] = rule

    def print_rule(self, number):

        assert(number in self.rule_dict.keys())

        print('Rule %d: %s' % (number, str(self.rule_dict[number])))

    def iterate_rules(self, rule_index):

        if rule_index == -1:

            indices = sorted(i for i in self.rule_dict.keys())

        else:

            indices = [rule_index]

        for i in indices:

            yield self.rule_dict[i], i
