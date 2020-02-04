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

cfg = config.parse()

R_PARAMS = cfg['rule_params']


class Rule:

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

            # Convert string representations to lists
            self.inserted = ast.literal_eval(inserted)
            self.modified = ast.literal_eval(modified)
            self.preserved = ast.literal_eval(preserved)

        def get_output_length(self):

            return len(self.inserted) + len(self.modified) + \
                len(self.preserved)

    def __init__(self, rule_text: list, header_text: list,
                 tag_languages: list):

        self.n_tags = len(tag_languages)

        self.number = int(rule_text[header_text.index(R_PARAMS['number'])])

        # Template phrases
        self.template_correct = rule_text[
            header_text.index(R_PARAMS['template_correct_phrase'])]
        self.template_error = rule_text[
            header_text.index(R_PARAMS['template_error_phrase'])]

        self.rule_string = '%s --> %s' % \
            (self.template_error, self.template_correct)

        # Retrieve unencoded part-of-speech tags of the template correct phrase

        syntactic_tags = rule_text[
            header_text.index(R_PARAMS['syntactic_tags'])]
        syntactic_tags = syntactic_tags.split(',')

        # Convert part-of-speech tags to index form
        self.n_correct_tokens = int(len(syntactic_tags) / self.n_tags)
        self.syntactic_tags = np.array(list(languages.parse_node_matrix(
            syntactic_tags[i * self.n_tags: i * self.n_tags + self.n_tags],
            tag_languages) for i in range(self.n_correct_tokens)))

        # Array of arrays denoting hows part-of-speech tags have been selected
        # This is marked as -1 = null, 0 = no match, 1 = match
        tag_mask = rule_text[header_text.index(R_PARAMS['syntactic_tag_mask'])]
        tag_mask = np.array(list(int(j) for j in tag_mask.split(',')))
        self.tag_mask = tag_mask.reshape(-1, self.n_tags)

        self.mapping = self.TemplateMapping(rule_text, header_text)
        self.n_error_tokens = self.mapping.get_output_length()

    def __str__(self):

        return self.rule_string

    def get_mapping(self):

        return self.mapping.inserted, self.mapping.modified, \
            self.mapping.preserved


class RuleList:

    def __init__(self, rule_file: str, tag_languages: list,
                 ignore_first: bool=True):

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
                rule = Rule(line, header, tag_languages)

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
