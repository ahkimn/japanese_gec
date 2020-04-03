# -*- coding: utf-8 -*-

# Filename: kana.py
# Date Created: 21/03/2020
# Description: KanaList class and associated functions; used to
#   manipulate and transform Japanese kana
# Python Version: 3.7

import csv

from . import config

from enum import Enum

cfg = config.parse()

DIRECTORIES = cfg['directories']
M_PARAMS = cfg['morpher_params']


class CharacterShift(Enum):

    MATCH = 0
    CROSS_ROW = 1
    CROSS_COLUMN = 2
    NONE = 3


class KanaList:

    def __init__(self, kana_file: str):

        f = open(kana_file, 'r')

        self.kana_row = dict()
        self.kana_col = dict()

        self.n_col = 0

        self.index_kana = dict()

        f = open(kana_file, 'r')
        csv_reader = csv.reader(f, delimiter=',')

        row = 0

        for line in csv_reader:

            col = 0

            for kana in line:

                if not kana:

                    continue

                if kana in self.kana_row:

                    raise ValueError('Duplicate entry for kana: %s' % kana)

                self.kana_row[kana] = row
                self.kana_col[kana] = col

                self.index_kana[(row, col)] = kana

                col += 1

            row += 1

            self.n_col = max(col, self.n_col)

        self.n_row = row + 1
        self.n_col += 1

    def get_row(self, kana):

        return self.kana_row.get(kana)

    def get_col(self, kana):

        return self.kana_col.get(kana)

    def get_kana(self, indices):

        return self.index_kana.get(indices)

    def convert_kana(self, kana, template_start, template_end):

        if self.get_row(template_start) == self.get_row(template_end):

            if self.get_col(kana) == self.get_col(template_start):

                return self.get_kana((self.get_row(kana),
                                      self.get_col(template_end)))

            else:

                return None

        elif kana == template_start:

            return template_end

        else:
            return None

    def get_character_shift(self, template_start, template_end):

        if self.get_row(template_start) == self.get_row(template_end):

            if self.get_col(template_start) == self.get_col(template_end):

                return CharacterShift.MATCH

            return CharacterShift.CROSS_ROW

        elif self.get_col(template_start) == self.get_col(template_end):

            return CharacterShift.CROSS_COLUMN

        else:

            return CharacterShift.NONE

    def get_same_col(self, kana, include_original=True):

        col = self.get_col(kana)
        ret = []

        for i in range(self.n_row):

            k = self.get_kana((i, col))

            # Return all same-column kana e
            if k is not None and \
                    (include_original or self.get_row(kana) != i):

                ret.append(k)

        return ret
