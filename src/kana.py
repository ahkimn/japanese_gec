import csv

from . import config

cfg = config.parse()

DIRECTORIES = cfg['directories']
M_PARAMS = cfg['morpher_params']


class KanaList:

    def __init__(self, kana_file: str):

        f = open(kana_file, 'r')

        self.kana_row = dict()
        self.kana_col = dict()

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

    def get_row(self, kana):

        return self.kana_row.get(kana)

    def get_col(self, kana):

        return self.kana_col.get(kana)

    def get_kana(self, indices):

        return self.index_kana.get(indices)
