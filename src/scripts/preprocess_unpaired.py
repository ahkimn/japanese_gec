# TODO

import csv
import os
import MeCab

from .. import languages

CONST_PARSER = MeCab.Tagger()


def pre_process_unpaired_files(
        input_dir, output_file,
        delimiter='。', indicators=None,
        ignore_header=True, ext='.csv'
):

    f_count = 1
    output_f = open(output_file, 'w+')

    if indicators is None:

        indicators = list(delimiter)

    for f_name in os.listdir(input_dir):

        data = dict()

        input_f = os.path.join(input_dir, f_name)
        input_f = open(input_f, 'r')

        csv_reader = csv.reader(input_f)

        row_count = -1

        for row in csv_reader:

            row_count += 1

            if ignore_header and not row_count:
                continue

            col = 0

            for cell in row:

                col += 1

                if any(i in cell for i in indicators):

                    if col not in data:

                        data[col] = list()

                    for sentence in cell.strip().split(delimiter):

                        if sentence == '':
                            continue

                        sentence += delimiter
                        tokens = languages.parse(sentence, CONST_PARSER, None)
                        data[col].append(' '.join(tokens))

        input_f.close()
        sentences = list()

        for key in sorted(data.keys()):

            sentences += set(data[key])

        for sentence in sentences:
            output_f.write(sentence)
            output_f.write(os.linesep)

        f_count += 1

    output_f.close()
