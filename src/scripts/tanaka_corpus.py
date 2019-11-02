"""
One-off script to split Tanaka corpus (http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)
    into component Japanese and English sentences.
"""
import csv
from .. import util


def split_tanaka_corpus(input_file, output_jp, output_en):
    """
    Function to split raw tanaka corpus (obtained from site
       ftp://ftp.monash.edu/pub/nihongo/examples.utf.gz)
       into parallel Japanese and English CSV files.

    Output files have one sentence per line.

    Args:
        input_file (str): Filepath to raw corpus file (examples.utf)
        output_jp (TYPE): Filepath to output CSV file of Japanese sentences
        output_en (TYPE): Filepath to output CSV file of English sentences
    """
    parallel_header = 'A: '
    parallel_delimiter = '\t'

    id_delimiter = '#ID='

    f_input = open(input_file, 'r')

    util.mkdir_p(output_jp, file=True)
    util.mkdir_p(output_en, file=True)

    jp_sentences = list()
    en_sentences = list()

    for line in f_input.readlines():

        if not line.startswith(parallel_header):

            continue

        line = line[len(parallel_header):]

        if id_delimiter not in line:

            continue

        id_idx = line.index(id_delimiter)
        line = line[:id_idx]

        parallel_text = line.split(parallel_delimiter)

        assert(len(parallel_text) == 2)

        jp_sentences.append([parallel_text[0]])
        en_sentences.append([parallel_text[1]])

    f_jp = open(output_jp, 'w+')
    jp_writer = csv.writer(f_jp)
    jp_writer.writerows(jp_sentences)

    f_en = open(output_en, 'w+')
    en_writer = csv.writer(f_en)
    en_writer.writerows(en_sentences)
