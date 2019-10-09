"""
Script to batch-translate input sentences using Google Translate API and
    the python library googletrans https://pypi.org/project/googletrans/
"""
import csv

from googletrans import Translator
from .. import util


def translate_sentences(input_file, output_file,
                        src='en', dst='ja', batch_size=32, display_every=100):
    """
    Function to batch-translate sentences from %input_file% and save outputs
        to file %output_file%

    Args:
        input_file (str): Filepath of file containing sentences to translate
        output_file (TYPE): Filepath to file where translations will be written
        src (str, optional): Language of %input_file%
        dst (str, optional): Desired output language
        batch_size (int, optional): Size of sentence batches to translate
        display_every (int, optional): Frequency of outputs
    """
    print('Reading text from file: %s' % input_file)

    f_input = open(input_file, 'r+')
    reader = csv.reader(f_input)

    translator = Translator()

    last_notified_at = 0
    n_completed = 0

    translations = list()

    for batch in util.iter_batch(reader, batch_size):

        batched_lines = list(line[0] for line in batch)
        results = translator.translate(batched_lines, src=src, dest=dst)

        translations += list([result.text] for result in results)

        n_completed += len(batched_lines)

        if n_completed - last_notified_at > display_every:

            print('Sentences completed: %2d' % n_completed)
            last_notified_at = n_completed

    f_input.close()

    util.mkdir_p(output_file, file=True)

    print('Writing translations to file: %s' % output_file)

    f_output = open(output_file, 'w+')
    writer = csv.writer(f_output)

    writer.writerows(translations)
    f_output.close()
