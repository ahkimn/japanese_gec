import MeCab
import time
import yaml

from gensim.models import FastText

from . import languages
from . import util

cfg = yaml.load(open('./config.yml', 'r'), Loader=yaml.FullLoader)

BREAK_LINE = cfg['print_params']['break_line']
DELIMITER_TOKEN = cfg['parser_params']['delimiter_token']
PARSER = MeCab.Tagger()


def construct_model(data_dir, file_type, n_files):

    start_time = time.time()

    print("Loading corpus text from: %s" % (data_dir))
    print(BREAK_LINE)

    # Read corpus data
    file_list = util.get_files(data_dir, file_type, n_files)

    n_sentences = 0
    n_completed = 0

    sentence_nodes = list()

    for filename in file_list[:]:

        n_completed += 1

        with open(filename, 'r', encoding='utf-8') as f:

            start_time_file = time.time()
            # print("Processing file: " + filename)

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i]

                nodes = languages.parse(
                    sentence, PARSER, DELIMITER_TOKEN, True)

                sentence_nodes.append(nodes)

            n_sentences += len(sentences)

            elapsed_time_file = time.time() - start_time_file
            print("\tFile %2d of %2d processed..." %
                  (n_completed, len(file_list)))
            print("\tTime Elapsed: %4f" % elapsed_time_file)

    print("\nCompleted processing of all files...")
    print(BREAK_LINE)

    elapsed_time = time.time() - start_time

    print("Total sentences checked: %2d" % (n_sentences))
    print("Total elapsed time: %4f" % (elapsed_time))

    print("\nTraining model...")

    model = FastText(min_count=5,
                     size=256, 
                     workers=8)

    print("Completed...")

    return model


def default():

    TEXT_DIR = cfg['directories']['raw_text']
    TEXT_FILETYPE = cfg['data_params']['raw_text_filetype']

    construct_model(TEXT_DIR, TEXT_FILETYPE, 100)
