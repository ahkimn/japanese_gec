import MeCab
import os
import time
import yaml

from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath

from . import languages
from . import util

cfg = yaml.load(open('./config.yml', 'r'), Loader=yaml.FullLoader)

BREAK_LINE = cfg['print_params']['break_line']
DELIMITER_TOKEN = cfg['parser_params']['delimiter_token']
PARSER = MeCab.Tagger()


class IterFile():

    def __init__(self, fpath):

        self.fpath = fpath

    def __iter__(self):

        with open(self.fpath, 'r') as f:
            for line in f:
                yield(list(line.strip().split()))


def construct_fasttext_model(src_save, tgt_save, data_dir=None, file_type='', filters=['type'], n_files=-1):

    start_time = time.time()

    util.mkdir_p(src_save, file=True)
    util.mkdir_p(tgt_save, file=True)

    print("Loading corpus text from: %s" % (data_dir))
    print(BREAK_LINE)

    assert(data_dir is not None and file_type != '')
    # Read corpus data
    # file_list = util.get_files_recursive(data_dir, file_type)

    # n_sentences = 0
    # n_completed = 0

    util.mkdir_p('./tmp')
    src_corpus = './tmp/src.cor'
    tgt_corpus = './tmp/tgt.cor'
    # f_src_corpus = open(src_corpus, 'w+')
    # f_tgt_corpus = open(tgt_corpus, 'w+')

    # for filename in file_list[:12000]:

    #     valid = False

    #     for j in filters:

    #         if j in filename:

    #             valid = True

    #     if not valid:

    #         continue

    #     n_completed += 1

    #     with open(filename, 'r', encoding='utf-8') as f:

    #         start_time_file = time.time()
    #         # print("Processing file: " + filename)

    #         sentences = f.readlines()

    #         for i in range(len(sentences)):

    #             sentence = sentences[i]
    #             paired_data = sentence.strip().split(',')

    #             if len(paired_data) != 2:
    #                 print('ERROR')
    #                 continue

    #             source_nodes = languages.parse(
    #                 paired_data[0], PARSER, DELIMITER_TOKEN, True)
    #             target_nodes = languages.parse(
    #                 paired_data[1], PARSER, DELIMITER_TOKEN, True)

    #             f_src_corpus.write(' '.join(source_nodes) + os.linesep)
    #             f_tgt_corpus.write(' '.join(target_nodes) + os.linesep)

    #         n_sentences += len(sentences)

    #         elapsed_time_file = time.time() - start_time_file
    #         print("\tFile %2d of %2d processed..." %
    #               (n_completed, len(file_list)))
    #         print("\tTime Elapsed: %4f" % elapsed_time_file)

    #         f.close()

    # print("\nCompleted processing of all files...")
    # print(BREAK_LINE)

    # f_src_corpus.close()
    # f_tgt_corpus.close()

    # elapsed_time = time.time() - start_time

    # print("Total sentences checked: %2d" % (n_sentences))
    # print("Total elapsed time: %4f" % (elapsed_time))

    print("\nTraining source model...")
    start_time = time.time()

    src_model = FastText(min_count=5,
                         size=256,
                         workers=8,
                         window=5)

    src = IterFile(src_corpus)
    src_model.build_vocab(sentences=src)
    src_wc = src_model.corpus_count
    src_model.train(sentences=src,
                    total_examples=src_wc, epochs=10)

    src_model.save(src_save)

    elapsed_time = time.time() - start_time
    print("Completed...")
    print("Total elapsed time: %4f" % (elapsed_time))

    print("\nTraining target model...")
    start_time = time.time()

    tgt_model = FastText(min_count=5,
                         size=256,
                         workers=8)

    tgt = IterFile(tgt_corpus)
    tgt_model.build_vocab(sentences=tgt)
    tgt_wc = tgt_model.corpus_count
    tgt_model.train(sentences=tgt,
                    total_examples=tgt_wc, epochs=10)

    tgt_model.save(tgt_save)

    elapsed_time = time.time() - start_time
    print("Completed...")
    print("Total elapsed time: %4f" % (elapsed_time))

    return src_model, tgt_model


def convert_embedding_fairseq(model_file, save_file):

    model = FastText.load(model_file)
    vocab = model.wv.vocab
    n_vocab = len(vocab)
    size = model.wv.vector_size

    f_write = open(save_file, 'w+')
    f_write.write(str(n_vocab) + ' ' + str(size) + os.linesep)

    for word in vocab.keys():

        vec = model.wv.get_vector(word)
        f_write.write(word + ' ' + ' '.join(list(str(i) for i in vec)) + os.linesep)

    f_write.close()




def construct_default_embedding():

    TEXT_DIR = cfg['directories']['raw_text']
    TEXT_FILETYPE = cfg['data_params']['raw_text_filetype']

    FASTTEXT_SAVEFILE = cfg['embedding_params']['fasttext_save']

    construct_fasttext_model(TEXT_DIR, TEXT_FILETYPE, FASTTEXT_SAVEFILE, -1)


def load_default_embedding():

    FASTTEXT_SAVEFILE = cfg['embedding_params']['fasttext_save']
    model = FastText.load('./model/fasttext/src.mdl')

    print(model.wv.similar_by_word('愛'))

    return model
