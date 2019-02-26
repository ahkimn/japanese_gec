import os
import numpy as np

import gensim
from gensim.models import Word2Vec
from gensim.models import Phrases

import configx


def get_embedding_sentences(language="error"):

    search_dir = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, configx.CONST_TEXT_OUTPUT_PREFIX, configx.CONST_CORPUS_SAVE_DIRECTORY)
    file_list = os.listdir(search_dir)

    sentences = list()

    for file_name in file_list:

        if language in file_name and "full" not in file_name:

            f = open(os.path.join(search_dir, file_name), "r")
            lines = f.readlines()            

            for line in lines:
                a = list(str(configx.CONST_SENTENCE_START_INDEX))
                a += list(i for i in line.strip().split(" "))
                sentences.append(a)

            f.close()

    return sentences

def pre_train_embedding(sentences, save, size=configx.CONST_EMBEDDING_SIZE, workers=8):

    W2Vmodel = Word2Vec(sentences, size=size, workers=workers, min_count=configx.CONST_MIN_FREQUENCY)
    W2Vmodel.wv.save_word2vec_format(save, binary=False)

def load_embedding(filename, size=configx.CONST_EMBEDDING_SIZE):

    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()

    embedding = dict()

    i = 0

    while i < len(lines):
        line = lines[i].split()

        try:
            if len(line) == 1:

                embedding[int(line[0])] = np.asarray(lines[i + 1].split(), dtype='float32')
                i += 1

            else:

                embedding[int(line[0])] = np.asarray(line[1:], dtype='float32')
        except:
            pass

        i += 1

    return embedding   

def reformat_embedding(embedding, save_file=configx.CONST_EMBEDDING_FAIRSEQ_SAVE):
    '''
    Reformat given dictionary embedding to file parseable by Fairseq
    '''

    size = len(embedding.keys())
    dimension = len(embedding[list(embedding.keys())[0]])

    with open(save_file, "w+") as f:

        f.write(str(size) + " " + str(dimension) + os.linesep)  

        for key in embedding.keys():

            arr = embedding[key]
            f.write(str(key) + " " + " ".join(list(str(i) for i in arr)) + os.linesep)


if __name__ == '__main__':

    error_sentences = get_embedding_sentences("error")
    pre_train_embedding(error_sentences, configx.CONST_EMBEDDING_SOURCE_SAVE)

    source_embedding = load_embedding(configx.CONST_EMBEDDING_SOURCE_SAVE)
    reformat_embedding(source_embedding)


    