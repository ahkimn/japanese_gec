
# Filename: w2v.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 05/03/2019
# Date Last Modified: 05/03/2019
# Python Version: 3.7

'''
Functions to generate, clean, and load indexed datasets from corpus text files
'''

import os
import time
from gensim.models import Word2Vec

from . import configx
from . import languages
from . import util

def construct_default_model(data_dir = configx.CONST_CORPUS_TEXT_DIRECTORY,
                            save_dir = configx.CONST_WORD_2_VEC_SAVE_DIRECTORY,
                            save_name = configx.CONST_WORD_2_VEC_MODEL_NAME,
                            file_type = configx.CONST_CORPUS_TEXT_FILETYPE, 
                            n_files = -1):

    w2v = construct_w2v_model(data_dir, file_type, n_files)

    if not os.path.isdir(save_dir):

        util.mkdir_p(save_dir)

    save_path = os.path.join(save_dir, save_name)

    w2v.save(save_path)

def construct_w2v_model(data_dir, file_type, n_files):
   
    start_time = time.time()

    print("Loading corpus text from: %s" % (data_dir))
    print(configx.BREAK_LINE)

    # Read corpus data
    file_list = util.get_files(data_dir, file_type, n_files)

    delimiter = configx.CONST_SENTENCE_DELIMITER_TOKEN

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
                nodes = util.tokenize(sentence, configx.CONST_PARSER, delimiter, True)

                sentence_nodes.append(nodes)

            n_sentences += len(sentences)

            elapsed_time_file = time.time() - start_time_file
            print("\tFile %2d of %2d processed..." % (n_completed, len(file_list)))

          
    print("\nCompleted processing of all files...")
    print(configx.BREAK_LINE)

    print(sentence_nodes)

    elapsed_time = time.time() - start_time

    print("Total sentences checked: %2d" % (n_sentences))    
    print("Total elapsed time: %4f" % (elapsed_time))

    model = Word2Vec(sentences, min_count=1)

    return model
    

