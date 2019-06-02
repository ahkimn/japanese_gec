# Filename: load.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 03/03/2019
# Date Last Modified: 03/03/2019
# Python Version: 3.7

'''
Functions to generate, clean, and load indexed datasets from corpus text files
'''

import os
import time
import numpy as np   

from . import configx
from . import languages
from . import util


def get_maximum_length(file_list, delimiter):
    """
    Function to obtain the length of the longest sentence in a set of files
    
    Args:
        file_list (arr): List of paths to corpus files
        delimiter (str): EOS string
    
    Returns:
        (int): Maximum length of any individual sentence
    """
    start_time = time.time()

    n_max = 0
    n_sentences = 0

    n_files = len(file_list)
    n_completed = 0

    print("Determining length of longest sentence...")
    print(configx.BREAK_LINE)

    for filename in file_list[:]:

        n_completed += 1
        
        with open(filename, 'r', encoding='utf-8') as f:

            start_time_file = time.time()
            # print("Processing file: " + filename)

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i]

                # Obtain nodes (tokens) from MeCab parser on sentence
                nodes, _ = languages.parse_full(sentence, configx.CONST_PARSER, delimiter)

                # Update maximum length
                n_max = max(n_max, len(nodes))

            n_sentences += len(sentences)

            elapsed_time_file = time.time() - start_time_file

            # print("\tSentences completed: %2d\t||\tTime elapsed: %4f" % (len(sentences), elapsed_time_file))
            print("\tFile %2d of %2d processed..." % (n_completed, len(file_list)))

    print("\nCompleted processing of all files...")
    print(configx.BREAK_LINE)

    elapsed_time = time.time() - start_time

    print("\nLength of longest sentence: %2d" % (n_max))
    print("Total sentences checked: %2d" % (n_sentences))    
    print("Total elapsed time: %4f" % (elapsed_time))

    return n_max, n_sentences


def construct_database(data_dir, file_type, n_files, language_dir, maximum_length=50):
    """
    Function to process a set of corpus text files into numpy arrays containing the tokenized and part-of-speech information of each sentence
    
    Args:
        data_dir (TYPE): Directory containing corpus files
        file_type (TYPE): Suffix of corpus files
        n_files (TYPE): Number of corpus files to use
        language_dir (TYPE): Directory containing Language class instances used to tag
    
    Returns:
        (tuple): Tuple containing the following arrays:
            token_matrix (np.ndarray): Three-dimensional numpy array containing tokenized and form information in two two-dimensional slices
            pos_matrix (np.ndarray): Three-dimensional numpy array containing part-of-speech information in four slices, 
                                     corresponding to each part-of-speech index
    """
    print("Constructing database with text from: %s, languages from: %s" % (data_dir, language_dir))
    print(configx.BREAK_LINE)

    # Load taggers for corpus data
    node_tagger, pos_taggers = languages.load_default_languages(language_dir)

    # Read corpus data
    file_list = util.get_files(data_dir, file_type, n_files)

    delimiter = node_tagger.stop_token

    # Very inefficient -_-
    # Should append to end of token_matrix and pos_matrix dynamically
    found_max, n_sentences = get_maximum_length(file_list, delimiter)

    n_max = maximum_length

    if (maximum_length <= 0):

        n_max = found_max

    n_pos_taggers = len(pos_taggers)

    # Initialize arrays to store data
    token_matrix = np.ndarray((n_sentences, n_max + 1, 2), dtype='uint32')
    pos_matrix = np.ndarray((n_sentences, n_max + 1, n_pos_taggers - 1), dtype='uint8')

    n_files = len(file_list)
    n_processed = 0
    n_files_processed = 0

    print("\nStarting token tagging...")
    print(configx.BREAK_LINE)

    for filename in file_list[:]:

        n_files_processed += 1

        with open(filename, 'r', encoding='utf-8') as f:

            start_time_file = time.time()
            # print("Processing file: " + filename)

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i]

                nodes, pos = languages.parse_full(sentence, configx.CONST_PARSER, delimiter)

                indices = node_tagger.parse_sentence(nodes)
                n_nodes = len(indices)

                if n_nodes > n_max:

                    continue

                else:

                    # Add SOS token and then copy index values to upper slice of token_matrix
                    token_matrix[n_processed, 0, 0] = node_tagger.start_index
                    token_matrix[n_processed, 1:1 + n_nodes, 0] = indices[:]

                    # Copy form indices from last part-of-speech tagger
                    form_indices = pos_taggers[-1].parse_sentence(pos[-1])

                    # ADD sos token and then copy index values to lower slice of token_matrix
                    token_matrix[n_processed, 0, 1] = pos_taggers[-1].start_index
                    token_matrix[n_processed, 1:1 + n_nodes, 1] = form_indices[:]

                    del form_indices
                    del nodes

                    # Copy part-of-speech indices to pos_matrix to each slice of pos_matrix
                    for j in range(len(pos_taggers) - 1):

                        ret = pos_taggers[j].parse_sentence(pos[j])

                        pos_matrix[n_processed, 0, j] = pos_taggers[j].unknown_index
                        pos_matrix[n_processed, 1:1 + n_nodes, j] = ret[:]  

                        del ret            

                    n_processed += 1

                    del pos

            elapsed_time_file = time.time() - start_time_file

            # print("\tSentences completed: %2d\t||\tTime elapsed: %4f" % (len(sentences), elapsed_time_file))
            print("\tFile %2d of %2d processed..." % (n_files_processed, len(file_list)))

            f.close()
            del sentences

    token_matrix = token_matrix[:n_processed]
    pos_matrix = pos_matrix[:n_processed]

    print("\nCompleted tagging corpus sentences...")
    print(configx.BREAK_LINE)

    return token_matrix, pos_matrix


def obtain_unique(tokens, pos, max_token, display_every=100000):
    """
    Function to obtain every possible unique token + part-of-speech tags + form combination
    
    Args:
        tokens (TYPE): Two-dimensional matrix of form (n, 2), where the second dimension is split to 
                       include both raw tokens and their forms
        pos (TYPE): Two-dimensional matrix of form (n, 4), where the second dimension size corresponds
                    to the number of part-of-speech taggers used
        max_token (TYPE): Maximum index of token to include in the output array
        display_every (int, optional): Determines how often a print message is called (in number of values processed)
    
    Returns:
        (arr): List of indices containing unique combinations
    """
    if max_token == -1:

        max_token = configx.CONST_MAX_SEARCH_TOKEN_INDEX

    assert(len(tokens) == len(pos))

    unique = set()
    indices = list()

    for i in range(len(tokens)):

        if (i + 1) % display_every == 0:            

            print("\tProcessing element %2d of %2d" % (i + 1, len(tokens)))

        h = tuple(tokens[i]) + tuple(pos[i])

        if h in unique or tokens[i][1] > max_token:

            continue

        else:

            unique.add(h)
            indices.append(i)

    return indices


def construct_default_database(save_dir = configx.CONST_DEFAULT_DATABASE_DIRECTORY,
                               data_dir = configx.CONST_CORPUS_TEXT_DIRECTORY,
                               file_type = configx.CONST_CORPUS_TEXT_FILETYPE, 
                               language_dir = configx.CONST_DEFAULT_LANGUAGE_DIRECTORY,
                               save_prefix = configx.CONST_UNCLEANED_DATABASE_PREFIX,
                               n_files = -1):
    """
    Construct a database of sentences (tokens, forms, and part-of-speech values) from a text corpus, tagged by a set of given languages 
    
    Args:
        save_dir (TYPE, optional): Path where the new data arrays are to be saved
        data_dir (TYPE, optional): Path to where the corpus text files are located
        file_type (TYPE, optional): Suffix of corpus text files
        language_dir (TYPE, optional): Path to where the Language instances used for tagging are saved
        save_prefix (TYPE, optional): Prefix used to save the data arrays
        n_files (TYPE, optional): Maximum number of corpus files to pass through
    """
    data_save_pos = os.path.join(save_dir, '_'.join([save_prefix, configx.CONST_POS_SUFFIX]))
    data_save_tokens = os.path.join(save_dir, '_'.join([save_prefix, configx.CONST_TOKENS_SUFFIX]))

    token_matrix, pos_matrix = construct_database(data_dir, file_type, n_files, language_dir)

    print("\nSaving database files...")
    print(configx.BREAK_LINE)

    if not os.path.isdir(save_dir):
        util.mkdir_p(save_dir)

    np.save(data_save_tokens, token_matrix)
    np.save(data_save_pos, pos_matrix)

    print("Done!")


def load_database(load_dir, load_prefix):
    """
    Load a database of tokens and part-of-speech tags from a given database directory and with a given prefix
    
    Args:
        load_dir (str): Location where the data arrays to load are stored
        load_prefix (str): Prefix of data arrays to load
    
    Returns:
        (tuple): Tuple containing the following arrays:
            tokens_matrix (np.ndarray): Matrix containing both the token and token form values for the sentences in the database
            pos_matrix (np.ndarray): Matrix containing the part-of-speech tag values for the sentences in the database
    """
    data_save_pos = os.path.join(load_dir, '_'.join([load_prefix, configx.CONST_POS_SUFFIX]))
    data_save_tokens = os.path.join(load_dir, '_'.join([load_prefix, configx.CONST_TOKENS_SUFFIX]))

    if not ".npy" in data_save_pos:

        data_save_pos += ".npy"  

    if not ".npy" in data_save_tokens:

        data_save_tokens += ".npy"  

    print("\nLoading database...") 

    tokens_matrix = np.load(data_save_tokens)
    pos_matrix = np.load(data_save_pos)
    
    print('\n========================================================\n')

    return tokens_matrix, pos_matrix


def clean_default_database(save_dir = configx.CONST_DEFAULT_DATABASE_DIRECTORY,
                           load_prefix = configx.CONST_UNCLEANED_DATABASE_PREFIX,
                           save_prefix = configx.CONST_CLEANED_DATABASE_PREFIX,
                           max_length = -1,
                           max_token = -1):
    """
    Function to clean the default-generated database, pruning non-unique sentences, and sentences of more than max_length length.
    Additionally, generate arrays of unique token + part-of-speech + form combinations for all tokens up to max_token.
    
    Args:
        save_dir (str, optional): Location where the data arrays to load are stored (and where to save the new data arrays)
        load_prefix (str, optional): Prefix of data arrays to load
        save_prefix (str, optional): Prefix of data arrays to save
        max_length (int, optional): Maximum length of sentences in cleaned data
        max_token (int, optional): Maximum index of token in unique arrays
    """
    print("\nCleaning database files...")
    print(configx.BREAK_LINE)

    tokens_matrix, pos_matrix = load_database(save_dir, load_prefix)

    print("\nDetermining unique sentences...")

    _, indices = np.unique(tokens_matrix[:, :, 0], return_index=True, axis=0)

    print("\tOriginal number of sentences: %d" % len(tokens_matrix))
    print("\tNumber of unique sentences: %d" % len(indices))

    tokens_matrix = tokens_matrix[indices]
    pos_matrix = pos_matrix[indices]

    if (max_length == -1):

        max_length = tokens_matrix.shape[1]

    else: 

        max_length = min(max_length, tokens_matrix.shape[1])

    # Split tokens matrix into tokens and forms
    tokens = tokens_matrix[:, :, 0]
    forms = tokens_matrix[:, :, 1]

    print("\nReducing matrix of unique sentences to those with length less than: %d..." % max_length)

    # Determine the first zeroed indices within each row (corresponds to index right after EOS)
    x = (tokens == 0).argmax(axis=1)
    # Determine which rows have their first zeroed indices below the max_length
    indices = (x <= max_length)

    # Slice arrays to contain only those sentences that satisfy the max_length criterion
    tokens = tokens[indices][:, :max_length]
    forms = forms[indices][:, :max_length]

    print("\tNumber of sentences satisfying the length requirement: %d..." % len(tokens))

    print("\nSaving reduced data arrays...")

    # Save reduced token and form arrays
    np.save(os.path.join(save_dir, configx.CONST_TOKENS_SUFFIX), tokens)
    np.save(os.path.join(save_dir, configx.CONST_FORM_SUFFIX), forms)

    # Save reduced part-of-speech arrays
    for i in range(4):
    
        reduced = pos_matrix[indices][:, :max_length, i]
        np.save(os.path.join(save_dir, configx.CONST_POS_SUFFIX + str(i)), reduced)

    np.save(os.path.join(save_dir, configx.CONST_LENGTHS_SUFFIX), x[indices])

    # Collapse first dimensions of token and position matrices (for one-dimensional searching)
    tokens_matrix = tokens_matrix.reshape(-1, tokens_matrix.shape[2])
    pos_matrix = pos_matrix.reshape(-1, pos_matrix.shape[2])

    print("\nDetermining unique tokens and part-of-speech tags for tokens up to index: %d..." % max_token)

    indices = obtain_unique(tokens_matrix, pos_matrix, max_token=max_token)

    unique_pos = pos_matrix[indices]
    unique_tokens = tokens_matrix[indices]

    print("\nSaving arrays of unique tokens...")

    sort_pos = unique_pos.argsort(axis=0)
    sort_form = unique_tokens[:, 1].argsort()

    save_pos_sort = os.path.join(save_dir, '_'.join([save_prefix, configx.CONST_POS_SUFFIX]))    
    np.save(save_pos_sort, unique_pos)
 
    save_tokens_sort = os.path.join(save_dir, '_'.join([save_prefix, configx.CONST_TOKENS_SUFFIX])) 
    np.save(save_tokens_sort, unique_tokens)

    save_sort = os.path.join(save_dir, configx.CONST_SORT_SUFFIX)
    np.save(save_sort, sort_pos)

    save_sort_form = os.path.join(save_dir, configx.CONST_SORT_FORM_SUFFIX)
    np.save(save_sort_form, sort_form)

    print("Done!")
