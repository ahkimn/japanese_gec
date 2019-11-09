# Filename: load.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 03/03/2019
# Date Last Modified: 03/03/2019
# Python Version: 3.7

'''
Functions to generate, clean, and load indexed datasets from corpus text files
'''

import gc
import os
import time
import numpy as np

from . import configx
from . import languages
from . import util


def _get_n_sentences(file_list, delimiter):
    """
    Function to obtain the length of the longest sentence in a set of files

    Args:
        file_list (arr): List of paths to corpus files
        delimiter (str): EOS string

    Returns:
        (int): Maximum length of any individual sentence
    """
    start_time = time.time()

    n_sentences = 0
    n_completed = 0

    print("Determining length of longest sentence...")
    print(configx.BREAK_LINE)

    for filename in file_list[:]:

        n_completed += 1

        f = open(filename, 'r', encoding='utf-8')

        start_time_file = time.time()
        # print("Processing file: " + filename)

        sentences = f.readlines()
        n_sentences += len(sentences)

        elapsed_time_file = time.time() - start_time_file

        print("\tSentences completed: %2d\t||\tTime elapsed: %4f" %
              (len(sentences), elapsed_time_file))
        print("\tFile %2d of %2d processed..." %
              (n_completed, len(file_list)))

        f.close()
        del sentences
        gc.collect()

    print("\nCompleted processing of all files...")
    print(configx.BREAK_LINE)

    elapsed_time = time.time() - start_time

    print("Total sentences checked: %2d" % (n_sentences))
    print("Total elapsed time: %4f" % (elapsed_time))

    return n_sentences


def _construct_database(data_dir, file_type, n_files,
                        language_dir, maximum_length=40):
    """
    Function to process a set of corpus text files into numpy arrays containing
        the tokenized and part-of-speech information of each sentence

    Args:
        data_dir (TYPE): Directory containing corpus files
        file_type (TYPE): Suffix of corpus files
        n_files (TYPE): Number of corpus files to use
        language_dir (TYPE): Directory containing Language class instances used
            to tag
        maximum_length (int, optional): Description

    Returns:
        (tuple): Tuple containing the following arrays:
            token_matrix (np.ndarray): Three-dimensional numpy array containing
                tokenized and form information in two two-dimensional slices
            pos_matrix (np.ndarray): Three-dimensional numpy array containing
                part-of-speech information in four slices, corresponding to
                each part-of-speech index
    """
    print("Constructing database with text from: %s, languages from: %s" %
          (data_dir, language_dir))
    print(configx.BREAK_LINE)

    # Load taggers for corpus data
    node_tagger, pos_taggers = languages.load_languages(language_dir)

    # Read corpus data

    if isinstance(data_dir, list):
        file_list = list()

        for _dir in data_dir:
            file_list += util.get_files(_dir, file_type, n_files)

    else:
        file_list = util.get_files(data_dir, file_type, n_files)

    delimiter = node_tagger.stop_token

    # Very inefficient -_-
    # Should append to end of token_matrix and pos_matrix dynamically
    n_sentences = _get_n_sentences(file_list, delimiter)

    n_max = maximum_length
    n_pos_taggers = len(pos_taggers)

    # Initialize arrays to store data
    token_matrix = np.ndarray((n_sentences, n_max + 1, 2), dtype='uint32')
    pos_matrix = np.ndarray(
        (n_sentences, n_max + 1, n_pos_taggers - 1), dtype='uint8')

    n_files = len(file_list)
    n_processed = 0
    n_files_processed = 0

    print("\nStarting token tagging...")
    print(configx.BREAK_LINE)

    for filename in file_list[:]:

        n_files_processed += 1

        f = open(filename, 'r', encoding='utf-8')

        start_time_file = time.time()
        # print("Processing file: " + filename)

        sentences = f.readlines()

        for i in range(len(sentences)):

            sentence = sentences[i]

            nodes, pos = languages.parse_full(
                sentence, configx.CONST_PARSER, delimiter)

            indices = node_tagger.parse_sentence(nodes)
            n_nodes = len(indices)

            if n_nodes > n_max:

                continue

            else:

                # Add SOS token and then copy index values to
                #   upper slice of token_matrix
                token_matrix[n_processed, 0, 0] = node_tagger.start_index
                token_matrix[n_processed, 1:1 + n_nodes, 0] = indices[:]

                # Copy form indices from last part-of-speech tagger
                form_indices = pos_taggers[-1].parse_sentence(pos[-1])

                # ADD sos token and then copy index values to lower
                #   slice of token_matrix
                token_matrix[n_processed, 0,
                             1] = pos_taggers[-1].start_index
                token_matrix[n_processed, 1:1 +
                             n_nodes, 1] = form_indices[:]

                del form_indices
                del nodes

                # Copy part-of-speech indices to pos_matrix to each
                #   slice of pos_matrix
                for j in range(len(pos_taggers) - 1):

                    ret = pos_taggers[j].parse_sentence(pos[j])

                    pos_matrix[n_processed, 0,
                               j] = pos_taggers[j].unknown_index
                    pos_matrix[n_processed, 1:1 + n_nodes, j] = ret[:]

                    del ret

                n_processed += 1

                del pos

        elapsed_time_file = time.time() - start_time_file

        print("\tSentences completed: %2d\t||\tTime elapsed: %4f" %
              (len(sentences), elapsed_time_file))
        print("\tFile %2d of %2d processed..." %
              (n_files_processed, len(file_list)))

        f.close()
        del sentences

    token_matrix = token_matrix[:n_processed]
    pos_matrix = pos_matrix[:n_processed]

    print("\nCompleted tagging corpus sentences...")
    print(configx.BREAK_LINE)

    return token_matrix, pos_matrix


def _obtain_unique(tokens, pos, max_token, display_every=100000):
    """
    Function to obtain every possible unique token + part-of-speech tags
        + form combination

    Args:
        tokens (np.array): Two-dimensional matrix of form (n, 2), where the
            second dimension is split to include both raw tokens and their
            forms
        pos (np.array): Two-dimensional matrix of form (n, 4), where the second
            dimension size corresponds to the number of part-of-speech
            taggers used
        max_token (int): Maximum index of token to include in the output array
        display_every (int, optional): Determines how often a print message
            is called (in number of values processed)

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


def construct_default_database():
    """
    Cosntruct database from files specified by configx
    """
    construct_database(configx.CONST_DEFAULT_DATABASE_DIRECTORY,
                       configx.CONST_CORPUS_TEXT_DIRECTORY,
                       configx.CONST_CORPUS_TEXT_FILETYPE,
                       configx.CONST_DEFAULT_LANGUAGE_DIRECTORY,
                       configx.CONST_UNCLEANED_DATABASE_PREFIX,
                       1)


def construct_database(save_dir,
                       data_dir,
                       file_type,
                       language_dir,
                       save_prefix,
                       n_files=-1):
    """
    Construct a database of sentences (tokens, forms, and part-of-speech
        values) from a text corpus, tagged by a set of given languages

    Args:
        save_dir (str, optional): Path where the new data arrays are
            to be saved
        data_dir (str, optional): Path to where the corpus text files
            are located
        file_type (str, optional): Suffix of corpus text files
        language_dir (str, optional): Path to where the Language
            instances used for tagging are saved
        save_prefix (str, optional): Prefix used to save the data
            arrays
        n_files (int, optional): Maximum number of corpus files to
            pass through
    """
    data_save_pos = os.path.join(save_dir, '_'.join(
        [save_prefix, configx.CONST_POS_SUFFIX]))
    data_save_tokens = os.path.join(save_dir, '_'.join(
        [save_prefix, configx.CONST_TOKENS_SUFFIX]))

    token_matrix, pos_matrix = _construct_database(
        data_dir, file_type, n_files, language_dir)

    print("\nSaving database files...")
    print(configx.BREAK_LINE)

    if not os.path.isdir(save_dir):
        util.mkdir_p(save_dir)

    np.save(data_save_tokens, token_matrix)
    np.save(data_save_pos, pos_matrix)

    print("Done!")


def load_database(load_dir, load_prefix):
    """
    Load a database of tokens and part-of-speech tags from a given database
        directory and with a given prefix

    Args:
        load_dir (str): Location where the data arrays to load are stored
        load_prefix (str): Prefix of data arrays to load

    Returns:
        (tuple): Tuple containing the following arrays:
            tokens_matrix (np.ndarray): Matrix containing both the token
                and token form values for the sentences in the database
            pos_matrix (np.ndarray): Matrix containing the part-of-speech tag
                values for the sentences in the database
    """
    data_save_pos = os.path.join(load_dir, '_'.join(
        [load_prefix, configx.CONST_POS_SUFFIX]))
    data_save_tokens = os.path.join(load_dir, '_'.join(
        [load_prefix, configx.CONST_TOKENS_SUFFIX]))

    if ".npy" not in data_save_pos:

        data_save_pos += ".npy"

    if ".npy" not in data_save_tokens:

        data_save_tokens += ".npy"

    print("\nLoading database...")

    tokens_matrix = np.load(data_save_tokens)
    pos_matrix = np.load(data_save_pos)

    print('\n========================================================\n')

    return tokens_matrix, pos_matrix


def clean_default_database():
    """
    Load and clean database at location specified in configx
    """
    clean_database(save_dir=configx.CONST_DEFAULT_DATABASE_DIRECTORY,
                   load_prefix=configx.CONST_UNCLEANED_DATABASE_PREFIX,
                   save_prefix=configx.CONST_CLEANED_DATABASE_PREFIX)


def clean_database(save_dir,
                   load_prefix,
                   save_prefix,
                   load_dir=None,
                   max_length=-1,
                   max_token=-1,
                   process_unique=True):
    """
    Function to clean the default-generated database, pruning non-unique
        sentences, and sentences of more than max_length length.
    Additionally, generate arrays of unique token + part-of-speech +
        form combinations for all tokens up to max_token.

    Args:
        save_dir (str, optional): Location where the data arrays to
            load are stored (and where to save the new data arrays)
        load_prefix (str, optional): Prefix of data arrays to load
        save_prefix (str, optional): Prefix of data arrays to save
        load_dir (None, optional): Description
        max_length (int, optional): Maximum length of sentences in cleaned data
        max_token (int, optional): Maximum index of token in unique arrays
        process_unique (bool, optional): Description
    """
    print("\nCleaning database files...")
    print(configx.BREAK_LINE)

    if load_dir is None:
        load_dir = save_dir

    tokens_matrix, pos_matrix = load_database(load_dir, load_prefix)

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

    print("\nReducing matrix of unique sentences to " +
          "those with length less than: %d..." % max_length)

    # Determine the first zeroed indices within each row
    #   (corresponds to index right after EOS)
    x = (tokens == 0).argmax(axis=1)
    # Determine which rows have their first zeroed indices below the max_length
    indices = (x <= max_length)

    # Slice arrays to contain only those sentences that satisfy the max_length
    #   criterion
    tokens = tokens[indices][:, :max_length]
    forms = forms[indices][:, :max_length]

    print("\tNumber of sentences satisfying the length requirement: %d..."
          % len(tokens))

    print("\nSaving reduced data arrays...")

    # Save reduced token and form arrays
    np.save(os.path.join(save_dir, configx.CONST_TOKENS_SUFFIX), tokens)
    np.save(os.path.join(save_dir, configx.CONST_FORM_SUFFIX), forms)

    # Save reduced part-of-speech arrays
    for i in range(4):

        reduced = pos_matrix[indices][:, :max_length, i]
        np.save(os.path.join(save_dir,
                             configx.CONST_POS_SUFFIX + str(i)), reduced)

    np.save(os.path.join(save_dir, configx.CONST_LENGTHS_SUFFIX), x[indices])

    if process_unique:

        # Collapse first dimensions of token and position matrices
        #   (for one-dimensional searching)
        tokens_matrix = tokens_matrix.reshape(-1, tokens_matrix.shape[2])
        pos_matrix = pos_matrix.reshape(-1, pos_matrix.shape[2])

        print('\nDetermining unique tokens and part-of-speech ' +
              'tags for tokens up to index: %d...' % max_token)

        indices = _obtain_unique(
            tokens_matrix, pos_matrix, max_token=max_token)

        unique_pos = pos_matrix[indices]
        unique_tokens = tokens_matrix[indices]

        print("\nSaving arrays of unique tokens...")

        sort_pos = unique_pos.argsort(axis=0)
        sort_form = unique_tokens[:, 1].argsort()

        save_pos_sort = os.path.join(save_dir, '_'.join(
            [save_prefix, configx.CONST_POS_SUFFIX]))
        np.save(save_pos_sort, unique_pos)

        save_tokens_sort = os.path.join(save_dir, '_'.join(
            [save_prefix, configx.CONST_TOKENS_SUFFIX]))
        np.save(save_tokens_sort, unique_tokens)

        save_sort = os.path.join(save_dir, configx.CONST_SORT_SUFFIX)
        np.save(save_sort, sort_pos)

        save_sort_form = os.path.join(save_dir, configx.CONST_SORT_FORM_SUFFIX)
        np.save(save_sort_form, sort_form)

        print("Done!")


def load_search_matrices(search_directory, pos_taggers):
    '''
    Load corpus matrices containing all unique sentences within database

    Args:
        search_directory (str): Directory where the sentence matrices
            are stored
        pos_taggers (arr): Array of Language instances used to tag
            part-of-speech

    Returns:
        (arr): A list containing the following objects:
            tokens (np.ndarray): Sentences in tokenized form (as a matrix
                with sentences being represented by individual rows)
            forms (np.ndarray): Form part-of-speech tag corresponding
                to each token in tokens
            lengths (np.ndarray): Lengths of each sentence (single array)
            set_pos (arr): List of np.ndarray, each corresponding to a single
                part-of-speech index
    '''
    # Get paths to files on disk
    tokens = os.path.join(search_directory, configx.CONST_TOKENS_SUFFIX)
    forms = os.path.join(search_directory, configx.CONST_FORM_SUFFIX)
    lengths = os.path.join(search_directory, configx.CONST_LENGTHS_SUFFIX)

    # Path prefix for part-of-speech arrays
    # pos_prefix = os.path.join(search_directory, configx.CONST_POS_PREFIX)

    set_pos = []

    # Load matrices from disk
    forms = np.load(forms + ".npy")
    tokens = np.load(tokens + ".npy")
    lengths = np.load(lengths + ".npy")

    # Load each individual part-of-speech matrix (two-dimensional)
    for i in range(len(pos_taggers) - 1):

        arr = np.load(os.path.join(search_directory,
                                   configx.CONST_POS_SUFFIX + str(i) + ".npy"))
        arr = arr.reshape((1,) + arr.shape)
        set_pos.append(arr)
        del arr

    set_pos = np.vstack(set_pos)

    matrices = dict()
    matrices['token'] = tokens
    matrices['form'] = forms
    matrices['pos'] = set_pos
    matrices['lengths'] = lengths

    return matrices


def load_unique_matrices(database_directory, pos_taggers,
                         save_prefix=configx.CONST_CLEANED_DATABASE_PREFIX):
    '''
    Loads data referencing unique tokens and part-of-speech tags from disk
        and produces derivative matrices that indicate all possible
        part-of-speech combinations and their sorted order

    Args:
        database_directory (str): Directory where the token/part-of-speech
            matrices are stored
        pos_taggers (arr): Array of Language instances used to tag
            part-of-speech
        save_prefix (str, optional): Prefix used for token/part-of-speech
            matrices

    Returns:
        (arr): A list containing the following objects:
            unique_tokens (np.ndarray): Matrix containing all unique tokens
                within the corpus data
            unique_pos (np.ndarray): Matrix containing all part-of-speech
                combinations (ordered in same manner as unique_tokens)
            sort (np.ndarray): Matrix denoting the order of the unique_pos if
                they were sorted by a specific part-of-speech index
            search_matrix (np.ndarray): Matrix denoting last positions of each
                specific part-of-speech tag index when the array is sorted
                along that index
            unique_pos_complete (dict): Dictionary containing all possible
                unique part-of-speech tag combinations (including form)
            unique_pos_classes (dict): Dictionary containing all possible
                unique part-of-speech tag combinations (excluding form)
    '''
    # Get paths to files on disk
    unique_tokens = os.path.join(database_directory, '_'.join(
        [save_prefix, configx.CONST_TOKENS_SUFFIX]))
    unique_pos = os.path.join(database_directory, '_'.join(
        [save_prefix, configx.CONST_POS_SUFFIX]))

    sort = os.path.join(database_directory, configx.CONST_SORT_SUFFIX)
    sort_form = os.path.join(
        database_directory, configx.CONST_SORT_FORM_SUFFIX)

    # Load files from disk
    unique_tokens = np.load(unique_tokens + ".npy")
    unique_pos = np.load(unique_pos + ".npy")

    sort = np.load(sort + ".npy")
    sort_form = np.load(sort_form + ".npy")

    n_tokens = len(unique_tokens)

    # Sort unique forms such that their corresponding tokens are in order
    # (from index = 0 to index = configx.CONST_MAX_SEARCH_TOKEN_INDEX)
    view = unique_tokens[sort_form, 1]

    # Array to store final indices (of token) where each form is found
    last_indices_form = [-1]

    # Calculate last instance for all forms (pos_taggers[-1]
    #   is the form tagger)
    for k in range(pos_taggers[-1].n_nodes):

        try:

            last_start_form = max(0, last_indices_form[-1])

            k_index = util.last(view, last_start_form, n_tokens, k, n_tokens)
            # last_start = k_index

            # If the form is not extant, set last index as equivalent
            #   to previous form
            if k_index == - 1:
                k_index = last_indices_form[-1]

            last_indices_form.append(k_index)

        # Once configx.CONST_MAX_SEARCH_TOKEN_INDEX are reached, exception
        #   is raised -> cancel loop
        except Exception:

            break

    search_matrix = []

    # For each part-of-speech index
    # Determine the final location in which a specific tag within that index
    #   (when sorted by the index) appears
    for j in range(sort.shape[1]):

        # Create a view of the part-of-speech matrix sorted by the index
        view = unique_pos[sort[:, j], j]
        last_indices_pos = [-1]
        last_start_pos = 0

        for k in range(pos_taggers[j].n_nodes):

            try:

                # last_start = max(0, last_indices_pos[-1])

                k_index = util.last(view, last_start_pos,
                                    n_tokens, k, n_tokens)
                last_start_pos = k_index

                if k_index == - 1:
                    k_index = last_indices_pos[-1]

                last_indices_pos.append(k_index)

            # If the part-of-speech tag is not extant, set last
            #   index as equivalent to previous form
            except Exception:

                break

        last_indices_pos.append(n_tokens)
        search_matrix.append(last_indices_pos)

    # The form is the final index of the data extracted from
    #   MeCab - order search_matrix accordingly
    search_matrix.append(last_indices_form)
    unique_pos_complete = dict()

    # Determine all possible unique part-of-speech tag combinations
    #   (including form)
    for k in range(len(unique_pos)):

        # Concatenate part-of-speech tags with token form
        up = tuple(unique_pos[k]) + tuple(unique_tokens[k, 1:])

        if up in unique_pos_complete:

            unique_pos_complete[up].append(k)

        else:
            unique_pos_complete[up] = [k]

    unique_pos_classes = dict()

    # Determine all possible unique part-of-speech tag combinations
    #   (excluding form)
    for k in range(len(unique_pos)):

        up = tuple(unique_pos[k])

        if up in unique_pos_classes:

            unique_pos_classes[up].append(k)

        else:
            unique_pos_classes[up] = [k]

    form_pos = dict()

    for k in range(len(unique_tokens)):

        _form = unique_tokens[k, 1]
        _token = unique_tokens[k, 0]
        _pos = tuple(unique_pos[k])

        if _form in form_pos:

            if _pos in form_pos[_form]:

                form_pos[_form][_pos].append(_token)

            else:

                form_pos[_form][_pos] = [_token]

        else:

            form_pos[_form] = dict()
            form_pos[_form][_pos] = [_token]

    sort = np.concatenate((sort, sort_form.reshape(-1, 1)), axis=1)

    matrices = dict()
    matrices['token'] = unique_tokens
    matrices['pos'] = unique_pos

    matrices['sort'] = sort
    matrices['search'] = search_matrix

    matrices['complete'] = unique_pos_complete
    matrices['classes'] = unique_pos_classes
    matrices['form_dict'] = form_pos

    return matrices
