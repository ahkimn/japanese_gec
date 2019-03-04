# Filename: load.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 19/06/2018
# Date Last Modified: 03/03/2019
# Python Version: 3.7

'''
Functions to convert CSV rules into text data via corpus manipulation
'''

import gc
import os
import csv
import ast
import psutil

import numpy as np

from . import configx
from . import generate
from . import languages
from . import save
from . import util


def load_search_matrices(search_directory, pos_taggers):
    '''
    Load corpus matrices containing all unique sentences within database
    
    Args:
        search_directory (str): Directory where the sentence matrices are stored
        pos_taggers (arr): Array of Language instances used to tag part-of-speech

    Returns:
        (arr): A list containing the following objects:
            tokens (np.ndarray): Sentences in tokenized form (as a matrix with sentences being represented by individual rows)
            forms (np.ndarray): Form part-of-speech tag corresponding to each token in tokens
            lengths (np.ndarray): Lengths of each sentence (single array)
            set_pos (arr): List of np.ndarray, each corresponding to a single part-of-speech index 
    '''
    # Get paths to files on disk
    tokens = os.path.join(search_directory, configx.CONST_TOKENS_SUFFIX)
    forms = os.path.join(search_directory, configx.CONST_FORM_SUFFIX)
    lengths = os.path.join(search_directory, configx.CONST_LENGTHS_SUFFIX)

    # Path prefix for part-of-speech arrays
    pos_prefix = os.path.join(search_directory, configx.CONST_POS_PREFIX)

    # Load matrices from disk
    forms = np.load(forms + ".npy")
    tokens = np.load(tokens + ".npy")
    lengths = np.load(lengths + ".npy")

    set_pos = []

    # Load each individual part-of-speech matrix (two-dimensional)
    for i in range(len(pos_taggers) - 1):

        arr = np.load(os.path.join(search_directory, configx.CONST_POS_SUFFIX + str(i) + ".npy"))
        arr = arr.reshape((1,) + arr.shape)
        set_pos.append(arr)   

        del arr     

    set_pos = np.vstack(set_pos)

    return [tokens, forms, lengths, set_pos]


def load_unique_matrices(database_directory, pos_taggers, save_prefix = configx.CONST_CLEANED_DATABASE_PREFIX):
    '''
    Loads data referencing unique tokens and part-of-speech tags from disk and produces
    derivative matrices that indicate all possible part-of-speech combinations and their sorted order
    
    Args:
        database_directory (str): Directory where the token/part-of-speech matrices are stored
        pos_taggers (arr): Array of Language instances used to tag part-of-speech
        save_prefix (str, optional): Prefix used for token/part-of-speech matrices
    
    Returns:
        (arr): A list containing the following objects:
            unique_tokens (np.ndarray): Matrix containing all unique tokens within the corpus data
            unique_pos (np.ndarray): Matrix containing all part-of-speech combinations (ordered in same manner as unique_tokens)
            sort (np.ndarray): Matrix denoting the order of the unique_pos if they were sorted by a specific part-of-speech index
            search_matrix (np.ndarray): Matrix denoting last positions of each specific part-of-speech tag index 
                when the array is sorted along that index
            unique_pos_complete (dict): Dictionary containing all possible unique part-of-speech tag combinations (including form)
            unique_pos_classes (dict): Dictionary containing all possible unique part-of-speech tag combinations (excluding form)    
    '''
    # Get paths to files on disk
    unique_tokens = os.path.join(database_directory, '_'.join([save_prefix, configx.CONST_TOKENS_SUFFIX]))
    unique_pos = os.path.join(database_directory, '_'.join([save_prefix, configx.CONST_POS_SUFFIX]))
    
    sort = os.path.join(database_directory, configx.CONST_SORT_SUFFIX)
    sort_form = os.path.join(database_directory, configx.CONST_SORT_FORM_SUFFIX)

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

    # Caiterationsulate last instance for all forms (pos_taggers[-1] is the form tagger)
    for k in range(pos_taggers[-1].n_nodes):

        try:

            last_start_form = max(0, last_indices_form[-1])

            k_index = util.last(view, last_start_form, n_tokens, k, n_tokens)
            last_start = k_index

            # If the form is not extant, set last index as equivalent to previous form
            if k_index == - 1:
                k_index = last_indices_form[-1]

            last_indices_form.append(k_index)

        # Once configx.CONST_MAX_SEARCH_TOKEN_INDEX are reached, exception is raised -> cancel loop
        except:

            break    

    search_matrix = []

    # For each part-of-speech index
    # Determine the final location in which a specific tag within that index (when sorted by the index) appears
    for j in range(sort.shape[1]):

        # Create a view of the part-of-speech matrix sorted by the index
        view = unique_pos[sort[:, j], j]
        last_indices_pos = [-1]
        last_start_pos = 0 

        for k in range(pos_taggers[j].n_nodes):

            try:

                last_start = max(0, last_indices_pos[-1])

                k_index = util.last(view, last_start_pos, n_tokens, k, n_tokens)
                last_start_pos = k_index

                if k_index == - 1:
                    k_index = last_indices_pos[-1]

                last_indices_pos.append(k_index)

            # If the part-of-speech tag is not extant, set last index as equivalent to previous form
            except:
                
                break

        last_indices_pos.append(n_tokens)
        search_matrix.append(last_indices_pos)

    # The form is the final index of the data extracted from MeCab - order search_matrix accordingly
    search_matrix.append(last_indices_form)
    unique_pos_complete = dict()

    # Determine all possible unique part-of-speech tag combinations (including form)
    for k in range(len(unique_pos)):

        # Concatenate part-of-speech tags with token form        
        up = tuple(unique_pos[k]) + tuple(unique_tokens[k, 1:])

        if up in unique_pos_complete:

            unique_pos_complete[up].append(k)
            
        else:
            unique_pos_complete[up] = [k]

    unique_pos_classes = dict()

    # Determine all possible unique part-of-speech tag combinations (excluding form)
    for k in range(len(unique_pos)):

        up = tuple(unique_pos[k])

        if up in unique_pos_classes:

            unique_pos_classes[up].append(k)
            
        else:
            unique_pos_classes[up] = [k]

    sort = np.concatenate((sort, sort_form.reshape(-1, 1)), axis=1) 

    return [unique_tokens, unique_pos, sort, search_matrix, unique_pos_complete, unique_pos_classes]



def match_template_tokens(unique_matrices, search_numbers, selected_cells, n_max):
    '''
    Function to generate possible substitute tokens given restrictions on which part-of-speech tags must be preserved
    
    Args:
        unique_matrices (arr): List of np.ndarrays containing information on unique tokens and part-of-speech tags
        search_numbers (arr): Indices of part-of-speech tags of the token
        selected_cells (arr): Array determining which part-of-speech indices need to be matched
        n_max (int): Determines maximal token index that is outputted. If this value is -1, any token
                               is allowed to be outputted
    Returns:
        (tuple): Tuple containing the following:
            (np.ndarray): Array containing all possible substitute part-of-speech combinations (including form)
            (np.ndarray): Array containing all possible substitute part-of-speech combinations (excluding form)
    '''    
    if n_max == -1:
        n_max = len(unique_matrices[2])

    else:
        n_max = min(len(unique_matrices[2]), n_max)

    # n_max = len(unique_matrices[2])

    # Part-of-speech indices that need to be matched exactly
    match_indices = np.where(selected_cells == 1)[0]
    # Part-of-speech indices that do not need to be matched
    randomize_indices = np.where(selected_cells == 0)[0]

    # If any indices must be 
    if len(match_indices) > 0:

        search_numbers_new = search_numbers[match_indices]
        possible_matches = search(unique_matrices, match_indices, search_numbers_new, n_max)

    else:

        # Any token class may substitute
        possible_matches = np.arange(n_max)
  
    classes = get_pos_classes(unique_matrices, randomize_indices, search_numbers, possible_matches)

    return classes


def search(unique_matrices, match_indices, search_numbers, n_max):
    """
    Function to obtain a list of possible substitute token indices given an input token and restrictions
    on which part-of-speech tags must be preserved
    
    Args:
        unique_matrices (arr): List of np.ndarrays containing information on unique tokens and part-of-speech tags
        match_indices (arr): Array of part-of-speech indices that require exact matching
        search_numbers (arr): Array of part-of-speech values corresponding to the match_indices
        n_max (int): Determines maximal token index that is outputted

    Returns:
        (tuple): Tuple containing the following:
            (np.ndarray): Array containing all possible substitute part-of-speech combinations (including form)
            (np.ndarray): Array containing all possible substitute part-of-speech combinations (excluding form)
    """
    pos = unique_matrices[1]
    sort = unique_matrices[2]
    search_matrix = unique_matrices[3]
   
    # Indices of token classes that may be used
    ret_indices = None

    # Iterate over each part-of-speech tag that needs matching
    for i in range(len(match_indices)):      

        # Tag value of original token at index
        search_number = search_numbers[i]

        # Maximum index to search
        max_search = len(search_matrix[match_indices[i]])

        if (search_number >= max_search):
            raise ("Illegal search number")

        elif (search_number == max_search - 1):

            start_index = search_matrix[match_indices[i]][search_number] + 1
            end_index = len(sort)

        else:

            start_index = search_matrix[match_indices[i]][search_number] + 1
            end_index = search_matrix[match_indices[i]][search_number + 1] + 1

        # From sorted array, determine all possible classes with part-of-speech at given index matching that of original token      
        possible_indices = sort[:, match_indices[i]][start_index:end_index]

        # If not first index matched, intersect along indices to determine which combinations match both part-of-speech indices
        if ret_indices is not None:

            ret_indices = np.intersect1d(ret_indices, possible_indices)

        else:

            ret_indices = possible_indices

    # Restrict output to valid index values
    ret_indices = ret_indices[ret_indices < n_max]

    return ret_indices


def get_pos_classes(unique_matrices, randomize_indices, original, possible_matches):
    """
    Function to determine unique part-of-speech combinations from possible substitute tokens
          
    Args:
        unique_matrices (arr): List of np.ndarrays containing information on unique tokens and part-of-speech tags
        randomize_indices (arr): Array of part-of-speech indices that require no matching whatsoever
        original (arr): Part-of-speech indices of original values
        possible_matches (np.ndarray): Array containing the indices of tokens that satisfy the criterion determined by indices and original

    Returns:
        (np.ndarray): Array containing all possible substitute part-of-speech combinations (including form)
        (np.ndarray): Array containing all possible substitute part-of-speech combinations (excluding form)
    """
    # Separate individual matrices for use
    tokens = unique_matrices[0]
    pos = unique_matrices[1]
    sort = unique_matrices[2]
    tags = unique_matrices[4]
    classes = unique_matrices[5]

    # Lists containing possible part-of-speech combinations that can be substituted
    all_tags = list()
    all_classes = list()   

    # Add part-of-speech combination of original token 
    all_tags.append(tuple(original))
    all_classes.append(tuple(original[:-1]))

    ret = list()    

    if len(randomize_indices) > 0:

        # If the rule is fully lenient, placing no bounds on possible substitute tokens
        if len(randomize_indices) == sort.shape[1]:

            all_dict = tags
            all_tags = list(all_dict.keys())   

            all_class = classes
            all_class_tags = list(classes.keys())

        # Otherwise iterate through possible substitute tokens to determine all possible substitute 
        # part-of-speech combinations
        else:      

            all_tags = dict() 
            all_classes = dict()

            matched_nodes = pos[possible_matches, :]
            matched_nodes_form = tokens[possible_matches, 1:] 

            # Determine possible substitute part-of-speech combinations (including form, stored on disk with tokens)
            for j in range(len(possible_matches)):                

                uc = tuple(matched_nodes[j]) + tuple(matched_nodes_form[j])

                if uc in all_tags:

                    all_tags[uc].append(possible_matches[j])

                else:

                    all_tags[uc] = [possible_matches[j]]            

            # Determine possible substitute part-of-speech combinations excluding form
            for j in range(len(possible_matches)):                

                uc = tuple(matched_nodes[j])

                if uc in all_classes:

                    all_classes[uc].append(possible_matches[j])

                else:

                    all_classes[uc] = [possible_matches[j]]

            # Retrieve keys from generated dictionaries
            all_tags = list(all_tags.keys())
            all_classes = list(all_classes.keys())

    return np.array(all_tags), np.array(all_classes)        


def match_template_sentence(search_matrices, search_numbers, selections, possible_classes, 
                            token_tagger, pos_taggers, n_max, n_search):
    """
    Given a template phrase, and matching leniency for part-of-speech tags, scan a corpus of text (search_matrices) for 
    sentences that contain phrases that match with the template phrase
    
    Args:
        search_matrices (arr): List of np.ndarrays containing the token and part-of-speech information for each sentence
        search_numbers (arr): Array of part-of-speech values corresponding to the phrase to match too
        selections (arr): Array determining which part-of-speech indices need to be matched
        possible_classes (arr): Array containing all possible substitute part-of-speech combinations to the template phrase (excluding form)
        token_tagger (Language): Language class instance used to tag tokens
        pos_taggers (arr): List of Language class instances used to tag each part-of-speech index
        n_max (int): Maximum index value of tokens matched
        n_search (int): Maximum number of sentences to search through (for testing efficiency mostly)
    
    Returns:
        (tuple): Tuple containing the following arrays:
            ret_sentences (arr): Three-dimensional list containing string/token representations of the matched sentences, separated by sub-rule
            ret_indices (arr): Three-dimensional list (lowest level np.ndarray) containing the token indices of each of the matched sentences,
                               separated by sub-rule
            starts (arr); List of lists denoting the start positions of each matched phrase within each matched sentence

    """
    # Separate individual matrices for use
    tokens = search_matrices[0]
    forms = search_matrices[1]
    lengths = search_matrices[2]
    pos = search_matrices[3]

    # Number of tokens in correct_sentence
    n_tokens = len(search_numbers)

    # Restrict number of sentences to search through
    if n_search == -1:

        n_search = len(forms)

    else:

        n_search = min(n_search, len(tokens))

    # Permutation to randomly select the sentences from the array
    perm = np.random.permutation(len(forms))[:n_search]
    match_array = None

    # Obtain a view of
    pos = pos[:, perm]

    # Iterate over each part-of-speech index, restricting possible matches by part-of-speech indices demanding exact matching
    for i in range(len(pos_taggers)):

        # Part-of-speech matching leniency and tags of index i for each token
        s_column = selections[:, i]
        n_column = search_numbers[:, i]
       
        # If there is leniency in the part-of-speech for all tokens, continue
        if np.all(s_column != 1):

            pass

        else:

            # Initialize match_array on first part-of-speech index
            if i == 0:

                match_array = util.search_template(pos[i], np.argwhere(s_column == 1), n_column, n_tokens)      

            # Perform array intersection on subsequent part-of-speech indices
            elif i != len(pos_taggers) - 1:

                match_array = np.logical_and(match_array, util.search_template(pos[i], np.argwhere(s_column == 1), n_column, n_tokens))
               
            # Form of token is on different data array
            else:

                match_array = np.logical_and(match_array, util.search_template(forms[perm], np.argwhere(s_column == 1), n_column, n_tokens))
           
    # Get indices of sentences that contain at least one match
    successes = np.any(match_array, axis=1)
    n_matched_sentences = np.sum(successes)

    # Extract the contents and length of sentences that contain matches
    sentences = tokens[perm][successes]
    lengths = lengths[perm][successes]
    pos = pos[:, successes]

    match_array = match_array[successes]

    # Get total number of matches (individual sentences may have more than one)
    n_per = np.sum(match_array, axis=1)
    n_matches = np.sum(match_array)

    print("\n\tNumber of sentences with a match: %s" % n_matched_sentences)
    print("\tTotal number of matches: %s" % n_matches)

    # Create new arrays to copy data for sentences with more than one match
    temp_match_array = np.ndarray((n_matches, match_array.shape[1]), match_array.dtype)
    temp_sentences = np.ndarray((n_matches, sentences.shape[1]), sentences.dtype)
    temp_lengths = np.ndarray((n_matches), lengths.dtype)   

    temp_pos = np.ndarray((pos.shape[0], n_matches, pos.shape[2]), pos.dtype)

    # Copy data to new array
    temp_match_array[:n_matched_sentences] = match_array[:] 
    temp_sentences[:n_matched_sentences] = sentences[:]
    temp_lengths[:n_matched_sentences] = lengths[:]

    temp_pos [:, :n_matched_sentences] = pos[:][:]

    insert_index = n_matched_sentences 

    print("\n\tProcessing sentences with more than one match")

    for j in range(len(match_array)):

        # Iterate over sentences with more than one match
        if n_per[j] > 1:

            copy = np.copy(match_array[j])

            # For each extra match
            while n_per[j] > 1:

                # Remove the first instance of the match in the match_array
                first_index = np.argmax(copy)
                copy[first_index] = 0

                # Copy the data for the sentence into each of the new arrays
                temp_match_array[insert_index][:] = copy[:] 
                temp_sentences[insert_index][:] = sentences[j][:]
                temp_lengths[insert_index] = lengths[j]
                
                temp_pos[:, insert_index] = pos[:, j]

                n_per[j] -= 1
                insert_index += 1

                assert(n_per[j] == np.sum(copy))

    print("\tCompleted...")

    assert(insert_index == n_matches)

    # Reset references
    match_array = temp_match_array
    sentences = temp_sentences
    lengths = temp_lengths
    pos = temp_pos

    # Get index to start search from per each sentence (note that argmax selects first index equivalent to max)
    # Copied sentences at end of match_array have had their first (k) instances of 1 removed 
    start = (match_array).argmax(axis=1)

    print("\n\tLimiting matches to those within first %d tokens" % (n_max))

    valid = None

    # Limit valid sentences to those in which all tokens within the matched phrase are limited to the first n_max indices
    for index in range(n_tokens):   

        # Determine where in each sentence to check for classes
        check = start + index

        # Obtain the token values for each matched phrase
        values = np.array(list(sentences[j][check[j]] for j in range(len(check))))
        # Boolean array mask
        values = (values < n_max)

        if valid is None:

            valid = values

        else:

            valid = np.logical_and(valid, values)

    n_valid = np.sum(valid)

    print("\tFinal number of valid sentences: %d" % (n_valid))

    match_array = match_array[valid]
    sentences = sentences[valid]
    lengths = lengths[valid]
    pos = pos[:, valid]

    # Get index to start search from per each sentence (note that argmax selects first index equivalent to max)
    # Copied sentences at end of match_array have had their first (k) instances of 1 removed 
    start = (match_array).argmax(axis=1)
    del match_array

    all_matches = list()
    all_counts = list()

    # Iterate over each token, checking which sentences match each sub-rule (as defined by tuples of possible_classes)
    for index in range(n_tokens):

        assert(len(possible_classes[index]) != 0)

        # Determine where in each sentence to check for classes
        check = start + index

        matches, counts = util.check_matched_indices(pos, check, possible_classes[index])
    
        all_matches.append(matches)
        all_counts.append(counts)

    rule_types = dict()    

    # Iterate over each matched sentence
    for sentence_number in range(n_valid):

        subrules = list()

        for index in range(len(possible_classes)):

            matches = all_matches[index]

            for k in range(len(matches)):

                # Extract the class of token per each index of each matched sentence
                if matches[k][sentence_number]:

                    subrules.append(k)

        # Each sub-rule represents unique combination of classes among matched sentences
        subrules = tuple(subrules)
        if len(subrules) == len(possible_classes):

            # If the sub-rule has already been seen
            if subrules in rule_types.keys():

                # Place indices corresponding to sub-rule
                rule_types[subrules].append(sentence_number)

            # If sub-rule has not already been seen
            else:

                rule_types[subrules] = [sentence_number]

        else:
            pass

    # All possible sub-rules within this rule
    subrules = list(rule_types.keys())

    print("\n\tNumber of possible sub-rules: %d" % (len(subrules)))

    ret_sentences = list()
    ret_indices = list()
    starts = list()

    # Iterate through each sub-rule
    for sub in range(len(subrules)):

        # Indices associated with each sub-rule
        selected_indices = np.array(rule_types[subrules[sub]])

        # Sentences associated with each sub-rule
        under = sentences[selected_indices]
        chosen_lengths = lengths[selected_indices]

        # Total number of sentences associated with each sub-rule
        n_under = len(under)

        # Only consider sub-rules with sentences matching
        if n_under != 0:  

            print("\t\tNumber of sentences under sub-rule %d: %d" % (sub + 1, n_under))

            # NOTE: Probably not necessary randomization but whatever
            # Randomize order of sentences (again)
            s_perm = np.random.permutation(len(under))[:n_under]

            under = under[s_perm]
            chosen_lengths = chosen_lengths[s_perm]

            ret_indices.append(list(under[i][1:chosen_lengths[i]] for i in range(n_under)))
            ret_sentences.append(list(token_tagger.parse_indices(under[i][1:chosen_lengths[i]]) for i in range(n_under)))
            starts.append(start[selected_indices][s_perm].tolist())

    return ret_sentences, ret_indices, starts


def convert_csv_rules(n_max = -1,
                      n_search = -1,
                      pause = True,
                      search_directory = configx.CONST_DEFAULT_SEARCH_DATABASE_DIRECTORY,
                      database_directory = configx.CONST_DEFAULT_DATABASE_DIRECTORY,
                      rule_file_directory = configx.CONST_RULE_CONFIG_DIRECTORY,
                      rule_file_name = configx.CONST_RULE_CONFIG):

    token_tagger, pos_taggers = languages.load_default_languages()
    attribute_indices = [0, 1, 4, 5, 6]
    n_pos = len(pos_taggers)

    print("Loading token database...")
    print(configx.BREAK_LINE)

    # Load matrices necessary for sentence generation
    search_matrices = load_search_matrices(search_directory, pos_taggers)
    unique_matrices = load_unique_matrices(database_directory, pos_taggers)

    print("Finished loading token databases...")
    print(configx.BREAK_LINE)
    
    # Load rule file
    rule_file = os.path.join(rule_file_directory, rule_file_name)

    # Process rule file
    with open(rule_file, 'r') as f:

        csv_reader = csv.reader(f, delimiter=',')
        
        # Counter for number of iterations (determines saved file names)
        iterations = 0

        # Read each line (rule) of CSV
        for rule_text in csv_reader:

            if iterations == 0:
                iterations += 1
                continue

            # Paired sentence data
            corrected_sentence = rule_text[0]
            error_sentence = rule_text[1]

            print("Reading Rule %2d: %s --> %s" % (iterations, corrected_sentence, error_sentence))
            print(configx.BREAK_LINE)
            
            # Retrieve unencoded part-of-speech tags of the correct sentence
            pos_tags = rule_text[2]
            pos_tags = pos_tags.split(',')

            print(pos_tags)

            # Convert part-of-speech tags to index form
            n_tokens = int(len(pos_tags) / n_pos)
            pos_tags = np.array(list(languages.parse_node_matrix(pos_tags[i * n_pos: i * n_pos + n_pos], pos_taggers) for i in range(n_tokens)))   
            print(pos_tags)

            # Array of arrays denoting hows part-of-speech tags have been selected
            # This is marked as -1 = null, 0 = no match, 1 = match
            selections = rule_text[3]
            selections = np.array(list(int(j) for j in selections.split(',')))
            selections = selections.reshape(-1, n_pos)

            # Arrays of tuples denoting token mappings between errored and correct sentence
            created = rule_text[4]
            altered = rule_text[5]
            preserved = rule_text[6]

            # Convert string representations to lists
            created = ast.literal_eval(created)
            altered = ast.literal_eval(altered)
            preserved = ast.literal_eval(preserved)

            # Aggregate mapping into single tuple
            mapping = (created, altered, preserved)   

            print("\tFinding potential substitute tokens...")
            print(configx.BREAK_SUBLINE)  

            # List of possible substitute token classes (part-of-speech combinations) per each index of correct sentence
            # as defined by the selections matrix
            possible_classes = list()

            # Iterate over each token
            for index in range(n_tokens):

                _, all_classes = match_template_tokens(unique_matrices, pos_tags[index], 
                                                              selections[index], n_max)
            
                possible_classes.append(all_classes)

            # Determine number of possible substitutes at each index
            n_possibilities = list(len(i) for i in possible_classes)

            print("\n\tSearching for sentences matching pair template...")
            print(configx.BREAK_SUBLINE)
            s_examples, _, starts, \
                = match_template_sentence(search_matrices, pos_tags, selections, possible_classes, 
                                          token_tagger, pos_taggers, n_max, n_search)



            print("\n\tGenerating new sentence pairs...")
            print(configx.BREAK_SUBLINE)
            error_examples, l_correct, l_error = \
                generate.create_errored_sentences(unique_matrices, token_tagger, pos_taggers, 
                                                  mapping, selections, s_examples, starts, error_sentence, 
                                                  corrected_sentence)

            raise

            # print("\tSaving new data...")
            # save.save_rule(corrected_sentence, error_sentence, s_examples, error_examples, iterations, rule_text)

            # print("\tPaired data saved successfully...\n")

            iterations += 1



if __name__ == '__main__':

    convert_csv_rule()
