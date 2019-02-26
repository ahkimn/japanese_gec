import os
import csv
import ast

import numpy as np

import configx
import generate
import languages
import save
import util

TOKEN_TAGGER, POS_TAGGERS = languages.load_default_languages()
ATTRIBUTE_INDICES = [0, 1, 4, 5, 6]
N_POS = len(POS_TAGGERS)


def load_unique_arrays():
    '''
    Loads data referencing unique part-of-speech (pos) and tokens from disk

    :returns: A list containing six numpy ndarays as follows:

        unique_pos: Matrix containing all part-of-speech combinations (ordered in same manner as unique_tokens)
        unique_tokens: Matrix containing all unique token combinations (includes token form)
        sort: Matrix denoting the order of the unique_pos if they were sorted by a specific part-of-speech index
        search_matrix: Matrix denoting last positions of each specific part-of-speech tag index 
            when the array is sorted along that index
        unique_pos_tags: Dictionary containing all possible unique part-of-speech tag combinations (including token form)
        unique_pos_classes: Dictionary containing all possible unique part-of-speech tag combinations (excluding token form)
    '''

    unique_pos = os.path.join(configx.CONST_DATABASE_DIRECTORY, "unique_nodes_pos.npy")
    unique_tokens = os.path.join(configx.CONST_DATABASE_DIRECTORY, "unique_nodes_tokens.npy")
    sort = os.path.join(configx.CONST_DATABASE_DIRECTORY, "sort.npy")
    sort_form = os.path.join(configx.CONST_DATABASE_DIRECTORY, "sort_form.npy")

    sort = np.load(sort)
    sort_form = np.load(sort_form)
    unique_pos = np.load(unique_pos)
    unique_tokens = np.load(unique_tokens)

    n_tokens = len(unique_tokens)

    search_matrix = []

    last_start_form = 0
    last_indices_form = [-1]

    view = unique_tokens[sort_form, 1]

    # Find the final index at which each token_form appears
    for k in range(POS_TAGGERS[-1].n_nodes):

        try:
            k_index = util.last(view, last_start_form, n_tokens, k, n_tokens)

            last_start = k_index

            if k_index == - 1:
                k_index = last_indices_form[-1]

            last_indices_form.append(k_index)

        except:

            continue 

    # For each part-of-speech index
    #   Determine the final location in which a specific tag within that index (when sorted by the index) appears
    for j in range(sort.shape[1]):

        # Create a view sorted by the index
        view = unique_pos[sort[:, j], j]
        last_indices = [-1]
        last_start = 0 

        for k in range(POS_TAGGERS[j].n_nodes):

            try:

                k_index = util.last(view, last_start, n_tokens, k, n_tokens)
                last_start = k_index

                if k_index == - 1:
                    k_index = last_indices[-1]

                last_indices.append(k_index)

            except:
                continue

        last_indices.append(n_tokens)
        search_matrix.append(last_indices)

    # The token form is the final index of the data extracted from MeCab
    search_matrix.append(last_indices_form)
    unique_pos_tags = dict()

    # Determine all possible unique part-of-speech tag combinations (including token form)
    for k in range(len(unique_pos)):

        # Concatenate part-of-speech tags with token form        
        up = tuple(unique_pos[k]) + tuple(unique_tokens[k, 1:])

        if up in unique_pos_tags:

            unique_pos_tags[up].append(k)
            
        else:
            unique_pos_tags[up] = [k]

    unique_pos_classes = dict()

    # Determine all possible unique part-of-speech tag combinations (excluding token form)
    for k in range(len(unique_pos)):

        up = tuple(unique_pos[k])

        if up in unique_pos_classes:

            unique_pos_classes[up].append(k)
            
        else:
            unique_pos_classes[up] = [k]

    sort = np.concatenate((sort, sort_form.reshape(-1, 1)), axis=1) 

    return [unique_tokens, unique_pos, sort, search_matrix, unique_pos_tags, unique_pos_classes]


def load_search_arrays():
    '''
    Load arrays containing all unique sentences within database
    
    :returns: a list of four arrays as follows:
        tokens: sentences in tokenized form (as a matrix with sentences being represented by individual rows)
        forms: token_forms corresponding to each token in tokens
        legnths: lengths of each sentence (single array)
        r_pos: list of matrices, each corresponding to a single part-of-speech index and in the form of tokens
    '''
    forms = os.path.join(configx.CONST_SEARCH_DATABASE_DIRECTORY, "forms.npy")
    tokens = os.path.join(configx.CONST_SEARCH_DATABASE_DIRECTORY, "tokens.npy")

    lengths = os.path.join(configx.CONST_SEARCH_DATABASE_DIRECTORY, "lengths.npy")
    pos = os.path.join(configx.CONST_SEARCH_DATABASE_DIRECTORY, "pos")

    forms = np.load(forms)
    tokens = np.load(tokens)
    lengths = np.load(lengths)
    r_pos = []

    for i in range(N_POS - 1):

        r_pos.append(np.load(pos + str(i) + ".npy"))

    return [tokens, forms, lengths, r_pos]


def match_template_tokens(unique_arrays, search_numbers, selected_cells, n_max=100):
    '''
    Function to determine possible classes (part-of-speech combinations) that may replace a given token

    :input unique_arrays: data arrays of unique part-of-speech and token information
    :input search_numbers: part-of-speech tags of original token
    :input selected_cells: array of 1's, 0's and -1's determining which part-of-speech tags need to be matched
    :input n_max: maximum number of output classes 

    :returns: array of possible substitute classes
    '''

    unique_arrays[2] = sort


    # Part-of-speech indices that need to be matched exactly
    match_indices = np.where(selected_cells == 1)[0]
    # Part-of-speech indices that do not need to be matched
    randomize_indices = np.where(selected_cells == 0)[0]

    # If any indices must be 
    if len(match_indices) > 0:

        search_numbers_new = search_numbers[match_indices]
        possible_matches = search(unique_arrays, match_indices, search_numbers_new)

    else:

        # Any token class may substitute
        possible_matches = np.arange(len(sort))

    tags = randomize(unique_arrays, randomize_indices, possible_matches, search_numbers)

    return tags


def randomize(unique_arrays, indices, matches, original):

    tokens = unique_arrays[0]
    pos = unique_arrays[1]
    sort = unique_arrays[2]
    tags = unique_arrays[4]
    classes = unique_arrays[5]

    all_tags = list()
    all_tags.append(tuple(original))
    all_classes = list()   
    all_classes.append(tuple(original[:-1]))
    ret = list()    

    if len(indices) > 0:

        if len(indices) == sort.shape[1]:

            all_dict = tags
            all_tags = list(all_dict.keys())   

            all_class = classes
            all_class_tags = list(classes.keys())

        else:      

            all_tags = dict()    
            matched_nodes = pos[matches, :]
            matched_nodes_form = tokens[matches, 1:] 

            for j in range(len(matches)):                

                uc = tuple(matched_nodes[j]) + tuple(matched_nodes_form[j])

                if uc in all_tags:

                    all_tags[uc].append(matches[j])

                else:

                    all_tags[uc] = [matches[j]]

            all_tags = list(all_tags.keys())

            all_classes = dict()

            for j in range(len(matches)):                

                uc = tuple(matched_nodes[j])

                if uc in all_classes:

                    all_classes[uc].append(matches[j])

                else:

                    all_classes[uc] = [matches[j]]

            all_classes = list(all_classes.keys())

    return np.array(all_tags), np.array(all_classes)        


def search(unique_arrays, indices, search_numbers):

    pos = unique_arrays[1]
    sort = unique_arrays[2]
    search_matrix = unique_arrays[3]
   
    # Indices of token classes that may be used
    ret_indices = None

    # Iterate over each part-of-speech tag that needs matching
    for i in range(len(indices)):      

        # Tag value of original token at index
        search_number = search_numbers[i]

        # Maximum index to search
        max_search = len(search_matrix[indices[i]])

        if (search_number >= max_search):
            raise ("Illegal search number")

        elif (search_number == max_search - 1):

            start_index = search_matrix[indices[i]][search_number] + 1
            end_index = len(sort)

        else:

            start_index = search_matrix[indices[i]][search_number] + 1
            end_index = search_matrix[indices[i]][search_number + 1] + 1

        # From sorted array, determine all possible classes with part-of-speech at given index matching that of original token      
        possible_indices = sort[:, indices[i]][start_index:end_index]

        # If not first index matched, intersect along indices to determine which combinations match both part-of-speech indices
        if ret_indices is not None:

            ret_indices = np.intersect1d(ret_indices, possible_indices)

        else:

            ret_indices = possible_indices

    return ret_indices


def match_template_sentence(search_arrays, search_numbers, selections, possible_classes, n_possibilities, n_search=0):

    tokens = search_arrays[0]
    forms = search_arrays[1]
    lengths = search_arrays[2]
    pos = search_arrays[3]

    n_tokens = len(search_numbers)

    if n_search == 0:

        n_search = len(forms)

    else:

        n_search = min(n_search, len(forms))

    perm = np.random.permutation(len(forms))[:n_search]

    match_array = None

    for i in range(N_POS):

        s_column = selections[:, i]
        n_column = search_numbers[:, i]

        if np.all(s_column != 1):

            pass

        else:

            if i == 0:

                match_array = util.search_template(pos[i][perm], np.argwhere(s_column == 1), n_column, n_tokens)      

            elif i != N_POS - 1:

                match_array = np.logical_and(match_array, util.search_template(pos[i][perm], np.argwhere(s_column == 1), n_column, n_tokens))
               
            else:

                match_array = np.logical_and(match_array, util.search_template(forms[perm], np.argwhere(s_column == 1), n_column, n_tokens))
           
    successes = np.any(match_array, axis = 1)
    n_total = np.sum(successes)

    match_array = match_array[successes]

    sentences = tokens[perm][successes]
    lengths = lengths[perm][successes]

    starts = list()

    start = (match_array).argmax(axis=1)

    all_matches = list()
    all_counts = list()

    for index in range(len(possible_classes)):

        assert(len(possible_classes[index]) != 0)
        check = start + index

        matches, counts = util.check_matched_indices(pos, successes, check, possible_classes[index], perm, N_POS)
    
        all_matches.append(matches)
        all_counts.append(counts)

    rule_types = dict()

    # Iterate over each matched sentence
    for sentence_number in range(n_total):

        subrules = list()

        for index in range(len(possible_classes)):

            matches = all_matches[index]

            for k in range(len(matches)):

                # Extract the class of token per each index of each matched sentence
                if matches[k][sentence_number]:

                    subrules.append(k)

        # Each subrule represents unique combination of classes among matched sentences
        subrules = tuple(subrules)
        if len(subrules) == len(possible_classes):

            # If the subrule has already been seen
            if subrules in rule_types.keys():

                # Place indices corresponding to subrule
                rule_types[subrules].append(sentence_number)

            # If subrule has not already been seen
            else:

                rule_types[subrules] = [sentence_number]

        else:
            pass

    # All possible subrules within this rule
    subrules = list(rule_types.keys())

    print(subrules)

    ret = list()
    ret_indices = list()
    tags_indices = list()

    # Iterate through each subrule
    for sub in range(len(subrules)):

        # Indices associated with each subrule
        selected_indices = np.array(rule_types[subrules[sub]])

        # Sentences associated with each subrule
        under = sentences[selected_indices]
        chosen_lengths = lengths[selected_indices]

        # Total number of sentences associated with each subrule
        z = len(under)

        if z != 0:  

            tags_indices.append(sub)

            s_perm = np.random.permutation(len(under))[:z]

            under = under[s_perm]
            chosen_lengths = chosen_lengths[s_perm]

            ret_indices.append(list(under[i][1:chosen_lengths[i]] for i in range(z)))
            ret.append(list(TOKEN_TAGGER.parse_indices(under[i][1:chosen_lengths[i]]) for i in range(z)))
            starts.append(start[selected_indices][s_perm].tolist())

    return ret, ret_indices, counts, starts, tags_indices

def convert_csv_rule():
    '''
    Function to load grammar rules from a CSV file on disk
    '''

    print("Loading token database...")
    print(configx.BREAK_LINE)

    search_arrays = load_search_arrays()
    unique_arrays = load_unique_arrays()

    print("Finished loading token databases...")
    print(configx.BREAK_LINE)
    
    # Load rule file
    rule_file = os.path.join(configx.CONST_RULE_CONFIG_DIRECTORY, configx.CONST_RULE_CONFIG)

    # Open file
    with open(rule_file, 'r') as f:

        csv_reader = csv.reader(f, delimiter=',')
        
        # Counter for number of iterations
        lc = 0

        # Read each line (rule) of CSV
        for rule_text in csv_reader:

            if lc == 0:
                lc += 1
                continue

            print("Reading Rule %2d: %s --> %s" % (lc, corrected_sentence, error_sentence))
            print(configx.BREAK_LINE)

            # Paired sentence data
            corrected_sentence = rule_text[0]
            error_sentence = rule_text[1]
            
            # Part of speech tags of the correct sentence
            pos_tags = rule_text[2]
            pos_tags = pos_tags.split(',')

            n_tokens = int(len(pos_tags) / N_POS)
            pos_tags = np.array(list(util.parse_pos_tags(pos_tags[i * N_POS: i * N_POS + N_POS], POS_TAGGERS) for i in range(n_tokens)))   


            # Array of arrays denoting hows part-of-speech tags have been selected
            #   This is marked as -1 = null, 0 = no match, 1 = match
            selections = rule_text[3]
            selections = np.array(list(int(j) for j in selections.split(',')))
            selections = selections.reshape(-1, N_POS)

            # Arrays of tuples denoting token mappings between errored and correct sentence
            created = rule_text[4]
            altered = rule_text[5]
            preserved = rule_text[6]

            created = ast.literal_eval(created)
            altered = ast.literal_eval(altered)
            preserved = ast.literal_eval(preserved)

            mapping = (created, altered, preserved)          

            print("\tFinding potential substitute tokens...")

            # List of possible substitute token classes (part-of-speech combinations) per each index of correct sentence
            possible_classes = list()

            for index in range(n_tokens):

                all_tags, all_classes = match_template_tokens(unique_arrays, pos_tags[index], selections[index])
                possible_classes.append(all_classes)

            # Determine number of possible substitutes at each index
            n_possibilities = list(len(i) for i in possible_classes)

            print("\tSearching for sentences matching pair template...")
            s_examples, s_indices, class_counts, starts, tag_indices = match_template_sentence(search_arrays, pos_tags, selections, possible_classes, n_possibilities)

            print("\tGenerating new sentence pairs...")
            error_examples, l_correct, l_error = \
                generate.create_errored_sentences(unique_arrays, 
                                                  ATTRIBUTE_INDICES, 
                                                  TOKEN_TAGGER, 
                                                  POS_TAGGERS, 
                                                  mapping, 
                                                  selections, 
                                                  s_examples,         
                                                  starts, 
                                                  error_sentence, 
                                                  corrected_sentence)

            print("\tSaving new data...")
            save.save_rule(corrected_sentence, error_sentence, s_examples, error_examples, lc, rule_text)

            print("\tPaired data saved succesfully...\n")

            lc += 1



if __name__ == '__main__':

    convert_csv_rule()
