# Filename: generate.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 23/06/2018
# Date Last Modified: 04/03/2019
# Python Version: 3.7

'''
Functions to generate new errored sentences from template sentences
'''

import numpy as np

from termcolor import colored

from . import configx
from . import languages
from . import util


def create_errored_sentences(unique_arrays, token_tagger, pos_taggers, 
                             mapping, selections, matched_sentences, start_indices, 
                             errored, corrected, verbose=True, max_per_colored=5):
    """
    Function to generate errored sentences from template sentences using a given mapping on the 
        correct -> errored rule designated by corrected and errored inputs
    
    Args:
        unique_arrays (arr): Arrays containing information on unique tokens within the corpus and their 
                              corresponding part-of-speech information
        token_tagger (Language): Language class instance used to tag tokens
        pos_taggers (arr): List of Language class instances used to tag each part-of-speech index
        mapping (TYPE): List of three arrays determining which tokens from the template sentence
        selections (arr): Array determining which part-of-speech indices need to be matched
        matched_sentences (TYPE): Template sentences to generate from, grouped by sub-rule
        start_indices (TYPE): Start indices of template phrases within each template sentence
        errored (str): Errored phrase of rule
        corrected (str): Corrected phrase of rule
        verbose (bool, optional): Determines whether debugging string output is printed to terminal or not
        max_per_colored (int, optional): Determines maximum number of colored output sentences per subrule outputted
    
    Returns:
        ret (arr): A list of list of pairs of sentences (template, generated), grouped by sub-rule
    """
    # Separate individual matrices for use
    tokens = unique_arrays[0]
    tags = unique_arrays[4] 

    created = mapping[0]
    altered = mapping[1]
    preserved = mapping[2]

    delimiter = token_tagger.stop_token  

    # Parse template phrases
    nodes_correct, _ = languages.parse_full(corrected, configx.CONST_PARSER, delimiter)
    nodes_error, pos_error = languages.parse_full(errored, configx.CONST_PARSER, delimiter)

    # Obtain 2D part-of-speech matrix for errored phrase
    # Of form (n, k), where n is the length of the phrase, 
    #   and k is the number of part-of-speech tags per token
    pos_error = np.array(list(languages.parse_node_matrix(pos_token, pos_taggers) \
                              for pos_token in np.array(pos_error).T))

    # Return array containing newly generated sentence pairs
    ret = []
    # Return coloured variant of generated pairs
    ret_coloured = []

    # Arrays containing lengths of the generated sentence pairs
    lengths_error = len(nodes_error)
    lengths_correct = len(nodes_correct)

    if verbose:
        print("\n\t\tCorrect: " + ' | '.join(nodes_correct))
        print("\t\tError: " + ' | '.join(nodes_error))

    valid_rule_starts = list()

    # Iterate over each sub-rule 
    for i in range(len(matched_sentences)):

        if verbose:

            print("\n\t\tProcessing sub-rule %d of %d..." % (i + 1, len(matched_sentences)))
            print(configx.BREAK_HALFLINE)

        current_sub_rule = matched_sentences[i]
        ret_sub_rule = []
        ret_sub_rule_coloured = []

        sub_rule_starts = list()

        # Iterate over each sentence in each subrule
        for j in range(len(current_sub_rule)):

            try:

                valid = True

                if verbose:

                    print("\n\t\t\tProcessing sentence %d of %d..." % (j + 1, len(current_sub_rule)))

                # Initialize new sentence with blank errored phrase
                generated_sentence = [''] * lengths_error

                coloured_correct = [''] * lengths_correct
                coloured_error = [''] * lengths_error

                template_sentence = current_sub_rule[j]
                template_start = start_indices[i][j] - 1

                # Get list representation of current template sentence
                template_sentence = template_sentence.split(',')
                temp = ''.join(template_sentence)

                nodes_template, pos_template = languages.parse_full(temp, configx.CONST_PARSER, delimiter)

                # Obtain 2D part-of-speech matrix for template sentence
                pos_template = np.array(list(languages.parse_node_matrix(pos_token, pos_taggers) \
                                          for pos_token in np.array(pos_template).T))    

                # Extract template phrase from sentence      
                pos_template = pos_template[template_start:template_start + lengths_correct]
                nodes_template = nodes_template[template_start:template_start + lengths_correct]

                if verbose:

                    print("\t\t\t\tTemplate: " + ' | '.join(nodes_template))

                # Created tokens -> copy tokens from errored
                for k in range(len(created)):

                    # NOTE:
                    # created[:] corresponds to indices of newly generated tokens in ERRORED phrase
                 
                    generated_sentence[created[k]] = nodes_error[created[k]]
                    coloured_error[created[k]] = colored(nodes_error[created[k]], 'green')


                # Preserved tokens -> copy tokens from correct
                for k in range(len(preserved)):

                    # NOTE:
                    # preserved[:][0] corresponds to indices of preserved tokens in ERRORED phrase
                    # preserved[:][1] corresponds to indices of preserved tokens in CORRECT phrase                
                    generated_sentence[preserved[k][0]] = nodes_template[preserved[k][1]]
                    coloured_error[preserved[k][0]] = nodes_template[preserved[k][1]]
                    coloured_correct[preserved[k][1]] = nodes_template[preserved[k][1]]


                # Altered token -> token search + replacement
                for k in range(len(altered)):

                    coloured_correct[altered[k][1]] = colored(nodes_template[altered[k][1]], 'blue')

                    # Boolean determining if a replacement token has been found by reverse-searching
                    #   the token tagger
                    discovered = False

                    # Boolean determining if the token replacement has never been seen previously
                    unique = False

                    # NOTE:
                    # altered[:][0] corresponds to indices of altered tokens in ERRORED phrase
                    # altered[:][1] corresponds to indices of altered tokens in CORRECT phrase
                    
                    # Difference in length (in characters) between the template and errored sentences
                    diff_length = len(nodes_correct[altered[k][1]]) - len(nodes_error[altered[k][0]])

                    # If the difference in length is equal to that of the template, the template should be replaced
                    #   by an empty string
                    # i.e. (作｜ますー＞作｜ます should result in し｜ますー＞｜ます where し is replaced by empty string)
                    if diff_length == len(nodes_template[altered[k][1]]):

                        discovered = True

                    # Generate template part-of-speech combination from errored token
                    first_template = pos_error[altered[k][0]][:].tolist()

                    # Replace the form of the combination with that of the template
                    # The replacement token should have the same form as that of template, 
                    #   so that even if the rule is for one form (i.e. 作る) it can be applied to
                    #   others (i.e. 守る、 知る、還る, etc.)
                    # NOTE: (作る、 作り、 作っ 作れ) all have same form: 作る            
                    first_template[-1] = pos_template[altered[k][1]][-1]

                    # if pos_template[altered[k][1]][-1] != pos_error[altered[k][0]][-1]:

                    #     pass

                    # Obtain the matching leniency of the correct token
                    altered_indices = selections[altered[k][1]]

                    # Get a list of those part-of-speech indices with matching leniency
                    variable_positions = list()
                    for q in range(len(altered_indices) - 1):                    

                        if altered_indices[q] != 1:

                            variable_positions.append(q)

                    # If there is no possible leniency, only possible final part-of-speech combination
                    #   is that of the current template (i.e. form of template token, part-of-speech of errored token)           
                    if variable_positions == list():

                        possibilities = []
                        possibilities.append(first_template)

                    # Otherwise, there is a possibility that the final part-of-speech combination can be
                    #   a permutation of the template and errored tokens' part of speech combinations
                    else:      

                        possibilities = [0] * (2 ** len(variable_positions))

                    # Fill in possible part-of-speech combinations on lenient matching indices
                    # Use binary numbers to determine indices which are swapped (to that of correct token)
                    for l in range(len(possibilities)):

                        possibilities[l] = first_template[:]

                        setting = str(format(l, 'b'))

                        for _ in range(len(variable_positions) - len(setting)):

                            setting = '0' + setting

                        for m in range(len(setting)):

                            if (setting[m] == '1'):

                                possibilities[l][variable_positions[m]] = pos_template[altered[k][1]][variable_positions[m]]

                    possibilities = list(tuple(p) for p in possibilities)

                    # If the token has not been replaced (i.e. is not a null token due to length differences)
                    if not discovered:

                        types_tested = 0
                         
                        # Check all permutated part-of-speech combinations
                        for template_pos in possibilities: 

                             # If the candidate combination has been seen in the corpus
                            if template_pos in tags:

                                # Reverse search for index of tokens matching part-of-speech combination
                                token_indices = tags[template_pos]

                                if (len(token_indices) > 1):

                                    print("\t\t\t\tWARNING: Multiple token matches for selected part-of-speech combination")

                                # Arbitrarily select first token
                                # NOTE: May be necessary to change
                                replacement = tokens[token_indices[0]][0]

                                # Place token in generated sentence
                                generated_sentence[altered[k][0]] = token_tagger.parse_index(replacement)
                                coloured_error[altered[k][0]] = colored(token_tagger.parse_index(replacement), 'yellow')

                                discovered = True

                                if verbose:

                                    print("\t\t\t\tMatched token on type: %d" % (types_tested + 1))

                                break

                            types_tested += 1

                    # If token has not been seen previously, must be generated
                    if not discovered:         

                        correct_token = nodes_correct[altered[k][1]]
                        errored_token = nodes_error[altered[k][0]]
                        template_token = nodes_template[altered[k][1]]

                        # Check if tokens aligned at front (MOST COMMON):
                        if correct_token[0] == errored_token[0]:

                            matched_count = 1

                            # Iterate to find last index of matching
                            for j in range(min(len(errored_token) - 1, len(correct_token) - 1)):

                                if correct_token[j + 1] == errored_token[j + 1]:

                                    matched_count += 1

                                else:

                                    break

                            # Number of characters deleted from correct token
                            n_deleted = len(correct_token) - matched_count
                            # Number of characters substituted in from errored token
                            n_change = len(errored_token) - matched_count

                            # Initialize new token with the correct token up to removed portion
                            new_token = template_token[:-n_deleted]

                            # If there is portion to add from errored
                            #   (correct token does not contain errored token as subsequence)
                            if n_change > 0:

                                new_token += errored_token[-n_change:]

                            # Place token in generated sentence
                            generated_sentence[altered[k][0]] = new_token
                            coloured_error[altered[k][0]] = colored(new_token, 'yellow')

                            unique = True

                        # Check if tokens aligned at end ():
                        # TODO: Maybe this might be issue
                        elif correct_token[-1] == errored_token[-1]:

                            print ("WARNING: End alignment --- Not implemented yet")
                            raise 

                            generated_sentence[altered[k][0]] = configx.CONST_UNKNOWN_TOKEN
                            coloured_error[altered[k][0]] = colored(configx.CONST_UNKNOWN_TOKEN, 'yellow')

                        # No surefire way of handling other cases (no alignment- likely error with rule)
                        else:

                            valid = False               

                    # Show exactly how token is altered
                    if verbose:

                        print_altered = generated_sentence[altered[k][0]]

                        # If the altered token is not valid 
                        if not valid:

                            print_altered = colored('ERROR', 'grey')

                        # If altered token is deleted (i.e. the alteration deletes a one-kana verb stem)
                        elif print_altered == '':

                            print_altered = colored('NULL', 'red')

                        # If the token is new (i.e. not in original dictionary)
                        elif unique:

                            print_altered = colored(print_altered, 'yellow')

                        # Otherwise, if the altered token was in the original dictionary
                        else:

                            print_altered = colored(print_altered, 'green')

                        print("\t\t\t\tCompleting alteration %s->%s||%s->%s" % \
                             (nodes_correct[altered[k][1]], nodes_error[altered[k][0]], nodes_template[altered[k][1]], print_altered))

                # Finish constructing generated sentence by placing generated phrase 
                #   within non-altered portions of tempalte sentence
                generated_sentence = template_sentence[:template_start] + generated_sentence + \
                    template_sentence[template_start + lengths_correct:]

                # Fill in red for deleted tokens on correct sentence
                for k in range(len(nodes_template)):

                    if coloured_correct[k] == '':

                        coloured_correct[k] = colored(nodes_template[k], 'red')

                # Finish constructing coloured sentences
                coloured_correct = template_sentence[:template_start] + coloured_correct + \
                    template_sentence[template_start + lengths_correct:]
                coloured_error = template_sentence[:template_start] + coloured_error + \
                    template_sentence[template_start + lengths_correct:]

                # If the rule is valid, append the new sentence pair to the return array
                if valid:

                    sub_rule_starts.append(start_indices[i][j])
                    ret_sub_rule.append((''.join(generated_sentence), ''.join(template_sentence)))
                    
                    if len(ret_sub_rule_coloured) < max_per_colored:
                        # Manually splice coloured sentences together
                        cc = ''
                        ce = ''

                        for l in range(len(coloured_correct)):
                            cc += coloured_correct[l]

                        for l in range(len(coloured_error)):
                            ce += coloured_error[l]

                        ret_sub_rule_coloured.append((ce, cc))

                else:

                    print("ERROR during generation - sentence will not be used")

            except:

                print("EXCEPTION")

                continue
        ret.append(ret_sub_rule)
        ret_coloured.append(ret_sub_rule_coloured)

        valid_rule_starts.append(sub_rule_starts)

    # TODO: Make output be paired
       
    return ret, ret_coloured, valid_rule_starts

