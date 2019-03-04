
import numpy as np

from termcolor import colored

from . import configx
from . import languages
from . import util


def create_errored_sentences(unique_arrays, token_tagger, pos_taggers, mapping, selections, 
                             matched_sentences, start_indices, errored, corrected, verbose=True):
    
    # Separate individual matrices for use
    tokens = unique_arrays[0]
    tags = unique_arrays[4] 

    created = mapping[0]
    altered = mapping[1]
    preserved = mapping[2]

    delimiter = token_tagger.stop_token  

    # Parse template phrases
    nodes_correct, _ = languages.parse_sentence(corrected, configx.CONST_PARSER, delimiter)
    nodes_error, pos_error = languages.parse_sentence(errored, configx.CONST_PARSER, delimiter)

    # Obtain 2D part-of-speech matrix for errored phrase
    # Of form (n, k), where n is the length of the phrase, 
    #   and k is the number of part-of-speech tags per token
    pos_error = np.array(list(languages.parse_node_matrix(pos_token, pos_taggers) \
                              for pos_token in np.array(pos_error).T))

    # Return array containing newly generated sentence pairs
    ret = []

    # Arrays containing lengths of the generated sentence pairs
    lengths_error = len(nodes_error)
    lengths_correct = len(nodes_correct)

    if verbose:
        print("\t\tCorrect: " + ' | '.join(nodes_correct))
        print("\t\tError: " + ' | '.join(nodes_error))

    # Iterate over each sub-rule 
    for i in range(len(matched_sentences)):

        if verbose:

            print("\n\t\tProcessing sub-rule %d of %d..." % (i + 1, len(matched_sentences)))
            print(configx.BREAK_HALFLINE)

        current_sub_rule = matched_sentences[i]
        ret_sub_rule = []

        # Iterate over each sentence in each subrule
        for j in range(len(current_sub_rule)):

            valid = True

            if verbose:

                print("\n\t\t\tProcessing sentence %d of %d..." % (j + 1, len(current_sub_rule)))

            # Initialize new sentence with blank errored phrase
            generated_sentence = [''] * lengths_error

            template_sentence = current_sub_rule[j]
            template_start = start_indices[i][j] - 1

            # Get list representation of current template sentence
            template_sentence = template_sentence.split(',')
            temp = ''.join(template_sentence)

            nodes_template, pos_template = languages.parse_sentence(temp, configx.CONST_PARSER, delimiter)

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

            # Preserved tokens -> copy tokens from correct
            for k in range(len(preserved)):

                # NOTE:
                # preserved[:][0] corresponds to indices of preserved tokens in ERRORED phrase
                # preserved[:][1] corresponds to indices of preserved tokens in CORRECT phrase

                generated_sentence[preserved[k][0]] = nodes_template[preserved[k][1]]

            # Altered token -> token search + replacement
            for k in range(len(altered)):

                # Boolean determining if a replacement token has been found by reverse-searching
                #   the token tagger
                discovered = False

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

                if not discovered:

                    types_tested = 0
                     
                    for template_pos in possibilities: 

                        print(template_pos)

                        # Using same base form, create altered candidate token matching errored sentence template
                        # template_pos = tuple(pos_error[altered[k][0]][:-1]) + tuple(pos_template[altered[k][1]][-1:])
                        
                        # If the modified candidate token has been seen before
                        if template_pos in tags:

                            q = tags[template_pos][0]

                            replacement = tokens[q][0]

                            generated_sentence[altered[k][0]] = token_tagger.parse_index(replacement)

                            discovered = True

                            print("\t\t\t\tMatched token on type: %d" % types_tested + 1)

                            break

                        types_tested += 1

                if not discovered:         

                    correct_token = nodes_correct[altered[k][1]]
                    errored_token = nodes_error[altered[k][0]]


                    generated_sentence[altered[k][0]] = "UNKNOWN"

                print("\t\t\t\tCompleting alteration %s->%s||%s->%s" % \
                     (nodes_correct[altered[k][1]], nodes_error[altered[k][0]], nodes_template[altered[k][1]], generated_sentence[altered[k][0]]))



            t = template_sentence[:template_start] + generated_sentence + template_sentence[template_start + lengths_correct:]
            
            if valid:

                ret_sub_rule.append(''.join(t))

            else:

                print("ERROR during generation - sentence will not be used")
            # current_sub_rule[j] = ''.join(template_sentence)


        ret.append(ret_sub_rule)

    # save_data = (matched_sentences, ret, errored, corrected)
    # save_data_full(save_data)
        
    return ret, lengths_correct, lengths_error