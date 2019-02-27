
import numpy as np

from . import parse
from . import util


def create_errored_sentences(unique_arrays, array_indices, tagger, languages, mapping, selections, 
                             new_sentences, start_indices, errored, corrected):
    """Summary
    
    Args:
        unique_arrays (TYPE): Description
        array_indices (TYPE): Description
        tagger (TYPE): Description
        languages (TYPE): Description
        mapping (TYPE): Description
        selections (TYPE): Description
        new_sentences (TYPE): Description
        start_indices (TYPE): Description
        errored (TYPE): Description
        corrected (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    tokens = unique_arrays[0]
    tags = unique_arrays[4]   

    c_error, nodes_error = parse.parse(errored)
    c_error = list(list(a[z] for z in array_indices) for a in c_error)
    error_pos = np.array(list(util.parse_pos_tags(a, languages) for a in c_error))
    
    _, nodes_correct = parse.parse(corrected)

    ret_new_sentences = []

    l_error = len(nodes_error)
    l_correct = len(nodes_correct)

    created = mapping[0]
    altered = mapping[1]
    preserved = mapping[2]

    for i in range(len(new_sentences)):

        current_set = new_sentences[i]

        ret_new_set = []

        for j in range(len(current_set)):

            generated_sentence = [''] * l_error

            current_sentence = current_set[j]
            current_start = start_indices[i][j] - 1

            current_sentence = current_sentence.split(',')

            temp = ''.join(current_sentence)

            c_new, nodes_new = parse.parse(temp)
            c_new = list(list(a[z] for z in array_indices) for a in c_new)
            new_pos = np.array(list(util.parse_pos_tags(a, languages) for a in c_new))

            new_pos = new_pos[current_start:current_start + l_correct]
            nodes_new = nodes_new[current_start:current_start + l_correct]

            # Created tokens
            for k in range(len(created)):
             
                generated_sentence[created[k]] = nodes_error[created[k]]

            # Altered token
            for k in range(len(altered)):

                x = selections[altered[k][0]]

                variable_positions = list()           

                for q in range(len(x) - 1):                    

                    if x[q] != 1:

                        variable_positions.append(q)

                first_template = error_pos[altered[k][0]][:].tolist()                
                first_template[-1] = new_pos[altered[k][1]][-1]

                if len(variable_positions) == list():

                    possibilities = []
                    possibilities.append(first_template)

                else:
      

                    possibilities = [0] * (2 ** len(variable_positions))

                for z in range(len(possibilities)):

                    possibilities[z] = first_template[:]

                    setting = str(format(z, 'b'))

                    for zz in range(len(variable_positions) - len(setting)):

                        setting = '0' + setting

                    for aa in range(len(setting)):

                        if (setting[aa] == '1'):

                            possibilities[z][variable_positions[aa]] = new_pos[altered[k][1]][variable_positions[aa]]

                possibilities = list(tuple(aaa) for aaa in possibilities)

                discovered = False
                     
                for template_pos in possibilities: 

                    # Using same base form, create altered candidate token matching errored sentence template
                    # template_pos = tuple(error_pos[altered[k][0]][:-1]) + tuple(new_pos[altered[k][1]][-1:])
                    
                    # If the modified candidate token has been seen before
                    if template_pos in tags:

                        q = tags[template_pos][0]

                        replacement = tokens[q][0]

                        generated_sentence[altered[k][0]] = tagger.parse_index(replacement)

                        discovered = True

                if not discovered:

                    generated_sentence[altered[k][0]] = "UNKNOWN"


                # # If the candidate token has not been seen before (use character by character replacement)
                # else:

                #     ret = token_replacement(nodes_error, template_pos, altered[k][0], nodes_correct[altered[k][1]], nodes_new[altered[k][1]])

                #     if ret != None:

                #         generated_sentence[altered[k][1]] = TOKEN_TAGGER.parse_index(ret)
                    
                #     else:

                #         generated_sentence[altered[k][1]] = "UNKNOWN"

            for k in range(len(preserved)):
                generated_sentence[preserved[k][0]] = nodes_new[preserved[k][1]]

            t = current_sentence[:current_start] + generated_sentence + current_sentence[current_start + l_correct:]
            ret_new_set.append(''.join(t))

            current_set[j] = ''.join(current_sentence)


        ret_new_sentences.append(ret_new_set)

    # save_data = (new_sentences, ret_new_sentences, errored, corrected)
    # save_data_full(save_data)
        
    return ret_new_sentences, l_correct, l_error