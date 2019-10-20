import csv
import os
import numpy as np

from . import configx
from . import languages
from . import util


def get_paired_phrases(tagger, source, target, start, top_k=-1):

    start = open(start, "r")
    start_lines = start.readlines()
    starts = list(list(int(i) for i in x.strip().split(','))
              for x in start_lines)

    target = open(target, "r")
    source = open(source, "r")

    source_lines = source.readlines()
    target_lines = target.readlines()

    assert(len(source_lines) == len(target_lines))
    n_lines = len(source_lines)

    source_strings = list()
    target_strings = list()

    for i in range(n_lines):

        text_source = source_lines[i].strip().split(' ')
        text_target = target_lines[i].strip().split(' ')

        indices_source = list(tagger.parse_node(j, top_k) for j in text_source)
        indices_target = list(tagger.parse_node(j, top_k) for j in text_target)

        tokens_source = list(tagger.parse_index(k) for k in indices_source)
        tokens_target = list(tagger.parse_index(k) for k in indices_target)

        if len(indices_target) == 0:

            n_lines -= 1

        else:

            if starts is not None:

                indices = starts[i]

                _source = ''.join(tokens_source[indices[0]: indices[1]])
                _target = ''.join(tokens_target[indices[2]: indices[3]])

                source_strings.append(_source)
                target_strings.append(_target)

    return source_strings, target_strings


def process_csv(input_file, output_source, output_target):

    source_text = list()
    target_text = list()

    with open(input_file, "r", encoding="utf-8") as in_file:

        for pair in in_file.readlines():

            pair = pair.strip().split(',')

            source_sentence = pair[0]

            if len(pair) == 1:

                target_sentence = ''

            elif len(pair) == 2:

                target_sentence = pair[1]

            else:

                print("WARNING: MORE THAN TWO SENTENCES IN LINE")
                target_sentence = pair[1]

            source_sentence = source_sentence.strip()
            target_sentence = target_sentence.strip()

            if target_sentence == '':
                target_sentence = source_sentence

            source_tokens = languages.parse(
                source_sentence, configx.CONST_PARSER, None)
            target_tokens = languages.parse(
                target_sentence, configx.CONST_PARSER, None)

            source_text.append(" ".join(source_tokens))
            target_text.append(" ".join(target_tokens))

        in_file.close()

    source_file = open(output_source, "w+")
    target_file = open(output_target, "w+")

    for i in range(len(source_text)):

        source_file.write(source_text[i])
        source_file.write(os.linesep)

        target_file.write(target_text[i])
        target_file.write(os.linesep)

    source_file.close()
    target_file.close()


def sort_sentences(input_file, output_file):

    ret = []

    with open(input_file, "r") as f:

        data = f.readlines()

        n_sentences = len(data)

        ret = [None] * n_sentences

        for sentence in data:

            tokens = sentence.split()

            ini = tokens[0]
            index = int(ini[2:])

            assert(index < n_sentences)

            # Hypothesis
            if ini[0] == 'H':

                tokens = tokens[2:]

            elif ini[0] == 'P':

                tokens = tokens[1:-1]

            else:

                tokens = tokens[1:]

            ret[index] = " ".join(tokens).strip()

    f.close()

    g = open(output_file, "w+")

    for i in range(n_sentences):

        g.write(ret[i])
        g.write(os.linesep)

    g.close()


def remove_pairs(original_source, original_target,
                 output_source, output_target, same):

    original_source = open(original_source, "r")
    original_target = open(original_target, "r")

    s_lines = original_source.readlines()
    t_lines = original_target.readlines()

    assert(len(s_lines) == len(t_lines))

    new_source = list()
    new_target = list()

    for i in range(len(s_lines)):

        if (s_lines[i] == t_lines[i]) == same:

            pass

        else:

            new_source.append(s_lines[i])
            new_target.append(t_lines[i])

    original_source.close()
    original_target.close()

    output_source = open(output_source, "w+")
    output_target = open(output_target, "w+")

    output_source.writelines(new_source)
    output_target.writelines(new_target)

    output_source.close()
    output_target.close()


def filter_probabilities(system_file, probability_file, source_file,
                         reference_file, output_dir):

    token_tagger, _ = languages.load_default_languages(
        load_dir=configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)

    f_out = open(system_file, "r")
    f_prob = open(probability_file, "r")

    f_org = open(source_file, "r")
    f_ref = open(reference_file, "r")

    sys_out = f_out.readlines()
    sys_prob = f_prob.readlines()

    _org = f_org.readlines()
    _ref = f_ref.readlines()

    f_out.close()
    f_prob.close()
    f_org.close()
    f_ref.close()

    assert(len(sys_out) == len(sys_prob))

    completed = 0

    # Probability assigned to correctly outputed token
    c_probs = list()
    # Probabilities assigned to incorrectly outputed with form change
    #   (diff. from original) (INCORRECT CORRECTION)
    fc_probs = list()
    fc_examples = list()
    # Probabilities assigned to incorrectly outputed tokens where ref
    #   is in original (RANDOM SUBSTITUTIONS)
    sub_probs = list()
    sub_examples = list()
    # Probabilities assigned to incorrectly outputed tokens where ref
    #   is not in original (INCORRECT)
    x_probs = list()
    x_examples = list()

    r_sub_probs = list()
    r_sub_examples = list()

    n_analyze = len(sys_out)
    # n_analyze = 20000

    for i in range(n_analyze):

        sys_sentence = list(_token for _token in sys_out[i].strip().split(' '))
        sys_string = ''.join(sys_sentence)
        l_probs = list(float(_prob)
                       for _prob in sys_prob[i].strip().split(' '))

        for j in range(len(l_probs)):

            if abs(l_probs[j]) < 1e-6:

                l_probs[j] = 0

        assert (len(l_probs) == len(sys_sentence))

        org_sentence = list(_token for _token in _org[i].strip().split(' '))
        org_string = ''.join(org_sentence)
        ref_sentence = list(_token for _token in _ref[i].strip().split(' '))
        ref_string = ''.join(ref_sentence)

        _, org_pos = languages.parse_full(
            org_string, configx.CONST_PARSER, None)
        _, ref_pos = languages.parse_full(
            ref_string, configx.CONST_PARSER, None)
        _, sys_pos = languages.parse_full(
            sys_string, configx.CONST_PARSER, None)

        org_forms = org_pos[-1]
        ref_forms = ref_pos[-1]
        sys_forms = sys_pos[-1]

        # No alignment of target/ref (more basic statistic)
        if len(ref_sentence) != len(sys_sentence):

            continue

        elif len(ref_forms) != len(ref_sentence):

            continue

        elif len(sys_forms) != len(sys_sentence):

            continue

        else:

            completed += 1

            n_tokens = len(ref_sentence)

            for j in range(n_tokens):

                if sys_sentence[j] != ref_sentence[j]:

                    if sys_forms[j] in org_forms or sys_forms[j] in ref_forms:

                        fc_probs.append(l_probs[j])
                        fc_examples.append(
                            ref_sentence[j] + "," + sys_sentence[j] +
                            os.linesep)

                    else:

                        # sys_index = token_tagger.parse_node(
                        #     sys_sentence[j], n_max=10000)
                        org_index = token_tagger.parse_node(
                            org_sentence[j], n_max=10000)

                        if org_index == 1 or org_index > 2000:

                            r_sub_probs.append(l_probs[j])
                            r_sub_examples.append(
                                ref_sentence[j] + "," + sys_sentence[j] +
                                os.linesep)

                        elif ref_forms[j] in org_forms:

                            sub_probs.append(l_probs[j])
                            sub_examples.append(
                                ref_sentence[j] + "," + sys_sentence[j] +
                                os.linesep)

                        else:

                            x_probs.append(l_probs[j])
                            x_examples.append(
                                ref_sentence[j] + "," + sys_sentence[j] +
                                os.linesep)

                else:

                    c_probs.append(l_probs[j])

    # Save example pairs:
    _f = open(os.path.join(output_dir, "form_change.csv"), "w+")
    _f.writelines(fc_examples)
    _f.close()
    _f = open(os.path.join(output_dir, "substitution.csv"), "w+")
    _f.writelines(sub_examples)
    _f.close()
    _f = open(os.path.join(output_dir, "rare_substitution.csv"), "w+")
    _f.writelines(r_sub_examples)
    _f.close()
    _f = open(os.path.join(output_dir, "other.csv"), "w+")
    _f.writelines(x_examples)
    _f.close()

    c_probs = np.array(c_probs)
    fc_probs = np.array(fc_probs)
    sub_probs = np.array(sub_probs)
    x_probs = np.array(x_probs)

    _f = open(os.path.join(output_dir, "out.log"), "w+")
    wl = list()

    arrays = [c_probs, fc_probs, sub_probs, r_sub_probs, x_probs]
    names = ["Correct Tokens", "Form Changes",
             "Incorrect Substitutions", "Rare Substitutions", "Other"]
    percentiles = np.arange(0, 100.0, 2.0)

    for i in range(len(arrays)):

        array = arrays[i]
        name = names[i]

        _str = "Array Data: %s\n===========================" % (name)
        wl.append(_str)
        print(_str)

        _str = "\n\tCount: %2d" % (len(array))
        wl.append(_str)
        print(_str)

        _str = "\nPercentiles: "
        wl.append(_str)
        print(_str)

        arr_percentiles = np.percentile(array, percentiles)

        print()
        for k in range(len(percentiles)):
            _str = "\t%2d: %4f" % (percentiles[k], arr_percentiles[k])
            wl.append(_str)
            print(_str)

        wl.append("\n\n")

    _f.writelines(wl)
    _f.close()


def replace_low_probability(source_file, align_file, system_file,
                            probability_file, output_file, threshold):

    source_file = open(source_file, "r")
    align_file = open(align_file, "r")
    system_file = open(system_file, "r")
    probability_file = open(probability_file, "r")

    source_sentences = source_file.readlines()
    source_file.close()

    alignments = align_file.readlines()
    align_file.close()

    system_sentences = system_file.readlines()
    system_file.close()

    probabilities = probability_file.readlines()
    probability_file.close()

    n_sentences = len(source_sentences)

    assert(len(alignments) == n_sentences)
    assert(len(system_sentences) == n_sentences)
    assert(len(probabilities) == n_sentences)

    ret = [None] * n_sentences

    for i in range(n_sentences):

        print(i)

        source = source_sentences[i].strip().split(' ')
        align = alignments[i].strip().split(' ')
        p = probabilities[i].strip().split(' ')
        sys = system_sentences[i].strip().split(' ')

        output = []

        for j in range(len(p)):

            c_p = abs(float(p[j]))

            if c_p > float(threshold):

                a = int(align[j])

                if a >= len(source):

                    output.append('。')

                    break

                else:

                    print("Current token: %s" % sys[j])
                    print("Loss: %2f" % c_p)
                    print("Aligned token: %s" % source[a])

                    output.append(source[a].strip())

            else:

                output.append(sys[j].strip())

        ret[i] = " ".join(output) + os.linesep

    output_file = open(output_file, "w+")
    output_file.writelines(ret)
    output_file.close()
