from . import languages
from . import configx
from . import util

from nltk import f_measure
import csv
import os


def eval_binary(src, ref, sys, corpus_name='', rule_label='', out_crt=None, out_err=None, start=None, rule=None,
                top_k=10000, language_dir=None):

    if language_dir is None:

        tagger, _ = languages.load_default_languages()

    else:

        tagger, _ = languages.load_languages(language_dir)

    model_name = os.path.splitext(os.path.basename(sys))[0]

    ref = open(ref, "r")
    sys = open(sys, "r")
    src = open(src, "r")

    if out_err is not None:

        util.mkdir_p(out_err, file=True)
        out_err = open(out_err, "w+")
        out_err.write('インデクス,誤り文,正しい文,モデル出力' + os.linesep)

    if out_crt is not None:

        util.mkdir_p(out_crt, file=True)
        out_crt = open(out_crt, "w+")
        out_crt.write('インデクス,誤り文,正しい文,モデル出力' + os.linesep)

    rule_strings = None
    starts = None

    if start is not None:

        starts = list()

        start = open(start, "r")
        start_lines = start.readlines()

        n_start = len(start_lines[0].split(','))

        if rule is not None:

            rule = open(rule, "r")
            rule_lines = rule.readlines()
            rule_strings = ''.join(rule_lines[0].split()).split(',')
            rule_lengths = list(int(i)
                                for i in rule_lines[1].strip().split(','))

        if n_start == 1:
            for i in start_lines:

                idx = int(i.strip())
                rule_starts = [idx, idx + rule_lengths[1],
                               idx, idx + rule_lengths[1]]
                starts.append(rule_starts)

        if n_start == 2:
            for i in start_lines:

                indices = list(int(x) for x in i.strip().split(','))
                rule_starts = [indices[1], indices[1] + rule_lengths[1],
                               indices[0], indices[0] + rule_lengths[0]]
                starts.append(rule_starts)

        else:

            starts = list(list(int(i) for i in x.strip().split(','))
                          for x in start_lines)

    src_lines = src.readlines()
    ref_lines = ref.readlines()
    sys_lines = sys.readlines()

    assert(len(ref_lines) == len(sys_lines))

    n_lines = len(ref_lines)
    ret = 0

    for i in range(n_lines):

        text_src = src_lines[i].strip().split(' ')
        text_ref = ref_lines[i].strip().split(' ')
        text_sys = sys_lines[i].strip().split(' ')

        indices_src = list(tagger.parse_node(j, top_k) for j in text_src)
        indices_ref = list(tagger.parse_node(j, top_k) for j in text_ref)
        indices_sys = list(tagger.parse_node(j, top_k) for j in text_sys)

        tokens_src = list(tagger.parse_index(k) for k in indices_src)
        tokens_ref = list(tagger.parse_index(k) for k in indices_ref)
        tokens_sys = list(tagger.parse_index(k) for k in indices_sys)

        if len(indices_ref) == 0:

            n_lines -= 1

        else:

            if starts is not None:

                indices = starts[i]

                _src = ' '.join(tokens_src[indices[0]: indices[1]])
                _ref = ' '.join(tokens_ref[indices[2]: indices[3]])
                _sys = ' '.join(tokens_sys[indices[2]: indices[3]])

            else:

                _src = ' '.join(tokens_src)
                _ref = ' '.join(tokens_ref)
                _sys = ' '.join(tokens_sys)

            if (_ref == _sys):

                if out_crt is not None:

                    out_crt.write(
                        ','.join([str(i), _src, _ref, _sys]) + os.linesep)

                ret += 1

            elif out_err is not None:

                out_err.write(
                    ','.join([str(i), _src, _ref, _sys]) + os.linesep)

    ret /= n_lines

    if out_crt is not None:

        out_crt.write('Score,%4f\n' % ret)

    if out_err is not None:

        out_err.write('Score,%4f\n' % ret)

    print('Binary accuracy on rule %s: %6f' % (rule_label, ret))

    if rule_strings is not None:
        print('Rule string: %s -> %s' % tuple(rule_strings))

    score_type = 'binary' if starts is not None else 'binary_full'

    if rule_label != '':
        __update_rule_file(score_type, model_name, ret, corpus_name, rule_label, rule_strings, n_lines)

    return ret


def eval_f(ref, sys, top_k=10000, alpha=0.5):

    target_language_dir = configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY

    tagger, _ = languages.load_default_languages(target_language_dir)

    ref = open(ref, "r")
    sys = open(sys, "r")

    ref_lines = ref.readlines()
    sys_lines = sys.readlines()

    assert(len(ref_lines) == len(sys_lines))

    n_lines = len(ref_lines)
    ret = 0

    for i in range(n_lines):

        text_ref = ref_lines[i].replace("\n", "").split(' ')
        text_sys = sys_lines[i].replace("\n", "").split(' ')

        indices_ref = list(tagger.parse_node(j, top_k) for j in text_ref)
        indices_sys = list(tagger.parse_node(j, top_k) for j in text_sys)

        indices_ref = list(k for k in indices_ref if k != tagger.unknown_index)
        indices_sys = list(k for k in indices_sys if k != tagger.unknown_index)

        # Error rate
        if alpha == -1:

            incorrect = 0.0

            n_ref = len(indices_ref)

            for j in range(n_ref):

                if j >= len(indices_sys):

                    incorrect += 1.0

                else:

                    if indices_ref[j] != indices_sys[j]:

                        incorrect += 1.0

            if n_ref == 0:

                val = None

            else:

                val = incorrect / n_ref

        else:

            val = f_measure(set(indices_ref), set(indices_sys), alpha=alpha)

        if val is None:

            n_lines -= 1

        else:

            ret += val

    ret /= n_lines

    return ret


def __update_rule_file(score_type, model_name, score, rule_dir, rule_label, rule_strings=None, n_lines=0):

    f = './comparison/%s/score_%s.csv' % (rule_dir, score_type)
    rule_label = str(rule_label)

    header = ['rule', 'rule_string', 'number_of_samples']

    data = list()
    found_rules = dict()

    if os.path.isfile(f):

        data_file = open(f, 'r')

        reader = csv.reader(data_file)

        idx = 0

        for line in reader:

            if idx == 0:

                if len(line) > 3:

                    header = line

                data.append(header)

            else:

                found_rules[line[0]] = idx
                data.append(line)

            idx += 1

        data_file.close()

    else:

        data.append(header)

    if model_name not in header:

        data[0].append(model_name)
        col_idx = len(data[0]) - 1

        for i in range(1, len(data)):

            data[i].append('')

    else:

        col_idx = header.index(model_name)

    n_cols = len(data[0])

    if rule_label in found_rules.keys():

        row_idx = found_rules[rule_label]

        if data[row_idx][col_idx] != '':
            print('WARNING OVERWRITE')

        data[row_idx][col_idx] = str(score)[:6]

        if data[row_idx][1] == '' and rule_strings is not None:
            data[row_idx][1] = ' '.join(rule_strings)

    else:

        text_rule = ' '.join(rule_strings) if rule_strings is not None else ''

        data.append([rule_label, text_rule, n_lines])

        for i in range(3, n_cols):

            data[-1].append('')

            if i == col_idx:

                data[-1][-1] = str(score)[:6]

    data_file = open(f, 'w+')
    writer = csv.writer(data_file)

    writer.writerows(data)

    data_file.close()


