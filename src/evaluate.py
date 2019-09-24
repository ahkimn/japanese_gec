from . import languages
from . import configx

from nltk import f_measure
import os


def eval_binary(ref, sys, out_crt=None, out_err=None, srt=None, rule=None,
                rule_label='', top_k=10000):

    target_language_dir = configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY

    tagger, _ = languages.load_default_languages(target_language_dir)

    ref = open(ref, "r")
    sys = open(sys, "r")

    if out_err is not None:

        out_err = open(out_err, "w+")
        out_err.write('インデクス,正しい文,モデル出力' + os.linesep)

    if out_crt is not None:

        out_crt = open(out_crt, "w+")
        out_crt.write('インデクス,正しい文,モデル出力' + os.linesep)

    rule_strings = None
    rule_lengths = None
    starts = list()

    if srt is not None:
        assert(rule is not None)
        srt = open(srt, "r")
        rule = open(rule, "r")

        srt_lines = srt.readlines()
        rule_lines = rule.readlines()

        rule_strings = ''.join(rule_lines[0].split()).split(',')
        rule_lengths = list(int(i) for i in rule_lines[1].strip().split(','))
        starts = list(int(i.strip()) for i in srt_lines)

    ref_lines = ref.readlines()
    sys_lines = sys.readlines()

    assert(len(ref_lines) == len(sys_lines))

    n_lines = len(ref_lines)
    ret = 0

    for i in range(n_lines):

        text_ref = ref_lines[i].strip().split(' ')
        text_sys = sys_lines[i].strip().split(' ')

        indices_ref = list(tagger.parse_node(j, top_k) for j in text_ref)
        indices_sys = list(tagger.parse_node(j, top_k) for j in text_sys)

        tokens_ref = list(tagger.parse_index(k) for k in indices_ref)
        tokens_sys = list(tagger.parse_index(k) for k in indices_sys)

        if len(indices_ref) == 0:

            n_lines -= 1

        else:

            if rule_lengths is not None:

                sys_start = starts[i] - 1
                sys_length = rule_lengths[0]

                _ref = ' '.join(tokens_ref[sys_start:sys_start + sys_length])
                _sys = ' '.join(tokens_sys[sys_start:sys_start + sys_length])

            else:

                _ref = ' '.join(tokens_ref)
                _sys = ' '.join(tokens_sys)

            if (_ref == _sys):

                if out_crt is not None:

                    out_crt.write(','.join([str(i), _ref, _sys]) + os.linesep)

                ret += 1

            elif out_err is not None:

                out_err.write(','.join([str(i), _ref, _sys]) + os.linesep)

    ret /= n_lines

    if out_crt is not None:

        out_crt.write('Score,%4f\n' % ret)

    if out_err is not None:

        out_err.write('Score,%4f\n' % ret)

    print('Binary accuracy on rule %s: %6f' % (rule_label, ret))
    print('Rule string: %s -> %s' % (rule_strings))
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
