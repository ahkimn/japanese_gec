# TODO

import csv
import os
import MeCab

from .. import evaluate

CONST_PARSER = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")


def split_eval_data(corpus_name,
                    comparison_dir='comparison',
                    output_subdir='tmp',
                    data_prefix='test',
                    src_suffix='source',
                    tgt_suffix='target',
                    sys_suffix='out',
                    index_suffix='indices'
    ):

    base_dir = os.path.join(comparison_dir, corpus_name)
    assert(os.path.isdir(base_dir))

    model_dir = os.path.join(base_dir, output_subdir)
    assert(os.path.isdir(model_dir))

    model_outputs = dict()

    for f in os.listdir(model_dir):

        try:

            model_output = os.path.join(model_dir, f)

            if not os.path.isfile(model_output):
                continue

            lines = open(model_output, 'r').readlines()

            model_outputs[f] = lines

        except Exception:

            print('Exception for file: %s' % model_output)
            continue
    for folder in os.listdir(base_dir):

        try:

            rule_dir = os.path.join(base_dir, folder)
            if not os.path.isdir(rule_dir):
                continue

            src_file = data_prefix + '.' + src_suffix
            tgt_file = data_prefix + '.' + tgt_suffix
            indices_file = data_prefix + '.' + index_suffix
            files = os.listdir(rule_dir)

            assert(src_file in files)
            assert(tgt_file in files)
            assert(indices_file in files)

            src_file = os.path.join(rule_dir, src_file)
            tgt_file = os.path.join(rule_dir, tgt_file)
            indices_file = os.path.join(rule_dir, indices_file)

            indices = open(indices_file, 'r').readlines()

            indices = list(int(x.strip()) for x in indices)

            for model in model_outputs.keys():

                model_output_file = os.path.join(rule_dir, model)
                model_output_file = open(model_output_file, 'w+')

                for i in indices:

                    model_output_file.write(model_outputs[model][i].strip() + os.linesep)

        except:
            print('Exception for folder: %s' % rule_dir)
            continue




def eval_binary_corpus(corpus_name,
                       comparison_dir="comparison",
                       data_prefix='test',
                       src_suffix='source',
                       tgt_suffix='target',
                       rule_suffix='rule',
                       start_suffix='start',
                       sys_suffix='out'):

    base_dir = os.path.join(comparison_dir, corpus_name)

    assert(os.path.isdir(base_dir))

    for folder in os.listdir(base_dir):

        try:

            rule_dir = os.path.join(base_dir, folder)
            if not os.path.isdir(rule_dir):
                continue

            src_file = data_prefix + '.' + src_suffix
            tgt_file = data_prefix + '.' + tgt_suffix
            rule_file = data_prefix + '.' + rule_suffix
            start_file = data_prefix + '.' + start_suffix

            files = os.listdir(rule_dir)

            if src_file not in files:
                continue
            if tgt_file not in files:
                continue

            src_file = os.path.join(rule_dir, src_file)
            tgt_file = os.path.join(rule_dir, tgt_file)

            rule_file = os.path.join(
                rule_dir, rule_file) if rule_file in files else None
            start_file = os.path.join(
                rule_dir, start_file) if start_file in files else None

            for f in files:

                if '.' + sys_suffix not in f:
                    continue

                if 'fairseq_full' in f:
                    os.remove(os.path.join(rule_dir, f))
                    continue

                sys_file = os.path.join(rule_dir, f)

                evaluate.eval_binary(
                    src=src_file, ref=tgt_file, sys=sys_file,
                    rule=rule_file, start=start_file,
                    rule_label=folder, corpus_name=corpus_name)

        except Exception:

            raise
            print('Exception for folder: %s' % rule_dir)
            continue
