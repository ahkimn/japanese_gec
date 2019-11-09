# TODO

import csv
import os
import MeCab

from .. import util
from .. import evaluate

CONST_PARSER = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

FAIRSEQ_PREFIX = 'fairseq'


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

                    model_output_file.write(
                        model_outputs[model][i].strip() + os.linesep)

        except:
            print('Exception for folder: %s' % rule_dir)
            continue


def _load_fairseq_dict(model_name):

    base_dir = os.path.join('model', 'fairseq', model_name, 'preprocessed')
    f_source = os.path.join(base_dir, 'dict.source.txt')
    f_target = os.path.join(base_dir, 'dict.target.txt')

    f_source = open(f_source, 'r').readlines()
    f_target = open(f_target, 'r').readlines()

    dict_source = set(i.strip().split()[0] for i in f_source)
    dict_target = set(i.strip().split()[0] for i in f_target)

    return (dict_source, dict_target)


def eval_binary_corpus(corpus_name,
                       comparison_dir="comparison",
                       data_prefix='test',
                       src_suffix='source',
                       tgt_suffix='target',
                       rule_suffix='rule',
                       start_suffix='start',
                       index_suffix='indices',
                       sys_suffix='out',
                       unique_predicates=True):

    base_dir = os.path.join(comparison_dir, corpus_name)

    rule_acc = dict()
    indices_rule = dict()
    correct_indices = dict()
    incorrect_indices = dict()

    assert(os.path.isdir(base_dir))

    for folder in os.listdir(base_dir)[:10]:

        try:

            rule_dir = os.path.join(base_dir, folder)
            if not os.path.isdir(rule_dir):
                continue

            src_file = data_prefix + '.' + src_suffix
            tgt_file = data_prefix + '.' + tgt_suffix
            rule_file = data_prefix + '.' + rule_suffix
            start_file = data_prefix + '.' + start_suffix
            indices_file = data_prefix + '.' + index_suffix

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
            indices_file = os.path.join(
                rule_dir, indices_file) if indices_file in files else None
            rule_idx = folder
            rule_acc[rule_idx] = dict()

            for f in files:

                if '.' + sys_suffix not in f:
                    continue

                model_name = f[:-(1 + len(sys_suffix))]

                if FAIRSEQ_PREFIX in model_name:

                    fairseq_name = model_name[len(FAIRSEQ_PREFIX) + 1:]
                    dicts = _load_fairseq_dict(fairseq_name)

                sys_file = os.path.join(rule_dir, f)

                ret = evaluate.eval_binary(
                    src=src_file, ref=tgt_file, sys=sys_file,
                    rule=rule_file, start=start_file, idx=indices_file,
                    rule_label=folder, corpus_name=corpus_name,
                    model_dicts=dicts)

                model_correct = ret['correct_indices']
                model_incorrect = ret['incorrect_indices']
                rule_acc[rule_idx][model_name] = \
                    [model_correct, model_incorrect]

                for x in model_correct:

                    indices_rule[x] = rule_idx

                for x in model_incorrect:

                    indices_rule[x] = rule_idx

                if model_name in correct_indices:

                    correct_indices[model_name].update(model_correct)
                    incorrect_indices[model_name].update(model_incorrect)

                else:

                    correct_indices[model_name] = set(model_correct)
                    incorrect_indices[model_name] = set(model_incorrect)

        except Exception:

            raise
            print('Exception for folder: %s' % rule_dir)
            continue

    if unique_predicates:

        corpus_source = os.path.join(base_dir, 'corpus.source')
        corpus_target = os.path.join(base_dir, 'corpus.target')

        source_lines = open(corpus_source, 'r').readlines()
        target_lines = open(corpus_target, 'r').readlines()

        idx_unique = dict()
        unique_idx = dict()

        rules_unique = set()
        nonrules_unique = set()

        idx_mapping = os.path.join(base_dir, 'mapping_indices.txt')

        mapping_file = open(idx_mapping, 'r')
        mapping_lines = mapping_file.readlines()
        mapping_file.close()

        for m in mapping_lines:

            m = list(int(i) for i in m.strip().split(','))

            idx_unique[m[0]] = m[1]
            if m[1] in unique_idx:
                unique_idx[m[1]].append(m[0])
            else:
                unique_idx[m[1]] = [m[0]]

        for rule in rule_acc.keys():

            try:

                _ = int(rule)

                for model in list(rule_acc[rule].keys())[:1]:
                    model_values = rule_acc[rule][model]
                    rules_unique.update(idx_unique[idx]
                                        for idx in model_values[0])
                    rules_unique.update(idx_unique[idx]
                                        for idx in model_values[1])

            except Exception:

                for model in list(rule_acc[rule].keys())[:1]:
                    model_values = rule_acc[rule][model]
                    nonrules_unique.update(
                        idx_unique[idx] for idx in model_values[0])
                    nonrules_unique.update(
                        idx_unique[idx] for idx in model_values[1])

        all_unique = rules_unique.union(nonrules_unique)
        out_of_rule = all_unique.difference(rules_unique)

        analysis_dir = os.path.join(base_dir, 'binary_acc')
        util.mkdir_p(analysis_dir)

        for model in correct_indices:

            print('MODEL: %s' % model)

            model_correct = correct_indices[model]
            model_incorrect = incorrect_indices[model]
            model_all_unique = set()
            model_in_rule = set()
            model_out_rule = set()

            rule_unique_seen = dict()
            rule_unique_correct = dict()

            n_correct_rule = 0
            n_correct_non_rule = 0

            n_incorrect_rule = 0
            n_incorrect_non_rule = 0

            model_output = os.path.join(base_dir, 'tmp', '%s.out' % model)
            model_output = open(model_output).readlines()

            in_rule_correct = \
                open(os.path.join(analysis_dir,
                                  '%s_rule_correct.csv' % model), 'w+')
            out_rule_correct = \
                open(os.path.join(analysis_dir,
                                  '%s_nonrule_correct.csv' % model), 'w+')

            in_rule_incorrect = \
                open(os.path.join(analysis_dir,
                                  '%s_rule_incorrect.csv' % model), 'w+')

            out_rule_incorrect = \
                open(os.path.join(analysis_dir,
                                  '%s_nonrule_incorrect.csv' % model), 'w+')

            for f in [in_rule_correct, out_rule_correct, in_rule_incorrect, out_rule_incorrect]:
                f.write(
                    ','.join(['source', 'target', 'system', 'rule']) + os.linesep)

            for ii in model_correct:

                idx = idx_unique[ii]

                model_all_unique.add(idx)

                if idx in rules_unique:

                    rule_index = indices_rule[ii]
                    if rule_index in rule_unique_seen:
                        rule_unique_seen[rule_index] += 1
                        rule_unique_correct[rule_index] += 1
                    else:
                        rule_unique_seen[rule_index] = 1
                        rule_unique_correct[rule_index] = 1

                    model_in_rule.add(idx)
                    in_rule_correct.write(
                        ','.join([source_lines[ii].strip(),
                                  target_lines[ii].strip(),
                                  model_output[ii].strip(),
                                  str(rule_index)]) + os.linesep)
                    n_correct_rule += 1

                elif idx in out_of_rule:

                    model_out_rule.add(idx)
                    n_correct_non_rule += 1
                    out_rule_correct.write(
                        ','.join([source_lines[ii].strip(),
                                  target_lines[ii].strip(),
                                  model_output[ii].strip(),
                                  str(indices_rule[ii])]) + os.linesep)

            for ii in model_incorrect:

                idx = idx_unique[ii]

                if idx in rules_unique:

                    rule_index = indices_rule[ii]

                    if rule_index in rule_unique_seen:
                        rule_unique_seen[rule_index] += 1

                    else:
                        rule_unique_seen[rule_index] = 0
                        rule_unique_correct[rule_index] = 0

                    n_incorrect_rule += 1

                    if idx not in model_all_unique:

                        in_rule_incorrect.write(
                            ','.join([source_lines[ii].strip(),
                                      target_lines[ii].strip(),
                                      model_output[ii].strip(),
                                      str(indices_rule[ii])]) + os.linesep)

                elif idx in out_of_rule:

                    n_incorrect_non_rule += 1

                    if idx not in model_all_unique:

                        out_rule_incorrect.write(
                            ','.join([source_lines[ii].strip(),
                                      target_lines[ii].strip(),
                                      model_output[ii].strip(),
                                      str(indices_rule[ii])]) + os.linesep)

            full_unique_acc = len(model_all_unique) / len(all_unique) \
                if len(all_unique) != 0 else 0

            rule_unique_acc = len(model_in_rule) / len(rules_unique) \
                if len(rules_unique) != 0 else 0

            nonrule_unique_acc = len(model_out_rule) / \
                len(out_of_rule) if len(out_of_rule) != 0 else 0

            total_rule = n_correct_rule + n_incorrect_rule
            total_nonrule = n_incorrect_rule + n_incorrect_non_rule

            rule_acc = n_correct_rule / total_rule \
                if total_rule != 0 else 0
            nonrule_acc = n_correct_non_rule / total_nonrule \
                if total_nonrule != 0 else 0

            evaluate.update_rule_file('unique', model, full_unique_acc,
                                      corpus_name, 'full', None,
                                      len(all_unique))
            evaluate.update_rule_file('unique', model, rule_unique_acc,
                                      corpus_name, 'in_rule', None,
                                      len(rules_unique))
            evaluate.update_rule_file('unique', model, nonrule_unique_acc,
                                      corpus_name, 'non_rule', None,
                                      len(out_of_rule))
            evaluate.update_rule_file('unique', model, rule_acc,
                                      corpus_name, 'raw_in_rule', None,
                                      total_rule)
            evaluate.update_rule_file('unique', model, nonrule_acc,
                                      corpus_name, 'raw_non_rule', None,
                                      total_nonrule)

            for iii in rule_unique_seen.keys():

                seen = rule_unique_seen[iii]
                correct = rule_unique_correct[iii]

                acc = correct / seen if seen != 0 else 0

                evaluate.update_rule_file('unique_rule', model, acc,
                                          corpus_name, 'unique_%s' % str(
                                              iii), None,
                                          seen)
