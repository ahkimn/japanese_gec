# -*- coding: utf-8 -*-

# Filename: datasets.py
# Date Created: 29/03/2020
# Description: Dataset class and associated functions; stores
#   and manipualtes tokenized data saved on disk storage
# Python Version: 3.7

import os
import numpy as np
import pandas as pd

from . import config
from . import sample
from . import util

from ast import literal_eval
from numpy.random import RandomState
from termcolor import colored

cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
SAVE_PARAMS = DS_PARAMS['save_names']


class Dataset:

    def __init__(self, data: dict=None, df: pd.DataFrame=None):

        assert(data is not None or df is not None)

        if data is not None:

            self.df = pd.DataFrame(data=data)

        else:

            self.df = df

        self.n_sentences = self.df.shape[0]

        columns = self.df.columns

        assert(DS_PARAMS['col_rules'] in columns)

        self.has_bounds = (DS_PARAMS['col_correct_bounds'] in columns and
                           DS_PARAMS['col_error_bounds'] in columns)
        self.has_subrules = DS_PARAMS['col_subrules'] in columns

        assert(self.has_bounds and self.has_subrules)

        self.setup = False
        self.save_file = None

    def _setup(self, force=False):

        if self.setup and not force:

            return

        self._get_rules()

        for rule in self.rules:

            self._get_subrules(rule)

        self.setup = True

    def _get_rules(self):

        self.rules = np.unique(self.df[DS_PARAMS['col_rules']].values)
        self.n_rules = len(self.rules)

        self.n_rule_sentences = []
        self.rule_idx = dict()

        for i in range(self.n_rules):

            self.rule_idx[self.rules[i]] = i

            self.n_rule_sentences.append(
                self.df.loc[self.df[DS_PARAMS['col_rules']] ==
                            self.rules[i]].shape[0])

    def _get_subrules(self, rule):

        if not hasattr(self, 'subrules'):
            self.subrules = dict()
            self.n_subrules = dict()
            self.n_subrule_sentences = dict()

        rule_data = self.df.loc[self.df[DS_PARAMS['col_rules']] == rule]

        subrules = np.unique(
            rule_data[DS_PARAMS['col_subrules']].values)
        n_subrules = len(subrules)

        n_subrule_sentences = []

        for j in range(n_subrules):

            n_subrule_sentences.append(
                rule_data.loc[rule_data[DS_PARAMS['col_subrules']] ==
                              subrules[j]].shape[0])

        self.subrules[rule] = subrules
        self.n_subrules[rule] = n_subrules
        self.n_subrule_sentences[rule] = n_subrule_sentences

    def print_rule_stats(self):

        self._setup()

        print('Displaying rule statistics')
        print(cfg['BREAK_LINE'])
        print('Name | Sentence Count | Subrule Count')

        for i in self.n_rules:

            rule = self.rules[i]
            count = self.n_rule_sentences[i]
            n_subrules = self.n_subrules[rule]

            print('%s | %d | %d' % (rule, count, n_subrules))

    def print_subrule_stats(self, rule):

        self._setup()

        assert(rule in self.rule_idx)

        print('Displaying subrule statistics for rule %s' % rule)
        print(cfg['BREAK_LINE'])
        print('Subrule ID | Sentence Count')

        n_subrules = self.n_subrules[rule]
        subrules = self.subrules[rule]
        subrule_counts = self.n_subrule_sentences[rule]

        for j in range(n_subrules):

            subrule = subrules[j]
            count = subrule_counts[j]

            print('%d | %d' % (subrule, count))

    def sample_rule_data(self, rule: str, n_per_subrule: int=5,
                         RS: RandomState=None):

        self._setup()

        rule_data = self.df.loc[self.df[DS_PARAMS['col_rules']] == rule]

        subrules = self.subrules[rule]
        n_subrules = self.n_subrules[rule]

        for i in range(n_subrules):

            subrule_data = \
                rule_data.loc[rule_data[DS_PARAMS['col_subrules']] ==
                              subrules[i]]

            print('\n\tSample sentences for sub-rule %d of %d\n'
                  % (i + 1, n_subrules))

            n_subrule = subrule_data.shape[0]
            perm = np.arange(n_subrule) if RS is None \
                else RS.permutation(n_subrule)
            perm = perm[:n_per_subrule]

            for j in perm:

                data = subrule_data.iloc[j]

                correct = data[DS_PARAMS['col_correct']]
                error = data[DS_PARAMS['col_error']]

                correct_bounds = data[DS_PARAMS['col_correct_bounds']]
                error_bounds = data[DS_PARAMS['col_error_bounds']]

                highlighted_error = ''.join(error[:error_bounds[0]]) \
                    + colored(''.join(error[
                        error_bounds[0]:error_bounds[1]]), 'red') \
                    + ''.join(error[error_bounds[1]:])

                highlighted_correct = ''.join(correct[:correct_bounds[0]]) \
                    + colored(''.join(correct[
                        correct_bounds[0]:correct_bounds[1]]), 'green') \
                    + ''.join(correct[correct_bounds[1]:])

                print('\tE: %s\n\tC: %s' %
                      (highlighted_error, highlighted_correct))

    def split(self, train_ratio=0.9, dev_ratio=0.05,
              max_per_rule=50000, min_per_rule=200, RS: RandomState=None):

        self._setup()

        if RS is None:

            RS = RandomState(seed=0)

        # Sanity check on data ratios
        assert(train_ratio + dev_ratio < 1)

        print('Total sentence pairs in dataset: %d' % self.n_sentences)
        print('Total number of rules: %d' % self.n_rules)

        print('Determining number of pairs to sample from each rule')
        print(cfg['BREAK_LINE'])

        rule_sample_counts = sample.balanced_rule_sample(
            self.rules, self.n_rule_sentences, max_per_rule,
            min_per_rule, sample_function=sample.linear_sampler)

        train_indices, dev_indices, test_indices = [], [], []

        for i in range(self.n_rules):

            rule = self.rules[i]
            subrules = self.subrules[rule]
            n_subrules = self.n_subrules[rule]
            subrule_counts = self.n_subrule_sentences[rule][:]

            n_sample_rule = rule_sample_counts[i]

            print('Rule %s: %d pairs' % (rule, n_sample_rule))

            assert(n_sample_rule >= 0)

            rule_indices = np.where(
                (self.df[DS_PARAMS['col_rules']] == rule))[0]
            rule_data = self.df.iloc[rule_indices]

            subrule_sample_counts = \
                sample.balanced_subrule_sample(subrule_counts, n_sample_rule)

            for j in range(n_subrules):

                subrule = subrules[j]
                n_sample_subrule = subrule_sample_counts[j]

                print('\tSubrule %d: %d pairs' % (subrule, n_sample_subrule))

                if n_sample_subrule == 0:
                    continue

                n_sr_train, n_sr_dev, n_sr_test = sample.split_subrule_count(
                    n_sample_subrule, train_ratio, dev_ratio)

                print('\t  Split: %d, %d, %d' %
                      (n_sr_train, n_sr_dev, n_sr_test))

                subrule_indices = np.where(
                    (rule_data[DS_PARAMS['col_subrules']] == subrule))[0]
                subrule_indices = rule_indices[subrule_indices]

                assert(subrule_counts[j] +
                       n_sample_subrule == len(subrule_indices))

                perm = RS.permutation(subrule_counts[j])

                sr_train = subrule_indices[perm[:n_sr_train]]
                sr_dev = subrule_indices[
                    perm[n_sr_train:n_sr_train + n_sr_dev]]
                sr_test = subrule_indices[
                    perm[n_sr_train + n_sr_dev:
                         n_sr_train + n_sr_dev + n_sr_test]]

                train_indices.append(sr_train)
                dev_indices.append(sr_dev)
                test_indices.append(sr_test)

        train_indices = np.hstack(train_indices)
        dev_indices = np.hstack(dev_indices)
        test_indices = np.hstack(test_indices)

        n_train = len(train_indices)
        n_dev = len(dev_indices)
        n_test = len(test_indices)

        print('Final number of training pairs: %d' % n_train)
        print('Final number of development pairs: %d' % n_dev)
        print('Final number of test pairs: %d' % n_test)

        df_train = self.df.iloc[train_indices]
        df_dev = self.df.iloc[dev_indices]
        df_test = self.df.iloc[test_indices]

        return Dataset(df=df_train), Dataset(df=df_dev), Dataset(df=df_test)

    def write(self, directory: str, token_delimiter: str,
              data_delimiter: str, include_tags: list,
              separation: str, max_per_rule=-1,
              save_prefix: str=''):

        # converted = self._convert_columns()
        self._setup()

        # if converted:
        #     self.save_default()

        if not os.path.isdir(directory):
            util.mkdir_p(directory)

        print("\t\tWriting to directory: %s" % directory)

        if save_prefix == '':
            save_prefix = SAVE_PARAMS['default_prefix']
        c_suffix = SAVE_PARAMS['correct_suffix']
        e_suffix = SAVE_PARAMS['error_suffix']

        rule_prefix = SAVE_PARAMS['rule_file_prefix']
        if rule_prefix != '':
            rule_prefix += '_'

        rule_folder_prefix = SAVE_PARAMS['rule_folder_prefix']
        if rule_folder_prefix != '':
            rule_folder_prefix += '_'

        subrule_prefix = SAVE_PARAMS['subrule_file_prefix']
        if subrule_prefix != '':
            subrule_prefix += '_'

        separation = separation.lower()

        if separation == 'none':

            file_correct = open(os.path.join(
                directory, '%s.%s' % (save_prefix, c_suffix)), 'w+')
            file_error = open(os.path.join(
                directory, '%s.%s' % (save_prefix, e_suffix)), 'w+')

        elif separation == 'rule':

            pass

        elif separation == 'subrule':

            pass

        else:

            raise ValueError(
                'Separation type must be one of \'%s\' \'%s\', or \'%s\'' %
                ('rule', 'subrule', 'none'))

        for i in range(self.n_rules):

            rule = self.rules[i]
            rule_data = self.df.loc[self.df[DS_PARAMS['col_rules']] == rule]

            n_rule = self.n_rule_sentences[i]

            if max_per_rule != -1:

                perm = np.random.permutation(n_rule)
                n_perm = min(n_rule, max_per_rule)
                rule_data = rule_data.iloc[perm[:n_perm]]

            if separation == 'rule':

                file_correct = open(os.path.join(
                    directory, '%s_%s.%s' % (rule_prefix, rule, c_suffix)),
                    'w+')
                file_error = open(os.path.join(
                    directory, '%s_%s.%s' % (rule_prefix, rule, e_suffix)),
                    'w+')

            rule_folder = '%s_%s' % (rule_folder_prefix, rule)

            subrules = self.subrules[rule]
            n_subrules = self.n_subrules[rule]
            subrule_counts = self.n_subrule_sentences[rule]

            for j in range(n_subrules):

                subrule_data = rule_data.loc[
                    rule_data[DS_PARAMS['col_subrules']] == subrules[j]]

                subrule_count = subrule_counts[j]

                if subrule_count == 0:

                    continue

                if separation == 'subrule':

                    if not os.path.isdir(rule_folder):

                        util.mkdir_p(rule_folder)

                    file_correct = open(os.path.join(
                        directory, rule_folder,
                        '%s_%d.%s' % (subrule_prefix,
                                      j + 1, c_suffix)), 'w+')

                    file_correct = open(os.path.join(
                        directory, rule_folder,
                        '%s_%d.%s' % (subrule_prefix,
                                      j + 1, e_suffix)), 'w+')

                for k in range(subrule_count):

                    data = subrule_data.iloc[k]

                    correct = data[DS_PARAMS['col_correct']]
                    error = data[DS_PARAMS['col_error']]

                    if isinstance(correct, str):
                        correct = literal_eval(correct)
                    if isinstance(error, str):
                        error = literal_eval(error)

                    correct = token_delimiter.join(correct)
                    error = token_delimiter.join(error)

                    correct_data = [correct]
                    error_data = [error]

                    file_correct.write(data_delimiter.join(
                        correct_data) + os.linesep)
                    file_error.write(data_delimiter.join(
                        error_data) + os.linesep)

    # def _convert_columns(self):

    #     converted = False

    #     print('Converting columns to correct datatypes...')

    #     first = self.df.iloc[0]

    #     for col_name in ['col_correct', 'col_error',
    #                      'col_correct_bounds', 'col_error_bounds']:

    #         col = DS_PARAMS[col_name]

    #         if not isinstance(first[col], list):

    #             print('\tConverting column: %s' % col)
    #             self.df[col] = self.df[col].apply(literal_eval)
    #             converted = True

    #     col_rules = DS_PARAMS['col_rules']

    #     if not isinstance(first[col_rules], str):

    #         print('\tConverting column: %s' % col_rules)
    #         self.df[col_rules] = self.df[col_rules].apply(str)
    #         converted = True

    #     col_subrules = DS_PARAMS['col_subrules']

    #     if not isinstance(first[col_subrules], np.int64):

    #         print('\tConverting column: %s' % col_subrules)
    #         self.df[col_subrules] = self.df[col_subrules].  apply(np.int64)
    #         converted = True

    #     return converted

    @classmethod
    def merge(cls, file_list: list, tmp_file='df_all.csv'):

        mode = 'w'
        header = True

        count = 1
        n_merge = len(file_list)

        for f in file_list:

            print('\tMerging instance %d of %d' % (count, n_merge))
            ds = cls.load(f)

            ds.df.to_csv(tmp_file, mode=mode, header=header, index=None)
            mode = 'a'
            header = False
            count += 1

            del ds

        df = pd.read_csv(tmp_file, index_col=None)
        os.remove(tmp_file)

        return cls(df=df)

    @classmethod
    def merge_directory(cls, directory: str, file_filter: str,
                        file_suffix: str):

        # Get files by extension
        file_list = util.get_files(directory, file_suffix)

        if file_filter != '':

            file_list = [f for f in file_list
                         if file_filter in os.path.basename(f).split('.')[-2]]

        print('Found %d files with extension \'.%s\' and \'%s\' in name' %
              (len(file_list), file_suffix, file_filter))

        print('\nMerging Datasets...')
        print(cfg['BREAK_LINE'])

        if file_list:

            return cls.merge(file_list)

        else:

            raise

    def save_default(self):

        assert(self.save_file is not None)

        self.save(self.save_file)

    def save(self, filename):

        self.df.to_pickle(filename)

    @classmethod
    def load(cls, filename):

        df = pd.read_pickle(filename)

        DS = cls(df=df)
        DS.save_file = filename

        return DS
