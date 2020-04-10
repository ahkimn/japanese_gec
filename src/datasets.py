# -*- coding: utf-8 -*-

# Filename: datasets.py
# Date Created: 29/03/2020
# Description: Dataset class and associated functions; stores
#   and manipualtes tokenized data saved on disk storage
# Python Version: 3.7

import os
import math
import numpy as np
import pandas as pd

from . import config
from . import generate
from . import sample
from . import util
from . import match

from . util import str_list
from . databases import Database
from . languages import Language
from . rules import RuleList
from . kana import KanaList
from . sorted_tag_database import SortedTagDatabase

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
        self.save_file = ''

    def _setup(self, force=False):

        if self.setup and not force:

            return

        self._get_rules()

        for rule in self.rules:

            self._get_subrules(rule)

        self.setup = True

    def _get_rules(self):

        self.rules = np.unique(self.df[DS_PARAMS['col_rules']].values)
        self.rule_type = type(self.rules[0])
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

        for i in range(self.n_rules):

            rule = self.rules[i]
            count = self.n_rule_sentences[i]
            n_subrules = self.n_subrules[rule]

            print('%s | %d | %d' % (rule, count, n_subrules))

    def print_subrule_stats(self, rule):

        self._setup()

        rule = self.rule_type(rule)
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

    def sample_rule_data(self, rule: str='', n_per_subrule: int=5,
                         RS: RandomState=None):

        self._setup()

        rule = self.rule_type(rule)

        if rule not in set(self.rules):
            raise ValueError('ERROR: rule %s not found in Dataset' % rule)

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

                correct = str_list(data[DS_PARAMS['col_correct']])
                error = str_list(data[DS_PARAMS['col_error']])

                correct_bounds = str_list(
                    data[DS_PARAMS['col_correct_bounds']])
                error_bounds = str_list(data[DS_PARAMS['col_error_bounds']])

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

            for i in range(self.n_sentences):

                data = self.df.iloc[i]

                correct = str_list(data[DS_PARAMS['col_correct']])
                error = str_list(data[DS_PARAMS['col_error']])

                correct = token_delimiter.join(correct)
                error = token_delimiter.join(error)

                correct_data = [correct]
                error_data = [error]

                file_correct.write(data_delimiter.join(
                    correct_data) + os.linesep)
                file_error.write(data_delimiter.join(
                    error_data) + os.linesep)

            return

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
                    directory, '%s%s.%s' % (rule_prefix, rule, c_suffix)),
                    'w+')
                file_error = open(os.path.join(
                    directory, '%s%s.%s' % (rule_prefix, rule, e_suffix)),
                    'w+')

            rule_folder = '%s_%s' % (rule_folder_prefix, rule)

            subrules = self.subrules[rule]
            n_subrules = self.n_subrules[rule]

            for j in range(n_subrules):

                subrule_data = rule_data.loc[
                    rule_data[DS_PARAMS['col_subrules']] == subrules[j]]

                subrule_count = subrule_data.shape[0]

                if subrule_count == 0:

                    continue

                if separation == 'subrule':

                    if not os.path.isdir(rule_folder):

                        util.mkdir_p(rule_folder)

                    file_correct = open(os.path.join(
                        directory, rule_folder,
                        '%s%d.%s' % (subrule_prefix,
                                     j + 1, c_suffix)), 'w+')

                    file_correct = open(os.path.join(
                        directory, rule_folder,
                        '%s%d.%s' % (subrule_prefix,
                                     j + 1, e_suffix)), 'w+')

                for k in range(subrule_count):

                    data = subrule_data.iloc[k]

                    correct = str_list(data[DS_PARAMS['col_correct']])
                    error = str_list(data[DS_PARAMS['col_error']])

                    correct = token_delimiter.join(correct)
                    error = token_delimiter.join(error)

                    correct_data = [correct]
                    error_data = [error]

                    file_correct.write(data_delimiter.join(
                        correct_data) + os.linesep)
                    file_error.write(data_delimiter.join(
                        error_data) + os.linesep)

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

    def save(self, filename: str=''):

        if filename == '':

            print(self.save_file)

            assert(self.save_file is not None)

            self.df.to_pickle(self.save_file)

        else:

            self.df.to_pickle(filename)

    @classmethod
    def load(cls, filename: str):

        df = pd.read_pickle(filename)

        DS = cls(df=df)
        DS.save_file = filename

        return DS

    def import_columns(self, data: list, column_name: str):

        assert(len(data) == self.n_sentences)
        kwargs = {column_name: data}
        self.df = self.df.assign(**kwargs)

    @classmethod
    def import_data(cls, error_sentences: list, correct_sentences: list,
                    error_bounds: list=None, correct_bounds: list=None,
                    rules: list=None, subrules: list=None):

        assert(error_sentences is not None)
        assert(correct_sentences is not None)
        n_sentences = len(error_sentences)

        assert(len(correct_sentences) == n_sentences)

        if error_bounds is not None:
            assert(len(error_bounds) == n_sentences)
        else:
            error_bounds = [''] * n_sentences

        if correct_bounds is not None:
            assert(len(correct_bounds) == n_sentences)
        else:
            correct_bounds = [''] * n_sentences

        if rules is not None:
            assert(len(rules) == n_sentences)
        else:
            rules = [''] * n_sentences

        if subrules is not None:
            assert(len(subrules) == n_sentences)
        else:
            subrules = [0] * n_sentences

        data = dict()

        data[DS_PARAMS['col_error']] = error_sentences
        data[DS_PARAMS['col_correct']] = correct_sentences
        data[DS_PARAMS['col_error_bounds']] = error_bounds
        data[DS_PARAMS['col_correct_bounds']] = correct_bounds
        data[DS_PARAMS['col_rules']] = rules
        data[DS_PARAMS['col_subrules']] = subrules

        return cls(data=data)

    def eval(self, column: str, full_sentence=False):

        self._setup()

        if column not in self.df.columns:
            raise ValueError('column %s not present' % column)

        in_rule_correct = []
        in_rule_incorrect = []
        out_of_rule_correct = []
        out_of_rule_incorrect = []

        rule_values = self.df[DS_PARAMS['col_rules']].values

        for rule in self.rules:

            indices = np.where(rule_values == rule)[0]

            for idx in indices:

                i_data = self.df.iloc[idx]

                model_correct = i_data[column]
                df_correct = i_data[DS_PARAMS['col_correct']]
                correct_bounds = i_data[DS_PARAMS['col_correct_bounds']]

                sentence_correct = (''.join(df_correct) ==
                                    ''.join(model_correct))

                # Use bounds if bounds are provided and user specifies to use
                #   phrase-level accuracy
                if not sentence_correct and not full_sentence and \
                        correct_bounds != []:

                    if len(model_correct) >= correct_bounds[1]:

                        df_correct_phrase = ''.join(
                            df_correct[correct_bounds[0]:correct_bounds[1]])
                        model_correct_phrase = ''.join(
                            model_correct[correct_bounds[0]:correct_bounds[1]])

                        sentence_correct = \
                            (df_correct_phrase == model_correct_phrase)

                if sentence_correct:

                    if rule != '':

                        in_rule_correct.append(idx)

                    else:
                        out_of_rule_correct.append(idx)

                else:

                    if rule != '':
                        in_rule_incorrect.append(idx)

                    else:
                        out_of_rule_incorrect.append(idx)

        UEM = UniqueErrorMapping(self)
        in_correct, in_total = UEM.resolve_unique_accuracy(
            in_rule_correct, in_rule_incorrect)
        out_correct, out_total = UEM.resolve_unique_accuracy(
            out_of_rule_correct, out_of_rule_incorrect)

        in_correct = in_correct.union(out_correct).intersection(in_total)
        out_total = out_total.difference(in_total)
        out_correct = out_correct.difference(
            in_correct).intersection(out_total)


        print('\n\tIn rule accuracy: %d / %d' % (len(in_correct), len(in_total)))
        print('\tOut of rule accuracy: %d / %d' % (len(out_correct), len(out_total)))

    def classify(self, character_language: Language, token_language: Language,
                 tag_languages: list, RL: RuleList, KL: KanaList,
                 STDB: SortedTagDatabase, tmp_db_dir: str,
                 check_rule: str='-1', clear: bool=False):

        self._setup()
        # unclassified = len(self.rules) == 1 and self.rules[0] == ''
        if clear:
            kwargs = {DS_PARAMS['col_rules']: '',
                      DS_PARAMS['col_subrules']: 0}
            self.df = self.df.assign(**kwargs)

        print('\nDetermining unique error-correct sentence pairs in Dataset')
        print(cfg['BREAK_LINE'])

        UEM = UniqueErrorMapping(self)

        print('\nCreating temporary Database instance for phrase searching')
        print(cfg['BREAK_LINE'])

        DB = self._make_tmp_db(character_language,
                               token_language, tag_languages, tmp_db_dir)

        print('\nGenerating \'ideal\' errors from rule list...')
        print(cfg['BREAK_LINE'])

        rule_match_indices = dict()
        rule_match_bounds = dict()
        rule_match_subrules = dict()

        for rule, idx in RL.iterate_rules(check_rule):

            print('\n\n')
            RL.print_rule(idx)
            print(cfg['BREAK_LINE'])

            matches = match.match_correct(
                rule, DB, STDB, n_max_out=-1, out_ratio=1.0,
                pre_merge_threshold=0)
            error_sentences, correct_sentences, error_bounds, \
                correct_bounds, rules, subrules, match_indices = \
                generate.generate_synthetic_pairs(
                    STDB, token_language, tag_languages, rule, matches,
                    KL=KL, ret_as_dataset=False)

            print('\tVerifying matched sentence pairs')
            print(cfg['BREAK_SUBLINE'])

            n_matches = len(error_sentences)

            valid_indices = []
            valid_bounds = []
            valid_subrules = []

            for i in range(n_matches):

                valid, match_error_bounds, match_correct_bounds = \
                    self._confirm_error_and_bounds(
                        match_indices[i],
                        error_sentences[i], correct_sentences[i],
                        error_bounds[i], correct_bounds[i])

                if not valid:
                    continue

                # Greedily update rules/subrules/bounds
                valid_indices.append(match_indices[i])
                valid_bounds.append([match_error_bounds, match_correct_bounds])
                valid_subrules.append(subrules[i])

            rule_match_indices[rule.name] = valid_indices
            rule_match_bounds[rule.name] = valid_bounds
            rule_match_subrules[rule.name] = valid_subrules

        inv_coverage = UEM.resolve_coverage(
            RL, rule_match_indices, rule_match_bounds)

        self._update_rules(inv_coverage, rule_match_indices, rule_match_bounds,
                           rule_match_subrules)

    def _confirm_error_and_bounds(self, idx, error_sentence, correct_sentence,
                                  error_bounds, correct_bounds,
                                  check_window=2):

        error_phrase = error_sentence[
            error_bounds[0]:error_bounds[1]]
        correct_phrase = correct_sentence[
            correct_bounds[0]:correct_bounds[1]]

        error_str = ''.join(error_phrase)
        correct_str = ''.join(correct_phrase)

        df_error = self.df[DS_PARAMS['col_error']].iloc[idx]
        if isinstance(df_error, str):
            df_error = df_error.split(' ')
        df_error_phrase = df_error[
            error_bounds[0]:error_bounds[1]]

        df_correct = self.df[DS_PARAMS['col_correct']].iloc[idx]
        if isinstance(df_correct, str):
            df_correct = df_correct.split(' ')
        df_correct_phrase = df_correct[
            correct_bounds[0]:correct_bounds[1]]

        df_error_str = ''.join(df_error)
        df_correct_str = ''.join(df_correct)

        # Both phrases should be present
        if correct_str not in df_correct_str or error_str not in df_error_str:

            return False, None, None

        # Error string must be present
        if error_str not in df_error_str:
            return False, None, None

        error_idx = df_error_str.index(error_str)
        correct_idx = df_correct_str.index(correct_str)

        error_end = error_idx + len(error_str)
        correct_end = correct_idx + len(correct_str)

        # Non errors should not be matched
        if error_str == correct_str:

            return False, None, None

        elif error_str in correct_str:

            sub_index = correct_str.index(error_str)
            if sub_index > 0:
                sc_error = max(error_idx - check_window, 0)
                sc_correct = max(correct_idx - check_window, 0)

                if df_correct_str[sc_correct:correct_idx] != \
                        df_error_str[sc_error:error_idx]:
                    return False, None, None
            else:
                ec_error = min(error_end + check_window, len(df_error_str))
                ec_correct = min(correct_end + check_window,
                                 len(df_correct_str))

                if df_correct_str[correct_end:ec_correct] != \
                        df_error_str[error_end:ec_error]:
                    return False, None, None

        right_offset = len(df_correct_str) - (correct_idx + len(correct_str))

        if df_error_phrase != error_phrase:

            error_idx = df_error_str.index(error_str)

            # Left boundaries must align
            if error_idx != correct_idx or \
                    len(df_error_str) - \
                    (error_idx + len(error_str)) != right_offset:
                return False, None, None

            error_bounds = _fix_bound_indices(df_error, error_str)
            df_error_phrase = df_error[
                error_bounds[0]:error_bounds[1]]

        if df_correct_phrase != correct_phrase:

            correct_bounds = _fix_bound_indices(df_correct, correct_str)
            df_correct_phrase = df_correct[
                correct_bounds[0]:correct_bounds[1]]

        print('\t\tIndex %d: %s -> %s' %
              (idx, ''.join(df_error_phrase), ''.join(df_correct_phrase)))

        return True, error_bounds, correct_bounds

    def _update_rules(self, indices_rule, rule_indices, rule_bounds,
                      rule_subrules):

        for rule in rule_indices.keys():

            indices = rule_indices[rule]
            bounds = rule_bounds[rule]
            subrules = rule_subrules[rule]

            n_indices = len(indices)

            for i in range(n_indices):

                idx = indices[i]

                r = indices_rule[idx]
                assert(r is not None)

                # If rule was not selected as coverer, continue
                if r != rule:
                    continue

                self.df.at[idx, DS_PARAMS['col_error_bounds']] = bounds[i][0]
                self.df.at[idx, DS_PARAMS['col_correct_bounds']] = bounds[i][1]
                self.df.at[idx, DS_PARAMS['col_rules']] = rule
                self.df.at[idx, DS_PARAMS['col_subrules']] = subrules[i]

    def _make_tmp_db(self, character_language: Language,
                     token_language: Language, tag_languages: list,
                     tmp_db_dir: str):

        DB = Database(tmp_db_dir)

        for sentences in self._iterate_sentences(DS_PARAMS['col_correct']):

            if isinstance(sentences.iloc[0], list):
                sentences = [''.join(sentence) for sentence in sentences]

            DB.add_sentences(sentences, character_language,
                             token_language, tag_languages,
                             force_save=True, allow_duplicates=True)

        return DB

    def _iterate_sentences(self, column: str, batch_size=1000):

        n_batch = math.ceil(self.n_sentences / batch_size)
        n_offset = 0

        col_data = self.df[column]

        for i in range(n_batch):

            n_end = min(self.n_sentences, n_offset + batch_size)

            data = col_data.iloc[n_offset:n_end]
            n_offset = n_end

            yield data


def _fix_bound_indices(sentence, phrase):

    sentence_str = ''.join(sentence)
    n_tokens = len(sentence)
    n_characters = len(sentence_str)

    # assert(phrase in sentence)
    phrase_start_idx = sentence_str.index(phrase)
    phrase_end_idx = phrase_start_idx + len(phrase)

    token_start_indices = []

    start_idx = 0
    for token in sentence:
        token_start_indices.append(start_idx)
        start_idx += len(token)
    token_start_indices.append(n_characters)

    token_start = -1 if phrase_start_idx != 0 else 0
    token_end = -1 if phrase_end_idx != n_characters else n_tokens

    for i in range(n_tokens):

        if token_start == -1 and phrase_start_idx < token_start_indices[i + 1]:
            token_start = i

        if token_end == -1 and phrase_end_idx <= token_start_indices[i + 1]:
            token_end = i + 1

    return [token_start, token_end]


def _display_coverage_supersets(match_indices: dict, RL: RuleList):
    """
    Determine if any rule of a RuleList contains another using set comparison
        on matched sentences

    Args:
        match_indices (dict): Dictionary, keyed by rule name, containing the
            indices of sentences matched by each rule
        RL (RuleList): RuleList instance containing rules that are the keys to
            %match_indices%
    """
    processed = set()

    for r_1, set_1 in match_indices.items():

        set_1 = set(set_1)

        s_1 = str(RL.get_rule(r_1))

        if len(set_1) == 0:

            print('\tWARNING: Rule %s (%s) is empty' % (r_1, s_1))

            continue

        processed.add(r_1)

        for r_2, set_2 in match_indices.items():

            set_2 = set(set_2)

            if len(set_2) == 0 or r_2 in processed:

                continue

            s_2 = str(RL.get_rule(r_2))

            if set_1 == set_2:

                print('\tERROR: Rule %s (%s) = %s (%s)' %
                      (r_1, s_1, r_2, s_2))
                continue

            diff = set_1.union(set_2)

            if diff == set_1:

                print('\tRule %s (%s) ⊆ %s (%s)' % (r_2, s_2, r_1, s_1))

            elif diff == set_2:

                print('\tRule %s (%s) ⊆ %s (%s)' % (r_1, s_1, r_2, s_2))


class UniqueErrorMapping:

    def __init__(self, DS: Dataset):

        self.n_sentences = DS.n_sentences
        self.n_unique_errors = 0

        error_sentences = DS.df[DS_PARAMS['col_error']]
        self.unique_errors = dict()
        self.unique_idx_unique_error = dict()
        self.unique_error_sentence_idx = dict()
        self.sentence_idx_unique_errors = dict()

        for i in range(self.n_sentences):

            error = error_sentences.iloc[i]
            if isinstance(error, list):
                error = ''.join(error)

            if error in self.unique_errors:
                unique_error_index = self.unique_errors[error]
                self.unique_error_sentence_idx[unique_error_index].add(i)
                self.sentence_idx_unique_errors[i] = unique_error_index

            else:
                self.unique_errors[error] = self.n_unique_errors
                self.unique_idx_unique_error[self.n_unique_errors] = error
                self.unique_error_sentence_idx[self.n_unique_errors] = set([i])
                self.sentence_idx_unique_errors[i] = self.n_unique_errors

                self.n_unique_errors += 1

        print('\tTotal number of error-correct pairs: %d' % self.n_sentences)
        print('\tNumber of unique error phrases: %d' % self.n_unique_errors)

    def _resolve_unique_coverage(self, sentence_indices: set):

        unique_covered = set()
        for idx in sentence_indices:
            unique_covered.add(self.sentence_idx_unique_errors[idx])
        return unique_covered

    def _display_multiple_coverage(self, unique_indices: dict):
        """
        Determine if any unique errors are covered by multiple rules

        Args:
            unique_indices (dict): Dictionary, keyed by rule name, that
                contains the sentence indices (of the source Dataset) covered
                by each rule
        """
        unique_error_rule = dict()

        for rule, rule_unique_indices in unique_indices.items():
            for i in rule_unique_indices:
                if i in unique_error_rule:
                    unique_error_rule[i].append(rule)
                else:
                    unique_error_rule[i] = [rule]

        for unique_idx, covering_rules in unique_error_rule.items():

            if len(covering_rules) > 1:

                print('\tError %d: %s' %
                      (unique_idx, self.unique_idx_unique_error[unique_idx]))
                print('\t\tMatched by rules: %s' %
                      ', '.join(str(x) for x in list(covering_rules)))

    def resolve_coverage(self, RL, match_indices, match_bounds,
                         display_multiple_coverage: bool=True,
                         display_supersets: bool=True):

        match_unique_indices = dict()

        valid_indices = set()
        inv_coverage = dict()
        for i in range(self.n_sentences):
            inv_coverage[i] = None

        for rule in match_indices.keys():
            covered = set(match_indices[rule])
            match_unique_indices[rule] = \
                self._resolve_unique_coverage(covered)
            valid_indices = valid_indices.union(covered)

            # For individual pairs with multiple covering rules,
            #   take last rule iterated over
            for idx in covered:
                inv_coverage[idx] = rule

        unique_indices = self._resolve_unique_coverage(valid_indices)
        n_valid = len(valid_indices)
        n_unique = len(unique_indices)

        if display_multiple_coverage:

            print('\nDisplaying unique errors covered by multiple rules...')
            print(cfg['BREAK_LINE'])
            self._display_multiple_coverage(match_unique_indices)

        if display_supersets:

            print('\nDisplaying empty and superset rules...')
            print(cfg['BREAK_LINE'])
            _display_coverage_supersets(match_indices, RL)

        print('\nPer rule coverage:')
        print(cfg['BREAK_LINE'])

        for rule in match_indices.keys():

            print('\tRule %s: %d total pairs; %d unique errors'
                  % (rule, len(match_indices[rule]),
                     len(match_unique_indices[rule])))

        print('\nOverall rule coverage:')
        print(cfg['BREAK_LINE'])

        print('\t# of sentence pairs covered by at least one rule: %d / %d' %
              (n_valid, self.n_sentences))
        print('\t# of unique errors covered by at least one rule: %d / %d' %
              (n_unique, self.n_unique_errors))

        return inv_coverage

    def resolve_unique_accuracy(self, rule_correct, rule_incorrect):

        unique_correct = set()
        unique_incorrect = set()

        for i in rule_correct:
            unique_correct.add(self.sentence_idx_unique_errors[i])

        for j in rule_incorrect:
            unique_incorrect.add(self.sentence_idx_unique_errors[j])

        unique_total = unique_correct.union(unique_incorrect)

        return unique_correct, unique_total
