import csv
import os
import numpy as np
import pandas as pd

from . import config
from . import util

from enum import Enum
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

        columns = self.df.columns

        assert(DS_PARAMS['col_rules'] in columns)

        self.has_bounds = (DS_PARAMS['col_correct_bounds'] in columns and
                           DS_PARAMS['col_error_bounds'] in columns)
        self.has_subrules = DS_PARAMS['col_subrules'] in columns

    def sample_rule_data(self, rule_name: str, n_per_subrule: int=5,
                         RS: RandomState=None):

        assert(self.has_subrules and self.has_bounds)

        loc = self.df.loc[self.df[DS_PARAMS['col_rules']] == rule_name]

        subrules = loc[DS_PARAMS['col_subrules']].values
        n_subrules = np.max(subrules) + 1

        for i in range(n_subrules):

            subrule_indices = np.where(subrules == i)[0]

            print('\n\tSample sentences for sub-rule %d of %d\n'
                  % (i + 1, n_subrules))

            n_subrule = len(subrule_indices)
            perm = np.arange(n_subrule) if RS is None \
                else RS.permutation(n_subrule)

            for j in subrule_indices[perm[:n_per_subrule]]:

                data = self.df.iloc[j]

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

    def write(self, directory: str, token_delimiter: str,
              data_delimiter: str, include_tags: list,
              separation: str, max_per_rule=-1):

        if not os.path.isdir(directory):
            util.mkdir_p(directory)

            print("\t\tWriting to directory: %s" % directory)

        rules = np.unique(self.df[DS_PARAMS['col_rules']].values)
        n_rules = len(rules)

        f_prefix = SAVE_PARAMS['default_prefix']
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
                directory, '%s.%s' % (f_prefix, c_suffix)), 'w+')
            file_error = open(os.path.join(
                directory, '%s.%s' % (f_prefix, e_suffix)), 'w+')

        elif separation == 'rule':

            pass

        elif separation == 'subrule':

            pass

        else:

            raise ValueError(
                'Separation type must be one of \'%s\' \'%s\', or \'%s\'' %
                ('rule', 'subrule', 'none'))

        for i in range(n_rules):

            rule = rules[i]
            rule_data = self.df.loc[self.df[DS_PARAMS['col_rules']] == rule]

            if max_per_rule != -1:

                n_rule = rule_data.shape[0]
                perm = np.random.permutation(n_rule)

                n_perm = min(n_rule, max_per_rule)
                rule_data = rule_data.iloc[perm[:n_perm]]

            if rule_data.shape[0] == 0:

                continue

            if separation == 'rule':

                file_correct = open(os.path.join(
                    directory, '%s%s.%s' % (rule_prefix, rule, c_suffix)),
                    'w+')
                file_error = open(os.path.join(
                    directory, '%s%s.%s' % (rule_prefix, rule, e_suffix)),
                    'w+')

            subrules = rule_data[DS_PARAMS['col_subrules']].values
            n_subrules = np.max(subrules) + 1

            rule_folder = '%s%s' % (rule_folder_prefix, rule)

            for j in range(n_subrules):

                subrule_data = rule_data.loc[
                    rule_data[DS_PARAMS['col_subrules']] == j]

                if subrule_data.shape[0] == 0:

                    continue

                if separation == 'subrule':

                    file_correct = open(os.path.join(
                        directory, rule_folder,
                        '%s%d.%s' % (subrule_prefix,
                                     j + 1, c_suffix)), 'w+')

                    file_correct = open(os.path.join(
                        directory, rule_folder,
                        '%s%d.%s' % (subrule_prefix,
                                     j + 1, e_suffix)), 'w+')

                n_sentences = subrule_data.shape[0]

                for k in range(n_sentences):

                    data = subrule_data.iloc[k]

                    correct = token_delimiter.join(
                        data[DS_PARAMS['col_correct']])
                    error = token_delimiter.join(
                        data[DS_PARAMS['col_error']])

                    correct_data = [correct]
                    error_data = [error]

                    file_correct.write(data_delimiter.join(
                        correct_data) + os.linesep)
                    file_error.write(data_delimiter.join(
                        error_data) + os.linesep)

    @classmethod
    def merge(cls, ds_list: list):

        df_list = [ds.df for ds in ds_list]

        return cls(df=pd.concat(df_list))

    @classmethod
    def merge_directory(cls, directory: str, file_prefix: str,
                        file_suffix: str):

        file_list = util.get_files(directory, file_suffix)

        if file_prefix != '':

            file_list = [f for f in file_list
                         if file_prefix in os.path.basename(f).split('.')[-2]]

        if file_list:

            return cls.merge([cls.load(ds) for ds in file_list])

        else:

            raise

    def save(self, filename):

        self.df.to_pickle(filename)

    @classmethod
    def load(cls, filename):

        df = pd.read_pickle(filename)

        return cls(df=df)
