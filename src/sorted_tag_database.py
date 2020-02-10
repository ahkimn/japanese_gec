# -*- coding: utf-8 -*-

# Filename: sorted_tag_database.py
# Date Created: 01/22/2020
# Description: SortedTagDatabase class
# Python Version: 3.7

import numpy as np
import os

from . import config
from . import databases
from . import util

cfg = config.parse()


class SortedTagDatabase:

    """Summary

    Attributes:
        constructed (TYPE): Description
        file_name_dict (TYPE): Description
        matrix_dict (dict): Description
        matrix_dir (TYPE): Description
        size (int): Description
        sort_form_prefix (TYPE): Description
        sort_tag_prefix (TYPE): Description
        tmp_data_dict (dict): Description
        ordered_form_prefix (TYPE): Description
        unique_tag_prefix (TYPE): Description
        unique_token_prefix (TYPE): Description
    """

    def __init__(self, matrix_dir, unique_token_prefix: str,
                 unique_tag_prefix: str, ordered_form_prefix: str,
                 sort_tag_prefix: str, sort_form_prefix: str):
        """Summary

        Args:
            matrix_dir (TYPE): Description
            unique_token_prefix (str): Description
            unique_tag_prefix (str): Description
            ordered_form_prefix (str): Description
            sort_tag_prefix (str): Description
            sort_form_prefix (str): Description
        """
        self.matrix_dir = matrix_dir

        self.unique_token_prefix = unique_token_prefix
        self.unique_tag_prefix = unique_tag_prefix
        self.ordered_form_prefix = ordered_form_prefix

        self.sort_tag_prefix = sort_tag_prefix
        self.sort_form_prefix = sort_form_prefix

        self.matrix_dict = {}
        self.tmp_data_dict = {}
        self.file_name_dict = self._unique_file_names()

        self.size = 0
        self.constructed = self._check_arrays()

        if self.constructed:
            self.size = len(self.load_matrix('sort_form'))

    def _check_arrays(self):
        """Summary

        Returns:
            TYPE: Description
        """
        constructed = True

        for key, file_name in self.file_name_dict.items():

            if not os.path.isfile(file_name):

                constructed = False

        return constructed

    def construct(self, db: databases.Database):
        """Summary

        Args:
            db (databases.Database): Description
        """
        if not os.path.isdir(self.matrix_dir):
            util.mkdir_p(self.matrix_dir)

        unique_matrices = list()

        print('Finding unique token/tag combinations')

        n_processed = 0

        # Obtain unique token/tag combinations for each partition
        for token_matrix, tag_matrix, _ in db.iterate_partitions():

            print('Processing partition %d...' % n_processed)
            print('\tRaw size: %d tokens' % db.partition_sizes[n_processed])

            token_matrix = token_matrix.reshape(-1, 1)
            tag_matrix = tag_matrix.reshape(-1, tag_matrix.shape[2])

            combined_matrix = np.hstack([token_matrix, tag_matrix])
            unique_matrix = np.unique(combined_matrix, axis=0)

            unique_matrices.append(unique_matrix)

            print('\tFiltered size: %d tokens' % len(unique_matrix))

            del unique_matrix

            n_processed += 1

        # Perform unique operation across all partition-level unique matrices
        full_unique = np.unique(np.vstack(unique_matrices), axis=0)
        print('Final unique combination count: %d' % len(full_unique))

        del unique_matrices

        unique_tokens = full_unique[:, 0]
        ordered_tags = full_unique[:, 1:-1]
        ordered_form = full_unique[:, -1]

        self._save_matrix('unique_tokens', unique_tokens)
        self._save_matrix('ordered_tags', ordered_tags)
        self._save_matrix('ordered_form', ordered_form)

        sort_tags = ordered_tags.argsort(axis=0)
        sort_form = ordered_form.argsort()

        self._save_matrix('sort_tags', sort_tags)
        self._save_matrix('sort_form', sort_form)

    def _unique_file_names(self):
        """Summary

        Returns:
            TYPE: Description
        """
        f_unq_tokens = os.path.join(self.matrix_dir,
                                    '%s.npy' % self.unique_token_prefix)
        f_unq_tags = os.path.join(self.matrix_dir,
                                  '%s.npy' % self.unique_tag_prefix)
        f_unq_form = os.path.join(self.matrix_dir,
                                  '%s.npy' % self.ordered_form_prefix)
        f_sort_tags = os.path.join(self.matrix_dir,
                                   '%s.npy' % self.sort_tag_prefix)
        f_sort_form = os.path.join(self.matrix_dir,
                                   '%s.npy' % self.sort_form_prefix)

        file_name_dict = {

            'unique_tokens': f_unq_tokens,
            'ordered_tags': f_unq_tags,
            'ordered_form': f_unq_form,
            'sort_tags': f_sort_tags,
            'sort_form': f_sort_form
        }

        return file_name_dict

    def load_matrix(self, name: str):
        """Summary

        Args:
            name (str): Description

        Returns:
            TYPE: Description
        """
        assert(name in self.file_name_dict)

        if name in self.matrix_dict:
            return self.matrix_dict[name]

        else:
            matrix = np.load(self.file_name_dict[name])
            self.matrix_dict[name] = matrix
            return matrix

    def _save_matrix(self, name: str, matrix: np.ndarray,
                     store: bool=True):
        """Summary

        Args:
            name (str): Description
            matrix (np.ndarray): Description
            store (bool, optional): Description
        """
        assert(name in self.file_name_dict)

        f_save = self.file_name_dict[name]
        np.save(f_save, matrix)

        if store:
            self.matrix_dict[name] = matrix

        else:
            del matrix

    def get_search_arrays(self):
        """Summary

        Returns:
            TYPE: Description
        """
        if 'search_arrays' not in self.tmp_data_dict:

            search_arrays = self._construct_tag_search_arrays()
            search_arrays.append(self._construct_form_search_array())

            self.tmp_data_dict['search_arrays'] = search_arrays

        return self.tmp_data_dict['search_arrays']

    def _construct_form_search_array(self):
        """Summary

        Returns:
            TYPE: Description
        """
        ordered_form = self.load_matrix('ordered_form')
        sort_form = self.load_matrix('sort_form')

        n = len(ordered_form)

        # Sort unique forms such that their corresponding tokens are in order
        view = ordered_form[sort_form]

        # Form of highest index
        max_form = view[-1]

        # Array to store final indices (of token) where each form is found
        search_form = [-1]
        last_start = 0

        # Calculate last instance for each form in sorted array
        for k in range(max_form + 1):

            try:

                k_index = util.last(view, last_start,
                                    n, k, n)
                last_start = k_index

                # If the form is not extant, set last index as equivalent
                #   to previous form
                if k_index == - 1:
                    k_index = search_form[-1]

                search_form.append(k_index)

            # Once configx.CONST_MAX_SEARCH_TOKEN_INDEX are reached, exception
            #   is raised -> cancel loop
            except Exception:

                raise

        # search_form.append(n)

        return np.array(search_form)

    def _construct_tag_search_arrays(self):
        """Summary

        Returns:
            TYPE: Description
        """
        search_tags = []

        ordered_tags = self.load_matrix('ordered_tags')
        sort_tags = self.load_matrix('sort_tags')

        n = len(ordered_tags)

        # For each tag index
        # Determine the final location in which a specific tag of that index
        #   (when sorted by the index) appears
        for j in range(sort_tags.shape[1]):

            # Create a view of the part-of-speech matrix sorted by the index
            view = ordered_tags[sort_tags[:, j], j]
            search_tag = [-1]
            last_start = 0

            # Form of highest index
            max_form = view[-1]

            for k in range(max_form):

                try:

                    k_index = util.last(view, last_start,
                                        n, k, n)
                    last_start = k_index

                    if k_index == - 1:
                        k_index = search_tag[-1]

                    search_tag.append(k_index)

                # If the part-of-speech tag is not extant, set last
                #   index as equivalent to previous form
                except Exception:

                    break

            # search_tag.append(n)
            search_tags.append(np.array(search_tag))

        return search_tags

    def get_unique_tags(self):
        """Summary

        Returns:
            TYPE: Description
        """
        if 'unique_tags' not in self.tmp_data_dict:

            ordered_tags = self.load_matrix('ordered_tags')
            self.tmp_data_dict['unique_tags'] = np.unique(ordered_tags, axis=0)

        return self.tmp_data_dict['unique_tags']

    def get_form_to_token(self):
        """Summary

        Returns:
            TYPE: Description
        """
        if 'form_to_token' not in self.tmp_data_dict:

            ordered_tags = self.load_matrix('ordered_tags')
            ordered_form = self.load_matrix('ordered_form')
            unique_tokens = self.load_matrix('unique_tokens')

            n = len(ordered_tags)

            form_to_token = dict()

            for k in range(n):

                _form = ordered_form[k]
                _token = unique_tokens[k]
                _tags = tuple(ordered_tags[k])

                if _form in form_to_token:

                    if _tags in form_to_token[_form]:

                        form_to_token[_form][_tags].append(_token)

                    else:

                        form_to_token[_form][_tags] = [_token]

                else:

                    form_to_token[_form] = dict()
                    form_to_token[_form][_tags] = [_token]

            self.tmp_data_dict['form_to_token'] = form_to_token

        return self.tmp_data_dict['form_to_token']

    def find_tokens(self, requisite_tags: np.ndarray,
                    requisite_indices: np.ndarray,
                    n_max: int):
        """
        Function to obtain a list of possible substitute token indices given
            a set of requisite tags and the indices they correspond to

        Args:
            requisite_tags (np.ndarray): Array of part-of-speech values
            requisite_indices (np.ndarray): Array of part-of-speech indices
            n_max (int): Determines maximal token index that is outputted
            that require  exact matching
            corresponding to the requisite indices

        Returns:
            TYPE: Description
        """
        sort_tags = self.load_matrix('sort_tags')
        sort_form = self.load_matrix('sort_form')
        search_matrix = self.get_search_arrays()

        n_tags = sort_tags.shape[1]

        # Indices of token classes that may be used
        token_indices = None

        # Iterate over each part-of-speech tag that needs matching
        for i in range(len(requisite_indices)):

            # Tag value of original token at index
            requisite_tag = requisite_tags[i]

            idx = requisite_indices[i]

            # Maximum index to search
            max_search = len(search_matrix[idx])

            if (requisite_tag >= max_search):

                raise ("Illegal search number")

            elif (requisite_tag == max_search - 1):

                start_index = search_matrix[idx][requisite_tag] + 1
                end_index = len(sort_tags)

            else:

                start_index = search_matrix[idx][requisite_tag] + 1
                end_index = search_matrix[idx][requisite_tag + 1] + 1

            # From sorted array, determine all possible tag combinations
            #   with tag at given index matching the requisite value
            possible_indices = sort_form[start_index:end_index] \
                if idx == n_tags else sort_tags[:, idx][start_index:end_index]

            # If not first index matched, intersect along indices to
            #   determine which combinations match both syntactic tag indices
            if token_indices is not None:

                token_indices = np.intersect1d(
                    token_indices, possible_indices)

            else:

                token_indices = possible_indices

        # Restrict output to valid index values
        token_indices = token_indices[token_indices < n_max]

        return token_indices

    def get_possible_tags(self, matched_tokens: np.ndarray,
                          template_tags: np.ndarray,
                          lenient_indices: np.ndarray):
        """
        Function to determine unique syntactic tag combinations from possible
            substitute tokens

        Args:
            lenient_indices (np.ndarray): Array of syntactic tag indices that
                are irrelevant in matching potential substitute tokens
            template_tags (np.ndarray): Syntactic tags of template phrase
            matched_tokens (np.ndarray): Array containing the indices of tokens
                that have satisfied the requisite syntactic tags

        Returns:
            (np.ndarray): Array containing all possible substitute syntactic
                tag combinations
        """
        if len(lenient_indices) == 0:

            return template_tags[:-1].reshape(1, -1)

        else:

            ordered_tags = self.load_matrix('ordered_tags')
            matched_tags = ordered_tags[matched_tokens]
            return np.unique(matched_tags, axis=0)

    def find_tokens_from_form(self, form: int, match_tags: np.ndarray,
                              match_indices: np.ndarray):

        form_token = self.get_form_to_token()
        form_info = form_token.get(form, None)

        valid_tokens = set()

        if form_info is None:
            return None

        for tags in form_info.keys():

            valid = True

            for idx in match_indices:

                if tags[idx] != match_tags[idx]:

                    valid = False

            if valid:

                for token in form_info[tags]:

                    valid_tokens.add(token)

        if len(valid_tokens) > 0:

            # Take most frequent substitute token
            sub = min(valid_tokens)
            return sub

        return None
