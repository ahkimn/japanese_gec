# -*- coding: utf-8 -*-

# Filename: databases.py
# Date Created: 21/12/2019
# Description: Database class and associated functions
# Python Version: 3.7

import numpy as np
import os
import time

from . import languages
from . import config
from . import parse
from . import util

cfg = config.parse()

D_PARAMS = cfg['data_params']
L_PARAMS = cfg['language_params']
P_PARAMS = cfg['parser_params']
DB_PARAMS = cfg['database_params']
DIRECTORIES = cfg['directories']


class Database:

    """Summary

    Attributes:
        fn_dict (dict): Description
        form_char_prefix (TYPE): Description
        form_len_prefix (TYPE): Description
        matrix_dict (dict): Description
        max_sentence_length (TYPE): Description
        max_token_length (TYPE): Description
        n_partitions (int): Description
        n_sentences (int): Description
        partition_dir (TYPE): Description
        partition_lengths (list): Description
        partition_sizes (list): Description
        sentence_len_prefix (TYPE): Description
        syntactic_tag_prefix (TYPE): Description
        token_char_prefix (TYPE): Description
        token_len_prefix (TYPE): Description
        token_prefix (TYPE): Description
    """

    def __init__(self,
                 form_char_prefix: str,
                 form_char_len_prefix: str,
                 max_sentence_length: int,
                 max_token_length: int,
                 sentence_len_prefix: str,
                 syntactic_tag_prefix: str,
                 token_char_prefix: str,
                 token_char_len_prefix: str,
                 token_prefix: str,
                 partition_dir: str):
        """Summary

        Args:
            form_char_prefix (str): Description
            form_char_len_prefix (str): Description
            max_sentence_length (int): Description
            max_token_length (int): Description
            sentence_len_prefix (str): Description
            syntactic_tag_prefix (str): Description
            token_char_prefix (str): Description
            token_char_len_prefix (str): Description
            token_prefix (str): Description
            partition_dir (str): Description
        """
        self.token_char_prefix = token_char_prefix
        self.form_char_prefix = form_char_prefix

        self.token_len_prefix = token_char_len_prefix
        self.form_len_prefix = form_char_len_prefix

        self.token_prefix = token_prefix
        self.syntactic_tag_prefix = syntactic_tag_prefix
        self.sentence_len_prefix = sentence_len_prefix

        self.max_sentence_length = max_sentence_length
        self.max_token_length = max_token_length

        self.partition_dir = partition_dir

        self.n_sentences = 0
        self.n_partitions = 0
        self.partition_lengths = []
        self.partition_sizes = []

        self.matrix_dict = {}
        self.fn_dict = {}
        self._check_partitions()

    def _character_matrix(self, size, pad: int):
        """Summary

        Args:
            size (TYPE): Description
            pad (int): Description

        Returns:
            TYPE: Description
        """
        return np.full((size,
                        self.max_sentence_length + 2,
                        self.max_token_length),
                       pad, dtype='uint32')

    def _len_matrix(self, size, pad: int):
        """Summary

        Args:
            size (TYPE): Description
            pad (int): Description

        Returns:
            TYPE: Description
        """
        return np.full((size,
                        self.max_sentence_length + 2),
                       pad, dtype='uint8')

    def _sentence_len_matrix(self, size):
        """Summary

        Args:
            size (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.zeros(size, dtype='uint8')

    def _tag_matrix(self, size, pad_indices: list):
        """Summary

        Args:
            size (TYPE): Description
            pad_indices (list): Description

        Returns:
            TYPE: Description
        """
        n_tags = len(P_PARAMS['parse_indices'])

        mat = np.zeros((size,
                        self.max_sentence_length + 2, n_tags),
                       dtype='uint32')

        for i in range(n_tags):
            mat[:, :, i] = pad_indices[i]

        return mat

    def _token_matrix(self, size, pad: int):
        """Summary

        Args:
            size (TYPE): Description
            pad (int): Description

        Returns:
            TYPE: Description
        """
        return np.full((size,
                        self.max_sentence_length + 2),
                       pad, dtype='uint32')

    def construct(self,
                  character_language: languages.Language,
                  token_language: languages.Language,
                  tag_languages: languages.Language,
                  source_corpus_dir: str,
                  source_corpus_filetype: str,
                  n_files: int=-1,
                  partition_size: int=50000):
        """
        Args:
            character_language (languages.Language): Description
            token_language (languages.Language): Description
            tag_languages (languages.Language): Description
            source_corpus_dir (str): Description
            source_corpus_filetype (str): Description
            n_files (int, optional): Description
            partition_size (int, optional): Description
        """
        file_list = util.get_files_recursive(
            source_corpus_dir,
            source_corpus_filetype)

        delimiter = P_PARAMS['delimiter']
        parser = parse.default_parser()

        n_files_processed = 0
        n_current = 0

        token_matrix = self._token_matrix(
            partition_size, pad=token_language.pad_index)
        tag_matrix = self._tag_matrix(
            partition_size,
            pad_indices=list(l.pad_index for l in tag_languages))
        sentence_len_matrix = self._sentence_len_matrix(partition_size)

        token_char_matrix = \
            self._character_matrix(
                partition_size, character_language.pad_index)
        form_char_matrix = self._character_matrix(
            partition_size, character_language.pad_index)

        token_len_matrix = self._len_matrix(partition_size, pad=0)
        form_len_matrix = self._len_matrix(partition_size, pad=0)

        partition_start = time.time()

        unique_sentences = set()

        print("\nStarting token tagging...")
        print(cfg['BREAK_LINE'])

        n_files = len(file_list) if n_files == -1 \
            else min(n_files, len(file_list))

        for file_name in file_list[:n_files]:

            f = open(file_name, 'r', encoding='utf-8')
            n_files_processed += 1

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i]

                # Skip previously seen sentences
                if sentence in unique_sentences:
                    continue

                tokens, tags = parse.parse_full(
                    sentence, parser, remove_delimiter=True,
                    delimiter=delimiter)

                token_indices = token_language.parse_nodes(tokens)
                n_tokens = len(token_indices)

                # Skip sentences exceeding maximum length
                if n_tokens > self.max_sentence_length:
                    continue

                unique_sentences.add(sentence)

                # Add SOS token and then copy token index values
                token_matrix[n_current, 0] = token_language.start_index
                token_matrix[n_current, 1:1 + n_tokens] = token_indices[:]
                token_matrix[n_current, 1 + n_tokens] = \
                    token_language.stop_index

                # Copy syntactic tag indices to tag matrix to each
                #   slice of tag_matrix
                for j in range(len(tag_languages)):

                    tag_indices = tag_languages[j].parse_nodes(tags[j])

                    tag_matrix[n_current, 0,
                               j] = tag_languages[j].start_index
                    tag_matrix[n_current, 1:1 + n_tokens, j] = tag_indices[:]
                    tag_matrix[n_current, 1 + n_tokens, j] = \
                        tag_languages[j].stop_index

                # Add length of current sentence
                sentence_len_matrix[n_current] = n_tokens

                forms = tags[-1]
                form_lengths = list(len(f) for f in forms)
                token_lengths = list(len(t) for t in tokens)

                form_len_matrix[n_current, 1:1 + n_tokens] = form_lengths
                token_len_matrix[n_current, 1:1 + n_tokens] = token_lengths

                for j in range(n_tokens):

                    form = forms[j]
                    token = tokens[j]

                    n_char_form = form_lengths[j]
                    n_char_token = token_lengths[j]

                    if n_char_form > self.max_token_length:

                        form_len_matrix[n_current, j + 1] = 0

                    else:

                        form_char_matrix[n_current, j + 1, :n_char_form] = \
                            character_language.parse_nodes(form)

                    if n_char_token > self.max_token_length:

                        token_len_matrix[n_current, j + 1] = 0

                    else:

                        token_char_matrix[n_current, j + 1, :n_char_token] = \
                            character_language.parse_nodes(token)

                self.n_sentences += 1
                n_current += 1

                # Save partition
                if n_current == partition_size:

                    self._save_partition(form_char_matrix,
                                         token_char_matrix,
                                         form_len_matrix,
                                         token_len_matrix,
                                         token_matrix,
                                         tag_matrix,
                                         sentence_len_matrix,
                                         self.n_partitions)
                    self.n_partitions += 1

                    partition_end = time.time()
                    print('\tTime elapsed: %4f' %
                          (partition_end - partition_start))
                    partition_start = partition_end

                    n_current = 0

                    token_char_matrix = \
                        self._character_matrix(
                            partition_size, character_language.pad_index)
                    form_char_matrix = self._character_matrix(
                        partition_size, character_language.pad_index)

                    token_len_matrix = self._len_matrix(partition_size, pad=0)
                    form_len_matrix = self._len_matrix(partition_size, pad=0)

                    token_matrix = self._token_matrix(
                        partition_size, pad=token_language.pad_index)
                    tag_matrix = self._tag_matrix(
                        partition_size,
                        pad_indices=list(l.pad_index for l in tag_languages))
                    sentence_len_matrix = self._sentence_len_matrix(
                        partition_size)

            f.close()

        # Remove padded non-sentences from arrays when saving
        #   on partition breaks
        if n_current:

            self._save_partition(form_char_matrix[:n_current],
                                 token_char_matrix[:n_current],
                                 form_len_matrix[:n_current],
                                 token_len_matrix[:n_current],
                                 token_matrix[:n_current],
                                 tag_matrix[:n_current],
                                 sentence_len_matrix[:n_current],
                                 self.n_partitions)
            self.n_partitions += 1

            partition_end = time.time()
            print('\tTime elapsed: %4f' % (partition_end - partition_start))
            partition_start = partition_end

    def _check_partitions(self):
        """Summary
        """
        n_partitions = 0
        n_sentences = 0
        partition_lengths = []

        found = True

        while found:

            self._add_partition_file_names(n_partitions)
            f_list = self.get_partition_files(n_partitions)

            if f_list and \
                    all(os.path.isfile(f_list[fn]) for fn in f_list.keys()):

                f_s_len = self.get_file(n_partitions, 'f_s_len')

                sentence_len_matrix = np.load(f_s_len)

                n_partition = len(sentence_len_matrix)
                n_sentences += n_partition
                partition_lengths.append(n_partition)

                size_partition = np.sum(sentence_len_matrix)
                self.partition_sizes.append(size_partition)

                n_partitions += 1

            else:

                found = False
                del self.fn_dict[n_partitions]

        self.n_partitions = n_partitions
        self.n_sentences = n_sentences
        self.partition_lengths = partition_lengths

    def _save_partition(self,
                        form_char_matrix: np.ndarray,
                        token_char_matrix: np.ndarray,
                        form_len_matrix: np.ndarray,
                        token_len_matrix: np.ndarray,
                        token_matrix: np.ndarray,
                        tag_matrix: np.ndarray,
                        sentence_len_matrix: np.ndarray,
                        n: int):
        """Summary

        Args:
            form_char_matrix (np.ndarray): Description
            token_char_matrix (np.ndarray): Description
            form_len_matrix (np.ndarray): Description
            token_len_matrix (np.ndarray): Description
            token_matrix (np.ndarray): Description
            tag_matrix (np.ndarray): Description
            sentence_len_matrix (np.ndarray): Description
            n (int): Description
        """
        if not os.path.isdir(self.partition_dir):
            util.mkdir_p(self.partition_dir)

        self._add_partition_file_names(n)

        print('Saving partition %d' % n)

        f_f_char = self.get_file(n, 'f_f_char')
        print('\tForm characters matrix file: %s' % f_f_char)
        np.save(f_f_char, form_char_matrix)

        f_t_char = self.get_file(n, 'f_t_char')
        print('\tToken characters matrix file: %s' % f_t_char)
        np.save(f_t_char, token_char_matrix)

        f_f_len = self.get_file(n, 'f_f_len')
        print('\tForm lengths matrix file: %s' % f_f_len)
        np.save(f_f_len, form_len_matrix)

        f_t_len = self.get_file(n, 'f_t_len')
        print('\tToken lengths matrix file: %s' % f_t_len)
        np.save(f_t_len, token_len_matrix)

        f_token = self.get_file(n, 'f_token')
        print('\tToken matrix file: %s' % f_token)
        np.save(f_token, token_matrix)

        f_tag = self.get_file(n, 'f_tag')
        print('\tTag matrix file: %s' % f_tag)
        np.save(f_tag, tag_matrix)

        f_s_len = self.get_file(n, 'f_s_len')
        print('\tSentence length matrix file: %s\n' % f_s_len)
        np.save(f_s_len, sentence_len_matrix)

    def _add_partition_file_names(self, n):
        """Summary

        Args:
            n (TYPE): Description
        """
        self.fn_dict[n] = {

            'f_f_char': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.form_char_prefix, n)),
            'f_t_char': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.token_char_prefix, n)),
            'f_f_len': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.form_len_prefix, n)),
            'f_t_len': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.token_len_prefix, n)),
            'f_token': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.token_prefix, n)),
            'f_tag': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.syntactic_tag_prefix, n)),
            'f_s_len': os.path.join(
                self.partition_dir, '%s%d.npy' %
                (self.sentence_len_prefix, n))
        }

    def get_file(self, n, fn):
        """Summary

        Args:
            n (TYPE): Description
            fn (TYPE): Description

        Returns:
            TYPE: Description
        """
        partition = self.fn_dict.get(n)

        return partition.get(fn) if partition else partition

    def get_partition_files(self, n):
        """Summary

        Args:
            n (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self.fn_dict.get(n)

    def iterate_partitions(self, fn_list):
        """Summary

        Args:
            fn_list (TYPE): Description

        Yields:
            TYPE: Description
        """
        for i in range(self.n_partitions):

            yield tuple(np.load(self.get_file(i, fn)) for fn in fn_list)

    def iterate_all_partitions(self):

        fn_list = ['f_f_char', 'f_t_char', 'f_f_len', 'f_t_len',
                   'f_token', 'f_tag', 'f_s_len']

        for i in range(self.n_partitions):

            yield tuple(np.load(self.get_file(i, fn)) for fn in fn_list)
