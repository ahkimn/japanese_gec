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

    def __init__(self, db_token_prefix: str,
                 db_syntactic_tag_prefix: str, length_prefix: str,
                 partition_dir: str, max_sentence_length: int=50,
                 partition_size: int=50000):

        self.length_prefix = length_prefix
        self.token_prefix = db_token_prefix
        self.syntactic_tag_prefix = db_syntactic_tag_prefix

        self.partition_dir = partition_dir

        self.max_sentence_length = max_sentence_length
        self.partition_size = partition_size

        self.n_sentences = 0
        self.n_partitions = 0
        self.partition_lengths = []
        self.partition_sizes = []

        self.matrix_dict = {}

        self._check_partitions()

    def _token_matrix(self, pad: int):

        return np.full((self.partition_size,
                        self.max_sentence_length + 2),
                       pad, dtype='uint32')

    def _tag_matrix(self, pad_indices: list):

        n_tags = len(P_PARAMS['parse_indices'])

        mat = np.zeros((self.partition_size,
                        self.max_sentence_length + 2, n_tags),
                       dtype='uint32')

        for i in range(n_tags):
            mat[:, :, i] = pad_indices[i]

        return mat

    def _len_matrix(self):

        return np.zeros(self.partition_size, dtype='uint8')

    def construct(self, token_language: languages.Language,
                  tag_languages: languages.Language,
                  source_corpus_dir: str,
                  source_corpus_filetype: str,
                  n_files: int=-1):
        """

        Args:
            source_corpus_dir (str): Description
            source_corpus_filetype (str): Description
        """
        file_list = util.get_files_recursive(
            source_corpus_dir,
            source_corpus_filetype)

        delimiter = P_PARAMS['delimiter']
        parser = parse.default_parser()

        n_files_processed = 0
        n_current = 0

        len_matrix = self._len_matrix()
        token_matrix = self._token_matrix(token_language.pad_index)
        tag_matrix = self._tag_matrix(list(l.pad_index for l in tag_languages))

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

                else:
                    unique_sentences.add(sentence)

                tokens, tags = parse.parse_full(
                    sentence, parser, remove_delimiter=True,
                    delimiter=delimiter)

                token_indices = token_language.parse_nodes(tokens)
                n_tokens = len(token_indices)

                # Skip sentences exceeding maximum length
                if n_tokens > self.max_sentence_length:

                    continue

                len_matrix[n_current] = n_tokens

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

                self.n_sentences += 1
                n_current += 1

                # Save partition
                if n_current == self.partition_size:

                    self._save_partition(token_matrix, tag_matrix, len_matrix,
                                         self.n_partitions)
                    self.n_partitions += 1

                    partition_end = time.time()
                    print('\tTime elapsed: %4f' %
                          (partition_end - partition_start))
                    partition_start = partition_end

                    n_current = 0

                    len_matrix = self._len_matrix()
                    token_matrix = self._token_matrix(token_language.pad_index)
                    tag_matrix = self._tag_matrix(
                        list(l.pad_index for l in tag_languages))

            f.close()

        # Save excess sentences
        if n_current:

            # Remove excess array space
            len_matrix = len_matrix[:n_current]
            token_matrix = token_matrix[:n_current]
            tag_matrix = tag_matrix[:n_current]

            self._save_partition(token_matrix, tag_matrix, len_matrix,
                                 self.n_partitions)
            self.n_partitions += 1

            partition_end = time.time()
            print('\tTime elapsed: %4f' % (partition_end - partition_start))
            partition_start = partition_end

    def _check_partitions(self):

        n_partitions = 0
        n_sentences = 0
        partition_lengths = []

        found = True

        while found:

            f_token, f_tag, f_len = self._partition_file_names(n_partitions)
            found = os.path.isfile(f_token) and os.path.isfile(f_tag) \
                and os.path.isfile(f_len)

            if found:

                len_matrix = np.load(f_len)

                n_partition = len(len_matrix)
                n_sentences += n_partition
                partition_lengths.append(n_partition)

                size_partition = np.sum(len_matrix)
                self.partition_sizes.append(size_partition)

                n_partitions += 1

        self.n_partitions = n_partitions
        self.n_sentences = n_sentences
        self.partition_lengths = partition_lengths

    def _save_partition(self, token_matrix: np.ndarray,
                        tag_matrix: np.ndarray, len_matrix: np.ndarray,
                        n: int):

        if not os.path.isdir(self.partition_dir):
            util.mkdir_p(self.partition_dir)

        f_token, f_tag, f_len = self._partition_file_names(n)

        print('Saving partition %d' % n)
        print('\tToken matrix file: %s' % f_token)
        print('\tTag matrix file: %s' % f_tag)
        print('\tLength matrix file: %s\n' % f_len)

        np.save(f_token, token_matrix)
        np.save(f_tag, tag_matrix)
        np.save(f_len, len_matrix)

    def _partition_file_names(self, n):

        f_token = os.path.join(self.partition_dir, '%s%d.npy' %
                               (self.token_prefix, n))
        f_tag = os.path.join(self.partition_dir, '%s%d.npy' %
                             (self.syntactic_tag_prefix, n))
        f_len = os.path.join(self.partition_dir, '%s%d.npy' %
                             (self.length_prefix, n))

        return f_token, f_tag, f_len

    def iterate_partitions(self):

        for i in range(self.n_partitions):

            f_token, f_tag, f_len = self._partition_file_names(i)

            token_matrix = np.load(f_token)
            tag_matrix = np.load(f_tag)
            len_matrix = np.load(f_len)

            yield token_matrix, tag_matrix, len_matrix
