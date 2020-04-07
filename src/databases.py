# -*- coding: utf-8 -*-

# Filename: databases.py
# Date Created: 21/12/2019
# Description: Database class and associated functions
# Python Version: 3.7

import numpy as np
import os
import time

from . import config
from . import parse
from . import util

from . languages import Language

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

    def __init__(
        self, partition_dir: str,
        form_char_prefix: str=DB_PARAMS['form_char_prefix'],
        form_char_len_prefix: str=DB_PARAMS['form_char_len_prefix'],
        max_sentence_length: int=DB_PARAMS['max_sentence_length'],
        max_token_length: int=DB_PARAMS['max_token_length'],
        sentence_len_prefix: str=DB_PARAMS['sentence_len_prefix'],
        syntactic_tag_prefix: str=DB_PARAMS['syntactic_tag_prefix'],
        token_char_prefix: str=DB_PARAMS['token_char_prefix'],
        token_char_len_prefix: str=DB_PARAMS['token_char_len_prefix'],
        token_prefix: str=DB_PARAMS['token_prefix'],
    ):
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

        self.m_dict = dict()
        self.has_partition_matrices = False

        self.fn_dict = dict()
        self._check_partitions()

    def _character_matrix(self, size: int, pad: int):
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

    def _len_matrix(self, size: int, pad: int):
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

    def _sentence_len_matrix(self, size: int):
        """Summary

        Args:
            size (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.zeros(size, dtype='uint8')

    def _tag_matrix(self, size: int, pad_indices: list):
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

    def _token_matrix(self, size: int, pad: int):
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

    def _get_matrix_dict(
            self, partition_size: int, character_language: Language,
            token_language: Language, tag_languages: list):

        self.m_dict['token'] = self._token_matrix(
            partition_size, pad=token_language.pad_index)
        self.m_dict['tag'] = self._tag_matrix(
            partition_size,
            pad_indices=list(l.pad_index for l in tag_languages))
        self.m_dict['sentence_len'] = self._sentence_len_matrix(
            partition_size)

        self.m_dict['token_char'] = \
            self._character_matrix(
                partition_size, character_language.pad_index)
        self.m_dict['form_char'] = self._character_matrix(
            partition_size, character_language.pad_index)

        self.m_dict['token_len'] = self._len_matrix(
            partition_size, pad=0)
        self.m_dict['form_len'] = self._len_matrix(
            partition_size, pad=0)

        self.has_partition_matrices = True

    def _reset_matrix_dict(self):

        self.m_dict.clear()
        self.has_partition_matrices = False

    def add_sentences(
            self, sentences: list, character_language: Language,
            token_language: Language, tag_languages: list,
            partition_size: int=50000, n_start: int=0,
            partition_start: float=None, unique_sentences: set=set(),
            force_save: bool=False, allow_duplicates: bool=False,
            remove_delimiter: bool=True):

        if partition_start is None:
            partition_start = time.time()

        delimiter = P_PARAMS['delimiter']
        parser = parse.default_parser()

        n_offset = n_start

        if not self.has_partition_matrices:
            self._get_matrix_dict(partition_size, character_language,
                                  token_language, tag_languages)

        for i in range(len(sentences)):

            sentence = sentences[i]

            # Skip previously seen sentences
            if not allow_duplicates and sentence in unique_sentences:
                continue

            tokens, tags = parse.parse_full(
                sentence, parser, remove_delimiter=remove_delimiter,
                delimiter=delimiter)

            token_indices = token_language.parse_nodes(tokens)
            n_tokens = len(token_indices)

            # Skip sentences exceeding maximum length
            if n_tokens > self.max_sentence_length:
                continue

            unique_sentences.add(sentence)

            # Add SOS token and then copy token index values
            self.m_dict['token'][n_offset, 0] = \
                token_language.start_index
            self.m_dict['token'][n_offset, 1:1 + n_tokens] = \
                token_indices[:]
            self.m_dict['token'][n_offset, 1 + n_tokens] = \
                token_language.stop_index

            # Copy syntactic tag indices to tag matrix to each
            #   slice of tag_matrix
            for j in range(len(tag_languages)):

                tag_indices = tag_languages[j].parse_nodes(tags[j])

                self.m_dict['tag'][n_offset, 0, j] = \
                    tag_languages[j].start_index
                self.m_dict['tag'][n_offset, 1:1 + n_tokens, j] = \
                    tag_indices[:]
                self.m_dict['tag'][n_offset, 1 + n_tokens, j] = \
                    tag_languages[j].stop_index

            # Add length of current sentence
            self.m_dict['sentence_len'][n_offset] = n_tokens

            forms = tags[-1]
            form_lengths = list(len(f) for f in forms)
            token_lengths = list(len(t) for t in tokens)

            self.m_dict['form_len'][n_offset, 1:1 + n_tokens] = \
                form_lengths
            self.m_dict['token_len'][n_offset, 1:1 + n_tokens] = \
                token_lengths

            for j in range(n_tokens):

                form = forms[j]
                token = tokens[j]

                n_char_form = form_lengths[j]
                n_char_token = token_lengths[j]

                if n_char_form > self.max_token_length:

                    self.m_dict['form_len'][n_offset, j + 1] = 0

                else:

                    self.m_dict['form_char'][n_offset, j + 1, :n_char_form] = \
                        character_language.parse_nodes(form)

                if n_char_token > self.max_token_length:

                    self.m_dict['token_len'][n_offset, j + 1] = 0

                else:

                    self.m_dict['token_char'][n_offset, j + 1, :n_char_token] = \
                        character_language.parse_nodes(token)

            self.n_sentences += 1
            n_offset += 1

            # Save partition
            if n_offset == partition_size:

                self._save_partition(self.n_partitions, last=partition_size)
                self._reset_matrix_dict()
                self.n_partitions += 1

                partition_end = time.time()
                print('\tTime elapsed: %4f' %
                      (partition_end - partition_start))
                partition_start = partition_end

                n_offset = 0

                self._get_matrix_dict(partition_size, character_language,
                                      token_language, tag_languages)

        # Remove padded non-sentences from arrays when saving
        #   on partition breaks
        if n_offset and force_save:

            self._save_partition(self.n_partitions, last=n_offset)
            self._reset_matrix_dict()
            self.n_partitions += 1

            partition_end = time.time()
            print('\tTime elapsed: %4f' % (partition_end - partition_start))
            partition_start = partition_end

        return n_offset, partition_start, unique_sentences

    def construct(self,
                  character_language: Language,
                  token_language: Language,
                  tag_languages: list,
                  source_corpus_dir: str,
                  source_corpus_filetype: str,
                  n_files: int=-1,
                  partition_size: int=50000):
        """
        Args:
            character_language (Language): Description
            token_language (Language): Description
            tag_languages (Language): Description
            source_corpus_dir (str): Description
            source_corpus_filetype (str): Description
            n_files (int, optional): Description
            partition_size (int, optional): Description
        """
        file_list = util.get_files_recursive(
            source_corpus_dir,
            source_corpus_filetype)

        n_files_processed = 0
        n_offset = 0

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

            force_save = (n_files_processed == n_files)

            n_offset, partition_start, unique_sentences = self.add_sentences(
                sentences, character_language, token_language, tag_languages,
                partition_size, n_offset, partition_start,
                unique_sentences=unique_sentences, force_save=force_save)

            f.close()

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

    def _save_partition(self, n: int, last: int):
        """Summary

        Args:
            n (int): Description
        """
        assert(self.has_partition_matrices)

        if not os.path.isdir(self.partition_dir):
            util.mkdir_p(self.partition_dir)

        self._add_partition_file_names(n)

        print('Saving partition %d' % n)

        f_f_char = self.get_file(n, 'f_f_char')
        print('\tForm characters matrix file: %s' % f_f_char)
        np.save(f_f_char, self.m_dict['form_char'][:last])

        f_t_char = self.get_file(n, 'f_t_char')
        print('\tToken characters matrix file: %s' % f_t_char)
        np.save(f_t_char, self.m_dict['token_char'][:last])

        f_f_len = self.get_file(n, 'f_f_len')
        print('\tForm lengths matrix file: %s' % f_f_len)
        np.save(f_f_len, self.m_dict['form_len'][:last])

        f_t_len = self.get_file(n, 'f_t_len')
        print('\tToken lengths matrix file: %s' % f_t_len)
        np.save(f_t_len, self.m_dict['token_len'][:last])

        f_token = self.get_file(n, 'f_token')
        print('\tToken matrix file: %s' % f_token)
        np.save(f_token, self.m_dict['token'][:last])

        f_tag = self.get_file(n, 'f_tag')
        print('\tTag matrix file: %s' % f_tag)
        np.save(f_tag, self.m_dict['tag'][:last])

        f_s_len = self.get_file(n, 'f_s_len')
        print('\tSentence length matrix file: %s\n' % f_s_len)
        np.save(f_s_len, self.m_dict['sentence_len'][:last])

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
