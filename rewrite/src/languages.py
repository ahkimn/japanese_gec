# -*- coding: utf-8 -*-

# Filename: languages.py

# Author: Alex Kimn
# Date Created: 19/12/2019
# Description: File containing Language class and related functions
# Python Version: 3.7

import os
import time
import pickle

import numpy as np

from . import config
from . import parse
from . import util

cfg = config.parse()

D_PARAMS = cfg['data_params']
L_PARAMS = cfg['language_params']
P_PARAMS = cfg['parser_params']
DIRECTORIES = cfg['directories']


class Language:
    """
    Class used to convert between string and integer representations of
        text

    Attributes:
        count (int): Total number of nodes processed
        index_node (dict): Dictionary mapping integers (indices)
            to strings (nodes)
        n_nodes (int): Number of unique nodes recognized in instance
        n_preserve (int): Number of default nodes (i.e. pad, unknown)
            to retain at front of indices
        node_count (dict): Dictionary mapping strings (nodes) to frequency
        node_index (dict): Dictionary mapping strings (nodes) to integers
            (indices)
        pad_index (int): Integer value representation of padding (default = 0)
        pad_token (str): String representation of padding (default = 'PAD')
        start_index (int): Integer value representation of the start token
            (default = 2)
        start_token (str): String representation of the start token
            (default = 'START')
        stop_index (int): Integer value representation of the stop token
            (default = 3)
        stop_token (str): String representation of the stop token
            (default = '。')
        unknown_index (int): Integer value representation of unknown tokens
            (default = 1)
        unknown_token (str): String representation of unknown tokens
            (default = 'UNKNOWN')
    """

    def __init__(self, pad_token: str=None, unknown_token: str=None,
                 start_token: str=None, stop_token: str=None):
        """
        Constructor for Language class

        Args:
            pad_token (str, optional): Description
            unknown_token (str, optional): Description
            start_token (str, optional): Description
            stop_token (str, optional): Description
        """
        self.pad_token = L_PARAMS['pad_token'] if pad_token is None \
            else pad_token
        self.pad_index = L_PARAMS['pad_index']

        self.unknown_token = L_PARAMS['unknown_token'] \
            if unknown_token is None else unknown_token
        self.unknown_index = L_PARAMS['unknown_index']

        # Use default values of start and stop tokens
        self.start_token = L_PARAMS['start_token'] \
            if start_token is None else start_token
        self.start_index = L_PARAMS['start_index']

        self.stop_token = L_PARAMS['stop_token'] \
            if stop_token is None else stop_token
        self.stop_index = L_PARAMS['stop_index']

        # Initialize dictionary mappings with default tokens
        self.node_index = {self.pad_token: self.pad_index,
                           self.unknown_token: self.unknown_index,
                           self.start_token: self.start_index,
                           self.stop_token: self.stop_index}

        self.node_count = {self.pad_token: 0,
                           self.unknown_token: 0,
                           self.start_token: 0,
                           self.stop_token: 0}

        self.index_node = {self.pad_index: self.pad_token,
                           self.unknown_index: self.unknown_token,
                           self.start_index: self.start_token,
                           self.stop_index: self.stop_token}

        self.n_preserve = len(self.index_node.keys())
        self.n_nodes = len(self.index_node.keys())
        self.count = 0

    def set_dicts(self, index_node: dict, node_count: dict,
                  node_index: dict):

        self.index_node = index_node
        self.node_count = node_count
        self.node_index = node_index

        self.n_nodes = len(self.index_node.keys())
        self.count = sum(list(self.node_count[i]
                              for i in self.node_count.keys()))

        self.sort()

    @classmethod
    def load(cls, load_path: str, load_prefix: str='', join: str='_'):
        """
        Load a Language instance from disk

        Args:
            load_path (str): Filepath to directory containing Language dicts
            load_prefix (str, optional): Identifying prefixes of Language dict
                filename
            join (str, optional): String joining prefix and suffix of
                Language dict filenames

        Returns:
            TYPE: Description
        """

        lg = cls()

        load_index_node = os.path.join(
            load_path, load_prefix + join + L_PARAMS['index_node'])
        load_node_count = os.path.join(
            load_path, load_prefix + join + L_PARAMS['node_count'])
        load_node_index = os.path.join(
            load_path, load_prefix + join + L_PARAMS['node_index'])

        index_node = pickle.load(open(load_index_node, 'rb'))
        node_count = pickle.load(open(load_node_count, 'rb'))
        node_index = pickle.load(open(load_node_index, 'rb'))

        lg.set_dicts(index_node, node_count, node_index)

        return lg

    def save(self, save_path: str, save_prefix: str='', join: str='_'):
        """
        Save a Language instance to disk

        Args:
            save_path (str): Filepath to save dictionaries of Language
                instance
            save_prefix (str, optional): Identifying prefixe for saving
                Language instance dicts
            join (str, optional): String joining prefix and suffix of
                Language dict filenames
        """
        self.sort()

        save_index_node = os.path.join(
            save_path, save_prefix + join + L_PARAMS['index_node'])
        save_node_count = os.path.join(
            save_path, save_prefix + join + L_PARAMS['node_count'])
        save_node_index = os.path.join(
            save_path, save_prefix + join + L_PARAMS['node_index'])

        pickle.dump(self.index_node, open(save_index_node, 'wb'))
        pickle.dump(self.node_count, open(save_node_count, 'wb'))
        pickle.dump(self.node_index, open(save_node_index, 'wb'))

    def add_nodes(self, nodes: list):
        """
        Add a list of nodes to self

        Args:
            nodes (list): List of nodes

        Returns:
            TYPE: Description
        """
        ret = list()

        for node in nodes:

            ret.append(self.add_node(node))

        return ret

    def add_node(self, node: str):
        """
        Add a single node to self

        Args:
            node (str): Single node (token or part-of-speech)
        """
        self.count += 1

        # If node has not previously been seen
        if node not in self.node_index:

            self.node_index[node] = self.n_nodes
            self.node_count[node] = 1

            self.index_node[self.n_nodes] = node

            self.n_nodes += 1

        # Do not increment default nodes
        elif self.node_index[node] < self.n_preserve:

            pass

        # If previously extant, update frequency of node
        else:

            self.node_count[node] += 1

        return self.node_index[node]

    def parse_indices(self, indices: list, n_max: int=-1,
                      delimiter: str=','):
        """
        Parse list of integer indices into string representation,
            separated by a delimiter.

        Args:
            indices (list): List of integers to parse
            n_max (int, optional): Integer threshold for unknown token.
                All values in %indices% exceeding this value are outputted as
                the instance's string representation for unknown tokens
            delimiter (str, optional): Delimiter used to separate individual
                nodes in output

        Returns:
            ret (str): String composed of nodes corresponding to integers
                in %indices%, separated by %delimiter%
        """
        ret = ''

        for idx in indices:

            ret += self.parse_index(idx, n_max)
            ret += delimiter

        return ret

    def parse_index(self, index: int, n_max: int=-1):
        """
        Convert integer index to corresponding string representation

        Args:
            index (int): Integer to parse
            n_max (int, optional): Integer threshold for unknown token.
                If %index% exceeds this value, the instance's string
                representation for unknown tokens is returned

        Returns:
            ret (str): String corresponding to %index%
        """
        n_max = min(n_max, self.n_nodes) if n_max > 0 \
            else self.n_nodes

        if index in self.index_node:

            node = self.index_node[index]

            return node if index < n_max else self.unknown_token

        else:
            return self.unknown_token

    def parse_nodes(self, nodes: list, n_max: int=-1):
        """
        Convert a list of nodes into a list of indices. Equivalent to
            add_nodes but does not update instance

        Args:
            nodes (list): List of strings (nodes)
            n_max (int, optional): Integer threshold for unknown index.
                All strings in %nodes% with index exceeding this value
                are replaced with the instance's unknown index

        Returns:
            ret (list): List of integer indices corresponding to strings
                in %nodes%
        """
        ret = []

        for node in nodes:

            idx = self.parse_index(node, n_max)
            ret.append(idx)

        return ret

    def parse_node(self, node: str, n_max: int=-1):
        """
        Convert single node to corresponding index

        Args:
            node (str): String (node) to convert
            n_max (int, optional): Integer threshold for unknown index.
                %node% is replaced by the instance's unknown index if
                its corresponding index exceeds this value.

        Returns:
            (int): Integer index coresponding to %node%
        """
        n_max = min(n_max, self.n_nodes) if n_max > 0 \
            else self.n_nodes

        if node in self.node_index:

            index = self.node_index[node]

            return index if index < n_max else self.unknown_index

        else:
            return self.unknown_index

    def sort(self):
        """
        Order dictionaries by frequency, with lower indices corresponding
            to more frequent nodes

        After operation, node with index = self.n_preserve + 1 is most frequent
        """
        temp = sorted(self.node_count.items(), key=lambda item: item[1])[
            self.n_preserve:]
        temp = temp[::-1]

        assert(len(temp) == self.n_nodes - self.n_preserve)

        for i in range(len(temp)):

            node, _ = temp[i]
            self.index_node[i + self.n_preserve] = node
            self.node_index[node] = i + self.n_preserve

    def sample(self, n_samples: int=50):
        """
        Display top-k most frequent nodes and frequencies

        Args:
            n_samples (int, optional): Number of nodes to display
        """

        # Re-order dicts
        self.sort()
        n_samples = min(n_samples, self.n_nodes - self.n_preserve)

        print('\nFixed nodes:')
        print('\n\tIndex: Node')

        for i in range(self.n_preserve):

            print('\t%d: %s' % (i, self.index_node[i]))

        print('\nRegular nodes:')
        print('\n\tIndex: Node, Count')

        for i in range(n_samples):

            j = i + self.n_preserve

            print('\t%d: %s, %d' % (j, self.index_node[j],
                                    self.node_count[self.index_node[j]]))


def compile_languages(source_corpus_dir: str, source_corpus_filetype: str,
                      save_dir: str, token_prefix: str,
                      syntactic_tag_prefix: str, n_files: int=-1):
    """
    Function to compile Language instances from a set of source
        corpus files.
    Contains one Language for tokens as well as additional Languages
        for each syntactic tags

    Args:
        source_corpus_dir (str): Directory containing corpus files
        source_corpus_filetype (str): Filetype of corpus files
        save_dir (str): Filepath to directory to save output Languages
        token_prefix (str): Save prefix for Language instance
            concerning source corpus tokens
        syntactic_tag_prefix (str): Save prefix for Language instances
            concerning source corpus syntactic tags
        n_files (int, optional): Maximum number of files to use

    Returns:
        (Language): Language of source corpus tokens
        (list): List of Language instances of source corpus syntactic tags

    """
    assert(os.path.isdir(source_corpus_dir))

    print('Obtaining list of corpus files...')
    print(cfg['BREAK_LINE'])
    file_list = util.get_files(source_corpus_dir,
                               source_corpus_filetype, n_files)

    print('Found %d files...\n' % len(file_list))

    token_language = Language()
    tag_languages = [Language() for i in range(len(P_PARAMS['parse_indices']))]

    print('Reading files...')
    print(cfg['BREAK_LINE'])

    delimiter = P_PARAMS['delimiter']
    count = 0

    for filename in file_list[:]:

        count += 1

        # Load sentences from each file and add to taggers
        with open(filename, 'r', encoding='utf-8') as f:

            start_time = time.time()

            print('Processing file: ' + filename)

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i].strip()

                nodes, pos = parse.parse_full(
                    sentence, parse.default_parser(), remove_delimiter=True,
                    delimiter=delimiter)

                token_language.add_nodes(nodes)

                for j in range(len(tag_languages)):

                    tag_languages[j].add_nodes(pos[j])

            elapsed_time = time.time() - start_time

            print('\tFile %2d of %2d processed...' % (count, len(file_list)))
            print('\tTime elapsed: %4f' % elapsed_time)

    print('\nCompleted processing corpus sentences...')
    print(cfg['BREAK_LINE'])

    print('\tSaving languages...')

    if not os.path.isdir(save_dir):
        util.mkdir_p(save_dir)

    token_language.sort()
    token_language.save(save_dir, token_prefix)
    token_language.sample()

    for i in range(len(tag_languages)):

        tag_languages[i].sort()
        tag_languages[i].save(save_dir, syntactic_tag_prefix + str(i))
        tag_languages[i].sample()

    print('\n\tCompleted...\n')

    return token_language, tag_languages


def load_languages(load_dir: str, token_prefix: str,
                   syntactic_tag_prefix: str, ):
    """
    Function to load the Language set defaulted by the configx.py configuration
        file
    Contains one token tagger as well as five part-of-speech taggers

    Args:
        load_dir (str): Filepath to directory to load Languages instances from
        token_prefix (str): Prefix of Language instance
            concerning source corpus tokens
        syntactic_tag_prefix (str): Prefix of Language instances
            concerning source corpus syntactic tags

    Returns:
        (Language): Language of source corpus tokens
        (list): List of Language instances of syntactic tags
    """
    token_language = Language()
    tag_languages = [Language() for i in range(len(P_PARAMS['parse_indices']))]

    print('\nLoading languages from %s...' % load_dir)

    token_language.load(load_dir, token_prefix)
    token_language.sort()

    parse_labels = P_PARAMS['parse_labels']

    for i in range(len(tag_languages)):

        print('\nLoading language of ' + parse_labels[i] + ' tags...')

        tag_languages[i].load(load_dir, syntactic_tag_prefix + str(i))
        tag_languages[i].sort()

    return token_language, tag_languages


def parse_node_matrix(syntactic_tags, languages):
    """
    Function to convert a one-dimensional matrix of syntactic tags
        into a corresponding matrix of indices using a list of
        languages, one language per row of output

    Args:
        syntactic_tags (arr): Array of arrays containing syntactic tags
        languages (arr): Array of Language class instances to use for parsing

    Returns:
        (np.array): A two-dimensional matrix of indices corresponding
            to the input array of arrays
    """
    assert(len(syntactic_tags) <= len(languages))

    return np.array(list(languages[i].parse_node(syntactic_tags[i])
                         for i in range(len(syntactic_tags))))
