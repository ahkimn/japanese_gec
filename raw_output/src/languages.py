# Filename: languages.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 11/06/2018
# Date Last Modified: 26/02/2019
# Python Version: 3.7

'''
Language class and related functions - for converting string representations of tokens/part-of-speech tags
to integer values (and vice-versa)
'''

import os
import time
import pickle

import numpy as np

from . import configx
from . import util

class Language:
    """
    Class used to determine mapping between unique nodes (tokens or parts-of-speech) and integers
    
    Attributes:
        count (int): Total number of nodes processed
        index_node (dict): Dictionary mapping integers (indices) to strings (nodes)
        n_nodes (int): Number of unique nodes recognized by the Language instance
        n_preserve (int): Number of default nodes (i.e. pad) to retain before adding actual nodes
        node_count (dict): Dictionary mapping strings (nodes) to their recorded frequency
        node_index (dict): Dictionary mapping strings (nodes) to integers (indices)
        pad_index (int): Integer value representation of padding (default = 0)
        pad_token (str): String representation of padding (default = 'PAD')
        start_index (int): Integer value representation of the start token (default = 2)
        start_token (str): String representation of the start token (default = 'START')
        stop_index (int): Integer value representation of the stop token (default = 3)
        stop_token (str): String representation of the stop token (default = '。')
        unknown_index (int): Integer value representation of unknown tokens (default = 1)
        unknown_token (str): String representation of unknown tokens (default = 'UNKNOWN')
        use_delimiter (bool, optional): Determines whether or not Language nodes are tokens  (which utilize '。' as sentence delimiters)
                                  or part-of-speech tags (which do not)
    """
    

    def __init__(self, use_delimiter=False):
        """
        Constructor
        
        Args:
            use_delimiter (bool, optional): Determines whether or not Language nodes are tokens  (which utilize '。' as sentence delimiters)
                                                or part-of-speech tags (which do not)
        """
        self.use_delimiter = use_delimiter

        self.pad_token = configx.CONST_PAD_TOKEN
        self.pad_index = configx.CONST_PAD_INDEX

        self.unknown_token = configx.CONST_UNKNOWN_TOKEN
        self.unknown_index = configx.CONST_UNKNOWN_INDEX

        # If nodes represent tokens
        if self.use_delimiter:

            # Use default values of start and stop tokens
            self.start_token = configx.CONST_SENTENCE_START_TOKEN
            self.start_index = configx.CONST_SENTENCE_START_INDEX

            self.stop_token = configx.CONST_SENTENCE_DELIMITER_TOKEN
            self.stop_index = configx.CONST_SENTENCE_DELIMITER_INDEX

            # Initialize dictionary mappings with default tokens
            self.node_index = {self.pad_token: self.pad_index,
                               self.unknown_token: self.unknown_index,
                               self.start_token: self.start_index, 
                               self.stop_token: self.stop_index}

            self.node_count = {self.pad_token:0,
                               self.unknown_token:0,
                               self.start_token: 0, 
                               self.stop_token: 0}
            
            self.index_node = {self.pad_index: self.pad_token,
                               self.unknown_index: self.unknown_token,
                               self.start_index: self.start_token, 
                               self.stop_index: self.stop_token}

        # Otherwise, if nodes are part-of-speech tags
        else:

            # Initialize dictionary mappings with asterisk to represent unused (padded) nodes
            # Start and stop indices are not included (as they are not part-of-speech tags)
            self.node_index = {'*': self.pad_index, self.unknown_token: self.unknown_index}
            self.node_count = {'*': 0, self.unknown_token:0}
            self.index_node = {self.pad_index: '*', self.unknown_index: self.unknown_token}


        self.n_preserve = len(self.index_node.keys())
        self.n_nodes = len(self.index_node.keys())
        self.count = 0


    def load_dicts(self, prefix):
        """
        Load a Language instance from disk
        
        Args:
            prefix (str): Language save prefix string
        """
        self.node_index = pickle.load(open("_".join((prefix, "ni.pkl")), 'rb'))
        self.node_count = pickle.load(open("_".join((prefix, "nc.pkl")), 'rb'))
        self.index_node = pickle.load(open("_".join((prefix, "in.pkl")), 'rb'))

        self.n_nodes = len(self.index_node.keys())
        self.count = sum(list(self.node_count[i] for i in self.node_count.keys()))


    def save_dicts(self, prefix):
        """
        Save a Language instance to disk
        
        Args:
            prefix (str): Language save prefix string
        """
        self.sort()

        pickle.dump(self.node_index, open("_".join((prefix, "ni.pkl")), 'wb'))
        pickle.dump(self.node_count, open("_".join((prefix, "nc.pkl")), 'wb'))
        pickle.dump(self.index_node, open("_".join((prefix, "in.pkl")), 'wb'))


    def add_sentence(self, nodes):
        """
        Add a list of nodes to the dictionary
        
        Args:
            nodes (arr): List of nodes
        """
        for node in nodes:

            self.add_node(node)


    def add_node(self, node):
        """
        Add a single node to the dictionary
        
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


    def parse_indices(self, indices, n_max=-1, delimiter=','):
        """
        Parse list of integer indices into string representation, separated by a delimiter
        
        Args:
            indices (arr): List of integers corresponding to node indices
            n_max (int, optional): Maximal integer value to output normally, all indices exceeding this value are outputted 
                                   as self.unknown_token (default = 'UNKNOWN')
            delimiter (str, optional): Delimiter used to separate individual nodes in output
        
        Returns:
            ret (str): String composed of nodes corresponding to input indices, delimited by the delimiter parameter
        """
        ret = ""

        if n_max > 0:

            n_max = min(n_max, self.n_nodes)

        else:

            n_max = self.n_nodes

        for i in range(len(indices)):

            index = indices[i]

            if index > n_max:

                ret += self.unknown_token

            else:

                ret += self.index_node[index]

            ret += delimiter

        return ret


    def sentence_from_indices(self, indices, n_max=-1):
        """
        Parse list of integer indices into string representation without delimiters
        
        Args:
            indices (arr): List of integers corresponding to node indices
            n_max (int, optional): Maximal integer value to output normally, all indices exceeding this value are outputted 
                                   as self.unknown_token (default = 'UNKNOWN')
        
        Returns:
            ret (str): String composed of nodes corresponding to input indices, delimited by the delimiter parameter
        """
        return self.parse_indices(indices, n_max, delimiter="")


    def parse_sentence(self, nodes, n_max=-1):
        """
        Convert a list of nodes into a list of indices
        
        Args:
            nodes (arr): List of strings (nodes)
            n_max (int, optional): Maximal index to convert normally, all nodes with indices exceeding this value are outputted 
                                   as self.unknown_index (default = 1)        
        
        Returns:
            ret (arr): List of integers (indices)
        """
        ret = []

        if n_max > 0:

            n_max = min(n_max, self.n_nodes)

        else:

            n_max = self.n_nodes

        for node in nodes:

            if node in self.node_index:

                index = self.node_index[node]

                ret.append(index) if index < n_max else ret.append(self.unknown_index)

            else:

                ret.append(self.unknown_index)

        return ret


    def decode_file(self, f_in, f_out):
        """
        Convert integer-encoded data file into equivalent text file
        
        Args:
            f_in (str): Path to input (encoded) file
            f_out (str): Path to output (decoded) file
        """
        data = f_in.readlines()

        for j in range(len(data)):

            line_in = list(i for i in data[j].strip().split(" "))

            for k in range(len(line_in)):

                # Process artifacts from fairseq Lua origins
                if '<Lua' in line_in[k]:

                    line_in[k] = self.unknown_index

                elif 'heritage>' in line_in[k]:

                    line_in[k] = self.unknown_index

                elif '<unk>' in line_in[k]:

                    line_in[k] = self.unknown_index

                else :

                    line_in[k] = int(line_in[k])

            line_out = self.sentence_from_indices(line_in)
            f_out.write(line_out + os.linesep)


    def parse_node(self, node, n_max=-1):
        """
        Convert single node to index, given an upper limit of indices
        
        Args:
            node (str): Input node
            n_max (int, optional): Maximum allowable index; any node with index above this value is outputted as 
                                    self.unknown_index (default = 1)
        
        Returns:
            (int): Corresponding index
        """
        if n_max > 0:

            n_max = min(n_max, self.n_nodes)

        else:

            n_max = self.n_nodes

        if node in self.node_index:

            index = self.node_index[node]

            return index if index < n_max else self.unknown_index

        else:
            return self.unknown_index


    def parse_index(self, index, n_max=-1):
        """
        Convert single index to its corresponding node, given an upper limit of indices
        
        Args:
            node (int): Input index
            n_max (int, optional): Maximum allowable index; any index above this value is outputted as 
                                    self.unknown_token (default = 'UNKNOWN')
        
        Returns:
            (str): Corresponding node
        """
        if n_max > 0:

            n_max = min(n_max, self.n_nodes)

        else:

            n_max = self.n_nodes

        if index in self.index_node:

            node = self.index_node[index]

            return node if index < n_max else self.unknown_token

        else:
            return self.unknown_token   

    def sort(self):
        """
        Refresh order of dictionaries, so that nodes are ordered in terms of decreasing frequency
        (i.e. node with index = self.n_preserve + 1 is most frequent)
        """
        temp = sorted(self.node_count.items(), key=lambda item: item[1])[self.n_preserve:]
        temp = temp[::-1]

        assert(len(temp) == self.n_nodes - self.n_preserve)

        for i in range(len(temp)):

            node, _ = temp[i]
            self.index_node[i + self.n_preserve] = node
            self.node_index[node] = i + self.n_preserve


def resolve_classification(classification):
    """
    Function to output relevant part-of-speech tags from a MeCab.Tagger()'s output
    
    Args:
        classification (arr): List of features from a MeCab.Tagger() 
    
    Returns:
        (tuple): Tuple of features containing only the major and first sub-division of part-of-speech,
                 as well as form features 
    """
    # Part of speech tags
    class1 = classification[0]
    class2 = classification[1]

    # Conjugation tags
    form1 = classification[4]
    form2 = classification[5]
    form3 = classification[6]

    return (class1, class2, form1, form2, form3)


def parse_full(sentence, tagger, delimiter, remove_delimiter = False):
    """
    Function to parse a given raw string into raw token and part-of-string tags using a given Mecab.Tagger()

    Args:
        sentence (str): Input string
        tagger (Language): Language class instance used to index individual tokens
        delimiter (str): Sentence end delimiter (i.e. period) 
        remove_delimiter (bool, optional): Determines whether or not the delimiter token is retained in output
    
    Returns:
        (tuple): A tuple containing the following:
            nodes (arr): A list of strings, correspoding to the tokens of the raw sentence
            pos (arr): A lsit of list of strings, corresponding to the part-of-speech tag of each index for the raw sentence
    """
    if remove_delimiter:

        sentence = sentence.replace(delimiter, '')

    sentence = sentence.strip()
    sentence = sentence.replace(' ', '')

    len_parsed = 0

    nodes = list()
    pos = [list(), list(), list(), list(), list()]

    tagger.parse('')
    res = tagger.parseToNode(sentence)

    while res:

        len_parsed += len(res.surface)
        
        if res.surface != '':

            c = res.feature.split(",")
            c = resolve_classification(c)  

            for i in range(len(pos)):

                pos[i].append(c[i])

            nodes.append(res.surface)

        res = res.next   

    assert(len_parsed == len(sentence))  

    return nodes, pos


def parse(sentence, tagger, delimiter, remove_delimiter = False):
    """
    Function to parse a given raw string into raw tokens using a given Mecab.Tagger()
    
    Args:
        sentence (str): Input string
        tagger (Language): Language class instance used to index individual tokens
        delimiter (str): Sentence end delimiter (i.e. period) 
        remove_delimiter (bool, optional): Determines whether or not the delimiter token is retained in output
    
    Returns:
        nodes(arr): A list of strings, correspoding to the tokens of the raw sentence
    """
    if remove_delimiter:

        sentence = sentence.replace(delimiter, '')

    sentence = sentence.strip()
    sentence = sentence.replace(' ', '')

    len_parsed  = 0

    nodes = list()

    tagger.parse('')
    res = tagger.parseToNode(sentence)

    while res:

        len_parsed += len(res.surface)
        
        if res.surface != '':

            nodes.append(res.surface)

        res = res.next 

    assert(len_parsed == len(sentence)) 
    
    return nodes



def load_default_languages(load_dir = configx.CONST_DEFAULT_LANGUAGE_DIRECTORY, 
                           node_save_prefix = configx.CONST_NODE_PREFIX, 
                           pos_save_prefix = configx.CONST_POS_PREFIX):
    """
    Function to load the Language set defaulted by the configx.py configuration file
    Contains one token tagger as well as five part-of-speech taggers
    
    Args:
        load_dir (str, optional): Path to directory containing the Language
        node_save_prefix (str, optional): Prefix used for saving the Language tagging tokens
        pos_save_prefix (str, optional): Prefix used for saving the Languages outputting part-of-speech tags
    
    Returns:
        token_tagger (Language): Language instance for converting tokens from MeCab output
        pos_taggers (arr): Array of Language instances for converting part-of-speech tags from MeCab output
    """
    token_tagger = Language(True)
    pos_taggers = [Language(), Language(), Language(), Language(), Language(True)]

    # print("\nLoading tokenizer...")

    node_prefix = os.path.join(load_dir, node_save_prefix)
    pos_prefix = os.path.join(load_dir, pos_save_prefix)

    token_tagger.load_dicts(node_prefix)
    token_tagger.sort()

    for i in range(len(pos_taggers)):

        # print("\nLoading part of speech tagger " + str(i) + " ...")

        pos_taggers[i].load_dicts(pos_prefix + str(i))
        pos_taggers[i].sort()

    # print('\n========================================================\n')

    return token_tagger, pos_taggers


def compile_default_languages(data_dir = configx.CONST_CORPUS_TEXT_DIRECTORY,
                              file_type = configx.CONST_CORPUS_TEXT_FILETYPE, 
                              save_dir = configx.CONST_DEFAULT_LANGUAGE_DIRECTORY, 
                              node_save_prefix = configx.CONST_NODE_PREFIX, 
                              pos_save_prefix = configx.CONST_POS_PREFIX,
                              n_files = -1):
    """
    Function to compile the default Language set from the corpus text defaulted by the configx.py configuration file
    Contains one token tagger as well as five part-of-speech taggers (the last of which contains the token forms)
    
    Args:
        data_dir (TYPE, optional): Directory to search for corpus files
        file_type (TYPE, optional): Corpus files suffix
        save_dir (str, optional): Path to directory containing the Language
        node_save_prefix (str, optional): Prefix used for saving the Language tagging tokens
        pos_save_prefix (str, optional): Prefix used for saving the Languages outputting part-of-speech tags
        n_files (TYPE, optional): Maximum number of files to use   
   
    """
    print("Obtaining list of corpus files...")
    print(configx.BREAK_LINE)
    file_list = util.get_files(data_dir, file_type, n_files)

    print("Found %d files...\n" % len(file_list) )

    token_tagger = Language(True)
    pos_taggers = [Language(), Language(), Language(), Language(), Language(True)]

    print("Reading files...")
    print(configx.BREAK_LINE)

    delimiter = token_tagger.stop_token
    count = 0

    for filename in file_list[:]:

        count += 1

        # Load sentences from each file and add to taggers
        with open(filename, 'r', encoding='utf-8') as f:

            start_time = time.time()

            # print("Processing file: " + filename)

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i].strip()

                nodes, pos = parse_sentence(sentence, configx.CONST_PARSER, delimiter)

                token_tagger.add_sentence(nodes)

                for j in range(len(pos_taggers)):

                    pos_taggers[j].add_sentence(pos[j])

            elapsed_time = time.time() - start_time

            # print("\tSentences completed: %2d\t||\tTime elapsed: %4f" % (len(sentences), elapsed_time))
            print("\tFile %2d of %2d processed..." % (count, len(file_list)))

    print("\nCompleted processing corpus sentences...")
    print(configx.BREAK_LINE)
    print("\tSaving languages...")

    if not os.path.isdir(save_dir):
        util.mkdir_p(save_dir)

    node_prefix = os.path.join(save_dir, node_save_prefix)
    pos_prefix = os.path.join(save_dir, pos_save_prefix)

    token_tagger.sort()
    token_tagger.save_dicts(node_prefix)

    for i in range(len(pos_taggers)):

        pos_taggers[i].sort()
        pos_taggers[i].save_dicts(pos_prefix + str(i))
        
    print("\n\tCompleted...\n")


def parse_node_matrix(pos_tags, languages):
    """
    Function to convert a one-dimensional matrix of nodes into a corresponding matrix of indices
    using a list of languages, one language per row of output
    
    Args:
        pos_tags (arr): Array of arrays containing nodes
        languages (arr): Array of Language class instances to use for parsing
    
    Returns:
        (np.ndarray): A one-dimensional matrix corresponding to the input array of arrays
    """
    assert(len(pos_tags) <= len(languages))

    return np.array(list(languages[i].parse_node(pos_tags[i]) for i in range(len(pos_tags))))
