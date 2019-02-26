import pickle
import os

from . import configx

# Language class used to determine mapping between unique tokens and integer values
class Language:

    def __init__(self, use_delimiter=False):

        self.use_delimiter = use_delimiter

        self.pad_token = configx.CONST_PAD_TOKEN
        self.pad_index = configx.CONST_PAD_INDEX

        self.unknown_token = configx.CONST_UNKNOWN_TOKEN
        self.unknown_index = configx.CONST_UNKNOWN_INDEX

        if self.use_delimiter:

            self.start_token = configx.CONST_SENTENCE_START_TOKEN
            self.start_index = configx.CONST_SENTENCE_START_INDEX

            self.stop_token = configx.CONST_SENTENCE_DELIMITER_TOKEN
            self.stop_index = configx.CONST_SENTENCE_DELIMITER_INDEX

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

        else:

            self.node_index = {'*': self.pad_index, self.unknown_token: self.unknown_index}
            self.node_count = {'*': 0, self.unknown_token:0}
            self.index_node = {self.pad_index: '*', self.unknown_index: self.unknown_token}


        self.n_preserve = len(self.index_node.keys())
        self.n_nodes = len(self.index_node.keys())
        self.count = 0

    def load_dicts(self, prefix):

        self.node_index = pickle.load(open("_".join((prefix, "ni.pkl")), 'rb'))
        self.node_count = pickle.load(open("_".join((prefix, "nc.pkl")), 'rb'))
        self.index_node = pickle.load(open("_".join((prefix, "in.pkl")), 'rb'))

        self.n_nodes = len(self.index_node.keys())
        self.count = sum(list(self.node_count[i] for i in self.node_count.keys()))

    def save_dicts(self, prefix):

        self.sort()

        pickle.dump(self.node_index, open("_".join((prefix, "ni.pkl")), 'wb'))
        pickle.dump(self.node_count, open("_".join((prefix, "nc.pkl")), 'wb'))
        pickle.dump(self.index_node, open("_".join((prefix, "in.pkl")), 'wb'))

    def add_sentence(self, nodes):

        for node in nodes:

            self.add_node(node)

    def add_node(self, node):

        self.count += 1

        if node not in self.node_index:

            self.node_index[node] = self.n_nodes
            self.node_count[node] = 1

            self.index_node[self.n_nodes] = node

            self.n_nodes += 1

        elif self.node_index[node] < self.n_preserve:

            pass

        else:

            self.node_count[node] += 1

    def parse_indices(self, indices, n_max=-1, delimiter=','):

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

        ret = self.parse_indices(indices, n_max)

        ret = "".join(ret.split(','))

        return ret

    def parse_sentence(self, nodes, n_max=-1):

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

        data = f_in.readlines()

        for j in range(len(data)):

            line_in = list(i for i in data[j].strip().split(" "))

            for k in range(len(line_in)):

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

        temp = sorted(self.node_count.items(), key=lambda item: item[1])[self.n_preserve:]
        temp = temp[::-1]

        # print("\tNumber of processed nodes: " + str(self.count))
        # print("\tMost common nodes: " + str(temp[:min(len(temp), 100)]))
        # print(list(self.index_node[i] for i in range(10)))

        assert(len(temp) == self.n_nodes - self.n_preserve)

        for i in range(len(temp)):

            node, _ = temp[i]
            self.index_node[i + self.n_preserve] = node
            self.node_index[node] = i + self.n_preserve


''' Function to extract part-of-speech as well as conjugation information from Mecab classification'''
def resolve_classification(classification):

    # Part of speech tags
    class1 = classification[0]
    class2 = classification[1]

    # Conjugation tags
    form1 = classification[4]
    form2 = classification[5]
    form3 = classification[6]

    return (class1, class2, form1, form2, form3)

''' Function to parse an input sentence using a Mecab tagger

        :returns: list of string containing nodes; list of tuples of part-of-speech tags
'''
def parse_sentence(sentence, tagger, delimiter, remove_periods=False):

    nodes = list()
    pos = [list(), list(), list(), list(), list()]

    sentence = sentence.strip()
    sentence = sentence.replace(' ', '')

    len_parsed = 0

    tagger.parse('')
    res = tagger.parseToNode(sentence)

    while res:

        len_parsed += len(res.surface)
        
        if res.surface != '' and not (res.surface == delimiter and remove_periods == True):

            c = res.feature.split(",")
            c = resolve_classification(c)  

            for i in range(len(pos)):

                pos[i].append(c[i])

            nodes.append(res.surface)

        res = res.next   

    assert(len_parsed == len(sentence))  

    return nodes, pos

def load_default_languages(load_dir=configx.CONST_DEFAULT_LANGUAGE_DIRECTORY, 
                           node_save_prefix=configx.CONST_NODE_PREFIX, 
                           pos_save_prefix=configx.CONST_POS_PREFIX):

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


if __name__ == '__main__':

    load_default_languages()

    pass