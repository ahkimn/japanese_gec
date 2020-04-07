# -*- coding: utf-8 -*-

# Filename: parse.py
# Date Created: 21/12/2019
# Description: Wrapper functions around MeCab's Tagger instances
# Python Version: 3.7


import MeCab
import re

from . import config

cfg = config.parse()
P_PARAMS = cfg['parser_params']


def default_parser():

    dict_dir = P_PARAMS['dictionary_dir']

    if dict_dir != '':

        parser = MeCab.Tagger('-d %s' % dict_dir)

    else:

        parser = MeCab.Tagger()

    parser.parse('')

    return parser


def resolve_syntactic_tags(tags: list):
    """
    Function to output config-defined syntatic tags from
        raw output from tagger

    Args:
        tags (list): List of syntatic tags

    Returns:
        (tuple): Tuple of selected syntatic tags
    """
    parse_indices = P_PARAMS['parse_indices']

    ret = list()

    for idx in parse_indices:

        ret.append(tags[idx])

    return tuple(ret)


def parse_full(sentence: str, parser: MeCab.Tagger,
               remove_delimiter: bool=False, delimiter: str=None):
    """
    Function to parse a given raw string into raw token and
        syntactic tags using a given MeCab tagger

    Args:
        sentence (str): Input string
        parser (MeCab.Tagger): Parser used to obtain syntactic tags
        remove_delimiter (bool, optional): If True, delimiter token is not
                present in output
        delimiter (str, optional): End-of-sentence delimiter token
                (i.e. period)

    Returns:
        (tuple): A tuple containing the following:
            nodes (list): A list of string tokens from the parsed %sentence%
            pos (list): A list of lists of strings. The nth list contains
                the syntactic tags corresponding to the nth token of %nodes%
    """
    if remove_delimiter:

        assert(delimiter is not None)
        sentence = sentence.replace(delimiter, '')

    sentence = re.sub(r'\s+', '', sentence.strip())

    len_parsed = 0

    nodes = list()
    pos = [list(), list(), list(), list(), list()]

    parser.parse('')
    res = parser.parseToNode(sentence)

    while res:

        len_parsed += len(res.surface)

        if res.surface != '':

            c = res.feature.split(",")
            c = resolve_syntactic_tags(c)

            for i in range(len(pos)):

                pos[i].append(c[i])

            nodes.append(res.surface)

        res = res.next

    assert(len_parsed == len(sentence))

    return nodes, pos
