# Filename: load.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 26/02/2019
# Date Last Modified: 26/02/2019
# Python Version: 3.7

'''
Functions to update pre-existing Language class instances
'''

import os

from . import configx
from . import languages


def update_languages(token_tagger, pos_taggers, sentence_list, save_dir, source=True):
    """
    Update and save languages using new data
    
    Args:
        token_tagger (Language): Language tagging tokens
        pos_taggers (arr): Array of languages tagging part-of-speech tags
        sentence_list (arr): Array of sentence pairs to add to the Langauge class instances
        save_dir (str): Relative path of save directory for the Language class instances
        source (bool, optional): Determines whether or not the Language is meant for the source (errored) sentences or target (correct) sentences
    """

    for j in range(len(sentence_list)):

        sentence = sentence_list[j][0]

        if not source:

            sentence = sentence_list[j][1]       

        tokens, pos_tags = languages.parse_sentence(sentence, configx.CONST_PARSER, None)

        # Update token tagger
        token_tagger.add_sentence(tokens)

        for k in range(len(pos_taggers)):

            pos_taggers[k].add_sentence(pos_tags[k])

    token_prefix = os.path.join(save_dir, configx.CONST_NODE_PREFIX)
    pos_prefix = os.path.join(save_dir, configx.CONST_POS_PREFIX)

    # Save updated token tagger
    token_tagger.sort()
    token_tagger.save_dicts(token_prefix)

    # Save updated pos_taggers
    for k in range(len(pos_taggers)):

        pos_taggers[k].sort()
        pos_taggers[k].save_dicts(pos_prefix + str(k))

