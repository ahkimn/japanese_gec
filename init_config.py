# -*- coding: utf-8 -*-

# Filename: init_config.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 11/06/2018
# Date Last Modified: 03/03/2019
# Python Version: 3.7

import yaml

cfg = dict()

cfg['directories'] = {

    'data': './data',
    'raw_text': './data/raw_text'

}

cfg['data_params'] = {

    'raw_text_filetype': '.txt'



}

cfg['embedding_params'] = dict()

cfg['parser_params'] = {

    'pad_token': 'PAD',
    'pad_index': 0,
    'unknown_token': 'UNKNOWN',
    'unknown_index': 1,
    'start_token': 'START',
    'start_index': 2,
    'delimiter_token': '。',
    'delimiter_index': 3
}

cfg['print_params'] = {

    'break_line': '==========================================================',
    'break_subline': '\t==================================================',
    'break_halfline': '\t\t=========================================='
}


with open('config.yml', 'w') as f_config:
    yaml.dump(cfg, f_config)
