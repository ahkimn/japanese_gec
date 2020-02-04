# -*- coding: utf-8 -*-

import yaml
import os

HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(
    HOME_DIR, 'Projects/aitutor/japanese_gec')
CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.yml')


def parse():

    f_config = open(CONFIG_FILE, 'r')
    cfg = yaml.load(f_config)
    f_config.close()

    return cfg


if __name__ == '__main__':

    cfg = dict()

    cfg['directories'] = {

        'data': os.path.join(PROJECT_DIR, 'data'),
        'source_corpora': os.path.join(PROJECT_DIR, 'data/source_corpora'),
        'languages': os.path.join(PROJECT_DIR, 'data/languages')
    }

    cfg['data_params'] = {

        'source_corpus_filetype': '.txt'
    }

    cfg['language_params'] = {

        'pad_token': '*',
        'pad_index': 0,
        'unknown_token': 'UNK',
        'unknown_index': 1,
        'start_token': 'START',
        'start_index': 2,
        'stop_token': 'STOP',
        'stop_index': 3,
        'index_node': 'in.pkl',
        'node_count': 'nc.pkl',
        'node_index': 'ni.pkl',
        'delimiter': '。',
        'token_prefix': 't',
        'syntactic_tag_prefix': 'st'
    }

    cfg['parser_params'] = {

        'delimiter': '。',
        'parse_indices': [0, 1, 4, 5, 6],
        'parse_labels':
        [
            'Part-of-speech',
            'Part-of-speech Sub-category',
            'Inflection',
            'Conjugation',
            'Root Form'
        ]
    }

    cfg['BREAK_LINE'] = '=====================\n'

    with open(CONFIG_FILE, 'w+') as f_config:
        yaml.dump(cfg, f_config)
