# -*- coding: utf-8 -*-

# Filename: config.py
# Date Created: 20/12/2019
# Description: Script to construct default parameters for repository
# Python Version: 3.7

import yaml
import os

HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(
    HOME_DIR, 'Projects/japanese_gec')
CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.yml')


def parse():

    f_config = open(CONFIG_FILE, 'r')
    cfg = yaml.load(f_config, Loader=yaml.SafeLoader)
    f_config.close()

    return cfg


if __name__ == '__main__':

    cfg = dict()

    cfg['directories'] = {

        'const': os.path.join(PROJECT_DIR, 'data/const'),
        'data': os.path.join(PROJECT_DIR, 'data'),
        'databases': os.path.join(PROJECT_DIR, 'data/databases'),
        'languages': os.path.join(PROJECT_DIR, 'data/languages'),
        'rules': os.path.join(PROJECT_DIR, 'data/rules'),
        'sorted_tag_databases':
            os.path.join(PROJECT_DIR, 'data/sorted_tag_databases'),
        'source_corpora': os.path.join(PROJECT_DIR, 'data/source_corpora'),
        'datasets': os.path.join(PROJECT_DIR, 'data/datasets'),
        'synthesized_data': os.path.join(PROJECT_DIR, 'data/synthesized'),
        'tmp': os.path.join(PROJECT_DIR, 'data/tmp'),
        'models': os.path.join(PROJECT_DIR, 'models'),
        'model_output': os.path.join(PROJECT_DIR, 'data/model_output'),
        'test_corpora': os.path.join(PROJECT_DIR, 'data/test_corpora')
    }

    cfg['data_params'] = {

        'source_corpus_filetype': '.txt',
    }

    cfg['dataset_params'] = {

        'col_correct': 'correct',
        'col_error': 'error',
        'col_prefix_output': 'output',
        'col_correct_bounds': 'correct_bounds',
        'col_error_bounds': 'error_bounds',
        'col_subrules': 'subrules',
        'col_rules': 'rules',

        'save_names': {
            'default_prefix': 'syn',
            'rule_folder_prefix': '',
            'rule_file_prefix': '',
            'subrule_file_prefix': 'sr',
            'correct_suffix': 'correct',
            'error_suffix': 'error',
            'ds_suffix': 'ds',
            'train_suffix': 'train',
            'test_suffix': 'test',
            'dev_suffix': 'dev'
        },

        'error_delimiters': ['<', '>'],
        'correct_delimiters': ['(', ')']
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
        'syntactic_tag_prefix': 'st',
        'character_prefix': 'c'
    }

    cfg['model_params'] = {

        'dictionary_size': 50000,
        'fairseq_output': 'generate-test.txt',
        'fconv_checkpoint': 'checkpoint_best.pt',
    }

    cfg['database_params'] = {

        'form_char_prefix': 'fc',
        'form_char_len_prefix': 'fcl',
        'max_sentence_length': 50,
        'max_token_length': 10,
        'partition_size': 50000,
        'sentence_len_prefix': 'sl',
        'sort_tag_prefix': 'stt',
        'sort_form_prefix': 'stf',
        'syntactic_tag_prefix': 'st',
        'token_char_prefix': 'tc',
        'token_char_len_prefix': 'tcl',
        'token_prefix': 't',
        'unique_token_prefix': 'ut',
        'unique_syntactic_tag_prefix': 'ust',
        'unique_form_prefix': 'uf'
    }

    cfg['morpher_params'] = {

        'kana_default': 'kana.csv'
    }

    cfg['parser_params'] = {

        'delimiter': '。',
        'dictionary_dir': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
        'parse_indices': [0, 1, 4, 5, 6],
        'parse_labels':
        [
            'part-of-speech',
            'part-of-speech sub-category',
            'inflection',
            'conjugation',
            'root form'
        ]
    }

    cfg['rule_params'] = {

        'class': 'Rule Class',
        'subclass': 'Rule Subclass',
        'name': '#',
        'rule_type': 'Rule Type',
        'type_token': 'Token',
        'type_character': 'Character',
        'syntactic_tags': 'Mecab Output',
        'syntactic_tag_mask': 'Requisite Syntactic Tags',
        'template_correct_phrase': 'Template Correct Phrase',
        'template_error_phrase': 'Template Error Phrase',
        'mapping_inserted': 'Inserted',
        'mapping_modified': 'Modified',
        'mapping_preserved': 'Preserved',
        'mapping_substituted': 'Substituted',
        'mapping_deleted': 'Deleted'
    }

    cfg['seed'] = 23
    cfg['BREAK_LINE'] = '=====================\n'
    cfg['BREAK_SUBLINE'] = '\t=================\n'
    cfg['BREAK_HALFLINE'] = '\t\t=============\n'

    for _, directory in cfg['directories'].items():
        if not os.path.isdir(directory):
            os.mkdir(directory)

    with open(CONFIG_FILE, 'w+') as f_config:
        yaml.dump(cfg, f_config)

