# -*- coding: utf-8 -*-

# Filename: gen_model_output.py
# Date Created: 01/04/2020
# Description: Script to generate Fairseq model output given either a saved
#   Dataset instance or a plain-text file.
# Python Version: 3.7

import argparse
import os
import subprocess
import shutil

from src import config
from src import convert
from src import util
from src.datasets import Dataset
from src.util import str_bool, literal_str


cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
DB_PARAMS = cfg['database_params']
MDL_PARAMS = cfg['model_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model given \
            train/dev/test Dataset instances')

    # ====================================================
    #       Parameters for loaded Dataset instances
    # ====================================================

    # Required for ds_eval/ds_generate
    parser.add_argument(
        '--ds_load_dir', metavar='DS_LOAD_DIR',
        type=str, help='sub-directory of ./data/datasets \
            where Dataset instances are saved', required=False, default=None)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='File extension of saved Dataset instances',
        required=False)

    # Required for ds_eval/ds_generate
    parser.add_argument(
        '--ds_name', metavar='DS_NAME', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            to load',
        required=False, default=None)

    # ====================================================
    #           Parameters for loaded file
    # ====================================================

    # Required for file_generate
    parser.add_argument(
        '--file_name', metavar='FILE_NAME',
        type=str, help='full filename (including extension) of file \
            within ./data/test_corpora directory', required=False,
        default=None)

    parser.add_argument(
        '--tokenized', metavar='TOKENIZED',
        type=str_bool, help='True if the file to be loaded is already \
            tokenized', default=False, required=False)

    parser.add_argument(
        '--token_delimiter', metavar='TOKEN_DELIMITER',
        type=literal_str, help='delimiter between tokens in file if the file is \
            tokenized', default=' ', required=False)

    parser.add_argument(
        '--sentence_delimiter', metavar='SENTENCE_DELIMITER',
        type=literal_str, help='delimiter between sentences in file if the file \
            contains sentence pairs', default=',', required=False)

    parser.add_argument(
        '--error_first', metavar='ERROR_FIRST',
        type=str_bool, help='for files with error/correct pairs. If True, the error \
            sentence is the first sentence of the pair', default=True,
        required=False)

    # ====================================================
    #         Parameters for model/model training
    # ====================================================

    parser.add_argument(
        '--cuda', metavar='CUDA', default=-1, type=int,
        help='if not -1, index of GPU to use',
        required=False)

    parser.add_argument(
        '--fp16', metavar='FP16', default=True, type=str_bool,
        help='if True, use half-precision \
            floating point values for training', required=False)

    # Required
    parser.add_argument(
        '--model_load_dir', metavar='MODEL_LOAD_DIR', type=str,
        default='mdl',
        help='sub-directory of \'./models\' to load model from',
        required=True)

    parser.add_argument(
        '--batch_size', metavar='BATCH_SIZE', default=32, type=str,
        help='batch size to use when training', required=False)

    # ====================================================
    #               Other parameters
    # ====================================================

    parser.add_argument(
        '--tmp_dir', metavar='TMP_DIR', type=str, default=DIRECTORIES['tmp'],
        help='sub-directory of (./data/tmp/) to write temporary files to',
        required=False)

    parser.add_argument(
        '--command', metavar='COMMAND', type=str,
        help='one of [\'ds_generate\', \'file_generate\']. Determines \
            which operation to run',
        required=True)

    parser.add_argument(
        '--output_file', metavar='OUTPUT_FILE', type=str,
        help='filepath within ./data/model_output to save generated data \
        to', default=None, required=False)

    parser.add_argument(
        '--output_ext', metavar='OUTPUT_EXT', type=str,
        help='file extension to use for output file', default='csv',
        required=False)

    args = parser.parse_args()
    command = args.command.lower()

    tmp_dir = args.tmp_dir

    write_dir = os.path.join(tmp_dir, 'write')
    prep_dir = os.path.join(tmp_dir, 'preprocess')
    gen_dir = os.path.join(tmp_dir, 'generate')

    model_load_dir = os.path.join(DIRECTORIES['models'],
                                  args.model_load_dir)

    if not os.path.isdir(tmp_dir):
        util.mkdir_p(tmp_dir)

    model_checkpoint = 'checkpoint_best.pt'
    model_file = os.path.join(model_load_dir, model_checkpoint)

    dict_source = 'dict.%s.txt' % SAVE_PARAMS['error_suffix']
    dict_target = 'dict.%s.txt' % SAVE_PARAMS['correct_suffix']
    dict_source = os.path.join(model_load_dir, dict_source)
    dict_target = os.path.join(model_load_dir, dict_target)

    assert(all(os.path.isfile(f) for f in
               [model_file, dict_source, dict_target]))

    max_tokens = args.batch_size * (DB_PARAMS['max_sentence_length'])

    output_file = args.output_file
    if output_file is None:
        output_file = args.model_load_dir

    output_model_file = os.path.join(DIRECTORIES['model_output'],
                               '%s.%s' % (output_file, args.output_ext))
    output_correct_file = os.path.join(DIRECTORIES['model_output'],
                                       '%s_correct.%s' %
                                       (output_file, args.output_ext))
    output_error_file = os.path.join(DIRECTORIES['model_output'],
                                     '%s_error.%s' %
                                     (output_file, args.output_ext))

    if command == 'ds_generate':

        ds_specified = all([args.ds_load_dir, args.ds_name])
        assert(ds_specified)

        if not os.path.isdir(write_dir):
            util.mkdir_p(write_dir)

        ds_load_file = os.path.join(DIRECTORIES['datasets'],
                                    args.ds_load_dir, '%s.%s' %
                                    (args.ds_name, args.ds_suffix))

        print('Loading dataset: %s' % ds_load_file)

        DS = Dataset.load(ds_load_file)
        DS.write(write_dir, token_delimiter=' ', data_delimiter='',
                 include_tags=[], separation='none', max_per_rule=-1,
                 save_prefix=args.ds_name)

        w_name = os.path.join(write_dir, args.ds_name)

    elif command == 'file_generate':

        assert(args.file_name is not None)

        load_file = os.path.join(DIRECTORIES['test_corpora'], args.file_name)

        assert(os.path.isfile(load_file))

        error_sentences, correct_sentences, _, _ = convert.process_file(
            load_file,
            token_delimiter=args.token_delimiter,
            sentence_delimiter=args.sentence_delimiter,
            tokenized=args.tokenized,
            error_first=args.error_first)

        error_out = os.path.join(write_dir, '%s.%s' %
                                 (SAVE_PARAMS['default_prefix'],
                                  SAVE_PARAMS['error_suffix']))

        correct_out = os.path.join(write_dir, '%s.%s' %
                                   (SAVE_PARAMS['default_prefix'],
                                    SAVE_PARAMS['correct_suffix']))
        convert.write_output_file(error_out, [error_sentences])
        convert.write_output_file(correct_out, [correct_sentences])

        w_name = os.path.join(write_dir, SAVE_PARAMS['default_prefix'])

    else:

        raise ValueError('Value %s for command argument not valid' % command)

    if os.path.isdir(prep_dir):
        shutil.rmtree(prep_dir)

    print('Preprocessing data to directory: %s' % prep_dir)

    prep = subprocess.Popen(
        ['fairseq-preprocess',
         '--source-lang', SAVE_PARAMS['error_suffix'],
         '--target-lang', SAVE_PARAMS['correct_suffix'],
         '--srcdict', dict_source,
         '--tgtdict', dict_target,
         '--testpref', w_name,
         '--destdir', prep_dir,
         '--workers', str(4)],
        stdout=subprocess.PIPE)

    output = prep.communicate()[0]

    print('Generating data to directory: %s' % gen_dir)

    gen_args = ['fairseq-generate', prep_dir,
                '--path', model_file,
                '--batch-size', str(args.batch_size),
                '--results-path', gen_dir,
                '--max-tokens', str(max_tokens)]

    if args.cuda != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

    if args.fp16:
        gen_args.append('--fp16')

    print('Running command: %s' % ' '.join(gen_args))

    gen = subprocess.Popen(gen_args,
                           stdout=subprocess.PIPE)

    output = gen.communicate()[0]

    gen_file = os.path.join(gen_dir, 'generate-test.txt')

    assert(os.path.isfile(gen_file))

    gen_sentences, _, crt_sentences, err_sentences = \
        convert.parse_fairseq_output(gen_file)

    print('Saving generated data to: %s' % output_file)

    convert.write_file(output_model_file, gen_sentences)
    convert.write_file(output_error_file, err_sentences)
    convert.write_file(output_correct_file, crt_sentences)
