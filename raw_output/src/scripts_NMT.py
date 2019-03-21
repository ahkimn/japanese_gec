import os
import sys
import languages
import subprocess

import numpy as np

import configx

def decodeCNN(commands):

    # Dataset name
    dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX

    if (len(commands) > 2):

        dataset_name = commands[2]

    n_rule = -1

    # Where model files will output to
    output_dir = configx.CONST_CNN_OUTPUT_DIRECTORY

    if (len(commands) > 3):

        output_dir = commands[3]

    # Which rule to test
    if (len(commands) > 1):

        try:
            n_rule = int(commands[1])
        except:
            print("Could not parse rule number")

    input_prefix = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY,
                          dataset_name,
                          configx.CONST_CORPUS_SAVE_DIRECTORY)

    output_prefix = os.path.join(output_dir, configx.CONST_TRANSLATE_OUTPUT_PREFIX)

    if not os.path.isdir(output_prefix):
        os.mkdir(output_prefix)

    if n_rule == -1:

        source_corpus = os.path.join(input_prefix, 'test_full_error')
        output_file = os.path.join(output_prefix, 'test_full_translated')
        target_corpus = os.path.join(input_prefix, 'test_full_correct')

        decoded_source_corpus = os.path.join(output_prefix, 'test_full_error_text')
        decoded_output_file = os.path.join(output_prefix, 'test_full_translated_text')
        decoded_target_corpus = os.path.join(output_prefix, 'test_full_correct_text')

    else:

        assert(n_rule >= 0)
        source_corpus = os.path.join(output_prefix, 'test_' + str(n_rule) + '_error.txt')
        output_file = os.path.join(output_prefix, 'test_' + str(n_rule) + '_translated.txt')
        target_corpus = os.path.join(output_prefix, 'test_' + str(n_rule) + '_correct.txt')

        decoded_source_corpus = os.path.join(output_prefix, 'test_' + str(n_rule) + '_error_text')
        decoded_output_file = os.path.join(output_prefix, 'test_' + str(n_rule) + '_translated_text')
        decoded_target_corpus = os.path.join(output_prefix, 'test_' + str(n_rule) + '_correct_text')

    source_corpus = open(source_corpus, "r")
    output_file = open(output_file, "r")
    target_corpus = open(target_corpus, "r")

    decoded_source_corpus = open(decoded_source_corpus, "w+")
    decoded_output_file = open(decoded_output_file, "w+")
    decoded_target_corpus = open(decoded_target_corpus, "w+")

    # Data to save/load taggers containing tokens from error sentences
    source_token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)
    target_token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)

    source_token_tagger.decode_file(source_corpus, decoded_source_corpus)
    target_token_tagger.decode_file(output_file, decoded_output_file)
    target_token_tagger.decode_file(target_corpus, decoded_target_corpus)

    source_corpus.close()
    output_file.close()
    target_corpus.close()

    decoded_source_corpus.close()
    decoded_output_file.close()
    decoded_target_corpus.close()


def truncatedDecode(commands, n_sentences=100, max_length=25):

    # Dataset name
    dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX

    if (len(commands) > 2):

        dataset_name = commands[2]

    n_rule = -1

    # Where model files will output to
    input_dir = configx.CONST_CNN_OUTPUT_DIRECTORY
    smt_dir = configx.CONST_SMT_OUTPUT_DIRECTORY

    if (len(commands) > 3):

        output_dir = commands[3]

    # Which rule to test
    if (len(commands) > 1):

        try:
            n_rule = int(commands[1])
        except:
            print("Could not parse rule number")

    input_prefix_nmt = os.path.join(input_dir, configx.CONST_TRANSLATE_OUTPUT_PREFIX)
    input_prefix_smt = os.path.join(smt_dir, configx.CONST_TRANSLATE_OUTPUT_PREFIX)

    output_dir = configx.CONST_TRUNCATED_OUTPUT_DIR

    if not os.path.isdir(output_dir):

        os.mkdir(output_dir)

    assert(n_rule >= 0)

    decoded_source_corpus = os.path.join(input_prefix_nmt, 'test_' + str(n_rule) + '_error_text')
    decoded_output_file = os.path.join(input_prefix_nmt, 'test_' + str(n_rule) + '_translated_text')
    decoded_target_corpus = os.path.join(input_prefix_nmt, 'test_' + str(n_rule) + '_correct_text')
    decoded_smt_output_file = os.path.join(input_prefix_smt, 'test_' + str(n_rule) + '_translated_text')

    output_source_corpus = os.path.join(output_dir, 'test_' + str(n_rule) + '_error_text')
    output_output_file = os.path.join(output_dir, 'test_' + str(n_rule) + '_nmt_translated_text')
    output_target_corpus = os.path.join(output_dir, 'test_' + str(n_rule) + '_correct_text')
    output_smt_output_file = os.path.join(output_dir, 'test_' + str(n_rule) + '_smt_translated_text')

    decoded_source_corpus = open(decoded_source_corpus, "r")
    decoded_output_file = open(decoded_output_file, "r")
    decoded_target_corpus = open(decoded_target_corpus, "r")
    decoded_smt_output_file = open(decoded_smt_output_file, "r")

    output_source_corpus = open(output_source_corpus, "w+")
    output_output_file = open(output_output_file, "w+")
    output_target_corpus = open(output_target_corpus, "w+")
    output_smt_output_file = open(output_smt_output_file, "w+")

    # Data to save/load taggers containing tokens from error sentences
    source_token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)
    target_token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)

    check = decoded_source_corpus.readlines()

    indices = list()

    for i in range(len(check)):

        sentence = check[i]

        if len(sentence) < max_length and len(sentence) > 0:

            indices.append(i)

    indices = np.array(indices)
    indices = np.random.permutation(indices)[:n_sentences]



    copy_truncate_data(decoded_source_corpus, output_source_corpus, indices)
    copy_truncate_data(decoded_output_file, output_output_file, indices)
    copy_truncate_data(decoded_target_corpus, output_target_corpus, indices)
    copy_truncate_data(decoded_smt_output_file, output_smt_output_file, indices)


    decoded_source_corpus.close()
    decoded_output_file.close()
    decoded_target_corpus.close()
    decoded_smt_output_file.close()

    output_source_corpus.close()
    output_output_file.close()
    output_target_corpus.close()
    output_smt_output_file.close()


def copy_truncate_data(input_file, output_file, indices):

    input_file.seek(0)
    input_data = input_file.readlines()

    for z in range(len(indices)):

        line_out = input_data[indices[z]]

        output_file.write(line_out)





if __name__ == '__main__':

    if (len(sys.argv) == 1):

        raise TypeError("ERROR: No command argument specified")

    command = sys.argv[1]


    if command.lower() == "cnn-decode" or command.lower() == "cnn_decode":

        decodeCNN(sys.argv[1:])

    elif command.lower() == "truncate" or command.lower() == "truncate":

        truncatedDecode(sys.argv[1:])

    else:

        raise TypeError("ERROR: Invalid command argument given")
