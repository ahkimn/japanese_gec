import os
import sys
import languages
import subprocess

import configx

def trainSMT(commands):

    # Dataset name
    dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX

    if (len(commands) > 1):

        dataset_name = commands[1]

    # Where model files will output to
    output_dir = configx.CONST_SMT_OUTPUT_DIRECTORY

    if (len(commands) > 2):

        output_dir = commands[2]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    tm_prefix = os.path.join(output_dir, configx.CONST_TRANLSATION_MODEL_DIRECTORY)
    lm_prefix = os.path.join(output_dir, configx.CONST_LANGUAGE_MODEL_DIRECTORY)

    if not os.path.isdir(tm_prefix):
        os.mkdir(tm_prefix)

    if not os.path.isdir(lm_prefix):
        os.mkdir(lm_prefix)

    input_prefix = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY,
                          dataset_name,
                          configx.CONST_CORPUS_SAVE_DIRECTORY)

    source_corpus = os.path.join(input_prefix, 'train_error')
    target_corpus = os.path.join(input_prefix, 'train_correct')

    print("Training language model for target sentences")
    print("============================================")

    # Train language models for source
    subprocess.run(['thot_lm_train',
                    '-c', target_corpus,
                    '-o', lm_prefix,
                    '-pr', configx.CONST_NUM_PROCESSORS,
                    '-n' , str(3),
                    '-unk'])

    print("Training translation model from source to target sentences")
    print("============================================")

    # Train translation model
    subprocess.run(['thot_tm_train',
                    '-s', source_corpus,
                    '-t', target_corpus,
                    '-o', tm_prefix,
                    '-pr', configx.CONST_NUM_PROCESSORS])


def tuneSMT(commands):

    # Dataset name
    dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX

    if (len(commands) > 1):

        dataset_name = commands[1]


     # Where model files will output to
    output_dir = configx.CONST_SMT_OUTPUT_DIRECTORY

    if (len(commands) > 2):

        output_dir = commands[2]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    tm_prefix = os.path.join(output_dir, configx.CONST_TRANLSATION_MODEL_DIRECTORY)
    lm_prefix = os.path.join(output_dir, configx.CONST_LANGUAGE_MODEL_DIRECTORY)
    tuned_prefix = os.path.join(output_dir, configx.CONST_TUNED_MODELS_DIRECTORY)

    # if not os.path.isdir(tuned_prefix):
    #     os.mkdir(tuned_prefix)

    config_prefix = os.path.join(output_dir, configx.CONST_SMT_CONFIG_DIRECTORY)

    if not os.path.isdir(config_prefix):
        os.mkdir(config_prefix)


    tm_desc = os.path.join(tm_prefix, "tm_desc")
    lm_desc = os.path.join(lm_prefix, "lm_desc")

    config_file = os.path.join(config_prefix, "initial.cfg")

    f = open(config_file, "w+")

    subprocess.run(['thot_gen_cfg_file',
                    lm_desc,
                    tm_desc], stdout=f)

    f.close()

    input_prefix = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY,
                          dataset_name,
                          configx.CONST_CORPUS_SAVE_DIRECTORY)

    source_corpus_validation = os.path.join(input_prefix, 'validation_full_error')
    target_corpus_validation = os.path.join(input_prefix, 'validation_full_correct')

    print(config_file)
    print(source_corpus_validation)
    print(target_corpus_validation)
    print(tuned_prefix)
    
    # # Tune translation model parameters
    # subprocess.run(['thot_smt_tune',
    #                 '-c', config_file,
    #                 '-s', source_corpus_validation,
    #                 '-t', target_corpus_validation,
    #                 '-o', tuned_prefix,
    #                 '-pr', configx.CONST_NUM_PROCESSORS])


def filterSMT(commnands):

    pass


def translateSMT(commands):

    # Dataset name
    dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX

    if (len(commands) > 2):

        dataset_name = commands[2]

    n_rule = -1

    # Where model files will output to
    output_dir = configx.CONST_SMT_OUTPUT_DIRECTORY

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
        target_corpus = os.path.join('test_full_correct')

    else:

        assert(n_rule >= 0)
        source_corpus = os.path.join(input_prefix, 'test_' + str(n_rule) + '_error')
        output_file = os.path.join(output_prefix, 'test_' + str(n_rule) + '_translated')
        target_corpus = os.path.join(input_prefix, 'test_' + str(n_rule) + '_correct')

    config_prefix = os.path.join(output_dir, configx.CONST_SMT_CONFIG_DIRECTORY)
    config_file = os.path.join(config_prefix, "initial.cfg")

    # Tune translation model parameters
    subprocess.run(['thot_decoder',
                    '-c', config_file,
                    '-t', source_corpus,
                    '-o', output_file,
                    '-pr', configx.CONST_NUM_PROCESSORS])

def decodeSMT(commands):

    # Dataset name
    dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX

    if (len(commands) > 2):

        dataset_name = commands[2]

    n_rule = -1

    # Where model files will output to
    output_dir = configx.CONST_SMT_OUTPUT_DIRECTORY

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
        source_corpus = os.path.join(input_prefix, 'test_' + str(n_rule) + '_error')
        output_file = os.path.join(output_prefix, 'test_' + str(n_rule) + '_translated')
        target_corpus = os.path.join(input_prefix, 'test_' + str(n_rule) + '_correct')

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







   
   
def translateSingle(commands, sentence, tagger_s, tagger_t):

    # Where model files will output to
    output_dir = configx.CONST_SMT_OUTPUT_DIRECTORY

    if (len(commands) > 1):

        output_dir = commands[1]

    if sentence[-1] != '。':
        sentence += '。'

    tokens, _ = languages.parse_sentence(sentence, configx.CONST_PARSER, '')

    tokens = tagger_s.parse_sentence(tokens)
    tokens = str(" ".join(list(str(i) for i in tokens)))

    if not os.path.isdir("temp"):
        os.mkdir("temp")

    input_file = os.path.join("temp", "temp")
    output_file = os.path.join("temp", "more_temp")

    f = open(input_file, "w+")
    
    for i in range(10):
        f.write(tokens+"\n")

    f.close()

    config_prefix = os.path.join(output_dir, configx.CONST_SMT_CONFIG_DIRECTORY)
    config_file = os.path.join(config_prefix, "initial.cfg")

    # Tune translation model parameters
    subprocess.run(['thot_decoder',
                    '-c', config_file,
                    '-t', input_file,
                    '-o', output_file,
                    '-pr', configx.CONST_NUM_PROCESSORS])

    f = open(output_file, "r")
    output = f.readlines()[0]
    f.close()

    output = list(int(i) for i in list(output.split(" ")))
    output = tagger_t.sentence_from_indices(output)

    return output







if __name__ == '__main__':

    if (len(sys.argv) == 1):

        raise TypeError("ERROR: No command argument specified")

    command = sys.argv[1]

    if command.lower() == "smt-train" or command.lower() == "smt_train":

        trainSMT(sys.argv[1:])

    elif command.lower() == "smt-tune" or command.lower() == "smt_tune":

        tuneSMT(sys.argv[1:])

        ''' Default Tuning Commmand 
        
        thot_smt_tune \
        -c smt_output/config/initial.cfg \
        -s output/type/corpus/validation_full_error \
        -t output/type/corpus/validation_full_correct \
        -o smt_output/tm_tuned \
        -pr 4
        '''

    elif command.lower() == "smt-filter" or command.lower() == "smt_filter":

        filterSMT(sys.argv[1:])

    elif command.lower() == "smt-translate" or command.lower() == "smt_translate":

        translateSMT(sys.argv[1:])

    elif command.lower() == "smt-decode" or command.lower() == "smt_decode":

        decodeSMT(sys.argv[1:])

    # Attempt to parse single sentence
    elif command.lower() == "smt-try" or command.lower() == "smt_try":

        # Data to save/load taggers containing tokens from error sentences
        source_token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)
        target_token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)

        print("Accepting terminal input to translate...")
        print("Enter your sentence or type \'exit\' to quit: \n\t")

        user_input = ''

        while (user_input.lower() != 'exit'):

            text_to_translate = input("\tInput: ")

            if (text_to_translate.lower() == 'exit'):
                user_input = text_to_translate

            else:

                output = translateSingle(sys.argv[1:], text_to_translate, source_token_tagger, target_token_tagger)
                print("\tOutput: " + output)

    else:

        raise TypeError("ERROR: Invalid command argument given")
