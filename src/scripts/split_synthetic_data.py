from .. import configx
from .. import languages
from .. import load
from .. import update_languages
from .. import util

import os


def split_synthetic_data_default():

    return split_synthetic_data(
        data_directory=configx.CONST_TEXT_OUTPUT_DIRECTORY,
        dataset_name=configx.CONST_TEXT_OUTPUT_PREFIX,
        data_file_prefix=configx.CONST_SENTENCE_FILE_PREFIX,
        data_file_type=configx.CONST_SENTENCE_FILE_SUFFIX,
        corpus_save_dir=configx.CONST_CORPUS_SAVE_DIRECTORY,
        source_language_dir=configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY,
        target_language_dir=configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)


def split_synthetic_data(
        data_directory,
        dataset_name,
        data_file_prefix,
        data_file_type,
        corpus_save_dir,
        source_language_dir,
        target_language_dir,
        language_dir=configx.CONST_DEFAULT_LANGUAGE_DIRECTORY):
    """
    Generate a dataset and accompanying languages

    Args:
        data_directory (str, optional): Directory containing datasets
        dataset_name (str, optional): Dataset folder name
        data_file_prefix (str, optional): Dataset text file prefix for
            rule subtypes
        data_file_type (str, optional): Datset text file type
        corpus_output_dir (str, optional): Output directory of dataset corpus
        source_language_dir (str, optional): Relative path of save directory
            for the source Language class instances
        target_language_dir (str, optional): Relative path of save directory
            for the target Language class instances
    """
    util.mkdir_p(corpus_save_dir)

    lengths_file = os.path.join(corpus_save_dir, 'split_lengths.csv')
    lengths_file = open(lengths_file, 'w+')

    # Obtain files for dataset
    data_files, start_files, rule_nodes, rule_counts, pair_counts = \
        load.get_dataset_files(
            data_directory, dataset_name, data_file_prefix, data_file_type)

    # Generate dataset
    dataset, dataset_starts, data_counts = load.generate_dataset(
        data_files, start_files, rule_counts, pair_counts)

    for k in data_counts.keys():

        st = [str(k)] + list(str(i) for i in data_counts[k])

        lengths_file.write(','.join(st) + os.linesep)

    lengths_file.close()

    print(configx.BREAK_LINE)
    print("Finished generating dataset...")

    train_data, validation_data, test_data, \
        full_validation, full_test = dataset
    train_starts, validation_starts, test_starts, \
        full_validation_starts, full_test_starts = dataset_starts

    concat_validation_data = list()
    for validation_rule in validation_data:
        concat_validation_data += validation_rule

    concat_test_data = list()
    for test_rule in test_data:
        concat_test_data += test_rule

    # # Uncomment this if running first time on new dataset
    # print("Updating source language")
    # # Generate and update source languages
    # token_tagger, pos_taggers = languages.load_languages(language_dir)

    # if not os.path.isdir(source_language_dir):
    #     util.mkdir_p(source_language_dir)

    # # Update languages with new tokens from training data
    # print("\tAdding data from train dataset")
    # update_languages.update_languages(
    #     token_tagger, pos_taggers, train_data, source_language_dir)

    # # Update languages with new tokens from validation data
    # print("\tAdding data from validation dataset")
    # update_languages.update_languages(
    #     token_tagger, pos_taggers, concat_validation_data, source_language_dir)

    # # Update languages with new tokens from test data
    # print("\tAdding data from test dataset")
    # update_languages.update_languages(
    #     token_tagger, pos_taggers, concat_test_data, source_language_dir)

    # print("Updating target language")
    # token_tagger, pos_taggers = languages.load_languages(language_dir)

    # if not os.path.isdir(target_language_dir):
    #     util.mkdir_p(target_language_dir)

    # # Update languages with new tokens from training data
    # print("\tAdding data from train dataset")
    # update_languages.update_languages(
    #     token_tagger, pos_taggers, train_data, target_language_dir, False)

    # # Update languages with new tokens from validation data
    # print("\tAdding data from validation dataset")
    # update_languages.update_languages(
    #     token_tagger, pos_taggers, concat_validation_data,
    #     target_language_dir, False)

    # # Update languages with new tokens from test data
    # print("\tAdding data from test dataset")
    # update_languages.update_languages(
    #     token_tagger, pos_taggers, concat_test_data, target_language_dir, False)

    # Location to save the corpus data
    if not os.path.isdir(corpus_save_dir):
        util.mkdir_p(corpus_save_dir)

    print("Finished updating languages")

    token_tagger_source, pos_taggers_source = \
        languages.load_languages(source_language_dir)
    token_tagger_target, pos_taggers_target = \
        languages.load_languages(target_language_dir)

    print("\nSaving training corpus data...")
    load.save_as_corpus(token_tagger_source, token_tagger_target,
                        train_data, train_starts, "train", corpus_save_dir)

    print("\nSaving full validation corpus data...")
    load.save_as_corpus(token_tagger_source, token_tagger_target,
                        full_validation, full_validation_starts,
                        "validation_full", corpus_save_dir)

    print("\nSaving full test corpus data...")
    load.save_as_corpus(token_tagger_source, token_tagger_target,
                        full_test, full_test_starts, "test_full",
                        corpus_save_dir)

    print('')
    for j in range(len(validation_data)):

        print("Saving validation data for rule: %d" % (j + 1))
        load.save_as_corpus(token_tagger_source, token_tagger_target,
                            validation_data[j], validation_starts[j],
                            "validation_" + str(j), corpus_save_dir,
                            rule_nodes=rule_nodes[j])

    print('')
    for j in range(len(test_data)):

        print("Saving test data for rule: %d" % (j + 1))
        load.save_as_corpus(token_tagger_source, token_tagger_target,
                            test_data[j], test_starts[j], "test_" + str(j),
                            corpus_save_dir, rule_nodes=rule_nodes[j])
