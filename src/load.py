import os
import csv
import numpy as np

from . import languages
from . import configx

# Function to grab dataset files
def get_dataset_files(data_directory, dataset_name, data_file_prefix, data_file_type):
'''
    :param: data_directory - relative path of directory to generate
'''

    ret = list()
    rule_counts = list()
    pair_counts = list()
    base_dir = os.path.join(data_directory, dataset_name)

    for sub_dir in os.listdir(base_dir):

        s_dir = os.path.join(base_dir, sub_dir)

        if os.path.isdir(s_dir):

            try:

                rule_ret = list()
                rule_pair_counts = list()

                i = int(sub_dir)
                rule_counts.append(0)

                for f in os.listdir(s_dir):

                    f_path = os.path.join(s_dir, f)

                    if os.path.isfile(f_path) and (data_file_prefix in f and data_file_type in f):

                        f_suffix = f[len(data_file_prefix):-len(data_file_type)]

                        try:

                            j = int(f_suffix)
                            rule_counts[-1] += 1

                            rule_ret.append(f_path)

                            with open(f_path, "r") as g:

                                csv_reader = csv.reader(g)
                                row_count = sum(1 for row in csv_reader)
                                rule_pair_counts.append(row_count)

                        except ValueError:

                            continue

                ret.append(rule_ret)
                pair_counts.append(rule_pair_counts)

            except ValueError:

                continue

    return ret, rule_counts, pair_counts

def generate_dataset(data_files, rule_counts, pair_counts, validation_ratio=0.1, max_per_rule=1000, min_per_rule=500):

    n_rules = len(rule_counts)
    valid_indices = list(range(n_rules))
    rule_pair_counts = list(sum(pair_counts[i]) for i in range(n_rules))

    # Remove rules with too few pairs
    for j in range(n_rules):

        if rule_pair_counts[j] < min_per_rule:

            valid_indices.remove(j)

    data_files = list(data_files[i] for i in valid_indices)
    rule_counts = list(rule_counts[i] for i in valid_indices)
    pair_counts = list(pair_counts[i] for i in valid_indices)

    n_valid_rules = len(valid_indices)

    # Two-dimensional list (t, 2) => t total pairs
    train_data = list()

    # Three-dimensional list, (r, ~k, 2) => r rules; ~k pairs per rule
    validation_data = list()
    test_data = list()

    full_validation = list()
    full_test = list()

    for i in range(n_valid_rules):

        r_validation = list()
        r_test = list()

        n_subrules = len(data_files[i])

        n_per = representative_sample(pair_counts[i], min(max_per_rule, sum(pair_counts[i])))

        for j in range(n_subrules):

            d_train, d_validation, d_test = sample_from_csv(data_files[i][j], n_per[j], validation_ratio)

            train_data += d_train

            r_validation += d_validation
            r_test += d_test

            full_validation += d_validation
            full_test += d_test

        validation_data.append(r_validation)
        test_data.append(r_test)

    return train_data, validation_data, test_data, full_validation, full_test


def sample_from_csv(file_path, count, validation_ratio, max_total=5000):

    with open(file_path, "r") as f:

        csv_reader = csv.reader(f)
        rows = list(row for row in csv_reader)

        perm = np.random.permutation(len(rows))

        n_validation = int(min(len(rows) - count, max_total) * validation_ratio)
        n_test = int(min(len(rows) - count, max_total) * (1 - validation_ratio))

        f.close()

        return list(rows[k] for k in perm[:count]), \
               list(rows[l] for l in perm[count:count + n_validation]), \
               list(rows[m] for m in perm[count + n_validation:count + n_validation + n_test])

def representative_sample(counts, max_total):

    n_subrules = len(counts)

    available_indices = list(range(n_subrules))
    sample_counts = [0] * n_subrules

    while sum(sample_counts) < max_total:

        remainder = max_total - sum(sample_counts)

        if remainder >= len(available_indices):

            per_each = int(min(remainder / len(available_indices), min(list(counts[j] for j in available_indices))))

            for j in available_indices:

                sample_counts[j] += per_each
                counts[j] -= per_each

        else:

            perm = np.random.permutation(len(available_indices))
            perm = perm[:remainder]

            for i in range(len(perm)):

                sample_counts[available_indices[perm[i]]] += 1
                counts[available_indices[perm[i]]] -= 1

        for i in range(n_subrules):

            if i in available_indices and counts[i] == 0:

                available_indices.remove(i)

    return sample_counts


def update_languages(token_tagger, pos_taggers, sentence_list, save_dir, source=True):

    for j in range(len(sentence_list)):

        if source:

            # Take only errored sentence
            error_sentence = sentence_list[j][0]

            tokens, pos_tags = languages.parse_sentence(error_sentence, configx.CONST_PARSER, None)

            # Update token tagger
            token_tagger.add_sentence(tokens)

            for k in range(len(pos_taggers)):

                pos_taggers[k].add_sentence(pos_tags[k])

        else:

            # Otherwise take only correct sentence
            correct_sentence = sentence_list[j][1]

            tokens, pos_tags = languages.parse_sentence(correct_sentence, configx.CONST_PARSER, None)

            # Update token tagger
            token_tagger.add_sentence(tokens)

            for k in range(len(pos_taggers)):

                pos_taggers[k].add_sentence(pos_tags[k])

    token_prefix = os.path.join(save_dir, configx.CONST_NODE_PREFIX)
    pos_prefix = os.path.join(save_dir, configx.CONST_POS_PREFIX)

    # Save updated tokken tagger
    token_tagger.sort()
    token_tagger.save_dicts(token_prefix)

    # Save updated pos_taggers
    for k in range(len(pos_taggers)):

        pos_taggers[k].sort()
        pos_taggers[k].save_dicts(pos_prefix + str(k))

def save_as_corpus(source_tagger, target_tagger, sentence_list, prefix, directory):

    # Make the directory if it does not exist already
    if not os.path.isdir(directory):
        os.mkdir(directory)

    error_data = []
    correct_data = []

    for j in range(len(sentence_list)):

        error_sentence = sentence_list[j][0]
        correct_sentence = sentence_list[j][1]

        # Extract tokenized data for each sentence
        error_tokens, _ = languages.parse_sentence(error_sentence, configx.CONST_PARSER, None)
        correct_tokens, _ = languages.parse_sentence(correct_sentence, configx.CONST_PARSER, None)

        # Convert tokens into integers
        error_data.append(source_tagger.parse_sentence(error_tokens))
        correct_data.append(target_tagger.parse_sentence(correct_tokens))
    
    assert(len(error_data) == len(correct_data))

    error_file = os.path.join(directory, prefix + "_" + configx.CONST_ERRORED_PREFIX)
    correct_file = os.path.join(directory, prefix + "_" + configx.CONST_CORRECT_PREFIX)

    error_file = open(error_file, "w+")
    correct_file = open(correct_file, "w+")

    for j in range(len(error_data)):

        # Write integer "sentences" into file
        error_file.write(" ".join(list(str(m) for m in error_data[j])))
        correct_file.write(" ".join(list(str(m) for m in correct_data[j])))

        error_file.write(os.linesep)
        correct_file.write(os.linesep)
    
def generate_languages_from_data(data_directory = configx.CONST_TEXT_OUTPUT_DIRECTORY,
                                 dataset_name = configx.CONST_TEXT_OUTPUT_PREFIX,
                                 data_file_prefix = configx.CONST_SENTENCE_FILE_PREFIX,
                                 data_file_type= configx.CONST_SENTENCE_FILE_SUFFIX):

    # Location to save the corpus data
    corpus_save_dir = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, 
                                   configx.CONST_TEXT_OUTPUT_PREFIX, 
                                   configx.CONST_CORPUS_SAVE_DIRECTORY)
  
    if not os.path.isdir(corpus_save_dir):
        mkdir_p(corpus_save_dir)

    data_files, rule_counts, pair_counts = get_dataset_files(data_directory, 
                                                             dataset_name, 
                                                             data_file_prefix, 
                                                             data_file_type)

    raise

    train_data, validation_data, test_data, full_validation, full_test = generate_dataset(data_files, rule_counts, pair_counts)

 


    '''
    Run code below if languages have not yet been created with tokens from test_data
    '''
    token_tagger = languages.Language(True)
    pos_taggers = [languages.Language(), languages.Language(), languages.Language(), languages.Language(), languages.Language(True)]

    if not os.path.isdir(configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY):
        os.mkdir(configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)

    # Update languages with new tokens from training data
    update_languages(token_tagger, pos_taggers, train_data, configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)

    # Update languages with new tokens from validation data
    for validation_rule in validation_data:
        update_languages(token_tagger, pos_taggers, validation_rule, configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)

    for test_rule in test_data:
        update_languages(token_tagger, pos_taggers, validation_rule, configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)

    token_tagger = languages.Language(True)
    pos_taggers = [languages.Language(), languages.Language(), languages.Language(), languages.Language(), languages.Language(True)]

    if not os.path.isdir(configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY):
        os.mkdir(configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)


    # Update languages with new tokens from training data
    update_languages(token_tagger, pos_taggers, train_data, configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY, False)

    # Update languages with new tokens from validation data
    for validation_rule in validation_data:
        update_languages(token_tagger, pos_taggers, validation_rule, configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY, False)

    for test_rule in test_data:
        update_languages(token_tagger, pos_taggers, validation_rule, configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY, False)

def save_numeric_data():

      # Location to save the corpus data
    corpus_save_dir = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, configx.CONST_TEXT_OUTPUT_PREFIX, configx.CONST_CORPUS_SAVE_DIRECTORY)
  
    data_files, rule_counts, pair_counts = get_dataset_files()
    train_data, validation_data, test_data, full_validation, full_test = generate_dataset(data_files, rule_counts, pair_counts)


    token_tagger_source, pos_taggers_source = languages.load_default_languages(configx.CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY)
    token_tagger_target, pos_taggers_target = languages.load_default_languages(configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY)
    save_as_corpus(token_tagger_source, token_tagger_target, train_data, "train", corpus_save_dir)
    save_as_corpus(token_tagger_source, token_tagger_target, full_validation, "validation_full", corpus_save_dir)
    save_as_corpus(token_tagger_source, token_tagger_target, full_test, "test_full", corpus_save_dir)

    print(len(validation_data))

    for j in range(len(validation_data)):
        save_as_corpus(token_tagger_source, token_tagger_target, validation_data[j], "validation_" + str(j), corpus_save_dir)

    for j in range(len(test_data)):
        save_as_corpus(token_tagger_source, token_tagger_target, test_data[j], "test_" + str(j), corpus_save_dir)


# Function to recursively generate directories
def mkdir_p(path):
'''
    :param: path - relative path of directory to generate
'''
    try:

        os.makedirs(path)

    except OSError as exc:

        if exc.errno == errno.EEXIST and os.path.isdir(path):

            pass

        else:

            raise


if __name__ == '__main__':

    generate_languages_from_data()
    save_numeric_data()



