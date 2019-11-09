# Filename: load.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 16/07/2018
# Date Last Modified: 26/02/2019
# Python Version: 3.7

'''
Functions to generate datasets from rule-separated text files
'''

import os
import csv
import numpy as np

from . import configx
from . import languages
from . import update_languages
from . import util


def get_dataset_files(data_directory, dataset_name, data_file_prefix, data_file_type):
    """
    Function to load data from a given set of text files

    Args:
        data_directory (str): Directory containing datasets
        dataset_name (str): Dataset folder name
        data_file_prefix (str): Dataset text file prefix for rule subtypes
        data_file_type (str): Datset text file type

    Returns:
        data_files (arr): Array containing all data files associated with dataset
        sub_rule_counts (arr): Array containing number of sub-rules per each rule
        pair_counts (arr): Array of arrays containing number of sentence pairs per each sub-rule
    """
    data_files = list()
    start_files = list()
    sub_rule_counts = list()
    pair_counts = list()

    # Obtain directory of dataset
    base_dir = os.path.join(data_directory, dataset_name)

    rules = []

    with open(os.path.join(base_dir, "rules.csv"), "r") as f:

        reader = csv.reader(f, delimiter='f')

        for line in reader:

            rules.append(line[0].split(',')[1])

    delimiter = configx.CONST_SENTENCE_DELIMITER_TOKEN
    rule_nodes = list()

    idx = 1

    # Iterate through each of the rules within the dataset
    for sub_dir in rules:

        correct, error = sub_dir.split('|||')
        print('Rule %d: %s --> %s' % (idx, error, correct))

        nodes_correct, _ = languages.parse_full(
            correct, configx.CONST_PARSER, None)
        nodes_error, _ = languages.parse_full(
            error, configx.CONST_PARSER, None)

        rule_nodes.append((nodes_correct, nodes_error))
        s_dir = os.path.join(base_dir, sub_dir)

        if os.path.isdir(s_dir):

            try:

                rule_data_files = list()
                rule_pair_counts = list()

                # Rules should be in folders with integer names
                sub_rule_counts.append(0)

                # Check all files within rule directory
                for f in os.listdir(s_dir):

                    f_path = os.path.join(s_dir, f)

                    # Obtain files containing the correct file prefix and suffix
                    if os.path.isfile(f_path) and (data_file_prefix in f and data_file_type in f):

                        # print(f_path)

                        f_suffix = f[len(data_file_prefix):-
                                     len(data_file_type)]

                        # Read file and update rule counts
                        try:

                            j = int(f_suffix[:-1])
                            sub_rule_counts[-1] += 1

                            rule_data_files.append(f_path)

                            with open(f_path, "r") as g:

                                csv_reader = csv.reader(g)
                                row_count = sum(1 for row in csv_reader)
                                rule_pair_counts.append(row_count)

                        except ValueError:

                            continue

                    elif os.path.isfile(f_path) and (configx.CONST_STARTS_FILE_PREFIX in f and data_file_type in f):

                        start_files.append(f_path)

                data_files.append(rule_data_files)
                pair_counts.append(rule_pair_counts)

                print('\tNumber of Pairs: %d' % sum(rule_pair_counts))

            except ValueError:

                continue

            idx += 1

    return data_files, start_files, rule_nodes, sub_rule_counts, pair_counts


def generate_dataset(data_files, start_files, rule_counts, pair_counts, training_ratio=0.9, validation_ratio=0.01,
                     max_per_rule=25000, min_per_rule=2000, minimum_rule=250):
    """
    Function to generate a full dataset (training/validation/test) from a given set of data files

    Args:
        data_files (arr): Array containing all data files associated with dataset
        rule_counts (arr): Array containing number of sub-rules per each rule
        pair_counts (arr): Array of arrays containing number of sentence pairs per each sub-rule
        validation_ratio (float, optional): Percentage of sentences going to validation set (relative to test size)
        max_per_rule (int, optional): Maximum number of training samples per rule
        min_per_rule (int, optional): Minimum number of training samples per rule (all rules with fewer are discarded)

    Returns:
        train_data (arr): Pairs of sentences for training set (aggregate)
        validation_data (arr): Pairs of sentences for validation set (separated by rule)
        test_data (arr): Pairs of sentences for test set (separated by rule)
        full_validation (arr): Pairs of sentences for validation set (aggregate)
        full_test (arr): Piars of sentences for test set (aggregate)
    """
    n_rules = len(rule_counts)
    # Array containing set of rules (numbers) that pass the max/min criterion
    valid_indices = list(range(n_rules))
    # Obtain total number of sentence pairs per each rule
    rule_pair_counts = list(sum(pair_counts[i]) for i in range(n_rules))
    data_counts = dict()

    # Remove rules with too few pairs
    for j in range(n_rules):

        if rule_pair_counts[j] < minimum_rule:

            raise ValueError('Rule %d has %d / %d required pairs' %
                             (j + 1, rule_pair_counts[j], minimum_rule))

            valid_indices.remove(j)

    # Update input arrays to reflect valid rule indices
    data_files = list(data_files[i] for i in valid_indices)
    start_files = list(start_files[i] for i in valid_indices)
    rule_counts = list(rule_counts[i] for i in valid_indices)
    pair_counts = list(pair_counts[i] for i in valid_indices)

    n_valid_rules = len(valid_indices)

    # Two-dimensional list (t, 2) => t total pairs
    train_data = list()
    train_starts = list()

    # Three-dimensional list, (r, ~k, 2) => r rules; ~k pairs per rule
    validation_data = list()
    test_data = list()

    validation_starts = list()
    test_starts = list()

    # Aggregate validation/test data (all rules one array)
    # Two-dimensional lists (t, 2) => t total pairs
    full_validation = list()
    full_test = list()

    full_validation_starts = list()
    full_test_starts = list()

    for i in range(n_valid_rules):

        prev_length = len(train_data)

        total_sentences = sum(pair_counts[i])

        if total_sentences < min_per_rule:
            rule_max = min_per_rule

        else:

            rule_max = int((total_sentences - min_per_rule) / 2) \
                + min_per_rule

        if rule_max > max_per_rule:
            rule_max = max_per_rule

        print(total_sentences, rule_max)

        print("RULE: %d" % (i + 1))

        r_validation = list()
        r_test = list()

        r_validation_starts = list()
        r_test_starts = list()

        n_subrules = len(data_files[i])

        sentence_starts = list()

        with open(start_files[i], 'r') as f_start:
            reader = csv.reader(f_start)
            for row in reader:

                sentence_starts.append(list(int(i) for i in row))

        # Obtain a representative sample of data over sub-rules
        n_per = representative_sample(
            pair_counts[i], min(sum(pair_counts[i]), rule_max))

        for j in range(n_subrules):

            subrule_file = data_files[i][j]
            subrule_index = os.path.splitext(os.path.basename(subrule_file))[0]
            subrule_index = int(subrule_index.replace(
                configx.CONST_SENTENCE_FILE_PREFIX, '')) - 1

            try:

                data, starts = sample_from_csv(
                    subrule_file, sentence_starts[subrule_index], n_per[j],
                    training_ratio, validation_ratio, rule_max)
            except:
                continue

            train_data += data[0]
            train_starts += starts[0]

            r_validation += data[1]
            r_test += data[2]

            r_validation_starts += starts[1]
            r_test_starts += starts[2]

            full_validation += data[1]
            full_test += data[2]

            full_validation_starts += starts[1]
            full_test_starts += starts[2]

        validation_data.append(r_validation)
        test_data.append(r_test)

        validation_starts.append(r_validation_starts)
        test_starts.append(r_test_starts)

        print("Lengths: %d, %d, %d" %
              (len(train_data) - prev_length, len(r_validation), len(r_test)))

        data_counts[valid_indices[i] + 1] = \
            [len(train_data) - prev_length, len(r_validation), len(r_test)]

    data = (train_data, validation_data, test_data,
            full_validation, full_test)
    data_starts = (train_starts, validation_starts, test_starts,
                   full_validation_starts, full_test_starts)

    return data, data_starts, data_counts


def sample_from_csv(file_path, starts, count, training_ratio, validation_ratio, max_total=10000):
    """
    Sample count sentence (pairs) from a given CSV file

    Args:
        file_path (str): Path to file
        count (int): Number of sentences to sample
        validation_ratio (float): Percentage of sentences going to validation set (relative to training size) 
        max_total (int, optional): Maximum number of sentences to sample per file 

    Returns:
        d_train (arr): Sampled sentences for the training set
        d_validation (arr): Sampled sentences for the validation set
        d_test (arr): Sampled sentences for the test set
    """
    f = open(file_path, "r")

    csv_reader = csv.reader(f)
    rows = list(row for row in csv_reader)
    assert(len(rows) == len(starts))
    rs = np.random.RandomState(seed=0)
    perm = rs.permutation(len(rows))

    n_sample = min(count, max_total)

    n_training = int(count * training_ratio)
    n_validation = int(count * validation_ratio)
    n_test = count - n_training - n_validation

    f.close()

    d_train = list(rows[k] for k in perm[:n_training])
    d_validation = list(rows[l] for l in
                        perm[n_training:n_training + n_validation])
    d_test = list(rows[m] for m in
                  perm[n_training + n_validation:n_training + n_validation + n_test])

    s_train = list(starts[k] for k in perm[:n_training])
    s_validation = list(starts[l] for l in
                        perm[n_training:n_training + n_validation])
    s_test = list(starts[m] for m in
                  perm[n_training + n_validation:n_training + n_validation + n_test])

    return (d_train, d_validation, d_test), (s_train, s_validation, s_test)


def representative_sample(counts, max_total):
    """
    Generate counts per sub-rule that constitute a representative sample of a rule

    Args:
        counts (arr): Array containing number of sentence pairs per sub-rule 
        max_total (int): Maximum number of pairs to be sampled from this rule

    Returns:
        sample_counts (arr): Number of sentence pairs per sub-rule to be sampled
    """
    n_subrules = len(counts)
    sample_counts = [0] * n_subrules
    rs = np.random.RandomState(seed=0)

    # Array containing the sub-rules that still have sentences to sample from
    available_indices = list(range(n_subrules))

    while sum(sample_counts) < max_total:

        # Recalculate remaining number of pairs to sample
        remainder = max_total - sum(sample_counts)

        # Check if it is possible to sample from all remaining sub-rules
        if remainder >= len(available_indices):

            # Sample equal number of sentences per remaining sub-rules
            per_each = int(min(remainder / len(available_indices),
                               min(list(counts[j] for j in available_indices))))

            for j in available_indices:

                sample_counts[j] += per_each
                counts[j] -= per_each

        else:

            # Pick a subset of sub-rules to sample from
            perm = rs.permutation(len(available_indices))
            perm = perm[:remainder]

            for i in range(len(perm)):

                sample_counts[available_indices[perm[i]]] += 1
                counts[available_indices[perm[i]]] -= 1

        # Update remaining number of sentence pairs available for sampling per sub-rule
        for i in range(n_subrules):

            if i in available_indices and counts[i] == 0:

                available_indices.remove(i)

    return sample_counts


def save_as_corpus(source_tagger, target_tagger, sentence_list, starts_list,
                   prefix, directory, raw_text=True, bpe=False, min_length=3, rule_nodes=None,
                   print_every=10000):
    """
    Save dataset files as corpus text

    Args:
        source_tagger (Language): Source Language instance for tokens
        target_tagger (Language): Target Language instance for tokens
        sentence_list (arr): Tokens to encode
        prefix (str): Prefix of data type (i.e. train/test/validation)
        directory (str): Directory to store strings
    """

    # Make the directory if it does not exist already
    if not os.path.isdir(directory):
        util.mkdir_p(directory)

    error_file = os.path.join(
        directory, prefix + "_" + configx.CONST_ERRORED_PREFIX)
    correct_file = os.path.join(
        directory, prefix + "_" + configx.CONST_CORRECT_PREFIX)
    starts_file = os.path.join(
        directory, prefix + "_" + configx.CONST_STARTS_PREFIX)

    error_file = open(error_file, "w+")
    correct_file = open(correct_file, "w+")
    starts_file = open(starts_file, "w+")

    n_failed = 0

    for j in range(len(sentence_list)):

        error_sentence = sentence_list[j][0]
        correct_sentence = sentence_list[j][1]

        # Extract tokenized data for each sentence
        error_tokens, _ = languages.parse_full(
            error_sentence, configx.CONST_PARSER, None)
        correct_tokens, _ = languages.parse_full(
            correct_sentence, configx.CONST_PARSER, None)

        if j % print_every == 0:
            print('\tProcessed sentence %d / %d' % (j, len(sentence_list)))

        if bpe:

            # TODO
            pass

        else:

            if raw_text:

                err = error_tokens
                crt = correct_tokens

            else:

                # Convert tokens into integers
                err = source_tagger.parse_sentence(error_tokens)
                crt = target_tagger.parse_sentence(correct_tokens)

        if len(err) >= min_length and len(crt) >= min_length:

            # Write integer "sentences" into file
            error_file.write(" ".join(list(str(m) for m in err)))
            correct_file.write(" ".join(list(str(m) for m in crt)))
            starts_file.write(str(starts_list[j]))

            error_file.write(os.linesep)
            correct_file.write(os.linesep)
            starts_file.write(os.linesep)

        else:

            n_failed += 1

            # print(err)
            # print(crt)

    print('Number of failed sentences: %d' % n_failed)

    if rule_nodes is not None:

        rules_file = os.path.join(
            directory, prefix + "_" + configx.CONST_RULES_PREFIX)

        rules_file = open(rules_file, "w+")
        rules_file.write(",".join(list(' '.join(i) for i in rule_nodes)))
        rules_file.write(os.linesep)
        rules_file.write(",".join(list(str(len(i)) for i in rule_nodes)))
        rules_file.write(os.linesep)
