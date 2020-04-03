# -*- coding: utf-8 -*-

# Filename: sample.py
# Date Created: 29/03/2020
# Description: Functions to sample data from a Dataset class instance
# Python Version: 3.7

import numpy as np

from numpy.random import RandomState
from statistics import median


def balanced_subrule_sample(subrule_counts: list, max_total: int=-1,
                            RS: RandomState=None):
    """
    Generate sentence counts per subrule that constitute a representative
        sample of a rule

    Weights each sub-rule equally and incrementally adds to each subrule's
        count until %max_total% is reached or all subrules are completely
        sampled

    Args:
        subrule_counts (list): Array containing number of sentence pairs per
            each subrule
        max_total (int): Maximum number of sentences to be sampled from
            this rule. If negative, all sentences are sampled

    Returns:
        sample_counts (arr): Number of sentence pairs per sub-rule to
            be sampled
    """
    n_subrules = len(subrule_counts)
    sample_counts = [0] * n_subrules

    if RS is None:
        RS = np.random.RandomState(seed=0)

    n_sentences = sum(subrule_counts)

    if max_total < 0:
        max_total = n_sentences

    else:
        max_total = min(n_sentences, max_total)

    # Array containing the sub-rules that still have sentences to sample from
    available_subrules = list(range(n_subrules))

    while sum(sample_counts) < max_total:

        # Recalculate remaining number of pairs to sample
        remainder = max_total - sum(sample_counts)

        # Check if it is possible to sample from all remaining sub-rules
        if remainder >= len(available_subrules):

            # Sample equal number of sentences per remaining sub-rules
            per_each = int(min(remainder / len(available_subrules),
                               min(list(subrule_counts[j] for j in
                                        available_subrules))))

            for j in available_subrules:

                sample_counts[j] += per_each
                subrule_counts[j] -= per_each

        else:

            # Pick a subset of sub-rules to sample from
            perm = RS.permutation(len(available_subrules))
            perm = perm[:remainder]

            for i in range(len(perm)):

                sample_counts[available_subrules[perm[i]]] += 1
                subrule_counts[available_subrules[perm[i]]] -= 1

        # Update remaining number of sentence pairs available for
        #   sampling per sub-rule
        for i in range(n_subrules):

            if i in available_subrules and subrule_counts[i] == 0:

                available_subrules.remove(i)

    return sample_counts


def split_subrule_count(n: int, train_ratio: float, dev_ratio: float):

    if n > 2:

        n_train = max(int(n * train_ratio), 1)
        n_dev = max(int(n * dev_ratio), 1)
        n_test = n - (n_train + n_dev)

        if n_test == 0:

            if n_train > n_dev:

                assert(n_train > 1)
                n_train -= 1

            else:

                assert(n_dev > 1)
                n_dev -= 1

            n_test = 1

        assert(n_train + n_dev + n_test == n)
        return n_train, n_dev, n_test

    elif n == 2:

        return 1, 0, 1

    elif n == 1:

        return 1, 0, 0


def balanced_rule_sample(rule_names: list, rule_counts: list,
                         max_per_rule: int=-1, min_per_rule: int=0,
                         sample_function=None):
    """
    Generate sentence counts per rule that constitute a representative
        sample of a dataset

    Args:
        rule_counts (list): Array containing number of sentence pairs per
            each rule
        max_per_rule (int, optional): Maximum number of sentences to sample
            per rule. If negative all sentences can be sampled.
        min_per_rule (int, optional): Minimum number of sentences to sample
            per rule. If the count of a rule falls below this, it is not
            sampled
        sample_function (float, optional): If not None, apply this function to
            rule count prior to applying bounds %max_per_rule% and
            %min_per_rule%
    """

    n_rules = len(rule_counts)

    sample_counts = [0] * n_rules

    if max_per_rule < 0:

        max_per_rule = median(rule_counts)

    if sample_function is not None:

        rule_counts = [sample_function(count, min_per_rule)
                       for count in rule_counts]

    for i in range(n_rules):

        count = min(rule_counts[i], max_per_rule)

        if count >= min_per_rule:

            sample_counts[i] = count

        else:

            print('Warning: rule %s will not be sampled' % rule_names[i])
            print('Contains %d/%d sentences required' % (count, min_per_rule))

    return sample_counts


def linear_sampler(count, floor, ratio=0.25):

    if count < floor:

        return count

    out = int(count * ratio)

    return max(out, floor)
