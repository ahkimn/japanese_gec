
import numpy as np
import os


def sample_lines(corpus_name, input_prefix, output_prefix, n_sample=10000):

    base_dir = os.path.join('input', corpus_name)
    src_file = os.path.join(base_dir, '%s.source' % input_prefix)
    tgt_file = os.path.join(base_dir, '%s.target' % input_prefix)
    start_file = os.path.join(base_dir, '%s.start' % input_prefix)

    out_src = os.path.join(base_dir, '%s.source' % output_prefix)
    out_tgt = os.path.join(base_dir, '%s.target' % output_prefix)
    out_start = os.path.join(base_dir, '%s.start' % output_prefix)

    src_lines = open(src_file, 'r').readlines()
    tgt_lines = open(tgt_file, 'r').readlines()
    start_lines = open(start_file, 'r').readlines()

    n_lines = len(src_lines)
    assert(len(tgt_lines) == n_lines)
    assert(len(start_lines) == n_lines)

    perm = np.random.permutation(n_lines)[:n_sample]

    out_src_lines = list()
    out_tgt_lines = list()
    out_start_lines = list()

    for i in perm:

        out_src_lines.append(src_lines[i])
        out_tgt_lines.append(tgt_lines[i])
        out_start_lines.append(start_lines[i])

    open(out_src, 'w+').writelines(out_src_lines)
    open(out_tgt, 'w+').writelines(out_tgt_lines)
    open(out_start, 'w+').writelines(out_start_lines)