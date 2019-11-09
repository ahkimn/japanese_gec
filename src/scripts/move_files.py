import os
import src.util as util
from shutil import copyfile


def mv_input(corpus_name):

    src_dir = os.path.join('corpus', corpus_name)
    dst_dir = os.path.join('input', corpus_name)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    copyfile(os.path.join(src_dir, "train_error"),
             os.path.join(dst_dir, "train.source"))
    copyfile(os.path.join(src_dir, "train_correct"),
             os.path.join(dst_dir, "train.target"))
    copyfile(os.path.join(src_dir, "train_start"),
             os.path.join(dst_dir, "train.start"))
    copyfile(os.path.join(src_dir, "validation_full_error"),
             os.path.join(dst_dir, "validation.source"))
    copyfile(os.path.join(src_dir, "validation_full_correct"),
             os.path.join(dst_dir, "validation.target"))
    copyfile(os.path.join(src_dir, "validation_full_start"),
             os.path.join(dst_dir, "validation.start"))
    copyfile(os.path.join(src_dir, "test_full_error"),
             os.path.join(dst_dir, "test.source"))
    copyfile(os.path.join(src_dir, "test_full_correct"),
             os.path.join(dst_dir, "test.target"))
    copyfile(os.path.join(src_dir, "test_full_start"),
             os.path.join(dst_dir, "test.start"))

    k = 0

    while (os.path.isfile(os.path.join(src_dir, "test_" + str(k) + "_error"))):

        for x in ["test", "validation"]:

            source_file = os.path.join(src_dir, x + "_" + str(k) + "_error")
            target_file = os.path.join(src_dir, x + "_" + str(k) + "_correct")
            start_file = os.path.join(src_dir, x + "_" + str(k) + "_start")
            rule_file = os.path.join(src_dir, x + "_" + str(k) + "_rule")

            test_location = os.path.join(dst_dir, str(k + 1))

            if not os.path.isdir(test_location):

                os.mkdir(test_location)

            copyfile(source_file, os.path.join(test_location, x + ".source"))
            copyfile(target_file, os.path.join(test_location, x + ".target"))
            copyfile(start_file, os.path.join(test_location, x + ".start"))
            copyfile(rule_file, os.path.join(test_location, x + ".rule"))

        k += 1


def mv_output(corpus_name, file_prefix):

    src_dir = os.path.join('input', corpus_name)
    dst_dir = os.path.join('comparison', corpus_name)

    k = 1

    while (os.path.isdir(os.path.join(src_dir, str(k)))):

        current_dir = os.path.join(src_dir, str(k))
        current_output_dir = os.path.join(dst_dir, str(k))

        source_file = os.path.join(current_dir, '%s.source' % file_prefix)
        target_file = os.path.join(current_dir, '%s.target' % file_prefix)
        start_file = os.path.join(current_dir, '%s.start' % file_prefix)
        rule_file = os.path.join(current_dir, '%s.rule' % file_prefix)

        if not os.path.isdir(current_output_dir):

            util.mkdir_p(current_output_dir)

        out_src = os.path.join(current_output_dir, '%s.source' % file_prefix)
        out_tgt = os.path.join(current_output_dir, '%s.target' % file_prefix)
        out_sys = os.path.join(current_output_dir, '%s.start' % file_prefix)
        out_rule = os.path.join(current_output_dir, '%s.rule' % file_prefix)

        copyfile(source_file, out_src)
        copyfile(target_file, out_tgt)
        copyfile(start_file, out_sys)
        copyfile(rule_file, out_rule)

        k += 1

