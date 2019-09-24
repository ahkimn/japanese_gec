import os
import csv
from . import configx

# Function saving rule files


def save_rule(corrected_sentence, error_sentence, paired_data, start_indices, n):

    if not os.path.isdir(configx.CONST_TEXT_OUTPUT_DIRECTORY):
        os.mkdir(configx.CONST_TEXT_OUTPUT_DIRECTORY)

    save_folder = os.path.join(
        configx.CONST_TEXT_OUTPUT_DIRECTORY, configx.CONST_TEXT_OUTPUT_PREFIX)

    print("\t\tSave directory: %s" % save_folder)

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # Create text-file named with rule format and containing rule text...
    rule_text = corrected_sentence + "|||" + error_sentence

    with open(os.path.join(save_folder, "rules.csv"), "a+") as f:

        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow([str(n), rule_text])
        f.close()

    rule_dir = os.path.join(save_folder, rule_text)

    if not os.path.isdir(rule_dir):
        os.mkdir(rule_dir)

    file_starts_name = configx.CONST_STARTS_FILE_PREFIX + \
        configx.CONST_SENTENCE_FILE_SUFFIX
    f_start = open(os.path.join(rule_dir, file_starts_name), "w+")
    csv_writer_starts = csv.writer(f_start, delimiter=',')

    # Iterate over each subrule
    for i in range(len(paired_data)):

        valid_start_indices = list()

        file_name = configx.CONST_SENTENCE_FILE_PREFIX + \
            str(i + 1) + configx.CONST_SENTENCE_FILE_SUFFIX

        with open(os.path.join(rule_dir, file_name), "w+") as f:

            csv_writer = csv.writer(f, delimiter=',')

            for j in range(len(paired_data[i])):

                if not "UNKNOWN" in paired_data[i][j][0] and not "UNKNOWN" in paired_data[i][j][1]:
                    csv_writer.writerow(
                        [paired_data[i][j][0], paired_data[i][j][1]])

                    valid_start_indices.append(start_indices[i][j])
                else:
                    print(paired_data[i][j][0])
                    print(paired_data[i][j][1])
                    print("=============")

        f.close()

        csv_writer_starts.writerow(valid_start_indices)
    f_start.close()
