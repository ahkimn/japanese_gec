import os
import csv
from . import configx

def save_rule(corrected_sentence, error_sentence, paired_data, n):

	save_folder = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, configx.CONST_TEXT_OUTPUT_PREFIX)

	print("\t\tSave directory: %s" % save_folder)

	if not os.path.isdir(save_folder):
		os.mkdir(save_folder)

	# Create text-file named with rule format and containing rule text...
	rule_text = corrected_sentence + "ー" + error_sentence
	
	with open(os.path.join(save_folder, "rules.csv"), "a+") as f:

		csv_writer = csv.writer(f, delimiter=',')
		csv_writer.writerow([str(n), rule_text])
		f.close()

	rule_dir = os.path.join(save_folder, rule_text)

	if not os.path.isdir(rule_dir):
		os.mkdir(rule_dir)

	# Iterate over each subrule
	for i in range(len(paired_data)):

		file_name = configx.CONST_SENTENCE_FILE_PREFIX + str(i + 1) + configx.CONST_SENTENCE_FILE_SUFFIX
		
		with open(os.path.join(rule_dir, file_name), "w+") as f:

			csv_writer = csv.writer(f, delimiter=',')

			for j in range(len(paired_data[i])):

				if not "UNKNOWN" in paired_data[i][j][0] and not "UNKNOWN" in paired_data[i][j][1]:
					csv_writer.writerow([paired_data[i][j][0], paired_data[i][j][1]])

				else:
					print(paired_data[i][j][0])
					print(paired_data[i][j][1])
					print("=============")

		f.close()

