import os
import csv
from . import configx

def save_rule(corrected_sentence, error_sentence, correct_examples,  error_examples, index, rule_text, sub_folder=configx.CONST_TEXT_OUTPUT_PREFIX):

	save_folder = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, sub_folder)

	if not os.path.isdir(save_folder):
		os.mkdir(save_folder)

	rule_dir = os.path.join(save_folder, str(index))

	if not os.path.isdir(rule_dir):
		os.mkdir(rule_dir)

	# Create text-file named with rule format and containing rule text...
	rule_text_file = corrected_sentence + "・・・" + error_sentence + ".txt"
	with open(os.path.join(rule_dir, rule_text_file), "w+") as f:

		csv_writer = csv.writer(f, delimiter=',')
		csv_writer.writerow(rule_text)
		f.close()

	assert(len(error_examples) == len(correct_examples))

	for i in range(len(error_examples)):

		assert(len(error_examples[i]) == len(correct_examples[i]))
		file_name = configx.CONST_TEXT_OUTPUT_PREFIX + str(i + 1) + ".txt"
		
		with open(os.path.join(rule_dir, file_name), "w+") as f:

			csv_writer = csv.writer(f, delimiter=',')

			for j in range(len(error_examples[i])):

				if not "UNKNOWN" in error_examples[i][j] and not "UNKNOWN" in correct_examples[i][j]:
					csv_writer.writerow([error_examples[i][j], correct_examples[i][j]])

				# else:
				# 	print(error_examples[i][j])
				# 	print(correct_examples[i][j])
				# 	print("=============")

		f.close()