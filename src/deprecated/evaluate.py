import os
import sys
import nltk.translate as translate
import numpy

import languages
import configx


def main(rule_no, analysis):

	load_dir = os.path.join(configx.CONST_SMT_OUTPUT_DIRECTORY, configx.CONST_TRANSLATE_OUTPUT_PREFIX)
	
	output_file = os.path.join(load_dir, "test_" + rule_no + "_translated")
	truth_file = os.path.join(load_dir, "test_" + rule_no + "_correct")

	o = open(output_file, "r")
	t = open(truth_file, "r")

	if True:

		hypotheses = o.readlines()
		references = t.readlines()

		h = list()
		r = list()

		assert(len(hypotheses) == len(references))

		for i in range(len(hypotheses)):

			h.append(hypotheses[i].strip().split(' '))

			temp = list()
			temp.append(references[i].strip().split(' '))

			r.append(temp)

		hypotheses = h
		references = r

		if analysis.lower() == "bleu":

			score = translate.bleu_score.corpus_bleu(references, hypotheses)
			print("Corpus %s Score: %8f" % (analysis.upper(), score))

		if analysis.lower() == "precision1":

			scores = list(translate.bleu_score.modified_precision(references[i], hypotheses[i], 1) for i in range(len(hypotheses)))
			print("Corpus %s Score: %8f" % (analysis.upper(), sum(scores) / len(scores) + .0))

		if analysis.lower() == "precision2":

			scores = list(translate.bleu_score.modified_precision(references[i], hypotheses[i], 2) for i in range(len(hypotheses)))
			print("Corpus %s Score: %8f" % (analysis.upper(), sum(scores) / len(scores) + .0))

		if analysis.lower() == 'all-or-nothing':

			scores = list(references[i][0] == hypotheses[i] for i in range(len(hypotheses)))
			print("Corpus %s Score: %8f" % (analysis.upper(), sum(scores) / len(scores) + .0))

	o.close()
	t.close()


def text():

	token_tagger, _ = languages.load_default_languages(configx.CONST_UPDATED_LANGUAGE_DIRECTORY)

	load_dir = os.path.join(configx.CONST_SMT_OUTPUT_DIRECTORY, configx.CONST_TRANSLATE_OUTPUT_PREFIX)
	save_dir = os.path.join(configx.CONST_SMT_OUTPUT_DIRECTORY, configx.CONST_TRANSLATE_TEXT_PREFIX)

	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	for file_path in os.listdir(load_dir):

		if not ".dec_" in file_path:

			load_file = os.path.join(load_dir, file_path)
			save_file = os.path.join(save_dir, file_path)

			load_file = open(load_file, "r")
			save_file = open(save_file, "w+")

			lines = load_file.readlines()
			lines = list(textify(line.strip(), token_tagger) + os.linesep for line in lines)

			save_file.writelines(lines)

			load_file.close()
			save_file.close()

def textify(line, token_tagger):

	line = list(int(i) for i in line.split(" "))
	line = token_tagger.sentence_from_indices(line)
	return line





if __name__ == '__main__':

	arguments = sys.argv
	assert(len(sys.argv) == 3)



	main(sys.argv[1], sys.argv[2])