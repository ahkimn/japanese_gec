import os

from . import configx
from . import languages

def pre_process_csv(input_file, output_source, output_target):

	source_text = list()
	target_text = list()

	with open(input_file, "r", encoding="utf-8") as in_file:

		for pair in in_file.readlines():

			pair = pair.strip().split(',')

			source_sentence = pair[0]

			if len(pair) == 1:

				target_sentence = ''

			elif len(pair) == 2:

				target_sentence = pair[1]

			else:

				print("WARNING: MORE THAN TWO SENTENCES IN LINE")
				target_sentence = pair[1]

			source_sentence = source_sentence.strip()
			target_sentence = target_sentence.strip()

			if target_sentence == '':
				target_sentence = source_sentence


			source_tokens = languages.parse(source_sentence, configx.CONST_PARSER, None)
			target_tokens = languages.parse(target_sentence, configx.CONST_PARSER, None)

			source_text.append(" ".join(source_tokens))
			target_text.append(" ".join(target_tokens))

		in_file.close()

	source_file = open(output_source, "w+")
	target_file = open(output_target, "w+")

	for i in range(len(source_text)):

		source_file.write(source_text[i])
		source_file.write(os.linesep)

		target_file.write(target_text[i])
		target_file.write(os.linesep)

	source_file.close()
	target_file.close()

def sort_sentences(input_file, output_file):

	ret = []

	with open(input_file, "r") as f:

		data = f.readlines()

		n_sentences = len(data)

		ret = [None] * n_sentences

		for sentence in data:

			tokens = sentence.split()

			ini = tokens[0]
			index = int(ini[2:])

			assert(index < n_sentences)

			# Hypothesis
			if ini[0] == 'H':

				tokens = tokens[2:]

			else:

				tokens = tokens[1:]

			ret[index] = " ".join(tokens).strip()

	f.close()

	g = open(output_file, "w+")

	for i in range(n_sentences):

		g.write(ret[i])
		g.write(os.linesep)

	g.close()

def remove_pairs(original_source, original_target, output_source, output_target, same):

	print(same)

	original_source = open(original_source, "r")
	original_target = open(original_target, "r")

	s_lines = original_source.readlines()
	t_lines = original_target.readlines()

	assert(len(s_lines) == len(t_lines))

	new_source = list()
	new_target = list()

	for i in range(len(s_lines)):

		if (s_lines[i] == t_lines[i]) == same:

			pass

		else:

			new_source.append(s_lines[i])
			new_target.append(t_lines[i])

	original_source.close()
	original_target.close()

	output_source = open(output_source, "w+")
	output_target = open(output_target, "w+")

	output_source.writelines(new_source)
	output_target.writelines(new_target)

	output_source.close()
	output_target.close()

