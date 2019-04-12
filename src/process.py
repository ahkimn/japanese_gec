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
				print(pair)

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

	print(output_source)

	for i in range(len(source_text)):

		source_file.write(source_text[i])
		source_file.write(os.linesep)

		target_file.write(target_text[i])
		target_file.write(os.linesep)

	source_file.close()
	target_file.close()