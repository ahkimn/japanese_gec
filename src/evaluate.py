from . import languages
from . import configx

from nltk import f_measure
import os

def eval_binary(ref, sys, top_k):

	target_language_dir = configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY

	tagger, _ = languages.load_default_languages(target_language_dir)

	ref = open(ref, "r")
	sys = open(sys, "r")

	ref_lines = ref.readlines()
	sys_lines = sys.readlines()

	assert(len(ref_lines) == len(sys_lines))

	n_lines = len(ref_lines)
	ret = 0

	for i in range(n_lines):

		text_ref = ref_lines[i].replace("\n", "").split(' ')
		text_sys = sys_lines[i].replace("\n", "").split(' ')

		indices_ref = list(tagger.parse_node(j, top_k) for j in text_ref)
		indices_sys = list(tagger.parse_node(j, top_k) for j in text_sys)

		indices_ref = list(str(k) for k in indices_ref if k != tagger.unknown_index)
		indices_sys = list(str(k) for k in indices_sys if k != tagger.unknown_index)

		if len(indices_ref) == 0:

			n_lines -= 1

		else:

			if ",".join(indices_ref) == ",".join(indices_sys):

				ret += 1

	ret /= n_lines

	return ret

def eval_f(ref, sys, top_k=10000, alpha=0.5):

	target_language_dir = configx.CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY

	tagger, _ = languages.load_default_languages(target_language_dir)

	ref = open(ref, "r")
	sys = open(sys, "r")

	ref_lines = ref.readlines()
	sys_lines = sys.readlines()

	assert(len(ref_lines) == len(sys_lines))

	n_lines = len(ref_lines)
	ret = 0

	for i in range(n_lines):

		text_ref = ref_lines[i].replace("\n", "").split(' ')
		text_sys = sys_lines[i].replace("\n", "").split(' ')

		indices_ref = list(tagger.parse_node(j, top_k) for j in text_ref)
		indices_sys = list(tagger.parse_node(j, top_k) for j in text_sys)

		indices_ref = list(k for k in indices_ref if k != tagger.unknown_index)
		indices_sys = list(k for k in indices_sys if k != tagger.unknown_index)

		# Error rate
		if alpha == -1:

			incorrect = 0.0

			n_ref = len(indices_ref)

			for j in range(n_ref):

				if j >= len(indices_sys):

					incorrect += 1.0

				else:

					if indices_ref[j] != indices_sys[j]:

						incorrect += 1.0

			if n_ref == 0:

				val = None

			else:

				val = incorrect / n_ref

		else:	

			val = f_measure(set(indices_ref), set(indices_sys), alpha=alpha)

		if val is None:

			n_lines -= 1

		else:

			ret += val


	ret /= n_lines

	return ret

