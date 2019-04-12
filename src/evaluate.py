from . import languages
from . import configx

from nltk import f_measure
import os

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

		text_ref = ref_lines[i].split(' ')
		text_sys = sys_lines[i].split(' ')

		indices_ref = list(tagger.parse_node(j, top_k) for j in text_ref)
		indices_sys = list(tagger.parse_node(j, top_k) for j in text_sys)

		indices_ref = list(k for k in indices_ref if k != tagger.unknown_index)
		indices_sys = list(k for k in indices_sys if k != tagger.unknown_index)

		val = f_measure(set(indices_ref), set(indices_sys), alpha=alpha)

		if val is None:

			n_lines -= 1

		else:

			ret += val


	ret /= n_lines

	return ret

