import sys
import src.process as process
import src.evaluate as evaluate
import src.convert as convert

def process_csv(input_file, output_source, output_target):

	process.pre_process_csv(input_file, output_source, output_target)

def remove_pairs(input_source, input_target, output_source, output_target, same):

	print(same)

	process.remove_pairs(input_source, input_target, output_source, output_target, bool(int(same)))

def sort_sentences(input_file, output_file):

	process.sort_sentences(input_file, output_file)

def eval_f(reference_file, system_file, top_k, alpha):

	ret = evaluate.eval_f(reference_file, system_file, int(top_k), float(alpha))
	print(ret)

def eval_binary(reference_file, system_file, top_k):

	ret = evaluate.eval_binary(reference_file, system_file, int(top_k))
	print(ret)

def find_semantic_pairs():

	convert.find_semantic_pairs()

def main():

	# n_files = -1
	# w2v.construct_default_model(n_files=n_files)

	# w2v.interactive()
	# languages.compile_default_languages(n_files=n_files)
	# database.construct_default_database(n_files=n_files)
	# database.clean_default_database(max_length = 30)

	# convert.convert_csv_rules(n_max=10000, n_search=1000000)

	# languages.load_default_languages()
	# load.save_dataset()

	pass

if __name__ == '__main__':

   # Accept input in form of (function_name, param1, param2, ..., paramX)
   params = []

   if len(sys.argv) > 2:

      for x in range(2, len(sys.argv)):

         params.append(sys.argv[x])

   if len(params) > 0:

      globals()[sys.argv[1]](*params)

   else:

      globals()[sys.argv[1]]()