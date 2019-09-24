import ast
import sys
import src.process as process
import src.evaluate as evaluate
import src.convert as convert
import src.embedding as embedding
import src.load as load


def process_csv(input_file, output_source, output_target):

    process.pre_process_csv(input_file, output_source, output_target)


def remove_pairs(input_source, input_target, output_source, output_target, same):

    process.remove_pairs(input_source, input_target,
                         output_source, output_target, bool(int(same)))


def sort_sentences(input_file, output_file):

    process.sort_sentences(input_file, output_file)


def eval_f(reference_file, system_file, top_k, alpha):

    ret = evaluate.eval_f(reference_file, system_file,
                          int(top_k), float(alpha))


def eval_binary(**kwargs):

    evaluate.eval_binary(**kwargs)


def find_semantic_pairs():

    convert.find_semantic_pairs()


def construct_fasttext_embedding():

    embedding.default()


def main():

    # n_files = -1
    # w2v.construct_default_model(n_files=n_files)

    # w2v.interactive()
    # languages.compile_default_languages(n_files=n_files)
    # database.construct_default_database(n_files=n_files)
    # database.clean_default_database(max_length = 30)

    # convert.convert_csv_rules(n_max=10000, n_search=2500000)

    load.save_dataset()

    pass


if __name__ == '__main__':

    # Accept input in form of (function_name, param1, param2, ..., paramX)
    params = []

    argv = sys.argv

    kwargs = {}
    method, args = argv[1], argv[2:]

    for arg in args:

        kwarg, val = arg.split('=')

        try:
        	val = ast.literal_eval(val)

        kwargs[kwarg] = val

    globals()[method](**kwargs)
