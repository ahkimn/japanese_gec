from src import load
from src import languages
from src import convert
from src import database
from src import w2v
from src import configx


def main():

	# n_files = -1
	# w2v.construct_default_model(n_files=n_files)

	# w2v.interactive()
	# languages.compile_default_languages(n_files=n_files)
	# database.construct_default_database(n_files=n_files)
	# database.clean_default_database(max_length = 30)

	# convert.convert_csv_rules(n_max=10000, n_search=1000000)

	# languages.load_default_languages()
	load.save_dataset()

	pass

	






if __name__ == '__main__':

	main()