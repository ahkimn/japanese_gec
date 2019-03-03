from src import load
from src import languages
from src import convert
from src import database


def main():

	n_files = 10
	# languages.compile_default_languages(n_files=n_files)
	# database.construct_default_database(n_files=n_files)
	# database.clean_default_database(max_length = 30)

	convert.convert_csv_rules(n_max=10000, n_search=10000)

	# languages.load_default_languages()
	# load.save_dataset()

	






if __name__ == '__main__':

	main()