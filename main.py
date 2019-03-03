from src import load
from src import languages
from src import convert


def main():

	convert.convert_csv_rules(n_max=10000, n_search=10000)

	# languages.load_default_languages()
	# load.save_dataset()

	






if __name__ == '__main__':

	main()