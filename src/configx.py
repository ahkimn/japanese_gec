import os
import MeCab

'''
File containing consants used throughout project
'''

'''
Special text
'''
BREAK_LINE = "==========================================================\n"
BREAK_SUBLINE = "\t================================================="

'''
Constants from previous code
'''
CONST_MAX_SEARCH_TOKEN_INDEX = 4000


'''
Specific directories
'''
CONST_DATABASE_DIRECTORY = "raw_data"
CONST_SEARCH_DATABASE_DIRECTORY = "raw_data"
CONST_RULE_CONFIG_DIRECTORY = "rules"

# Folder to store text file outputs
CONST_TEXT_OUTPUT_DIRECTORY = "generated_text"
# Essentially the name of the dataset (within /CONST_TEXT_OUTPUT_DIRECTORY)
CONST_TEXT_OUTPUT_PREFIX = "d_15"
# Output folder of dataset corpus within CONST
CONST_CORPUS_SAVE_DIRECTORY = "init_15"

# Prefix/Suffix for each text file (within /CONST_TEXT_OUTPUT_DIRECTORY/CONST_TEXT_OUTPUT_PREFIX)
CONST_SENTENCE_FILE_PREFIX = "type"
CONST_SENTENCE_FILE_SUFFIX = ".txt"

CONST_ERRORED_PREFIX = "error"
CONST_CORRECT_PREFIX = "correct"

# Default location of SMT files
CONST_SMT_OUTPUT_DIRECTORY = "smt_output"

# Default location of CNN files
CONST_CNN_OUTPUT_DIRECTORY = "cnn_output"
# Default location of translated output (within SMT_OUTPUT_DIRECTORY)
CONST_TRANSLATE_OUTPUT_PREFIX = "output"
CONST_TRANSLATE_TEXT_PREFIX = "text"

# Default language/translation/tuned model directories (within SMT_OUTPUT_DIRECTORY)
CONST_LANGUAGE_MODEL_DIRECTORY = "lm"
CONST_TRANLSATION_MODEL_DIRECTORY = "tm"
CONST_TUNED_MODELS_DIRECTORY = "tm_tuned"

# Default directory of SMT config files (within SMT_OUTPUT_DIRECTORY)
CONST_SMT_CONFIG_DIRECTORY = "config"

# Default number of processors used
CONST_NUM_PROCESSORS = str(8)


CONST_MODEL_SAVE_DIRECTORY = "model"
# Default embedding save locations
CONST_EMBEDDING_TOKENS = os.path.join(CONST_MODEL_SAVE_DIRECTORY, "embedding_tokens.npy")
CONST_EMBEDDING = os.path.join(CONST_MODEL_SAVE_DIRECTORY, "embedding.npy")

CONST_EMBEDDING_SIZE = 128
CONST_EMBEDDING_SOURCE_SAVE = os.path.join(CONST_TEXT_OUTPUT_DIRECTORY, CONST_TEXT_OUTPUT_PREFIX, "embedding_source.txt")
CONST_EMBEDDING_FAIRSEQ_SAVE = os.path.join(CONST_TEXT_OUTPUT_DIRECTORY, CONST_TEXT_OUTPUT_PREFIX, "embedding_fairseq.txt")

CONST_MAX_SENTENCE_LENGTH = 40


'''
Specific files
'''
CONST_RULE_CONFIG = "pair_data.csv"


'''
Constants for Languages
'''
CONST_PARSER = MeCab.Tagger()

CONST_PAD_TOKEN = "PAD"
CONST_PAD_INDEX = 0
CONST_UNKNOWN_TOKEN = "UNKNOWN"
CONST_UNKNOWN_INDEX = 1
CONST_SENTENCE_START_TOKEN = "START"
CONST_SENTENCE_START_INDEX = 2
CONST_SENTENCE_DELIMITER_TOKEN = '。'
CONST_SENTENCE_DELIMITER_INDEX = 3

CONST_LANGUAGES_SAVE_DIRECTORY = "languages"
CONST_DEFAULT_LANGUAGE_PREFIX = "original"
CONST_DEFAULT_LANGUAGE_DIRECTORY = \
	os.path.join(CONST_LANGUAGES_SAVE_DIRECTORY, CONST_DEFAULT_LANGUAGE_PREFIX)

CONST_UPDATED_SOURCE_LANGUAGE_PREFIX = "updated_source"
CONST_UPDATED_SOURCE_LANGUAGE_DIRECTORY = \
	os.path.join(CONST_LANGUAGES_SAVE_DIRECTORY, CONST_UPDATED_SOURCE_LANGUAGE_PREFIX)

CONST_UPDATED_TARGET_LANGUAGE_PREFIX = "updated_target"
CONST_UPDATED_TARGET_LANGUAGE_DIRECTORY = \
	os.path.join(CONST_LANGUAGES_SAVE_DIRECTORY, CONST_UPDATED_TARGET_LANGUAGE_PREFIX)

CONST_POS_PREFIX = "pos"
CONST_NODE_PREFIX = "node"

CONST_TRUNCATED_OUTPUT_DIR = "truncated"


'''
Constants for Fairseq
'''
CONST_MIN_FREQUENCY = 10
CONST_CNN_SAVE_DIRECTORY = "cnn_model"
CONST_SOURCE_DICTIONARY_NAME = "source_dict.txt"
CONST_TARGET_DICTIONARY_NAME = "target_dict.txt"
CONST_MAX_DICT_SIZE = 5000