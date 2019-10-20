import ast
import sys

from src.scripts.eval_binary import *
from src.scripts.synthetic_data import *
from src.scripts.preprocess_unpaired import *
from src.scripts.translate_azure import *
from src.scripts.translate_google import *
from src.scripts.tanaka_corpus import *

from src.process import *
from src.evaluate import *
from src.embedding import *
from src.search import *

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

        except Exception:

            pass

        kwargs[kwarg] = val

    print('Method: %s' % method)
    print('Kwargs: ')
    for key in sorted(kwargs.keys()):

        print(key + ': ' + str(kwargs[key]) + ', ' + str(type(kwargs[key])))

    globals()[method](**kwargs)
