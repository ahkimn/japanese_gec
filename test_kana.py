import argparse
import os

from src import config
from src import kana

cfg = config.parse()

DIRECTORIES = cfg['directories']
M_PARAMS = cfg['morpher_params']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Hiragana/Katakana List')

    parser.add_argument('--kana_file', metavar='KANA_FILE',
                        default = M_PARAMS['kana_default'],
                        type=str, help='File within data/const containing organized \
                        list of kana', required=False)

    args = parser.parse_args()

    kana_file = os.path.join(DIRECTORIES['const'], args.kana_file)

    KL = kana.KanaList(kana_file)

