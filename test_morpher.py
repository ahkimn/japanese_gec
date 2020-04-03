# Filename: test_morpher.py
# Date Created: 25/01/2019
# Description: Script to test basic functionality of Morpher class
#   (excluding syntatic-tag constrained morphing)
# Python Version: 3.7

import argparse

from src import morph

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compile Languages')
    parser.add_argument('--template_start', metavar='START', default='分かっ',
                        type=str, help='template start token for morpher, \
                            representative of tokens to be morphed')

    parser.add_argument('--template_end', metavar='END', default='分かる',
                        type=str, help='template end token for morpher, \
                            representative of morphed tokens')

    parser.add_argument('--input', metavar='INPUT', default='知っ',
                        type=str, help='test token to input into morpher ')

    args = parser.parse_args()

    start = args.template_start
    end = args.template_end

    base = args.input

    template = tuple([start, end])

    morpher = morph.Morpher(template)

    print(morpher)
    print('Morphed Result: %s' % morpher.morph(base))
