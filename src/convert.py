import os
import re

from . import config
from . import parse
from . import util

cfg = config.parse()

DS_PARAMS = cfg['dataset_params']


def parse_fairseq_output(input_file: str, output_delimiter: str=' ',
                         fairseq_delimiter: str=' '):

    f_in = open(input_file, 'r')

    mdl_header = re.compile(r'H-\d+')
    multispace = re.compile(r'\s+')

    data = f_in.readlines()
    f_in.close()

    mdl_data = dict()
    mdl_p = dict()

    for line in data:

        line = multispace.sub(fairseq_delimiter, line)

        values = line.strip().split(fairseq_delimiter)

        header = values[0]

        if mdl_header.match(header) is not None:
            mdl_data[int(header[2:])] = values[2:]
            mdl_p[int(header[2:])] = float(values[1])

    n_output = max(mdl_data.keys()) + 1

    print('Number of output sentences found: %d' % n_output)

    output_sentences = [''] * n_output
    output_p = [0] * n_output

    for i in range(n_output):

        output_sentences[i] = output_delimiter.join(mdl_data[i])
        output_p[i] = mdl_p[i]

    return output_sentences, output_p


def process_file(
        filepath: str, token_delimiter: str=' ',
        sentence_delimiter: str=',', tokenized: bool = False,
        error_first: bool=True):
    """
    Read file from disk and output error/correct sentences

    Args:
        filepath (str): Filepath to load from
        token_delimiter (str, optional): If %tokenized% is true, the
            character(s) used to delimit individual tokens
        sentence_delimiter (str, optional): If the file contains paired data,
            the character(s) used to delimit the error and correct data
        tokenized (bool, optional): If True, the sentences of the loaded
            file are already tokenized
        error_first (bool, optional): If True, the error sentence is the first
            of the pair of sentences in each line

    Returns:
        error_data (list): List of tokenized error sentences
        correct_data (list): List of tokenized correct sentences

    Raises:
        ValueError: If the number of sentences in a given line is not one
            or two
    """
    f_in = open(filepath, 'r')
    tagger = parse.default_parser()

    data = f_in.readlines()

    correct_data = []
    error_data = []

    n_lines = 0

    for line in data:

        sentences = line.strip().split(sentence_delimiter)

        line_sentences = []

        for sentence in sentences:

            if sentence != '':

                if tokenized:

                    line_sentences.append(
                        sentence.split(token_delimiter))

                else:

                    tokens, _ = parse.parse_full(
                        sentence, tagger)
                    line_sentences.append(tokens)

                    pass

        n_lines += 1

        print('Parsed line %d: %s' %
              (n_lines,
               '\t'.join(' '.join(t for t in sentence)
                         for sentence in line_sentences)))

        if len(line_sentences) == 0:

            print('WARNING: Line: %d contains no sentences' % n_lines)

        elif len(line_sentences) == 1:

            error_data.append(line_sentences[0])
            correct_data.append([''])

        elif len(line_sentences) == 2:

            if error_first:
                error_data.append(line_sentences[0])
                correct_data.append(line_sentences[1])

            else:
                error_data.append(line_sentences[1])
                correct_data.append(line_sentences[0])

        else:
            raise ValueError(
                'Line: %d contains more than two sentences' % n_lines)

    return error_data, correct_data


def write_output_file(file_path, sentence_lists: list,
                      sentence_delimiter: str=',',
                      token_delimiter: str=' '):
    """
    Write tokenized list of series of sentences to file. Each series must
        contain the same number of sentences.

    Output format consists of one sentence per each series in order
        separated by %sentence_delimiter%. Tokens within each sentence
        are separated by %token_delimiter%.

    Args:
        file_path (TYPE): File to write output to
        sentence_lists (list): List of lists of tokenized sentences to write
            to output file
        sentence_delimiter (str, optional): Delimiter to use between sentences
        token_delimiter (str, optional): Delimiter to use between tokens in
            each sentence
    """

    util.mkdir_p(file_path, file=True, verbose=True)
    f_out = open(file_path, 'w')

    n_series = len(sentence_lists)
    n_sentences = len(sentence_lists[0])

    assert(all(len(s) == n_sentences) for s in sentence_lists)

    for i in range(n_sentences):

        sentences = []

        for j in range(n_series):

            sentences.append(token_delimiter.join(sentence_lists[j][i]))

        line = sentence_delimiter.join(sentences) + os.linesep
        f_out.write(line)

    f_out.close()


def _split_sentence(sentence: str, delimiters: list):
    """
    Split a sentence given a pair of delimiter character(s)

    Args:
        sentence (str): Sentence to parse
        delimiters (list): List containing a pair of delimiter characters

    Returns:
        phrase (str): Phrase contained within delimiters
        bounds (tuple): Character indices of left- and right-hand boundaries
            of phrase string within overall sentence
        filtered (str): Sentence with delimiters removed

    """
    temp = list(i for i in sentence.split(delimiters[0]) if i)
    assert(len(temp) <= 2)

    start_index = 0

    filtered = ''
    if len(temp) == 2:
        filtered += temp[0]
        start_index = len(temp[0])

    temp = list(i for i in temp[-1].split(delimiters[1]) if i)
    assert(len(temp) <= 2)

    phrase = temp[0]
    filtered += phrase

    end_index = start_index + len(phrase)
    if len(temp) == 2:
        filtered += temp[1]

    bounds = (start_index, end_index)

    return phrase, bounds, filtered


def _match_token_indices(tokens: list, template_indices: tuple):
    """
    Determine if character indices given by %template_indices% align
        with token boundaries

    Args:
        tokens (list): List of tokens within sentence
        template_indices (tuple): Tuple of phrase boundaries (character
            indices)

    Returns:
        start (int): Index of token matching left-hand phrase boundary
        end (int): Index of token matching right-hand phrase boundary
    """
    chars = 0
    start = -1
    end = -1

    for i in range(len(tokens)):

        if chars == template_indices[0]:
            start = i
        chars += len(tokens[i])
        if chars == template_indices[1]:
            end = i + 1

    assert(start != -1)
    assert(end != -1)

    return start, end


def process_annotated_file(
    input_file: str, sentence_delimiter: str=',',
    error_delimiters: list=DS_PARAMS['error_delimiters'],
    correct_delimiters: list=DS_PARAMS['correct_delimiters'],
    error_first: bool=True, raise_on_error: bool=False,
    ignore_punctuation: bool=True, punctuation_list: list=['。']
):
    """
    Read file containing annotations (error/correct phrase boundaries)
        from disk and sentence pairs and phrase boundaries.

    Assumes data is not tokenized. TODO: Maybe add support for tokenized

    Args:
        input_file (str): Filepath to read from
        sentence_delimiter (str, optional): Character(s) that delimiter
            sentences of pair in each line
        error_delimiters (list, optional): List containing pair of delimiter
            characters that denote error phrase boundaries
        correct_delimiters (list, optional): List containing pair of delimiter
            characters that denote correct phrase boundaries
        error_first (bool, optional): If True, the error sentence is the first
            of the pair of sentences in each line
        raise_on_error (bool, optional): If True, raise errors each time a
            formatting assumption is violated
    """
    tagger = parse.default_parser()

    seen_dict = dict()

    source_sentences = list()
    target_sentences = list()

    source_bounds = list()
    target_bounds = list()

    with open(input_file, "r", encoding="utf-8") as f_in:

        n_processed = 1
        line_number = 0

        for line in f_in.readlines():

            line_number += 1

            print('line %d: %s' % (line_number, line))
            line = line.strip().split(sentence_delimiter)

            if len(line) != 2:
                print('ERROR: Line %d does not have two sentences.' %
                      line_number)
                # missed_indices.append(line_number)
                continue

            if error_first:

                source_line, target_line = line[0], line[1]

            else:

                target_line, source_line = line[0], line[1]

            if source_line == '' or target_line == '':
                print('ERROR IN LINE %2d: EMPTY SOURCE OR TARGET SENTENCE' %
                      line_number)
                if raise_on_error:
                    raise
                continue

            source_punctuation, target_punctuation = '', ''
            if ignore_punctuation:
                if source_line[-1] in punctuation_list:
                    source_punctuation = source_line[-1]
                    source_line = source_line[:-1]

                if target_line[-1] in punctuation_list:
                    target_punctuation = target_line[-1]
                    target_line = target_line[:-1]

            base_source = source_line
            base_target = target_line

            for i in error_delimiters:

                base_source = base_source.replace(i, '')

            for i in correct_delimiters:

                base_target = base_target.replace(i, '')

            base = base_source + '|||' + base_target

            if base in seen_dict:
                print("ERROR IN LINE %2d: BASE SENTENCE PREVIOUSLY SEEN" %
                      line_number)
                if raise_on_error:
                    raise
                continue
            else:
                seen_dict[base] = 1

            try:

                err, err_indices, source = _split_sentence(
                    source_line, error_delimiters)
                crt, crt_indices, target = _split_sentence(
                    target_line, correct_delimiters)

            except Exception:

                print('ERROR IN LINE %2d: UNABLE TO PARSE SENTENCE' %
                      line_number)
                if raise_on_error:
                    raise
                continue

            source_tokens, _ = parse.parse_full(source, tagger)
            target_tokens, _ = parse.parse_full(target, tagger)

            try:

                err_start, err_end = _match_token_indices(
                    source_tokens, err_indices)
                crt_start, crt_end = _match_token_indices(
                    target_tokens, crt_indices)

            except Exception:

                print(
                    'ERROR IN LINE %2d: ERROR AND CORRECTION PHRASES DO NOT \
                    COINCIDE WITH TOKEN BOUNDARIES' % line_number)
                if raise_on_error:
                    raise
                continue

            if ignore_punctuation:
                if source_punctuation != '':
                    source_tokens.append(source_punctuation)
                if target_punctuation != '':
                    target_tokens.append(target_punctuation)

            source_sentences.append(source_tokens)
            target_sentences.append(target_tokens)

            source_bounds.append([err_start, err_end])
            target_bounds.append([crt_start, crt_end])

            n_processed += 1

    print('Sentence pairs succesfully processed: %d' % n_processed)

    f_in.close()

    return source_sentences, target_sentences, source_bounds, target_bounds
