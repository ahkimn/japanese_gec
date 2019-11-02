import os
import re

from .. import configx
from .. import languages
from .. import util


def _split_sentence(sentence, delimiters):

    temp = list(i for i in sentence.split(delimiters[0]) if i)
    assert(len(temp) <= 2)

    start_index = 0

    raw = ''
    if len(temp) == 2:
        raw += temp[0]
        start_index = len(temp[0])

    temp = list(i for i in temp[-1].split(delimiters[1]) if i)
    assert(len(temp) <= 2)

    filtered = temp[0]
    raw += filtered

    end_index = start_index + len(filtered)
    if len(temp) == 2:
        raw += temp[1]

    return filtered, (start_index, end_index), raw


def _match_token_indices(tokens, template_indices):

    print(tokens)
    print(template_indices)

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


def pre_process_delimited_txt(
    input_file, output_source,
    output_target, output_start,
    err_delimiters=('<', '>'), crt_delimiters=('(', ')'),
    sentence_delimiter='\t', sentence_end='。'
):

    seen_dict = dict()

    source_text = list()
    target_text = list()
    starts = list()

    util.mkdir_p(output_source, file=True)
    util.mkdir_p(output_target, file=True)
    util.mkdir_p(output_start, file=True)

    with open(input_file, "r", encoding="utf-8") as in_file:

        line_number = 1
        for line in in_file.readlines():

            print('line: %s' % line)
            line = line.replace(sentence_end, '')
            line = line.strip().split(sentence_delimiter)

            if len(line) != 2:
                print("ERROR IN LINE %2d: MORE THAN TWO SENTENCES IN LINE" %
                      line_number)
                continue

            source_line = re.sub(r'\s+|　', '', line[0])
            target_line = re.sub(r'\s+|　', '', line[1])

            base_source = source_line
            base_target = target_line

            for i in err_delimiters:

                base_source = base_source.replace(i, '')
                base_target = base_target.replace(i, '')

            for i in crt_delimiters:

                base_source = base_source.replace(i, '')
                base_target = base_target.replace(i, '')

            base = base_source + '|||' + base_target

            if base in seen_dict:
                print("ERROR IN LINE %2d: BASE SENTENCE PREVIOUSLY SEEN" %
                      line_number)
                continue
            else:
                seen_dict[base] = 1

            if source_line == '' or target_line == '':
                print('ERROR IN LINE %2d: EMPTY SOURCE OR TARGET SENTENCE' %
                      line_number)
                continue

            try:

                err, err_indices, source = _split_sentence(
                    source_line, err_delimiters)
                crt, crt_indices, target = _split_sentence(
                    target_line, crt_delimiters)

            except Exception:

                print('ERROR IN LINE %2d: UNABLE TO PARSE SENTENCE' %
                      line_number)
                continue

            source_tokens = languages.parse(source, configx.CONST_PARSER, None)
            target_tokens = languages.parse(target, configx.CONST_PARSER, None)

            if source_tokens[-1] != sentence_end:
                source_tokens.append(sentence_end)
            if target_tokens[-1] != sentence_end:
                target_tokens.append(sentence_end)

            try:

                err_start, err_end = _match_token_indices(
                    source_tokens, err_indices)
                crt_start, crt_end = _match_token_indices(
                    target_tokens, crt_indices)

            except Exception:

                print(
                    'ERROR IN LINE %2d: ERROR AND CORRECTION PHRASES DO NOT \
                    COINCIDE WITH TOKEN BOUNDARIES' % line_number)
                continue

            source_text.append(' '.join(source_tokens))
            target_text.append(' '.join(target_tokens))

            starts.append([err_start, err_end, crt_start, crt_end])

            line_number += 1
            print(line_number)

    in_file.close()
    source_file = open(output_source, "w+")
    target_file = open(output_target, "w+")
    start_file = open(output_start, "w+")

    for i in range(len(source_text)):

        source_file.write(source_text[i])
        source_file.write(os.linesep)

        target_file.write(target_text[i])
        target_file.write(os.linesep)

        start_file.write(','.join(list(str(j) for j in starts[i])))
        start_file.write(os.linesep)

    source_file.close()
    target_file.close()
    start_file.close()
