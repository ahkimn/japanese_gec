# -*- coding: utf-8 -*-

# Filename: morph.py
# Date Created: 25/01/2019
# Description: Morpher class and associated functions/helper classes;
#   performs token-level manipulation to search for/generate tokens
#   matching a correct-to-error template and MeCab syntactic tags
# Python Version: 3.7

from difflib import SequenceMatcher
from enum import Enum

from . import parse


class Morph(Enum):

    TAIL_REPLACE = 0
    HEAD_REPLACE = 1
    OTHER = 2


class Operation(Enum):

    DELETION = 0
    ADDITION = 1
    SUBSTITUTION = 3
    OTHER = 2


VALID_KANA = {

    'a': set(['あ', 'か', 'さ', 'た', 'な', 'は', 'ま', 'や', 'ら',
              'が', 'ざ', 'だ', 'ば', 'ぱ', 'わ', 'ん']),
    'i': set(['い', 'き', 'し', 'ち', 'に', 'ひ', 'み', 'り', 'ぎ',
              'じ', 'ぢ', 'び', 'ぴ', 'ゐ']),
    'u': set(['う', 'く', 'す', 'つ', 'ぬ', 'ふ', 'む', 'ゆ', 'る',
              'ぐ', 'ず', 'づ', 'ぶ', 'ぷ']),
    'e': set(['え', 'け', 'せ', 'て', 'ね', 'へ', 'め', 'れ', 'げ',
              'ぜ', 'で', 'べ', 'ぺ', 'ゑ']),
    'o': set(['お', 'こ', 'そ', 'と', 'の', 'ほ', 'も', 'よ', 'ろ',
              'ご', 'ぞ', 'ど', 'ぼ', 'ぽ', 'を']),
    'ya': set(['きゃ', 'しゃ', 'ちゃ', 'にゃ', 'ひゃ', 'みゃ',
               'りゃ', 'ぎゃ', 'じゃ', 'びゃ', 'ぴゃ']),
    'yu': set(['きゅ', 'しゅ', 'ちゅ', 'にゅ', 'ひゅ', 'みゅ',
               'りゅ', 'ぎゅ', 'じゅ', 'びゅ', 'ぴゅ']),
    'yo': set(['きょ', 'しょ', 'ちょ', 'にょ', 'ひょ', 'みょ',
               'りょ', 'ぎょ', 'じょ', 'びょ', 'ぴょ']),
    'aa': set(['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'っ', 'ゕ', 'ゖ',
               'ゃ', 'ゅ', 'ょ'])
}

UNIQUE_KANA = set()
KANA_TO_SET = dict()

for kana_set in VALID_KANA.keys():

    _current = VALID_KANA[kana_set]

    UNIQUE_KANA = UNIQUE_KANA.union(_current)

    for kana in _current:

        KANA_TO_SET[kana] = kana_set


def _get_search_order(kana):

    order = list(kana)
    kana_set = KANA_TO_SET[kana]

    for k in VALID_KANA[kana_set]:

        if k != kana:
            order.append(k)

    for s in VALID_KANA.keys():

        if s != kana_set:

            order += list(VALID_KANA[s])

    return order


class Morpher:

    def __init__(self, template):

        self.start = template[0]
        self.end = template[1]

        self.n_start = len(self.start)
        self.n_end = len(self.end)

        if self.end in self.start and self.n_end < self.n_start:

            self._operation = Operation.DELETION

        elif self.start in self.end and self.n_end > self.n_start:

            self._operation = Operation.ADDITION

        elif self.n_end == self.n_start:

            self._operation = Operation.SUBSTITUTION

        else:

            self._operation = Operation.OTHER

        self.match = \
            SequenceMatcher(None, self.start, self.end).find_longest_match(
                0, self.n_start, 0, self.n_end)

        self.morph_type = None
        self.params = dict()

        if self.match.a + self.match.size == self.n_start \
                and self.match.b + self.match.size == self.n_end:

            self.morph_type = Morph.HEAD_REPLACE

        elif self.match.a == self.match.b and self.match.a == 0:

            self.morph_type = Morph.TAIL_REPLACE

        else:

            raise ValueError("ERROR: %s is of unsupported morph type"
                             % self.print_template())

    def print_template(self):

        return 'Morph: %s -> %s' % (self.start, self.end)

    def print_morph_type(self):

        return 'Morph Type: %s' % str(self.morph_type)

    def print_operation(self):

        return 'Operation: %s' % str(self._operation)

    def __str__(self):

        return '\n'.join([self.print_template(), self.print_operation(),
                         self.print_morph_type()])

    def del_length(self):

        return self.n_start - self.n_end

    def morph(self, base):

        ret = ''

        if self.morph_type == Morph.HEAD_REPLACE:

            ret += self.end[:self.match.b]
            ret += base[self.match.a:]

        elif self.morph_type == Morph.TAIL_REPLACE:

            tail_start = self.n_start - (self.match.size)

            ret += base[:-tail_start]
            ret += self.end[self.match.size:]

        return ret

    def _dir_morph(self, base, template):

        if self.morph_type == Morph.HEAD_REPLACE:

            return template + base[self.match.a:]

        elif self.morph_type == Morph.TAIL_REPLACE:

            tail_start = self.n_start - (self.match.size)

            return base[:-tail_start] + template

    def morph_pos(self, base, base_form, token_language, tag_languages,
                  mecab_tagger, template, match_indices):

        if not self.can_morph():

            return None

        elif self._operation == Operation.DELETION:

            return self.morph(base)

        elif self._operation == Operation.ADDITION:

            if self.n_end - self.n_start > 1:

                return None

            else:

                if self.morph_type == Morph.TAIL_REPLACE:

                    sub_template = self.end[self.n_start:]

                elif self.morph_type == Morph.HEAD_REPLACE:

                    sub_template = self.end[:self.n_end - self.n_start]

                else:

                    return None

        elif self._operation == Operation.SUBSTITUTION:

            if self.n_end - self.match.size > 1:

                return None

            else:

                if self.morph_type == Morph.TAIL_REPLACE:

                    sub_template = self.end[self.match.size:]

                elif self.morph_type == Morph.HEAD_REPLACE:

                    sub_template = self.end[:self.n_end - self.match.size]

                else:

                    return None

        else:

            return None

        n_attempts = 0
        search_order = _get_search_order(sub_template)

        while n_attempts < len(search_order):

            gen_token = self._dir_morph(base, search_order[n_attempts])
            gen_token, gen_tags = \
                parse.parse_full(gen_token, mecab_tagger)

            if len(gen_token) == 1:

                gen_token = gen_token[0]
                gen_tags = list(tag_languages[q].add_node(
                    gen_tags[q][0]) for q in range(len(tag_languages)))

                valid = (gen_tags[-1] == base_form)

                # TODO
                # Add check if token equals base form (and same is true for
                #   template)
                # i.e. kana-based parsing (i.e. ただよう may be correctly generated
                #   but won't be parsed correctly by mecab)

                for idx in match_indices:

                    if gen_tags[idx] != template[idx]:

                        valid = False

                if valid:

                    return gen_token

            n_attempts += 1

    def is_deletion(self):

        return self._operation == Operation.DELETION

    def is_addition(self):

        return self._operation == Operation.ADDITION

    def is_substitution(self):

        return self._operation == Operation.SUBSTITUTION

    def get_rule(self):

        return '%s -> %s' % (self.start, self.end)

    def can_morph(self):

        return (self.morph_type != Morph.OTHER)

    def verify(self, base_token, final_token):

        valid = self.can_morph()

        if self.is_deletion():

            if len(base_token) - len(final_token) != self.del_length():

                valid = False

        # TODO other checks

        return valid
