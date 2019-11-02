from difflib import SequenceMatcher
from enum import Enum


class Morph(Enum):

    TAIL_REPLACE = 0
    HEAD_REPLACE = 1
    OTHER = 2


class Operation(Enum):

    DELETION = 0
    ADDITION = 1
    SUBSTITUTION = 3
    OTHER = 2


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

            raise ValueError("ERROR: Transformation %s -> %s \
             is of unsupported morph type" % (self.start, self.end))

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

    def is_deletion(self):

        return self._operation == Operation.DELETION

    def is_addition(self):

        return self._operation == Operation.ADDITION

    def is_substitution(self):

        return self._operation == Operation.SUBSTITUTION

    def get_rule(self):

        return '%s -> %s' % (self.start, self.end)

    def can_morph(self):

        return (self.morph_type == Morph.HEAD_REPLACE
                or self.morph_type == Morph.TAIL_REPLACE)


if __name__ == '__main__':

    start = '分かっ'
    end = '分かる'

    base = 'しなかっ'

    template = tuple([start, end])

    a = Morpher(template)

    print(a.morph(base))

    start = 'おみやげ'
    end = 'みやげ'

    base = 'お皿'

    template = tuple([start, end])

    b = Morpher(template)

    print(b.morph(base))
