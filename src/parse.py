from . import configx

def main():

    parser = configx.CONST_PARSER

    ret = parse(u'振り返ると、満面の笑みの彼女と、僕ら二人を訝しがるクラスメイト達の顔が見えた。\n', parser)

def parse(text, parser=configx.CONST_PARSER):

    # print(text)

    nodes = list()
    ret = list()

    text = text.strip()

    res = parser.parseToNode('')
    res = parser.parseToNode(text)

    len_parsed = 0

    while res:

        len_parsed += len(res.surface)

        if res.surface != '':

            classifications = res.feature.split(",")
            classifications = resolve_classification(classifications)  
            
            nodes.append(res.surface)
            ret.append(classifications)

        res = res.next

    assert(len_parsed == len(text))

    return ret, nodes

''' Function to extract part-of-speech as well as conjugation information from Mecab classification'''
def resolve_classification(classification):

    # Part of speech tags
    class1 = classification[0]
    class2 = classification[1]
    class3 = classification[2]
    class4 = classification[3]

    # Conjugation tags
    form1 = classification[4]
    form2 = classification[5]

    # Original form
    origin = classification[6]

    return [class1, class2, class3, class4, form1, form2, origin]

if __name__ == '__main__':
    
    main()


    