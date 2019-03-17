# Filename: w2v.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 05/03/2019
# Date Last Modified: 09/03/2019
# Python Version: 3.7

'''
Functions to generate, clean, and load indexed datasets from corpus text files
'''

import os
import time

from termcolor import colored
from gensim.models import Word2Vec

from . import configx
from . import languages
from . import util


def construct_default_model(n_files=-1):

    data_dir = configx.CONST_BCCWJ_NT_TEXT_DIRECTORY
    save_dir = configx.CONST_WORD_2_VEC_SAVE_DIRECTORY
    save_name = configx.CONST_WORD_2_VEC_MODEL_NAME
    file_type = configx.CONST_CORPUS_TEXT_FILETYPE

    w2v = construct_model(data_dir, file_type, n_files)

    if not os.path.isdir(save_dir):

        util.mkdir_p(save_dir)

    save_path = os.path.join(save_dir, save_name)

    w2v.save(save_path)


def load_default_model():

    return load_model(configx.CONST_WORD_2_VEC_SAVE_DIRECTORY, configx.CONST_WORD_2_VEC_MODEL_NAME)


def construct_model(data_dir, file_type, n_files):

    start_time = time.time()

    print("Loading corpus text from: %s" % (data_dir))
    print(configx.BREAK_LINE)

    # Read corpus data
    file_list = util.get_files(data_dir, file_type, n_files)

    delimiter = configx.CONST_SENTENCE_DELIMITER_TOKEN

    n_sentences = 0
    n_completed = 0

    sentence_nodes = list()

    for filename in file_list[:]:

        n_completed += 1

        with open(filename, 'r', encoding='utf-8') as f:

            start_time_file = time.time()
            # print("Processing file: " + filename)

            sentences = f.readlines()

            for i in range(len(sentences)):

                sentence = sentences[i]

                nodes = languages.parse(
                    sentence, configx.CONST_PARSER, delimiter, True)

                sentence_nodes.append(nodes)

            n_sentences += len(sentences)

            elapsed_time_file = time.time() - start_time_file
            print("\tFile %2d of %2d processed..." %
                  (n_completed, len(file_list)))

    print("\nCompleted processing of all files...")
    print(configx.BREAK_LINE)

    elapsed_time = time.time() - start_time

    print("Total sentences checked: %2d" % (n_sentences))
    print("Total elapsed time: %4f" % (elapsed_time))

    print("\nTraining model...")

    model = Word2Vec(sentence_nodes, size=256, min_count=20,
                     sorted_vocab=1, workers=8, window=4)

    print("Completed...")

    return model


def load_model(save_dir, save_name):

    save_path = os.path.join(save_dir, save_name)

    return Word2Vec.load(save_path)


def find_similar(model, context, input, k=20, max_index=5000):

    print("\t最も近い言葉:\n")

    vocabulary = list(model.wv.vocab.keys())

    delimiter = configx.CONST_SENTENCE_DELIMITER_TOKEN
    nodes = languages.parse(context, configx.CONST_PARSER, delimiter, True)

    print(nodes)
    similar_words = model.wv.most_similar(positive=nodes, topn=k, restrict_vocab=max_index)

    for i in range(k):

        replacement, similarity = similar_words[i][0], similar_words[i][1]
        # replacement = reverse_dictionary[replacement]

        print('\t\t' + replacement + ": " + str(similarity)[:5])

    _continue = ''

    while _continue not in ['y', 'ｙ']:

        _continue = input("この結果を保存しますか？（ｙ）： ")


def replace_similar(model, context_words, context_classifications, threshold, topk=300, max_per_category=20):
    """
    Function to use a pre-trained Word-2-Vec model to obtain a list of possible replacement tokens (with the same part-of-speech tags)
    as an original token given a context of words near it

    Args:
        model (gensim.Word2Vec.model): Pre-trained model instance 
        context_words (arr): List of tokens (with the original token first) that form the context
        context_classifications (arr): List of part-of-speech tags (excluding form) for each token in context_words
        threshold (float): Similarity threshold to use for finding similar words
        topk (int, optional): Maximum number of similar words to search through
        max_per_category (int, optional): Maximum number of output category (same_class, most_similar), etc. printed to console

    Returns:
        (tuple): A tuple of lists of similar words
            same_class_replacements (arr): List of similar words, with the same part-of-speech tags as the original token, 
                                           most similar to the original token over others in the context,
                                           sorted in order of similarity to the context
            similar_class_replacements (arr): List of similar words, with only the same 品詞 as the original token,
                                              most similar to the original token over others in the context,
                                              sorted in order of similarity to the context
    """
    delimiter = configx.CONST_SENTENCE_DELIMITER_TOKEN

    # Obtain replacement token classification
    classification = context_classifications[0]

    # Vocabulary of model
    vocabulary = list(model.wv.vocab.keys())

    # Words with the same classification as original (i.e. 品詞 and 品詞細分類1)
    similar_class_words = list()
    # Words in the model's vocabulary
    valid_context_words = list()
    # List of valid replacements
    same_class_replacements = list()
    similar_class_replacements = list()

    # Number of examples printed out per non-saved category
    n_same_not_max = 0
    n_similar_not_max = 0
    n_diff = 0

    for index in range(len(context_words)):

        # Determine whether the context word token has the same 品詞 as original
        similar_class = False

        if (context_classifications[index][:1] == classification[:1]):

            similar_class = True

        word = context_words[index]

        # Only tokens within model's vocabulary are valid
        if word in vocabulary:

            valid_context_words.append(word)

            if similar_class:

                similar_class_words.append(word)

            else:

                continue

        else:
            # If the original token is not within model vocabulary, no possibility for replacement tokens
            if index == 0:

                return possible

    assert(len(valid_context_words) > 0)

    # Obtain similar words from model
    similar_words = model.wv.most_similar(
        positive=valid_context_words, topn=topk)

    # Print out all tokens of same class as original
    print('\t同じ品詞の言葉： ' + colored(similar_class_words[0], "green") + ', ' + ', '.join(
        [colored(word, "yellow") for word in similar_class_words[1:]]))

    print('\tキャンディデートの言葉: ')

    # Iterate through most similar words
    for i in range(topk):

        # print(i)

        replacement, similarity = similar_words[i][0], similar_words[i][1]

        # Ignore candidate word if below similarity threshold
        if similarity < threshold:

            break

        else:
            # Determines if replacement token is most similar to original (or some other token within context)
            most_similar = True

            # Replacement words with similar classification
            valid_similarities = list()

            for j in range(len(similar_class_words)):

                valid_similarities.append(model.similarity(
                    replacement, similar_class_words[j]))

            if (max(valid_similarities) != valid_similarities[0]):

                most_similar = False

            tokens, nodes = languages.parse_full(
                replacement, configx.CONST_PARSER, delimiter, False)

            # Reduce specificity of verb part-of-speech tags (i.e. 五段・ラ行アルー＞'*')
            if nodes[0][0] == '動詞':
                nodes[2] = list('*')

            classification_replacement = tuple(
                nodes[j][0] for j in range(len(nodes) - 1))

            # print(classification)
            # print(classification_replacement)

            # If the replacement token is most similar to the original token and has the same part-of-speech tags
            # Output colour: green, bold
            if classification == classification_replacement and most_similar:

                if len(same_class_replacements) < max_per_category:

                    print('\t\t' + colored(replacement, "green", attrs=["bold"]) + ": " + str(similarity)[:5] +
                          '\t(' +
                          colored(str(valid_similarities[0])[
                                  :5], "green") + ', '
                          + ', '.join([colored(str(q)[:5], "yellow") for q in valid_similarities[1:]]) + ')')

                same_class_replacements.append(
                    [replacement, similarity, valid_similarities])

            # If the replacement token is not most similar to the original token (but has the same part-of-speech tags)
            # Output colour: green
            elif classification == classification_replacement:

                if n_same_not_max < max_per_category:

                    print('\t\t' + colored(replacement, "green") + ": " + str(similarity)[:5] +
                          '\t(' +
                          colored(str(valid_similarities[0])[
                                  :5], "green") + ', '
                          + ', '.join([colored(str(q)[:5], "yellow") for q in valid_similarities[1:]]) + ')')

                n_same_not_max += 1

            # If the replacement token has the same 品詞 and 品詞細分類1 tags as the original (but not the same form) and
            #   is most similar to the original token (over other tokens in context)
            # Output colour: blue, bold
            elif classification[0] == classification_replacement[0] and most_similar:

                if len(similar_class_replacements) < max_per_category:

                    print('\t\t' + colored(replacement, "blue", attrs=["bold"]) + ": " + str(similarity)[:5] +
                          '\t(' +
                          colored(str(valid_similarities[0])[
                                  :5], "green") + ', '
                          + ', '.join([colored(str(q)[:5], "yellow") for q in valid_similarities[1:]]) + ')')

                similar_class_replacements.append(
                    [replacement, similarity, valid_similarities])

            # If the replacement token has the same 品詞 and 品詞細分類1 tags as the original (but not the same form) and
            #   is most similar to some other token in the context
            # Output colour: blue
            elif classification[:1] == classification_replacement[:1]:

                if n_similar_not_max < max_per_category:

                    print('\t\t' + colored(replacement, "blue") + ": " + str(similarity)[:5] +
                          '\t(' +
                          colored(str(valid_similarities[0])[
                                  :5], "green") + ', '
                          + ', '.join([colored(str(q)[:5], "yellow") for q in valid_similarities[1:]]) + ')')

                n_similar_not_max += 1

            # If the replacement token does not share its part-of-speech
            # Output colour: N/A (terminal default)
            else:

                if n_diff < max_per_category:

                    print('\t\t' + replacement + ": " + str(similarity)[:5])

                n_diff += 1

    return same_class_replacements, similar_class_replacements


def replace(model, sentence, max_separation=3, threshold=0.700, max_per_token=20):
    """
    Parse a given sentence/phrase, and for each token that is a noun, verb, or adjective,
        find and return similar tokens (with the same or similar part-of-speech tags)
        within max_separation distance

    Args:
        model (gensim.Word2Vec.model): Pre-trained model instance 
        sentence (str): Sentence/phrase to parse
        max_separation (int, optional): Maximum separation (in tokens) of context tokens
        threshold (float, optional): Minimum similarity value in returned tokens
        max_per_token (int, optional): Maximum number of tokens per class 
                                      (same part-of-speech, similar part-of-speech) returned
    """
    print("\n\t類似性閾値: " + str(threshold))

    delimiter = configx.CONST_SENTENCE_DELIMITER_TOKEN

    tokens, nodes = languages.parse_full(
        sentence, configx.CONST_PARSER, delimiter, False)

    n_tokens = len(tokens)
    length = 0

    # Reduce specificity of verb part-of-speech tags (i.e. 五段・ラ行アルー＞'*')
    for j in range(n_tokens):

        if nodes[0][j] == '動詞':

            nodes[2][j] = '*'

    # Array storing (character) indices where each token starts
    token_start_indices = list()
    # Array storing (token) indices of each token that can be replaced
    valid_indices = list()
    # Array storing strings of tokens that can be replaced
    valid_tokens = list()
    # Array storing part-of-speech tags of tokens (excluding form) of tokens in valid_tokens
    valid_classifications = list()

    # Iterate over the sentence tokens
    for i in range(n_tokens):

        # Check if the token is a candidate for replacement
        if nodes[0][i] in configx.CONST_WORD_2_VEC_CANDIDATES:

            if nodes[1][i] != u'接尾':

                token_start_indices.append(length)

                valid_indices.append(i)
                valid_tokens.append(tokens[i])

                # Add part-of-speech tags of valid token (excluding token form)
                valid_classifications.append(
                    tuple(nodes[j][i] for j in range(len(nodes) - 1)))

        length += len(tokens[i])

    same_replacements = list()
    similar_replacements = list()

    phrase_contexts = list()

    # For each token, obtain tokens within context
    for i in range(len(valid_tokens)):

        # Each token is in its context
        phrase_context = [tokens[valid_indices[i]]]
        phrase_classifications = [valid_classifications[i]]

        # Find other valid tokens within max_separation of current token
        for j in range(len(valid_tokens)):

            if abs(valid_indices[j] - valid_indices[i]) <= max_separation and i != j:

                phrase_context.append(tokens[valid_indices[j]])
                phrase_classifications.append(valid_classifications[j])

        # Show replacement token as green
        replacement_word = colored(phrase_context[0], "green")
        context_words = ""

        # Show other words in context as yellow
        if len(phrase_context) > 1:

            context_words = ", " + \
                colored(", ".join(phrase_context[1:]), "yellow")

        print("\n\t入れ替えの文脈: " + replacement_word + context_words)

        # Get possible replacment tokens (with same part-of-speech tags)
        same_class, similar_class = replace_similar(
            model, phrase_context, phrase_classifications, threshold)

        same_replacements.append(same_class)
        similar_replacements.append(similar_class)

        phrase_contexts.append(phrase_context)

    print()
    save = ''

    while save not in ['y', 'n', 'ｙ', 'ｎ']:

        save = input("この結果を保存しますか？（ｙ｜ｎ）： ")

    if save in ['y', 'ｙ']:

        # Save the returned replacements to disk
        save_dir = configx.CONST_WORD_2_VEC_OUTPUT_DIRECTORY

        # Make the save directory, if it does not already exist
        if not os.path.isdir(save_dir):
            util.mkdir_p(save_dir)

        # File path to save to
        file_name = os.path.join(save_dir, sentence + '.txt')

        print('\n出力ファイルの名前： ' + file_name)

        with open(file_name, 'w+') as f:

            # Headers
            f.write('入力フレーズ： %s\n' % sentence)
            f.write('\n')
            f.write('調査されたトーケン： %s\n' % ' | '.join(valid_tokens))
            f.write('類似性閾値： %.3f\n' % threshold)
            f.write('=' * 64 + '\n\n\n')

            # Per token information
            for i in range(len(valid_tokens)):

                f.write('ターゲットのトーケン： %s\n' % valid_tokens[i])
                f.write('文脈内のトーケン: %s\n' % ' | '.join(phrase_contexts[i]))
                f.write('=' * 64 + '\n\n')

                column_width = 15

                header_one = 'トーケン'
                header_two = '文脈に対する類似性'
                header_three = valid_tokens[i] + 'に対する類似性'

                header_one += ' ' * (column_width - len(header_one))

                f.write('\t同じ（品詞、品詞細分類1、活用形）\n\n')

                f.write(
                    '\t' + ' ｜ '.join([header_one, header_two, header_three]) + '\n')
                f.write('\t' + '=' * 60 + '\n')

                # Tokens that share 品詞、品詞細分類1、活用形 with target token
                for j in range(min(len(same_replacements[i]), max_per_token)):

                    first_column = same_replacements[i][j][0]
                    first_column += ' ' * (len(header_one) - len(first_column))

                    second_column = str(same_replacements[i][j][1])[:5]
                    second_column += ' ' * \
                        (len(header_two) - len(second_column))

                    third_column = str(same_replacements[i][j][2][0])[:5]
                    third_column += ' ' * \
                        (len(header_three) - len(third_column))

                    f.write('\t' + first_column + ' ｜ ' +
                            second_column + ' ｜ ' +
                            third_column + '\n')

                f.write('\n\t同じ（品詞）、別の（品詞細分類1、活用形)\n\n')

                f.write(
                    '\t' + ' ｜ '.join([header_one, header_two, header_three]) + '\n')
                f.write('\t' + '=' * 60 + '\n')

                # Tokens that share only 品詞 with target token
                for j in range(min(len(similar_replacements[i]), max_per_token)):

                    first_column = similar_replacements[i][j][0]
                    first_column += ' ' * (len(header_one) - len(first_column))

                    second_column = str(similar_replacements[i][j][1])[:5]
                    second_column += ' ' * \
                        (len(header_two) - len(second_column))

                    third_column = str(similar_replacements[i][j][2][0])[:5]
                    third_column += ' ' * \
                        (len(header_three) - len(third_column))

                    f.write('\t' + first_column + ' ｜ ' +
                            second_column + ' ｜ ' +
                            third_column + '\n')

                f.write('\n\n')

            f.close()


def interactive():

    model = load_default_model()

    while(True):

        # Clear the screen
        util.clear_()
        print('コマンドの選択肢：\n')
        print('類似性によるトーケン検索： 1')
        print('入れ替えが可能のトーケンをハイライトする： 2')
        print('プログラムを閉じる： 3')

        x = input('\nコマンドの番号を入力してください: \n\t')

        if x == '2':

            # threshold = 0.500
            # max_separation = 3

            context = input('\n\t文を入力してください： ')
            threshold = input('\n\t類似性閾値を入力してください:')
            max_separation = input('\n\t類似性の計算に使う最大距離を入力してください：')

            replace(model, context, threshold=threshold,
                    max_separation=max_separation)

        elif x == '1':

            util.clear_()
            context = input('\n\t文脈を入力してください： ')
            find_similar(model, context, 20)

        elif x == '3':

            break

        else:

            continue
