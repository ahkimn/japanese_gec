# # TODO: Rule Classification
# def create_search_templates(pos_tags, selections, n_gram_max):

#     n_to_match = len(pos_tags)
#     assert(n_to_match == 2)
#     free_spaces = n_gram_max - n_to_match

#     ret_pos_tags = list()
#     ret_selections = list()

#     for i in(range(free_spaces + 1)):

#         new_pos_tags = list()
#         new_selections = list()

#         new_pos_tags.append(pos_tags[0])
#         new_selections.append(selections[0])

#         for k in range(i):

#             new_pos_tags.append(
#                 np.zeros(pos_tags.shape[1], dtype=pos_tags.dtype))
#             new_selections.append(
#                 np.zeros(selections.shape[1], dtype=selections.dtype))

#         new_pos_tags.append(pos_tags[1])
#         new_selections.append(selections[1])

#         new_pos_tags = np.array(new_pos_tags, dtype=pos_tags.dtype)
#         new_selections = np.array(new_selections, dtype=selections.dtype)

#         ret_pos_tags.append(new_pos_tags)
#         ret_selections.append(new_selections)

#     return ret_pos_tags, ret_selections

# # TODO: Semantic Rule Generation


# def find_semantic_pairs(
#         n_max=-1,
#         n_search=-1,
#         n_gram_max=3,
#         pause=True,
#         search_directory=configx.CONST_DEFAULT_SEARCH_DATABASE_DIRECTORY,
#         database_directory=configx.CONST_DEFAULT_DATABASE_DIRECTORY,
#         rule_file_directory=configx.CONST_RULE_CONFIG_DIRECTORY,
#         semantic_pairs_file=configx.CONST_SEMANTIC_PAIRS):

#     token_tagger, pos_taggers = languages.load_languages(language_dir)
#     # attribute_indices = [0, 1, 4, 5, 6]
#     n_pos = len(pos_taggers)

#     print("\nLoading token database...")
#     print(configx.BREAK_LINE)

#     # Load matrices necessary for sentence generation
#     # search_matrices = load_search_matrices(search_directory, pos_taggers)
#     # unique_matrices = load_unique_matrices(database_directory, pos_taggers)

#     print("\nFinished loading token databases...")
#     print(configx.BREAK_LINE)

#     # Load rule file
#     semantic_pairs_file = os.path.join(
#         rule_file_directory, semantic_pairs_file)

#     # Process rule file
#     with open(semantic_pairs_file, 'r') as f:

#         csv_reader = csv.reader(f, delimiter=',')

#         # Counter for number of iterations (determines saved file names)
#         iterations = 1

#         # Read each line (rule) of CSV
#         for rule_text in csv_reader:

#             # Paired sentence data
#             base_pair = rule_text[0]

#             print("\nFinding Semantic Pair Type %2d: %s " %
#                   (iterations, base_pair))
#             print(configx.BREAK_LINE)

#             # Retrieve unencoded part-of-speech tags of the correct sentence
#             pos_tags = rule_text[2]
#             pos_tags = pos_tags.split(',')

#             # Convert part-of-speech tags to index form
#             n_tokens = int(len(pos_tags) / n_pos)
#             pos_tags = np.array(list(languages.parse_node_matrix(
#                 pos_tags[i * n_pos: i * n_pos + n_pos], pos_taggers) for i in range(n_tokens)))

#             # Array of arrays denoting hows part-of-speech tags have been selected
#             # This is marked as -1 = null, 0 = no match, 1 = match
#             selections = rule_text[3]
#             selections = np.array(list(int(j) for j in selections.split(',')))
#             selections = selections.reshape(-1, n_pos)

#             pos_tags, selections = create_search_templates(
#                 pos_tags, selections, n_gram_max)

#             print("\n\tFinding potential substitute tokens...")
#             print(configx.BREAK_SUBLINE)

#             # List of possible substitute token classes (part-of-speech combinations) per each index of correct sentence
#             # as defined by the selections matrix
#             possible_classes = list()

#             # # Iterate over each token
#             # for index in range(n_tokens):



#     rule_file = os.path.join(rule, '%s.csv' % rule_file_name)
#     corpus_save_dir = os.path.join('corpus', save_name)
#     tmp_src_language = os.path.join(tmp_dir, 'src')
#     tmp_tgt_language = os.path.join(tmp_dir, 'tgt')

#     util.mkdir_p(tmp_dir)

#     database.construct_default_database(save_dir=tmp_dir,
#                                         data_dir=data_dir, file_type=file_type)
#     database.clean_default_database(save_dir=tmp_dir, process_unique=False)
#     # load.save_dataset(dataset_name=save_name, corpus_save_dir=corpus_save_dir,
#     #                   source_language_dir=tmp_src_language,
#     #                   target_language_dir=tmp_tgt_language)


# #             #     _, all_classes = match_template_tokens(unique_matrices, pos_tags[index],
# #             #                                                   selections[index], n_max)

# #             #     possible_classes.append(all_classes)

# #             # # Determine number of possible substitutes at each index
# #             # n_possibilities = list(len(i) for i in possible_classes)

# #             # print("\n\tSearching for sentences matching pair template...")
# #             # print(configx.BREAK_SUBLINE)
# #             # s_examples, _, starts, \
# #             #     = match_template_sentence(search_matrices, pos_tags, selections, possible_classes,
# #             #                               token_tagger, pos_taggers, n_max, n_search)
#     pass
