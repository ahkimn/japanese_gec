from .. import configx
from .. import database
from .. import languages
from .. import update_languages
from .. import util

import os
import csv


def gen_default_text_database():

    corpus_dir = 'raw_data/scrape'
    corpus_lang_dir = 'database/scrape'
    database_dir = 'database/scrape'
    gen_text_dir = 'generated_text/full'

    gen_text_database(corpus_dir, corpus_lang_dir, database_dir,
                      gen_text_dir)


def gen_text_database(corpus_dir, corpus_lang_dir,
                      database_dir,
                      gen_text_dir,
                      corpus_file_type='txt',
                      gen_file_type='csv',
                      _filter='type', o_every=10000, o_prefix='o'):

    tmp_dir = os.path.join('tmp', 'database')

    if not os.path.isdir(tmp_dir):
        util.mkdir_p(tmp_dir)

    languages.compile_languages(data_dir=corpus_dir,
                                file_type=corpus_file_type, save_dir=tmp_dir)

    o_file_num = 1
    lines_completed = 0
    last_completed = 0

    o_file_name = \
        os.path.join(tmp_dir,
                     '%s%d.%s' % (o_prefix, o_file_num,
                                  corpus_file_type))

    o_file = open(o_file_name, 'w+')

    gen_files = util.get_files_recursive(gen_text_dir, gen_file_type)
    update_sentences = list()

    for file_name in gen_files:

        if _filter not in file_name:

            continue

        else:

            f = open(file_name, 'r')
            reader = csv.reader(f)

            for line in reader:

                err = line[0]

                o_file.write(err + os.linesep)

                if lines_completed - last_completed >= o_every:
                    last_completed = lines_completed
                    o_file_num += 1
                    o_file_name = \
                        os.path.join(tmp_dir,
                                     '%s%d.%s' % (o_prefix, o_file_num,
                                                  corpus_file_type))

                    print('Number of files written: %d' % (o_file_num - 1))
                    o_file.close()
                    o_file = open(o_file_name, 'w+')

                lines_completed += 1
                update_sentences.append(err)

    token_tagger, pos_taggers = \
        languages.load_languages(tmp_dir, configx.CONST_NODE_PREFIX,
                                 configx.CONST_POS_PREFIX)

    token_tagger.sample()

    update_languages.update_languages(
        token_tagger, pos_taggers,
        update_sentences, corpus_lang_dir, source=None)

    del update_sentences

    database.construct_database(save_dir=tmp_dir,
                                data_dir=[tmp_dir, corpus_dir],
                                file_type=corpus_file_type,
                                language_dir=corpus_lang_dir,
                                save_prefix='tmp',
                                n_files=1500)

    database.clean_database(load_dir=tmp_dir,
                            save_dir=database_dir,
                            load_prefix='tmp',
                            save_prefix='cleaned')
