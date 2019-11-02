from .. import convert
from .. import database
from .. import load
from .. import util

import os


def gen_synthetic_data(data_dir, save_name, tmp_dir='tmp/synth', file_type='.csv'):

    corpus_save_dir = os.path.join('corpus', save_name)
    tmp_src_language = os.path.join(tmp_dir, 'src')
    tmp_tgt_language = os.path.join(tmp_dir, 'tgt')

    util.mkdir_p(tmp_dir)

    # database.construct_default_database(save_dir=tmp_dir,
    #                                     data_dir=data_dir, file_type=file_type)
    # database.clean_default_database(save_dir=tmp_dir, process_unique=False)

    convert.convert_csv_rules(n_max=10000, search_directory=tmp_dir, save_name=save_name)

    # load.save_dataset(dataset_name=save_name, corpus_save_dir=corpus_save_dir,
    #                   source_language_dir=tmp_src_language,
    #                   target_language_dir=tmp_tgt_language)


def gen_synthetic_data_default():   

    data_dir = 'tmp/tanaka_azure'
    save_name = 'tanaka_azure'
    gen_synthetic_data(data_dir=data_dir, save_name=save_name)
