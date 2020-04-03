import argparse
import os
import subprocess

from src import config
from src import util
from src.datasets import Dataset
from src.util import str_bool


cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
MDL_PARAMS = cfg['model_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model given \
            train/dev/test Dataset instances')

    # Required
    parser.add_argument(
        '--ds_load_dir', metavar='DS_LOAD_DIR',
        type=str, help='sub-directory of ./data/datasets \
            where Dataset instances are saved', required=True)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='File extension of saved Dataset instances',
        required=False)

    # Required
    parser.add_argument(
        '--ds_name_train', metavar='DS_NAME_TRAIN', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            containing training data',
        required=True)

    # Required
    parser.add_argument(
        '--ds_name_dev', metavar='DS_NAME_DEV', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            containing development data',
        required=True)

    # Required
    parser.add_argument(
        '--ds_name_test', metavar='DS_NAME_TRAIN', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            containing test data',
        required=True)

    parser.add_argument(
        '--tmp_dir', metavar='TMP_DIR', type=str, default=DIRECTORIES['tmp'],
        help='sub-directory of (./data/tmp/) to write temporary files to',
        required=False)

    parser.add_argument(
        '--n_words_model', metavar='N_WORDS_MODEL',
        default=MDL_PARAMS['dictionary_size'], help='number of tokens \
        to use in model training', required=False)

    parser.add_argument(
        '--fp16', metavar='FP16', default=True, type=str_bool,
        help='if True, use half-precision \
            floating point values for training', required=False)

    parser.add_argument(
        '--command', metavar='COMMAND', type=str,
        help='one of [\'write\', \'preprocess\', \
            \'train\', or \'all\']. Determines which operation to run',
        required=True)

    parser.add_argument(
        '--cuda', metavar='CUDA', default=-1, type=int,
        help='if not -1, index of GPU to use',
        required=False)

    parser.add_argument(
        '--model-save_dir', metavar='MODEL_SAVE_DIR', type=str,
        default='mdl',
        help='sub-directory of \'./models\' to save model to',
        required=False)

    parser.add_argument(
        '--model-arch', metavar='MODEL_ARCH', type=str,
        help='architecture of model (registered in Fairseq) to use',
        default='fconv_jp_current', required=False)

    args = parser.parse_args()

    command = args.command.lower()

    assert(command in ['write', 'preprocess', 'train', 'all'])
    tmp_dir = args.tmp_dir

    if not os.path.isdir(tmp_dir):
        util.mkdir_p(tmp_dir)

    f_names = [args.ds_name_train, args.ds_name_dev, args.ds_name_test]
    w_names = [os.path.join(tmp_dir, f) for f in f_names]

    if command == 'write' or command == 'all':

        for f in f_names:
            ds_load_file = os.path.join(DIRECTORIES['datasets'],
                                        args.ds_load_dir, '%s.%s' %
                                        (f, args.ds_suffix))

            print('Loading dataset: %s' % ds_load_file)

            DS = Dataset.load(ds_load_file)
            DS.write(tmp_dir, token_delimiter=' ', data_delimiter='',
                     include_tags=[], separation='none', max_per_rule=-1,
                     save_prefix=f)

    if command == 'preprocess' or command == 'all':

        prep = subprocess.Popen(
            ['fairseq-preprocess',
             '--source-lang', SAVE_PARAMS['error_suffix'],
             '--target-lang', SAVE_PARAMS['correct_suffix'],
             '--trainpref', w_names[0],
             '--validpref', w_names[1],
             '--testpref', w_names[2],
             '--destdir', tmp_dir,
             '--workers', str(4)],
            stdout=subprocess.PIPE)
        output = prep.communicate()[0]

    if command == 'train' or command == 'all':

        model_save_dir = os.path.join(DIRECTORIES['models'],
                                      args.model_save_dir)

        t_args = ['fairseq-train', tmp_dir,
                  '--lr', str(0.1),
                  '--clip-norm', str(0.1),
                  '--dropout', str(0.1),
                  '--max-tokens', str(args.n_words_model),
                  '--save-dir', model_save_dir,
                  '--batch-size', str(64),
                  '--arch', args.model_arch]

        if args.cuda != -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

        if args.fp16:
            t_args.append('--fp16')

        print(t_args)

        train = subprocess.Popen(t_args,
                                 stdout=subprocess.PIPE)
        output = train.communicate()[0]
