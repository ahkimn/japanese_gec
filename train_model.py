import argparse
import os
import subprocess
import shutil

from src import config
from src import util
from src import architectures
from src.datasets import Dataset
from src.util import str_bool

cfg = config.parse()
DS_PARAMS = cfg['dataset_params']
MDL_PARAMS = cfg['model_params']
DIRECTORIES = cfg['directories']

SAVE_PARAMS = DS_PARAMS['save_names']

os.environ['KMP_INIT_AT_FORK'] = 'FALSE'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model given \
            train/dev/test Dataset instances')

    # ====================================================
    #    Parameters for loaded Dataset instances
    # ====================================================

    # Required for write/preprocess
    parser.add_argument(
        '--ds_load_dir', metavar='DS_LOAD_DIR',
        type=str, help='sub-directory of ./data/datasets \
            where Dataset instances are saved', required=False, default=None)

    parser.add_argument(
        '--ds_suffix', metavar='DS_SUFFIX', default=SAVE_PARAMS['ds_suffix'],
        type=str, help='File extension of saved Dataset instances',
        required=False)

    # Required for write/preprocess
    parser.add_argument(
        '--ds_name_train', metavar='DS_NAME_TRAIN', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            containing training data',
        required=False, default=None)

    # Required for write/preprocess
    parser.add_argument(
        '--ds_name_dev', metavar='DS_NAME_DEV', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            containing development data',
        required=False, default=None)

    # Required for write/preprocess
    parser.add_argument(
        '--ds_name_test', metavar='DS_NAME_TRAIN', type=str,
        help='filename (preceding extension) of saved Dataset instance \
            containing test data',
        required=False, default=None)

    # ====================================================
    #         Parameters for model/model training
    # ====================================================

    parser.add_argument(
        '--cuda', metavar='CUDA', default=-1, type=int,
        help='if not -1, index of GPU to use',
        required=False)

    parser.add_argument(
        '--cpu', metavar='CUDA', default=True, type=str_bool,
        help='if True, use the CPU instead of CUDA',
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
        '--model_save_dir', metavar='MODEL_SAVE_DIR', type=str,
        default='mdl',
        help='sub-directory of \'./models\' to save model to',
        required=False)

    parser.add_argument(
        '--model_arch', metavar='MODEL_ARCH', type=str,
        help='architecture of model (registered in Fairseq) to use',
        default='fconv_jp_current', required=False)

    parser.add_argument(
        '--batch_size', metavar='BATCH_SIZE', default=32, type=str,
        help='batch size to use when training', required=False)

    # ====================================================
    #               Other parameters
    # ====================================================

    parser.add_argument(
        '--tmp_dir', metavar='TMP_DIR', type=str, default=DIRECTORIES['tmp'],
        help='sub-directory of (./data/tmp/) to write temporary files to',
        required=False)

    parser.add_argument(
        '--command', metavar='COMMAND', type=str,
        help='one of [\'write\', \'preprocess\', \'train\', \'clean\', \
            or \'all\']. Determines which operation to run',
        required=True)

    args = parser.parse_args()
    command = args.command.lower()

    assert(command in ['write', 'preprocess', 'train', 'clean', 'all'])
    tmp_dir = args.tmp_dir

    write_dir = os.path.join(tmp_dir, 'write')
    prep_dir = os.path.join(tmp_dir, 'preprocess')

    model_save_dir = os.path.join(DIRECTORIES['models'],
                                  args.model_save_dir)

    if not os.path.isdir(tmp_dir):
        util.mkdir_p(tmp_dir)

    ds_specified = all([args.ds_load_dir, args.ds_name_train,
                        args.ds_name_dev, args.ds_name_test])

    if ds_specified:

        f_names = [args.ds_name_train, args.ds_name_dev, args.ds_name_test]
        w_names = [os.path.join(write_dir, f) for f in f_names]

    if command == 'write' or command == 'all':

        assert(ds_specified)

        if not os.path.isdir(write_dir):
            util.mkdir_p(write_dir)

        for f in f_names:
            ds_load_file = os.path.join(DIRECTORIES['datasets'],
                                        args.ds_load_dir, '%s.%s' %
                                        (f, args.ds_suffix))

            print('Loading dataset: %s' % ds_load_file)

            DS = Dataset.load(ds_load_file)
            DS.write(write_dir, token_delimiter=' ', data_delimiter='',
                     include_tags=[], separation='none', max_per_rule=-1,
                     save_prefix=f)

    if command == 'preprocess' or command == 'all':

        assert(ds_specified)

        if os.path.isdir(prep_dir):
            shutil.rmtree(prep_dir)

        prep = subprocess.Popen(
            ['fairseq-preprocess',
             '--source-lang', SAVE_PARAMS['error_suffix'],
             '--target-lang', SAVE_PARAMS['correct_suffix'],
             '--trainpref', w_names[0],
             '--validpref', w_names[1],
             '--testpref', w_names[2],
             '--destdir', prep_dir,
             '--workers', str(4)],
            stdout=subprocess.PIPE)

        output = prep.communicate()[0]

    if command == 'train' or command == 'all':\

        if os.path.isdir(model_save_dir):

            print('WARNING: Model directory exists')
            dec = input('\tDelete model_dir (y/n)?:')
            if dec.lower() == 'y':
                shutil.rmtree(model_save_dir)
            else:
                raise ValueError('Model directory not empty')

        t_args = ['fairseq-train', prep_dir,
                  '--lr', str(0.1),
                  '--clip-norm', str(0.1),
                  '--dropout', str(0.1),
                  '--max-tokens', str(args.n_words_model),
                  '--save-dir', model_save_dir,
                  '--batch-size', str(args.batch_size),
                  '--arch', args.model_arch]

        if args.cpu:
            t_args.append('--cpu')

        if args.cuda != -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

        if args.fp16:
            t_args.append('--fp16')

        train = subprocess.Popen(t_args,
                                 stdout=subprocess.PIPE)

        output = train.communicate()[0]

    # Clean tmp/model directories, move dicts to model directory
    if command == 'clean':

        assert(os.path.isdir(model_save_dir))
        assert(os.path.isdir(prep_dir))

        dict_source = 'dict.%s.txt' % SAVE_PARAMS['error_suffix']
        dict_target = 'dict.%s.txt' % SAVE_PARAMS['correct_suffix']

        model_best = 'checkpoint_best.pt'

        prep_dir_files = os.listdir(prep_dir)
        model_dir_files = os.listdir(model_save_dir)

        assert(dict_source in prep_dir_files)
        assert(dict_target in prep_dir_files)
        assert(model_best in model_dir_files)

        new_dict_source = os.path.join(model_save_dir, dict_source)
        new_dict_target = os.path.join(model_save_dir, dict_target)

        dict_source = os.path.join(prep_dir, dict_source)
        dict_target = os.path.join(prep_dir, dict_target)

        # Delete non-best checkpoints
        for f in os.listdir(model_save_dir):

            if f not in [model_best, dict_source, dict_target]:
                f = os.path.join(model_save_dir, f)
                os.remove(f)

        shutil.copyfile(dict_source, new_dict_source)
        shutil.copyfile(dict_target, new_dict_target)

        # Remove tmemporary directory
        shutil.rmtree(tmp_dir)
