import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        metavar='LR',
        help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--tau',
        type=float,
        default=1.00,
        metavar='T',
        help='parameter for GAE (default: 1.00)')
    parser.add_argument(
        '--beta',
        type=float,
        default=1e-2,
        help='entropy regularization term')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--action-space',
        type=int,
        default=6,
        metavar='AS',
        help='# of actions (default: 5)')
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        metavar='W',
        help='how many training processes to use (default: 32)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=30,
        metavar='NS',
        help='number of forward steps in A3C (default: 50)')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=100,
        metavar='M',
        help='maximum length of an episode (default: 100)')
    parser.add_argument(
        '--env-config',
        default='config.json',
        metavar='EC',
        help='environment to crop and resize info (default: config.json)')
    parser.add_argument(
        '--shared-optimizer',
        default=True,
        metavar='SO',
        help='use an optimizer with shared statistics.')
    parser.add_argument(
        '--load_model',
        type=str,
        default='',
        help='Path to load a saved model.')
    parser.add_argument(
        '--save-freq',
        type=int,
        default=1e+7,
        help='save model after this # of training frames (default: 1e+6)')
    parser.add_argument(
        '--optimizer',
        default='SharedAdam',
        metavar='OPT',
        help='shared optimizer choice of SharedAdam or SharedRMSprop')
    parser.add_argument(
        '--save-model-dir',
        default='trained_models/',
        metavar='SMD',
        help='folder to save trained navigation')
    parser.add_argument(
        '--log-dir',
        default='logs/',
        metavar='LG',
        help='folder to save logs')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument(
        '--amsgrad',
        default=True,
        metavar='AM',
        help='Adam optimizer amsgrad parameter')
    parser.add_argument(
        '--grid_size',
        type=float,
        default=0.25,
        metavar='GS',
        help='The grid size used to discretize AI2-THOR maps.')
    parser.add_argument(
        '--docker_enabled',
        action='store_true',
        help='Whether or not to use docker.')
    parser.add_argument(
        '--x-display',
        type=str,
        default=None,
        help='The X display to target, if any.')
    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='If true, logs will contain more information.')
    parser.add_argument(
        '--num-train-episodes',
        type=float,
        default=100000,
        help='maximum # of episodes')
    parser.add_argument(
        '--num-test-episodes',
        type=float,
        default=1000,
        help='maximum # of episodes')

    parser.add_argument(
        '--model',
        type=str,
        default='BaseModel',
        help='Which model to use.')
    parser.add_argument(
        '--train-thin',
        type=int,
        default=10,
        help='Frequency of logging.')
    parser.add_argument(
        '--test-thin',
        type=int,
        default=10,
        help='Frequency of logging.')
    parser.add_argument(
        '--hindsight-replay',
        type=bool,
        default=False,
        help='whether or not to use hindsight replay.')
    parser.add_argument(
        '--title',
        type=str,
        default='default',
        help='Title for logging.')
    parser.add_argument(
        '--train-scenes',
        type=str,
        default='[1-20]',
        help='scenes for training.')
    parser.add_argument(
        '--hidden-state-sz',
        type=int,
        default=512,
        help='size of hidden state of LSTM.')
    parser.add_argument(
        '--fov',
        type=float,
        default=90.0,
        help='The field of view to use.')
    parser.add_argument(
        '--enable-logging',
        action='store_true',
        default=True,
        help='Use tensorboard for logging.')
    parser.add_argument(
        '--randomize-objects',
        action='store_true',
        default=False,
        help='Randomize object locations at start of episode.')
    parser.add_argument(
        '--arch',
        type=str,
        choices=['osx', 'linux'],
        default='linux',
        help='OS.')
    parser.add_argument(
        '--scenes',
        type=int,
        choices=[1,2,3],
        default=3,
        help='# of scenes to use')
    parser.add_argument(
        '--train_threshold',
        type=float,
        default=0.98,
        help='Success rate required to finish training.')
    parser.add_argument(
        '--load-model',
        type=str,
        default='',
        help='which model file to start from.')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='whether or not to resume from ./model/last_model')

    return parser.parse_args()
