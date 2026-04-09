import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import numpy as np
import random
import os
from tensorboardX import SummaryWriter

def main(args):
    import jittor as jt
    from model import DRL_GAT
    from tools import backup, load_policy
    from envs import make_vec_envs
    from train_tools import train_tools

    # The name of this experiment, related file backups and experiment tensorboard logs will
    # be saved to '.\logs\experiment' and '.\logs\runs'
    custom = args.exp_name if getattr(args, 'exp_name', None) else input('Please input the experiment name\n')
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if args.no_cuda:
        jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = 1

    device = 'cpu' if args.no_cuda else 'cuda'

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Backup all py files and create tensorboard logs
    backup(timeStr, args, None)
    log_writer_path = './logs/runs/{}'.format('PCT-' + timeStr)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(logdir=log_writer_path)

    # Create parallel packing environments to collect training samples online
    envs = make_vec_envs(args, './logs/runinfo', True)

    # Create the main actor & critic networks of PCT
    PCT_policy =  DRL_GAT(args)
    # Jittor manages device automatically

    # Load the trained model, if needed
    if args.load_model:
        PCT_policy = load_policy(args.model_path, PCT_policy)
        print('Loading pre-train model', args.model_path)

    # Perform all training.
    trainTool = train_tools(writer, timeStr, PCT_policy, args)
    trainTool.train_n_steps(envs, args, device)

if __name__ == '__main__':
    from tools import get_args, registration_envs
    registration_envs()
    args = get_args()
    main(args)