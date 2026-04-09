import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import jittor as jt
from model import *
from tools import *
from evaluation_tools import evaluate
import gym

def main(args):
    # The name of this evaluation, related file backups and experiment tensorboard logs will
    # be saved to '.\logs\evaluation' and '.\logs\runs'
    custom = args.exp_name if getattr(args, 'exp_name', None) else input('Please input the evaluate name\n')
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if args.no_cuda:
        jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = 1

    device = 'cpu' if args.no_cuda else 'cuda'

    jt.set_global_seed(args.seed)

    # Create single packing environment and load existing dataset.
    envs = gym.make(args.id,
                    setting = args.setting,
                    container_size=args.container_size,
                    item_set=args.item_size_set,
                    data_name=args.dataset_path,
                    load_test_data = args.load_dataset,
                    internal_node_holder=args.internal_node_holder,
                    leaf_node_holder=args.leaf_node_holder,
                    LNES = args.lnes,
                    shuffle=args.shuffle,
                    sample_from_distribution=args.sample_from_distribution,
                    sample_left_bound=args.sample_left_bound,
                    sample_right_bound=args.sample_right_bound
                   )

    # Create the main actor & critic networks of PCT
    PCT_policy =  DRL_GAT(args)
    # Jittor manages device automatically

    # Load the trained model
    if args.load_model:
        PCT_policy = load_policy(args.model_path, PCT_policy)
        print('Pre-train model loaded!', args.model_path)

    # Backup all py file
    backup(timeStr, args, None)
    
    # Perform all evaluation.
    evaluate(PCT_policy, envs, timeStr, args, device,
             eval_freq=args.evaluation_episodes, factor=args.normFactor)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)
