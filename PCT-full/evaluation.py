import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import os
import numpy as np
import gym


def _preset_cuda_visible_device():
    argv = sys.argv[1:]
    if '--no_cuda' in argv:
        return

    device_idx = None
    for i, arg in enumerate(argv):
        if arg == '--device' and i + 1 < len(argv):
            device_idx = argv[i + 1]
            break
        if arg.startswith('--device='):
            device_idx = arg.split('=', 1)[1]
            break

    if device_idx is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_idx)
        os.environ['PCT_JITTOR_DEVICE_ID'] = str(device_idx)


_preset_cuda_visible_device()

import jittor as jt

from model import *
from tools import *
from evaluation_tools import evaluate_PCT, evaluate_CDRL
from arguments import get_args


def _move_module_to_device(module, device):
    if hasattr(module, 'to'):
        try:
            return module.to(device)
        except Exception:
            pass
    return module


def main(args):
    if args.custom is not None:
        custom = args.custom
    else:
        custom = input('Please input the evaluate name\n')

    timeStr = custom
    if not args.without_time_str:
        timeStr += '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
    args.timeStr = timeStr

    if args.no_cuda:
        device = 'cpu'
        jt.flags.use_cuda = 0
        args.device_id = None
    else:
        device = 'cuda'
        jt.flags.use_cuda = 1
        args.device_id = int(os.environ.get('PCT_JITTOR_DEVICE_ID', str(args.device)))
    args.device = device

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.make(args.id, args=args, disable_env_checker=True)

    if args.drl_method != 'rainbow':
        if args.model_architecture == 'PCT':
            PCT_policy = DRL_GAT(args)
            if args.load_model:
                PCT_policy = load_policy(args.model_path, PCT_policy, args)
                print('Loading pre-train model', args.model_path)
        else:
            import CDRL_model
            channel = 6
            if args.setting == 2:
                channel = 10
            if args.setting == 3:
                channel = 7
            if args.practical_constrain is not None and 'category' in args.practical_constrain:
                channel += 1
            PCT_policy = CDRL_model.Policy(
                envs.observation_space.shape, envs.action_space, channel=channel,
                base_kwargs={'recurrent': False, 'hidden_size': 256})
            PCT_policy = _move_module_to_device(PCT_policy, device)
            if args.load_model:
                PCT_policy = load_policy(args.model_path, PCT_policy, args)
                print('Loading pre-train model', args.model_path)
    else:
        from rainbow_model import RainbowAgent
        PCT_policy = RainbowAgent(args)

    backup(timeStr, args, None)

    if args.model_architecture == 'CDRL':
        print('CDRL')
        evaluate_CDRL(PCT_policy, envs, timeStr, args, device, eval_freq=args.evaluation_episodes)
    elif args.simple_test:
        print('Simple test')
        evaluate_PCT(PCT_policy, envs, timeStr, args, device, eval_freq=args.evaluation_episodes, factor=args.normFactor)


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)
