import time as mytime
import numpy as np
import random

def main(args):
    # Import Jittor-dependent modules lazily so spawned env workers do not
    # initialize Jittor/CUDA again when they import this module.
    import gym
    import jittor as jt
    from models.graph_attention import DRL_GAT
    from envs import make_vec_envs
    from ppo import PPO_Training

    custom = args.custom
    timeStr = custom + '-' + mytime.strftime('%Y.%m.%d-%H-%M-%S', mytime.localtime(mytime.time()))

    if args.no_cuda:
        jt.flags.use_cuda = 0
        device = 'cpu'
    else:
        jt.flags.use_cuda = 1
        device = 'cuda'

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    envs = make_vec_envs(args, None, True)

    bppObs_size = (args.num_box + args.num_next_box + args.num_candidate_action) * args.node_dim
    BPP_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(bppObs_size, )),
                         gym.spaces.Discrete(n=args.num_candidate_action),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         args.num_candidate_action,
                         args.node_dim,
                         policy_ctg='place',
                         )

    advObs_size = (args.num_box + args.num_next_box) * args.node_dim
    adv_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(advObs_size, )),
                         gym.spaces.Discrete(n=args.num_next_box),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         0,
                         args.node_dim,
                         policy_ctg='permutation',
                         )

    balObs_size = (args.num_box + args.num_next_box) * args.node_dim
    bal_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(balObs_size,)),
                         gym.spaces.Discrete(n=args.num_next_box),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         0,
                         args.node_dim,
                         policy_ctg='permutation',
                         )

    BPP_policy = BPP_policy
    adv_policy = adv_policy
    bal_policy = bal_policy


    train_model = PPO_Training(BPP_policy,
                               adv_policy,
                               bal_policy,
                               args,)

    train_model.train_n_steps(envs, args, device, timeStr)


if __name__ == '__main__':
    from envs import registration_envs
    from utils import get_args

    registration_envs()
    args = get_args()
    main(args)


