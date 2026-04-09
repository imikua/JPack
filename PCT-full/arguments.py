import argparse
import numpy as np
import givenData
import sys

def parse_comma_separated_floats(value):
    return [float(item) for item in value.split(',')]


def _flag_was_provided(flag):
    argv = sys.argv[1:]
    return any(arg == flag or arg.startswith(flag + '=') for arg in argv)

def get_args():
    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='PCT arguments')
    parser.add_argument('--custom', type=str, default = None, help='Custom name for this training/test')
    parser.add_argument('--without_time_str', action='store_true', help='Save log without time str addation')

    # todo parameters for the PCT method
    parser.add_argument('--setting', type=int, default=2, help='Experiment setting, please see our paper for details')
    parser.add_argument('--lnes', type=str, default='EMS',
                        help='Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC')
    parser.add_argument('--internal_node_holder', type=int, default=80, help='Maximum number of internal nodes')
    parser.add_argument('--leaf_node_holder', type=int, default=50, help='Maximum number of leaf nodes')
    parser.add_argument('--shuffle', type=bool, default=True, help='Randomly shuffle the leaf nodes')
    parser.add_argument('--continuous', action='store_true',
                        help='Use continuous enviroment, otherwise the enviroment is discrete')

    parser.add_argument('--no_cuda', action='store_true', help='Forbidden cuda')
    parser.add_argument('--device', type=int, default=0, help='Which GPU will be called')
    parser.add_argument('--seed', type=int, default=4, help='Random seed')

    parser.add_argument('--num_processes', type=int, default=64,
                        help='The number of parallel processes used for training')
    parser.add_argument('--num_steps', type=int, default=5, help='The rollout length for n_step training')
    parser.add_argument('--learning_rate', type=float, default=1e-6, metavar='η',
                        help='Learning rate, only works for A2C')
    parser.add_argument('--actor_loss_coef', type=float, default=1.0, help='The coefficient of actor loss')
    parser.add_argument('--critic_loss_coef', type=float, default=1.0, help='The coefficient of critic loss')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max norm of gradients')

    parser.add_argument('--embedding_size', type=int, default=64, help='Dimension of input embedding')

    parser.add_argument('--hidden_size', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--gat_layer_num', type=int, default=1, help='The number GAT layers')
    parser.add_argument('--head_num',      type=int, default=1, help='The number of attention heads')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='γ', help='Discount factor')

    parser.add_argument('--model_save_interval', type=int, default=200, help='How often to save the model')
    parser.add_argument('--model_update_interval', type=int, default=20e3, help='How often to create a new model')
    parser.add_argument('--model_save_path', type=str, default='./logs/experiment',
                        help='The path to save the trained model')
    parser.add_argument('--regular_model_save_path', type=str, default='./logs/experiment',
                        help='The regular path to save the trained model')
    parser.add_argument('--print_log_interval', type=int, default=10, help='How often to print training logs')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation_episodes', type=int, default=100, metavar='N',
                        help='Number of episodes evaluated')
    parser.add_argument('--load_model', action='store_true', help='Load the trained model')
    parser.add_argument('--model_path', type=str, default='', help='The path to load model')
    parser.add_argument('--load_dataset', action='store_true',
                        help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset_path', type=str, help='The path to load dataset')

    parser.add_argument('--sample_from_distribution', action='store_true',
                        help='Sample continuous item size from a uniform distribution U(a,b), otherwise sample items from \'item_size_set\' in \'givenData.py\'')
    parser.add_argument('--sample_left_bound', type=float, metavar='a', help='The parametre a of distribution U(a,b)')
    parser.add_argument('--sample_right_bound', type=float, metavar='b', help='The parametre b of distribution U(a,b)')

    parser.add_argument('--distribution', type=str, default='uniform', help='uniform, normal, mixture, icra, deli, deli_prob, deli_deploy, or mixLength')
    parser.add_argument('--normal_mean', type=float, metavar='μ', default=0.5, help='The parametre μ of normal distribution')
    parser.add_argument('--normal_std', type=float, metavar='σ',  default=0.5, help='The parametre σ of normal distribution')
    parser.add_argument('--height_reward', action='store_true', help='Add a low height preference reward.')
    parser.add_argument('--distribution_supervision', type=str, default='none', help='sup, con, none, Train packing policy with distribution supervision, con means supcon here')
    # parser.add_argument('--attention_without_leaf', action='store_true',   help='Pointer attention without leaf node')
    parser.add_argument('--feature_aggregation', type=str, default='mean', help='feature aggregation method: mean, no_leaf, item_attention or item')
    parser.add_argument('--no_internal_node_input', action='store_true', help='Set the internal node input to zero')
    parser.add_argument('--repeat_embed', type=int, default=1, help='Repeat the embedding for multiple times')
    parser.add_argument('--new_attention', action='store_true', help='Encode the input with new attention')
    parser.add_argument('--inner_leaf_attention_eliminate', action='store_true', help='Eliminate the inner leaf attention for better generazation')
    parser.add_argument('--specified_mask_id', type=int, default=0, help='The id of specified mask to explore which parts impact the most')
    parser.add_argument('--pred_value_with_heightmap', action='store_true', help='Critic only predict the value with heightmap')
    parser.add_argument('--practical_constrain', type=str, default=None, help='bridge, variance, physics, robot, stability, loading, category, or None')

    parser.add_argument('--draw_attention', action='store_true', help='Draw attention with color map')
    parser.add_argument('--reward_evolution', action='store_true', help='Train agent with different upper bound with reward evolution')
    parser.add_argument('--evolution_reward_datasize', type=int, default=200, help='Dataset size for training agent with reward evolution')
    # parser.add_argument('--leaf_without_maxz', action='store_true', help='Train agent without set the leaf node to maxZ')
    parser.add_argument('--linear_net', action='store_true',               help='Expect better generazation with linear net')
    parser.add_argument('--cate_attn',  action='store_true',               help='Sort attention by category')
    parser.add_argument('--add_distribution',  action='store_true',        help='add distribution for discover generation')

    parser.add_argument('--training_without_evaluate', action='store_true', help='Training without evaluate')
    parser.add_argument('--evaluate_with_large_data', action='store_true', help='Evaluate with large scale data')
    parser.add_argument('--evaluate_with_no_interval_node', action='store_true', help='Evaluate with no interval node')
    parser.add_argument('--evaluate_with_given_data', action='store_true', help='Evaluate with given data')
    parser.add_argument('--drl_method', type=str, default='acktr', help='acktr, a2c, rainbow, rehearsal, incre')

    # todo Parameters for the rainbow reinforcement learning agent
    parser.add_argument('--noisy_std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=31, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V_min', type=float, default=-1, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V_max', type=float, default=8, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--target_update', type=int, default=int(1e3), metavar='τ',
                        help='Number of steps after which to update target network')
    parser.add_argument('--multi_step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--reward_clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--rainbow_batch_size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm_clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--memory_capacity', type=int, default=int(1e5), metavar='CAPACITY',
                        help='Experience replay memory capacity')
    parser.add_argument('--replay_frequency', type=int, default=4, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--priority_exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority_weight', type=float, default=1.0, metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T_max', type=int, default=int(50e6), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--max_episode_length', type=int, default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history_length', type=int, default=1, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'dataset-efficient'],
                        metavar='ARCH', help='Network architecture')
    parser.add_argument('--learn_start', type=int, default=int(5e2), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--evaluation_size', type=int, default=500, metavar='N',
                        help='Number of transitions to use for validating Q')

    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable_cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')

    parser.add_argument('--checkpoint_interval', default=10000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--save_interval', default=1000, help='How often to save the model.')

    parser.add_argument('--disable_bzip_memory', action='store_true',
                        help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--heuristic', type=str, default='LSAH', help='Options: LSAH DBL MACS OnlineBPH HM BR RANDOM')
    parser.add_argument('--simple_test', action='store_true', help='Test with simple online packing problem')
    parser.add_argument('--env_version', type=int, default=0, help='Debug with new environment')
    parser.add_argument('--draw', action='store_true', help='Draw the packing process')
    parser.add_argument('--actor_with_q', action='store_true', help='Train actor critic with Q intergeration')
    parser.add_argument('--with_four_corner', action='store_true', help='Train packing policy with four corners')
    parser.add_argument('--box_bound', type=int, default=40, help='How many boxes each subcontainer can have')
    # funsion function: return, v, q, LSAH, minZ, LLM, DBL, TD_error
    parser.add_argument('--fusion_function', type=str, default='minZ+', help='The function to be fuse subcontainers')
    parser.add_argument('--local_height', action='store_true', help='Use local height as subcontainer')

    # extend BPP-1 to BPP-k
    parser.add_argument('--preview', type=int, default=1, help='The number of objects we can preview')
    parser.add_argument('--select', type=int, default=1, help='The number of objects we can select')
    parser.add_argument('--pruning_threshold', type=float, default=0.1, help='pruning threshold')
    parser.add_argument('--use_cached_tree', action='store_true', help='Use cached mcts to accelerate search')

    # for offline
    parser.add_argument('--offline', action='store_true', help='Set the problem to be offline packing problem, otherwise online')
    parser.add_argument('--known', type=int, default=50, help='Total box number, only used in offline packing problem')

    # for large scale
    parser.add_argument('--large_scale', action='store_true', help='Set the problem to be large scale packing problem')

    # for learnable
    parser.add_argument("--load_mcts_model", action="store_true", help="Load pretrained learnable mcts model")
    parser.add_argument("--mcts_model_path", type=str, default='', help="The path to load pretrained learnable mcts model")

    parser.add_argument('--policy_positional_encoding',  action='store_true',  help='Add positional encoding to the policy input')
    # for packing policy training
    parser.add_argument('--ppo_epoch',                   type=int, default= 4,   help='How many epoches the ppo agent is trained')
    parser.add_argument('--num_mini_batch',              type=int, default= 32,   help='How many mini batch for ppo epoch')

    parser.add_argument('--use_distilled_critic', action='store_true', help='Use MCTS with distilled critic')
    parser.add_argument('--container_size', type=parse_comma_separated_floats,  default=[1, 1, 1], help='The size of the container')
    parser.add_argument('--item_size_set',  type =str, default='discrete', help='The item size set')
    parser.add_argument('--update_container_method',  type =str, default='recursive', help='recursive, cheb')
    parser.add_argument('--no_next_item_input',       action='store_true',  help='not input the next item as obs')
    parser.add_argument('--model_architecture',       type =str, default='PCT', help='PCT, CDRL, Attend2Pack, RCQL, Pack-E')

    args = parser.parse_args()
    learning_rate_was_explicit = _flag_was_provided('--learning_rate')

    args.load_memory_path = None
    args.save_memory_path = None
    args.container_size = [int(x) for x in args.container_size]
    if args.evaluate:
        args.num_processes = 1
    else:
        args.evaluation_episodes = 25

    if args.drl_method == 'rainbow':
        args.learning_rate = 0.0000625

    if (
        args.drl_method == 'a2c'
        and args.model_architecture in {'CDRL', 'RCQL', 'Attend2Pack', 'PackE'}
        and not learning_rate_was_explicit
        and args.learning_rate == 1e-6
    ):
        args.learning_rate = 1e-4
        print(f"[runtime] auto-set learning_rate={args.learning_rate} for {args.model_architecture}+A2C")

    if args.model_architecture == 'CDRL' and args.drl_method == 'a2c':
        print("[runtime] note: original torch commands primarily use CDRL with ACKTR; A2C is a fallback/smoke path.")

    if args.no_cuda: args.device = 'cpu'

    if args.item_size_set == 'discrete':
        args.item_size_set = givenData.get_discrete_dataset()

    if args.distribution == 'deli_deploy':
        args.item_size_set  = givenData.get_deli_prob_dataset()
        args.container_size = [120, 100, 170]

    # add new dataset
    if args.distribution == 'pg':
        args.item_size_set  = [[43, 20,  7],[45, 30,  6]]
        args.container_size = [134, 125, 100]
        args.env_version = 3
        args.continuous = False

    if args.distribution == 'deli':
        args.item_size_set  = [[43, 20,  7],[45, 30,  6]]
        args.container_size = [120, 100, 170]
        args.env_version = 3
        args.continuous = False

    if args.distribution == 'opai':
        args.item_size_set  = [[43, 20,  7],[45, 30,  6]]
        args.container_size = [250, 120, 100]
        args.env_version = 3
        args.continuous = False

    if args.distribution == 'real_opai':
        args.item_size_set  = givenData.get_real_opai_dataset()
        args.container_size = [250, 120, 100]
        args.env_version = 3
        args.continuous = False

    if args.sample_from_distribution and args.sample_left_bound is None:
        args.sample_left_bound = 0.1 * min(args.container_size)
    if args.sample_from_distribution and args.sample_right_bound is None:
        args.sample_right_bound = 0.5 * min(args.container_size)

    if args.continuous:
            args.id = 'PctContinuous-v{}'.format(args.env_version)
    else:
        args.id = 'PctDiscrete-v{}'.format(args.env_version)

    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    args.leaf_node_length = 6
    if args.practical_constrain is not None:
        if 'category' in args.practical_constrain:
            args.internal_node_length = 8

    if args.evaluate:
        args.num_processes = 1
    args.normFactor = 1.0 / np.max(args.container_size)

    assert args.select <= args.preview

    if args.offline:
        args.preview = args.known
        args.select = args.known

    if args.large_scale and args.load_model:
        if args.model_path == '':
            args.model_path = 'model/conti-01.pt'
        args.normFactor = 1.0

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    print(args.id)
    return args