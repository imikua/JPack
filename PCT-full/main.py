import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import random
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def _preset_cuda_visible_device():
    # Jittor chooses visible GPUs during import/initialization. To keep the
    # CLI semantics close to the original torch version, parse `--device`
    # before importing jittor and map it through CUDA_VISIBLE_DEVICES.
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

import numpy as np


def _warmup_jittor(args, packing_policy, jt):
    """Run a single forward pass to trigger JIT compilation before training."""
    try:
        print('[warmup] starting JIT warmup ...')
        if args.drl_method == 'rainbow':
            dummy = jt.randn((1, args.internal_node_holder + args.leaf_node_holder + 1, 9))
            out = packing_policy.online_net(dummy, log=False)
            _ = out.mean()
            jt.sync_all()
            print('[warmup] done (rainbow)')
            return

        if args.model_architecture == 'PCT':
            dummy = jt.randn((1, args.internal_node_holder + args.leaf_node_holder + 1, 9))
            action_log_prob, pointers, dist_entropy, value = packing_policy(dummy, normFactor=args.normFactor)
            # Force evaluation to trigger compilation of forward kernels
            _ = value.numpy()
            jt.sync_all()
            print('[warmup] done (PCT forward)')
    except Exception as e:
        print(f'[warn] warmup skipped due to: {e}')


def _move_module_to_device(module, device):
    # Jittor uses global CUDA flags / visible devices, so most modules do not
    # implement torch-style `.to(device)`. Keep this helper for parity with the
    # original torch entrypoint while remaining no-op on Jittor modules.
    if hasattr(module, 'to'):
        try:
            return module.to(device)
        except Exception:
            pass
    return module

# PackE CDRL RCQL Attend2Pack
# --setting 2 --custom packe_pg --drl_method acktr --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --training_without_evaluate --distribution pg --model_architecture PackE --drl_method a2c --num_processes 16 --load_dataset --dataset_path data/time_series/pg.xlsx
# --setting 2 --custom packe_deli --drl_method acktr --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --training_without_evaluate --distribution deli --model_architecture PackE --drl_method a2c --num_processes 16 --load_dataset --dataset_path data/occupancy/deli.xlsx
# --setting 2 --custom packe_opai --drl_method acktr --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --training_without_evaluate --distribution opai --model_architecture PackE --drl_method a2c --num_processes 16 --load_dataset --dataset_path data/flat_long/opai.txt

def main(args):
    # Lazy imports: keep jittor and all model/training modules OUT of the
    # top level so that spawn worker processes (which re-execute this module)
    # never accidentally import jittor and initialise CUDA.
    import jittor as jt
    import math
    from tensorboardX import SummaryWriter
    from model import DRL_GAT
    from tools import backup, load_policy
    from envs import make_vec_envs
    from train_tools import train_tools
    from rainbow_model import RainbowAgent

    try:
        jt.flags.compile_threads = 1
    except Exception:
        pass
    try:
        jt.flags.use_parallel_op_compiler = 0
    except Exception:
        pass

    if args.custom is not None:
        custom = args.custom
    else:
        custom = input('Please input the experiment name\n')

    # custom = "debug-offline"
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

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
    random.seed(args.seed)

    # Backup all py files and create tensorboard logs
    backup(timeStr, args, None)
    log_writer_path = './logs/runs/{}'.format('PCT-' + timeStr)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(logdir=log_writer_path)

    vec_mode = 'spawn-shmem' if args.num_processes > 1 else 'dummy-single-process'
    print(f'[runtime] vec env mode: {vec_mode}')
    print(f'[runtime] requested num_processes: {args.num_processes}')

    print('[DEBUG] Entering make_vec_envs...')
    # Create parallel packing environments to collect training samples online
    envs = make_vec_envs(args, './logs/runinfo', True)
    print('[DEBUG] Exited make_vec_envs.')

    # Create the main actor & critic networks of PCT
    if args.drl_method == 'rainbow':
        packing_policy = RainbowAgent(args)
    elif args.model_architecture == 'PCT':
        packing_policy = DRL_GAT(args)
    elif args.model_architecture == 'CDRL':
        import CDRL_model
        if args.setting == 1: channel = 6
        if args.setting == 2: channel = 6
        if args.setting == 3: channel = 7
        packing_policy = CDRL_model.Policy(
            envs.observation_space.shape, envs.action_space, channel=channel, container_size=args.container_size,
            base_kwargs={'recurrent': False, 'hidden_size' : 256})
        packing_policy = _move_module_to_device(packing_policy, device)
    elif args.model_architecture == 'PackE':
        import PackE_model
        packing_policy = PackE_model.Policy(
            envs.observation_space.shape, envs.action_space, args.container_size,
            base_kwargs={'recurrent': False, 'hidden_size' : 256})
        packing_policy = _move_module_to_device(packing_policy, device)
    elif args.model_architecture == 'Attend2Pack':
        from models.model_Attend2Pack import Attend2Pack
        originaldim = 3
        embedingdim = 128
        num_head = 8
        FFNdim = 512
        AFFNnum = 3
        inchannel = 2
        outchannel = 4
        C = 10
        outdim = 2 * args.container_size[0]
        L, W = args.container_size[0:2]
        scale = 1 / math.sqrt(embedingdim // num_head)
        packing_policy = Attend2Pack(originaldim, embedingdim, num_head, FFNdim, AFFNnum, inchannel, outchannel, C, outdim,
                                  L, W, scale)
        packing_policy = _move_module_to_device(packing_policy, device)
    elif args.model_architecture == 'RCQL':
        from models.model_RCQL import RCQL
        state_size = 6
        hidden_size = 128
        nb_heads = 8
        encoder_nb_layers = 3
        attn_span = 20
        inner_hidden_size = 512
        src_head_hidden_size = 128
        pos_head_hidden_size = 512
        s_res_size = 1
        r_res_size = 2
        x_res_size, y_res_size = args.container_size[0:2]
        decoder_nb_layers = 1
        item_state_size = 3
        packing_policy = RCQL(state_size, hidden_size, nb_heads, encoder_nb_layers, attn_span, inner_hidden_size,
                            src_head_hidden_size, pos_head_hidden_size, s_res_size, r_res_size,
                            x_res_size, y_res_size, decoder_nb_layers, item_state_size)
        packing_policy = _move_module_to_device(packing_policy, device)
    else:
        assert False

    # Load the trained model, if needed
    # args.load_model = True
    if args.load_model:
        # args.model_path = '/media/wzf/goodluck/Workspace/rl/packe/packe_pg_2025.01.15-05-56-56.pt'
        packing_policy = load_policy(args.model_path, packing_policy, args)
        print('Loading pre-train model', args.model_path)

    _warmup_jittor(args, packing_policy, jt)

    # Perform all training.
    trainTool = train_tools(writer, timeStr, packing_policy, args)
    if args.drl_method == 'rainbow':
        trainTool.train_q_value(envs, args)
    elif args.model_architecture == 'CDRL':
        trainTool.train_CDRL(envs, args)
    elif args.model_architecture == 'PCT':
        trainTool.train_PCT(envs, args)
    elif args.model_architecture == 'Attend2Pack':
        trainTool.train_Attend2Pack(envs, args)
    elif args.model_architecture == 'RCQL':
        trainTool.train_RCQL(envs, args)
    elif args.model_architecture == 'PackE':
        trainTool.train_PackE(envs, args)

if __name__ == '__main__':
    from envs import registration_envs
    from arguments import get_args
    registration_envs()
    args = get_args()
    main(args)