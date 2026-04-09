import IPython
import gymnasium as gym
import numpy as np
import tapnet
import tianshou as ts
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
import argparse
import jittor as jt
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.data import Batch, to_numpy
from tapnet.jittor_policy import JittorA2CPolicy, JittorPPOPolicy, JittorCategorical

'''
--box-num
10
--box-range
2
4
--container-size
5
5
10
--train
0
--test-num
10
--model
'tnpp'
--prec-type
'none'
--fact-type
'box'
--data-type
'rand'
--ems-type
'ems-id-stair'
--stable-rule
'hard_before_pack'
--rotate-axes
'z'
--hidden-dim
128
--world-type
'real'
--container-type
'single'
--pack-type
'last'
--stable-predict
0
--note
'4corner'
--corner-num
4
--reward-type
'H'
--resume-path
"/home/wzf/Workspace/rl/maduo-tap/log/result/z_5_5_2-4_10_box_rand/real_single_last/ppo_tnpp_none_ems-id-stair_hard_before_pack_4corner/checkpoint_4.pth"
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tapnet/TAP-v0")
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--model", type=str, default='tnpp') # tnpp, tn, greedy, tnex

    parser.add_argument("--corner-num", type=int, default=4)

    parser.add_argument("--box-num", type=int, default=10)
    parser.add_argument("--init-ctn-num", type=int, default=None)
    # parser.add_argument("--container-size", type=int, nargs="*", default=[5, 5, 5])
    # parser.add_argument("--box-range", type=int, nargs="*", default=[1, 3])
    # parser.add_argument("--box-num", type=int, default=20)
    parser.add_argument("--container-size", type=int, nargs="*", default=[5, 5, 10])
    # parser.add_argument("--box-range", type=int, nargs="*", default=[10, 60])
    parser.add_argument("--box-range", type=int, nargs="*", default=[2, 4])
    # parser.add_argument("--box-range", type=int, nargs="*", default=[20, 70])

    # parser.add_argument("--container-size", type=int, nargs="*", default=[5, 5, 2000])
    # parser.add_argument("--box-range", type=int, nargs="*", default=[1, 4])

    parser.add_argument("--save", type=int, default=0)

    # parser.add_argument("--fact-type", type=str, default='tap_fake')
    # parser.add_argument("--data-type", type=str, default='rand')
    # parser.add_argument("--prec-type", type=str, default='attn')
    # # parser.add_argument("--rotate-axes", type=str, nargs="*", default=['z'])
    # parser.add_argument("--rotate-axes", type=str, nargs="*", default=[ 'x', 'y', 'z'])
    # parser.add_argument("--ems-type", type=str, default='ems-id-stair')
    parser.add_argument("--gripper-size", type=int, nargs="*", default=None)

    parser.add_argument("--require-box-num", type=int, default=0)

    parser.add_argument("--world-type", type=str, default='real') # ideal / real
    parser.add_argument("--container-type", type=str, default='single') # single / multi
    parser.add_argument("--stable-rule", type=str, default="hard_after_pack")
    parser.add_argument("--pack-type", type=str, default='last') # all / last 
    parser.add_argument("--stable-predict", type=int, default=0)

    parser.add_argument("--fact-type", type=str, default='box')
    parser.add_argument("--data-type", type=str, default='rand')
    parser.add_argument("--prec-type", type=str, default='none')
    parser.add_argument("--rotate-axes", type=str, nargs="*", default=['z'])
    parser.add_argument("--ems-type", type=str, default='ems-id-stair')
    # parser.add_argument("--gripper-size", type=int, nargs="*", default=[40,40,70])

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-type", type=str, default="H")

    parser.add_argument("--use-bridge", type=int, default=0)
    parser.add_argument("--min-ems-width", type=int, default=0)
    parser.add_argument("--min-height-diff", type=int, default=0)
    parser.add_argument("--same-height-threshold", type=float, default=0)

    parser.add_argument("--note", type=str, default='4corner')
    parser.add_argument("--seed", type=int, default=666)

    # parser.add_argument("--buffer-size", type=int, default=1000)
    # parser.add_argument("--max-epoch", type=int, default=200)
    # parser.add_argument("--step-per-epoch", type=int, default=1000)
    # parser.add_argument("--step-per-collect", type=int, default=200)
    # parser.add_argument("--repeat-per-collect", type=int, default=10)
    # parser.add_argument("--episode-per-test", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=128)
    # parser.add_argument("--train-num", type=int, default=1)
    # parser.add_argument("--test-num", type=int, default=1)

    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=2000)
    parser.add_argument("--step-per-collect", type=int, default=1024)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--episode-per-test", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--value-mini-batch", type=int, default=64)
    parser.add_argument("--train-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)

    # ppo special
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--action-bound-method", type=str, default="clip")
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument(
        "--device", type=str, default="cuda:0" if jt.has_cuda else "cpu"
    )
    parser.add_argument("--method", type=str, default='ppo')
    # parser.add_argument("--resume-path", type=str, default="/home/wzf/Workspace/rl/maduo-tap/log/result/z_5_5_2-4_10_box_rand/real_single_last/ppo_tnpp_none_ems-id-stair_hard_before_pack_4corner/checkpoint_4.pth")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--tnex-checkpoints", type=str, nargs="*", default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--custom", type=str, default=None)
    # parser.add_argument("--resume-path", type=str, default="./log/a2c/tapnet_[100_100]_C+S_1_debug")
    args = parser.parse_args()

    rotate_axes = ''
    for ax in args.rotate_axes:
        rotate_axes += ax

    # log_path = f"./log/result/{rotate_axes}_{args.container_size[0]}_{args.container_size[1]}_{args.box_range[0]}-{args.box_range[1]}_{args.box_num}_{args.fact_type}_{args.data_type}/{args.world_type}_{args.container_type}_{args.pack_type}/{args.method}_{args.model}_{args.prec_type}_{args.ems_type}_{args.stable_rule}_{args.note}"
    log_path = f"./log/result/{args.custom}_{args.stable_rule}"

    print(args.dataset_path)
    print(args.custom)
    print(log_path)

    if args.stable_predict == 1:
        log_path += '_pred'

    args.log_path = log_path

    if 'hard' in args.stable_rule:
        args.allow_unstable = False     # here
    else:
        args.allow_unstable = True

    if args.require_box_num == 0:
        args.require_box_num = None
    
    return args


def configure_jittor_device(device_str):
    if "CUDA_VISIBLE_DEVICE" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(
            "WARNING: found `CUDA_VISIBLE_DEVICE`; did you mean `CUDA_VISIBLE_DEVICES`? "
            "Current GPU selection may be ignored."
        )
    if device_str and str(device_str).startswith("cuda"):
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0

def get_policy(args):
    
    box_dim = 3
    # ems_dim = 7 if args.use_bridge else 6    
    ems_dim = 6 + (args.container_type == 'multi')
    # ems_dim = 7 + (args.container_type == 'multi')

    args.ems_dim = ems_dim

    rot_num = 2
    device = args.device
    
    if args.require_box_num is not None:
        box_state_num = rot_num * len(args.rotate_axes) * args.require_box_num
    else:
        box_state_num = rot_num * len(args.rotate_axes) * args.box_num
        
    # ems_per_num = 6
    ems_per_num = 10
    max_ems_num = args.box_num * ems_per_num
    args.ems_per_num = ems_per_num

    args.fact_data_folder = None
    
    # source_scale_rate = 1.4
    # source_container_size = [ int(args.container_size[i] * source_scale_rate) for i in range(3) ]
    # gripper_width = int(np.ceil(args.container_size[0] * 0.1))
    # fact_data_folder = f"./tapnet/data/{args.fact_type}/{args.data_type}/{args.box_num}/[{source_container_size[0]}_{source_container_size[1]}]_[{args.box_range[0]}_{args.box_range[1]}]_{gripper_width}"
    # print('load data from ', fact_data_folder)
    # args.fact_data_folder = fact_data_folder
    

    prec_dim = 2
    # if args.prec_type == 'cnn':
    #     if args.train == 0 and args.require_box_num is not None:
    #         prec_dim = args.require_box_num * prec_dim
    #     else:
    #         prec_dim = args.box_num * prec_dim

    args.stable_predict = args.stable_predict == 1

    if args.model == 'tnpp':
        from tapnet.models.network import Net, Critic

        actor = Net(box_dim, ems_dim, args.hidden_dim, prec_dim, args.prec_type, args.stable_predict, args.corner_num, device)
        critic = Critic(box_dim, ems_dim, box_state_num, max_ems_num, args.hidden_dim, prec_dim, args.prec_type, device=device)
        args.action_type = 'box-ems'

    elif args.model == 'greedy':
        from tapnet.models.greedy import Greedy, Critic
        actor = Greedy( pack_type=args.pack_type, container_height=args.container_size[2], device=device)
        critic = Critic(device=device)
        args.action_type = 'box-ems'

    elif args.model == 'tn':
        raise NotImplementedError("Jittor root does not migrate `tapnet.models.old` yet; please use `torch_code` for `--model tn`.")

    elif args.model == 'tnex':
        from tapnet.models.network import Critic
        from tapnet.models.critic import NewActor

        if not args.tnex_checkpoints:
            raise ValueError("`--model tnex` requires `--tnex-checkpoints ckpt1.pkl ckpt2.pkl ...`.")

        actor = NewActor(
            args.tnex_checkpoints,
            box_dim,
            ems_dim,
            args.hidden_dim,
            prec_dim,
            args.prec_type,
            args.stable_predict,
            args.corner_num,
            device,
            container_height=args.container_size[2],
        )
        critic = Critic(box_dim, ems_dim, box_state_num, max_ems_num, args.hidden_dim, prec_dim, args.prec_type, device=device)
        args.action_type = 'box-ems'

    else:
        raise ValueError(f"Unsupported model for Jittor root: {args.model}")
    optim = jt.optim.Adam(actor.parameters() + critic.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        max_update_num = int(np.ceil(args.step_per_epoch / args.step_per_collect) * args.max_epoch)
        lr_scheduler = jt.optim.LambdaLR(
            optim, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / max_update_num)
        )

    def dist_fn(*probs):
        return JittorCategorical(probs=probs[0])

    value_mini_batch = getattr(args, "value_mini_batch", min(args.batch_size, 64))

    if args.method == 'a2c':
        policy = JittorA2CPolicy(actor, critic, optim, dist_fn=dist_fn, \
                discount_factor=args.gamma,
                max_grad_norm=args.max_grad_norm,
                action_bound_method=args.action_bound_method,
                lr_scheduler=lr_scheduler,
                gae_lambda=args.gae_lambda,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                reward_normalization=args.rew_norm,
                action_scaling=True,
                norm_adv=args.norm_adv,
                recompute_adv=args.recompute_adv,
                value_mini_batch=value_mini_batch,
        )
    else:
        policy = JittorPPOPolicy(actor, critic, optim, dist_fn=dist_fn, \
                discount_factor=args.gamma,
                max_grad_norm=args.max_grad_norm,
                action_bound_method=args.action_bound_method,
                lr_scheduler=lr_scheduler,
                gae_lambda=args.gae_lambda,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                reward_normalization=args.rew_norm,
                action_scaling=True,
                eps_clip=args.eps_clip,
                value_clip=args.value_clip,
                dual_clip=args.dual_clip,
                norm_adv=args.norm_adv,
                recompute_adv=args.recompute_adv,
                value_mini_batch=value_mini_batch,
        )
    
    if args.resume_path is not None and args.model != 'greedy':
        policy_pth = args.resume_path
        print(f"loading {policy_pth}")
        policy.actor.load(policy_pth) # Jittor uses load instead of load_state_dict

    return policy


def run(args, envs, actor, test_num):

    save_path = "./render/results_ems"
    save_path = "./render/debug"
    debug_run = os.environ.get("TAPNET_DEBUG_RUN", "0") == "1"

    all_rew = []
    all_ctn = []
    all_di = []
    all_df = []
    all_box_num = []

    env_num = len(envs.workers)
    step_num = test_num // env_num

    if test_num < env_num:
        step_num = 1

    for i in tqdm(range(step_num)):
        hidden = None
        obs, info = envs.reset()
        if debug_run:
            print("init obs")
            print(obs)

        # if args.save == 1:
        #     envs.workers[0].env.factory.source_container.save_states(save_dir=f"{save_path}/data/{args.data_type}/{i}/init")

        obs = Batch(obs)
        if debug_run:
            print("batch obs")
            print(obs)
            print("end")
        
        batch_logp = []
        while True:
            # batch x action_num
            logits, hidden = actor(obs, state=hidden)
            
            dist = JittorCategorical(probs=logits)
            if not actor.training:
                act_np = np.asarray(to_numpy(logits)).argmax(axis=-1).astype(np.int64, copy=False).reshape(-1)
                act = jt.array(act_np).astype(jt.int32)
                prob = jt.gather(logits, 1, act.reshape((-1, 1))).reshape(-1)
                logp = jt.log(prob + 1e-12)
            else:
                act = dist.sample()
                logp = dist.log_prob(act)

            batch_logp.append(logp.unsqueeze(1))

            if debug_run:
                print(act)
            obs, reward, terminated, truncated, info = envs.step(act.numpy())

            if terminated[0] == True:
                break
        
            obs = Batch(obs)

        if args.save == 1:
            envs.workers[0].env.container.save_states(save_dir=f"{save_path}/data/{args.data_type}/{i}/{args.container_type}/{args.model}_{args.pack_type}")

        info = Batch(info)
        all_rew += list(reward)
        all_ctn += list(info.ctn)
        all_di += list(info.delta_int)
        all_df += list(info.delta_float)
        all_box_num += list(info.box_num)

    # delta_float: {np.mean(all_df):.2f}
    print(f'Reward: {np.mean(all_rew)}, ctn: {np.mean(all_ctn):.2f}, delta_int: {np.mean(all_di):.2f}, box num: {np.mean(all_box_num):.2f}' )
    
    if args.save == 1:
        os.makedirs(f'{save_path}/reward/{args.data_type}/{i}/{args.container_type}', exist_ok=True)
        os.makedirs(f'{save_path}/ctn/{args.data_type}/{i}/{args.container_type}', exist_ok=True)
        np.save(f"{save_path}/reward/{args.data_type}/{i}/{args.container_type}/{args.model}_{args.pack_type}", all_rew)
        np.save(f"{save_path}/ctn/{args.data_type}/{i}/{args.container_type}/{args.model}_{args.pack_type}", all_ctn)

    return reward, batch_logp

if __name__ == "__main__":
    args = get_args()
    configure_jittor_device(args.device)

    policy = get_policy(args)

    print(args)
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))

    def save_best_fn(policy):
        policy.actor.save(os.path.join(args.log_path, "policy.pkl"))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(args.log_path, f"checkpoint_{epoch%5}.pkl")
        policy.actor.save(ckpt_path)
        return ckpt_path

    # ems_dim = 7 if args.use_bridge else 6
    # ems_dim = 6 + args.

    use_bridge = args.use_bridge == 1

    test_in_train = False

    train_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(args.task, 
                box_num=args.box_num, 
                ems_dim=args.ems_dim,
                container_size=args.container_size, 
                box_range=args.box_range,
                stable_rule=args.stable_rule,
                allow_unstable=args.allow_unstable,
                use_bridge=use_bridge,
                same_height_threshold = args.same_height_threshold, 
                min_ems_width = args.min_ems_width, 
                min_height_diff = args.min_height_diff,
                fact_type=args.fact_type,
                data_type=args.data_type,
                ems_type=args.ems_type,
                rotate_axes=args.rotate_axes,
                fact_data_folder=args.fact_data_folder,
                action_type=args.action_type,
                require_box_num=args.require_box_num,
                world_type=args.world_type,
                container_type=args.container_type,
                pack_type=args.pack_type,
                ems_per_num = args.ems_per_num,
                init_ctn_num = args.init_ctn_num,
                stable_predict=args.stable_predict,
                gripper_size=args.gripper_size,
                corner_num=args.corner_num,
                reward_type=args.reward_type,
                dataset_path=args.dataset_path) for _ in range(args.train_num)] )

    if args.train == 0:
        test_num = 1
    else:
        test_num = 1
    
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(args.task, 
                box_num=args.box_num,  
                ems_dim=args.ems_dim,
                container_size=args.container_size, 
                box_range=args.box_range,
                stable_rule=args.stable_rule,
                allow_unstable=args.allow_unstable,
                use_bridge=use_bridge,
                same_height_threshold = args.same_height_threshold, 
                min_ems_width = args.min_ems_width, 
                min_height_diff = args.min_height_diff,
                fact_type=args.fact_type,
                data_type=args.data_type,
                ems_type=args.ems_type,
                rotate_axes=args.rotate_axes,
                action_type=args.action_type,
                require_box_num=args.require_box_num,
                world_type=args.world_type,
                container_type=args.container_type,
                pack_type=args.pack_type,
                ems_per_num = args.ems_per_num,
                init_ctn_num = args.init_ctn_num,
                stable_predict=args.stable_predict,
                gripper_size=args.gripper_size,
                corner_num=args.corner_num,
                reward_type='H',
                dataset_path=args.dataset_path) for _ in range(test_num)] )

    train_envs.seed(args.seed)
    test_envs.seed(args.seed)


    import time
    start = time.time()

    if args.train == 0:
        # Let's watch its performance!
        policy.eval()
        print(args.box_range, args.box_num)
        # test_collector = Collector(policy, test_envs)
        # test_collector.reset()
        # result = test_collector.collect(n_episode=args.test_num, render=None)
        # print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}, {len(result["rews"])}')
        run(args, test_envs, policy.actor, args.test_num)
        
    else:
        # collector
        if args.train_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(args.buffer_size)
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, test_envs)

        logger = None
        if SummaryWriter is not None:
            writer = SummaryWriter(args.log_path)
            logger = TensorboardLogger(writer)
        
        result = ts.trainer.onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch = args.max_epoch,
            step_per_epoch = args.step_per_epoch,
            repeat_per_collect = args.repeat_per_collect,
            episode_per_test = args.episode_per_test,
            batch_size = args.batch_size,
            step_per_collect = args.step_per_collect,
            test_in_train = test_in_train,
            logger = logger,
            save_best_fn = save_best_fn,
            save_checkpoint_fn = save_checkpoint_fn,
        )


        print('----over----')
        
        policy.eval()
        test_envs.seed(args.seed)
        print(args.box_range, args.box_num)
        run(args, test_envs, policy.actor, 200)

    end = time.time()
    print(args.log_path)
    print("Running time: %.2fh / %.2fm" % ((end-start) / 60.0 / 60.0, (end-start) / 60.0) )
