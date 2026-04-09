import os, sys
import numpy as np
import time as mytime
from time import strftime, localtime, time
from collections import deque
import random

import jittor as jt
import jittor.nn as nn

import utils
from utils import safe_save
from storage import PPO_RolloutStorage
from models.graph_attention import DRL_GAT


class PPO_Training():
    def __init__(self,
                 BPP_policy,
                 adv_policy,
                 bal_policy,
                 args,
                 use_clipped_value_loss=True,
                 ):

        self.BPP_policy = BPP_policy
        self.adv_policy = adv_policy
        self.bal_policy = bal_policy

        self.alpha = args.alpha
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.lr = args.learning_rate
        self.eps = args.eps

        self.args = args
        self.factor = args.normFactor

        self.bpp_optimizer = jt.optim.Adam(self.BPP_policy.parameters(), lr=self.lr, eps=self.eps)
        self.adv_optimizer = jt.optim.Adam(self.adv_policy.parameters(), lr=self.lr, eps=self.eps)
        self.bal_optimizer = jt.optim.Adam(self.bal_policy.parameters(), lr=self.lr, eps=self.eps)

        if args.seed is not None:
            jt.set_global_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

    def train_n_steps(self, envs, args, device, timeStr,):
        self.timeStr = timeStr
        self.sub_time_str = timeStr
        self.model_save_path = os.path.join(args.model_save_path, self.timeStr)
        self.BPP_policy.train()
        self.adv_policy.train()
        self.bal_policy.train()

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        rot_num = 2 if args.setting != 2 else 6
        batchX = self.batchX = jt.arange(num_processes)
        self.device = device

        self.bpp_rollout = PPO_RolloutStorage(
            num_steps,
            num_processes,
            obs_shape=(num_box + num_next_box + num_candidate_action, node_dim),
            action_shape=(1, ),
        )
        self.bpp_rollout.to(device)

        self.adv_rollout = PPO_RolloutStorage(
            num_steps,
            num_processes,
            obs_shape=(num_box + num_next_box, node_dim),
            action_shape=(1, ),
        )
        self.adv_rollout.to(device)

        self.bal_rollout = PPO_RolloutStorage(
            num_steps,
            num_processes,
            obs_shape=(num_box + num_next_box, node_dim),
            action_shape=(1,),
        )
        self.bal_rollout.to(device)


        self.bpp_ratio_recorder = 0
        self.bpp_episode_rewards = deque(maxlen=10)
        self.bpp_episode_ratio = deque(maxlen=10)
        self.bpp_episode_counter = deque(maxlen=10)
        self.bpp_step_counter = 1

        self.adv_ratio_recorder = 0
        self.adv_episode_rewards = deque(maxlen=10)
        self.adv_episode_ratio = deque(maxlen=10)
        self.adv_episode_counter = deque(maxlen=10)
        self.adv_step_counter = 1

        self.bal_ratio_recorder = 0
        self.bal_episode_rewards = deque(maxlen=10)
        self.bal_episode_ratio = deque(maxlen=10)
        self.bal_episode_counter = deque(maxlen=10)
        self.bal_step_counter = 1

        max_update_num = int(args.num_env_steps // num_steps // num_processes)
        self.bpp_start = self.adv_start = self.bal_start = time()
        while True:
            self.train_adv_policy(
                envs, rot_num, batchX, max_update_num, args, device
            )
            self.train_bal_policy(
                envs, rot_num, batchX, max_update_num, args, device
            )
            self.train_bpp_policy(
                envs, rot_num, batchX, max_update_num, args, device
            )

        return


    def train_bpp_policy(self, envs, rot_num, batchX, max_update_num, args, device):

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        pmt_obs = envs.reset()
        # Torch baseline expects base obs (bin+next) here; slice if env returns full obs.
        if len(pmt_obs.shape) == 2:
            obs_dim = int(pmt_obs.shape[1])
            expected_dim = int((num_box + num_next_box) * node_dim)
            if obs_dim < expected_dim:
                raise ValueError(f"pmt_obs dim {obs_dim} < expected {expected_dim}. Check env observation.")
            if obs_dim != expected_dim:
                pmt_obs = pmt_obs[:, :expected_dim]
        pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box + num_next_box, node_dim)
        bpp_obs, box_idx = self.execute_permute_policy(envs, pmt_obs, batchX, device)
        self.bpp_rollout.obs[0] = bpp_obs.numpy() if isinstance(bpp_obs, jt.Var) else bpp_obs

        for bpp_step in range(10):
            utils.update_linear_schedule(
                self.bpp_optimizer, self.bpp_step_counter, max_update_num, args.learning_rate,)
            self.bpp_step_counter += 1
            for step in range(num_steps):
                with jt.no_grad():
                    action_log_probs, action, entropy = self.BPP_policy.forward_actor(
                        bpp_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.BPP_policy.forward_critic(
                        bpp_obs, deterministic=False, normFactor=self.factor
                    )

                location = bpp_obs.split(
                    [num_box, num_next_box, num_candidate_action], dim=1
                )[-1][batchX, action.squeeze(1)][:, :7]
                zero_padding = jt.zeros((location.shape[0], 1))
                execution = jt.concat((location, box_idx, zero_padding), dim=-1)

                pmt_obs, reward, done, infos = envs.step(execution.data)
                # Torch baseline expects base obs (bin+next) here; slice if env returns full obs.
                if len(pmt_obs.shape) == 2:
                    obs_dim = int(pmt_obs.shape[1])
                    expected_dim = int((num_box + num_next_box) * node_dim)
                    if obs_dim < expected_dim:
                        raise ValueError(f"pmt_obs dim {obs_dim} < expected {expected_dim}. Check env observation.")
                    if obs_dim != expected_dim:
                        pmt_obs = pmt_obs[:, :expected_dim]
                pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box + num_next_box, node_dim)
                bpp_obs, box_idx = self.execute_permute_policy(envs, pmt_obs, batchX, device)

                masks = jt.array(np.array([[0.0] if done_ else [1.0] for done_ in done], dtype=np.float32))
                bad_masks = jt.array(np.array([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos], dtype=np.float32))
                self.bpp_rollout.insert(bpp_obs, action, action_log_probs, value, reward, masks, bad_masks,)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.bpp_episode_rewards.append(infos[_]['reward'])
                        if 'ratio' in infos[_].keys():
                            self.bpp_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.bpp_episode_counter.append(infos[_]['counter'])

            with jt.no_grad():
                next_value = self.BPP_policy.forward_critic(jt.array(self.bpp_rollout.obs[-1]), normFactor=self.factor)

            self.bpp_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = self.update(
                self.bpp_rollout, self.bpp_optimizer, self.BPP_policy
            )
            self.bpp_rollout.after_update()

            # ad by wzf : Save the trained policy model
            if (self.bpp_step_counter % args.model_save_interval == 0) and args.model_save_path != "":
                # if self.bpp_step_counter % args.model_update_interval == 0:
                #     self.sub_time_str = mytime.strftime('%Y.%m.%d-' \
                #                                         '-H-%M-%S', mytime.localtime(mytime.time()))

                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)

                safe_save(self.BPP_policy.state_dict(),
                           os.path.join(self.model_save_path, 'AR2L-BPP-policy-' + self.timeStr + '_' + self.sub_time_str + ".pt"))

            if self.bpp_step_counter % args.print_log_interval == 0 and len(self.bpp_episode_rewards) > 1:
                total_num_steps = self.bpp_step_counter * num_processes * num_steps
                end = time()

                if len(self.bpp_episode_ratio) != 0:
                    self.bpp_ratio_recorder = max(self.bpp_ratio_recorder, np.max(self.bpp_episode_ratio))

                episodes_training_results = \
                    "Train BPP policy\n" \
                    "Updates {}, num timesteps {}, FPS {}\n" \
                    "Last {} training episodes:\n" \
                    "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                    "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                    "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                    "The ratio threshold is {}\n" \
                    "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                        .format(self.bpp_step_counter, total_num_steps, int(total_num_steps / (end - self.bpp_start)),
                                len(self.bpp_episode_rewards),
                                np.mean(self.bpp_episode_rewards), np.median(self.bpp_episode_rewards),
                                np.min(self.bpp_episode_rewards), np.max(self.bpp_episode_rewards),
                                np.mean(self.bpp_episode_ratio), np.median(self.bpp_episode_ratio),
                                np.min(self.bpp_episode_ratio), np.max(self.bpp_episode_ratio),
                                np.mean(self.bpp_episode_counter), np.median(self.bpp_episode_counter),
                                np.min(self.bpp_episode_counter), np.max(self.bpp_episode_counter),
                                self.bpp_ratio_recorder,
                                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                                )
                print(episodes_training_results)


    def train_adv_policy(self, envs, rot_num, batchX, max_update_num, args, device):

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        adv_obs = envs.reset()
        # Keep behavior identical to torch baseline: adv_obs is always (num_box+num_next_box)*node_dim
        # If env returns extra candidate-action nodes, slice them off here.
        if len(adv_obs.shape) == 2:
            obs_dim = int(adv_obs.shape[1])
            expected_dim = int((num_box + num_next_box) * node_dim)
            if obs_dim < expected_dim:
                raise ValueError(f"adv_obs dim {obs_dim} < expected {expected_dim}. Check env observation.")
            if obs_dim != expected_dim:
                adv_obs = adv_obs[:, :expected_dim]
            adv_obs = adv_obs.reshape(adv_obs.shape[0], num_box + num_next_box, node_dim)
        else:
            # Some env wrappers may return (N, graph, node_dim) already
            adv_obs = adv_obs.reshape(adv_obs.shape[0], num_box + num_next_box, node_dim)

        self.adv_rollout.obs[0] = adv_obs.numpy() if isinstance(adv_obs, jt.Var) else adv_obs

        for adv_step in range(10):
            utils.update_linear_schedule(
                self.bpp_optimizer, self.bpp_step_counter, max_update_num, args.learning_rate,
            )
            self.adv_step_counter += 1

            for step in range(num_steps):
                with jt.no_grad():
                    action_log_probs, action, entropy = self.adv_policy.forward_actor(
                        adv_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.adv_policy.forward_critic(
                        adv_obs, deterministic=False, normFactor=self.factor
                    )

                # Jittor is fragile with split(...)[1][batch, idx] advanced indexing; do gather manually.
                next_boxes = adv_obs[:, num_box:num_box + num_next_box, :]  # [B, num_next_box, node_dim]
                act_idx = action.squeeze(1)
                if act_idx.dtype != jt.int32 and act_idx.dtype != jt.int64:
                    act_idx = act_idx.int32()
                if batchX.dtype != jt.int32 and batchX.dtype != jt.int64:
                    batchX = batchX.int32()
                assert int(act_idx.min().data) >= 0 and int(act_idx.max().data) < num_next_box, (
                    f"action index out of range: min={int(act_idx.min().data)} max={int(act_idx.max().data)} num_next_box={num_next_box}"
                )
                box = next_boxes[batchX, act_idx]  # [B, node_dim]
                box = box[:, :7]

                one_padding = jt.ones((box.shape[0], 1))
                execution = jt.concat((box, action, one_padding), dim=-1)

                bpp_obs, _, _, _, = envs.step(execution.data)
                # bpp_obs may include candidate actions; reshape with full length
                bpp_obs = bpp_obs.reshape(num_processes, num_box + num_next_box + num_candidate_action, node_dim)
                adv_obs, reward, done, infos = self.execute_bpp_policy(envs, bpp_obs, batchX, device, action)

                masks = jt.array(np.array([[0.0] if done_ else [1.0] for done_ in done], dtype=np.float32))
                bad_masks = jt.array(np.array([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos], dtype=np.float32))
                self.adv_rollout.insert(adv_obs, action, action_log_probs, value, reward, masks, bad_masks)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.adv_episode_rewards.append(infos[_]['reward'])
                        if 'ratio' in infos[_].keys():
                            self.adv_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.adv_episode_counter.append(infos[_]['counter'])

            with jt.no_grad():
                next_value = self.adv_policy.forward_critic(jt.array(self.adv_rollout.obs[-1]), normFactor=self.factor)

            self.adv_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )
            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = self.update(
                self.adv_rollout, self.adv_optimizer, self.adv_policy
            )
            self.adv_rollout.after_update()

            # ad by wzf : Save the trained policy model
            if (self.adv_step_counter % args.model_save_interval == 0) and args.model_save_path != "":
                if self.adv_step_counter % args.model_update_interval == 0:
                    self.sub_time_str = mytime.strftime('%Y.%m.%d-%H-%M-%S', mytime.localtime(mytime.time()))

                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)

                safe_save(self.adv_policy.state_dict(),
                           os.path.join(self.model_save_path,
                                        'AR2L-adv-policy-' + self.timeStr + '_' + self.sub_time_str + ".pt"))

            if self.adv_step_counter % args.print_log_interval == 0 and len(self.adv_episode_rewards) > 1:
                total_num_steps = self.adv_step_counter * num_processes * num_steps
                end = time()

                if len(self.adv_episode_ratio) != 0:
                    self.adv_ratio_recorder = max(self.adv_ratio_recorder, np.max(self.adv_episode_ratio))

                episodes_training_results = \
                    "Train Adv policy\n" \
                    "Updates {}, num timesteps {}, FPS {}\n" \
                    "Last {} training episodes:\n" \
                    "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                    "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                    "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                    "The ratio threshold is {}\n" \
                    "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                        .format(self.adv_step_counter, total_num_steps, int(total_num_steps / (end - self.adv_start)),
                                len(self.adv_episode_rewards),
                                np.mean(self.adv_episode_rewards), np.median(self.adv_episode_rewards),
                                np.min(self.adv_episode_rewards), np.max(self.adv_episode_rewards),
                                np.mean(self.adv_episode_ratio), np.median(self.adv_episode_ratio),
                                np.min(self.adv_episode_ratio), np.max(self.adv_episode_ratio),
                                np.mean(self.adv_episode_counter), np.median(self.adv_episode_counter),
                                np.min(self.adv_episode_counter), np.max(self.adv_episode_counter),
                                self.adv_ratio_recorder,
                                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                                )
                print(episodes_training_results)

    def train_bal_policy(self, envs, rot_num, batchX, max_update_num, args, device):

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        bal_obs = envs.reset()
        # Keep behavior identical to torch baseline: bal_obs is always (num_box+num_next_box)*node_dim
        if len(bal_obs.shape) == 2:
            obs_dim = int(bal_obs.shape[1])
            expected_dim = int((num_box + num_next_box) * node_dim)
            if obs_dim < expected_dim:
                raise ValueError(f"bal_obs dim {obs_dim} < expected {expected_dim}. Check env observation.")
            if obs_dim != expected_dim:
                bal_obs = bal_obs[:, :expected_dim]
            bal_obs = bal_obs.reshape(bal_obs.shape[0], num_box + num_next_box, node_dim)
        else:
            bal_obs = bal_obs.reshape(bal_obs.shape[0], num_box + num_next_box, node_dim)

        self.bal_rollout.obs[0] = bal_obs.numpy() if isinstance(bal_obs, jt.Var) else bal_obs

        for bal_step in range(10):
            utils.update_linear_schedule(
                self.bpp_optimizer, self.bpp_step_counter, max_update_num, args.learning_rate,
            )
            self.bal_step_counter += 1

            for step in range(num_steps):
                with jt.no_grad():
                    action_log_probs, action, entropy = self.bal_policy.forward_actor(
                        bal_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.bal_policy.forward_critic(
                        bal_obs, deterministic=False, normFactor=self.factor
                    )
                box = bal_obs.split([num_box, num_next_box, ], dim=1)[1][batchX, action.squeeze(1)][:, :7]
                one_padding = jt.ones((box.shape[0], 1))
                execution = jt.concat((box, action, one_padding), dim=-1)

                bpp_obs, _, _, _, = envs.step(execution.data)
                bpp_obs = bpp_obs.reshape(num_processes, num_box + num_next_box + num_candidate_action, node_dim)
                bal_obs, reward, done, infos = self.execute_bpp_policy(envs, bpp_obs, batchX, device, action, inv_reward=False)

                masks = jt.array(np.array([[0.0] if done_ else [1.0] for done_ in done], dtype=np.float32))
                bad_masks = jt.array(np.array([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos], dtype=np.float32))
                self.bal_rollout.insert(bal_obs, action, action_log_probs, value, reward, masks, bad_masks)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.bal_episode_rewards.append(infos[_]['reward'])
                        if 'ratio' in infos[_].keys():
                            self.bal_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.bal_episode_counter.append(infos[_]['counter'])

            with jt.no_grad():
                next_value = self.bal_policy.forward_critic(jt.array(self.bal_rollout.obs[-1]), normFactor=self.factor)

            self.bal_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            distance_loss_epoch, normal_loss_epoch, worst_loss_epoch = self.update(
                self.bal_rollout, self.bal_optimizer, self.bal_policy, dist_loss=True
            )
            self.bal_rollout.after_update()

            # ad by wzf : Save the trained policy model
            if (self.bal_step_counter % args.model_save_interval == 0) and args.model_save_path != "":
                # if self.bal_step_counter % args.model_update_interval == 0:
                #     sub_time_str = mytime.strftime('%Y.%m.%d-' \
                #                                    '-H-%M-%S', mytime.localtime(mytime.time()))

                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)

                safe_save(self.bal_policy.state_dict(),
                           os.path.join(self.model_save_path,
                                        'AR2L-bal-policy-' + self.timeStr + '_' + self.sub_time_str + ".pt"))

            if self.bal_step_counter % args.print_log_interval == 0 and len(self.bal_episode_rewards) > 1:
                total_num_steps = self.bal_step_counter * num_processes * num_steps
                end = time()

                if len(self.bal_episode_ratio) != 0:
                    self.bal_ratio_recorder = max(self.bal_ratio_recorder, np.max(self.bal_episode_ratio))

                episodes_training_results = \
                    "Train Bal policy\n" \
                    "Updates {}, num timesteps {}, FPS {}\n" \
                    "Last {} training episodes:\n" \
                    "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                    "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                    "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                    "The ratio threshold is {}\n" \
                    "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                    "The distance loss {:.5f}, the normal loss {:.5f}, the worst loss {:.5f}\n" \
                        .format(self.bal_step_counter, total_num_steps, int(total_num_steps / (end - self.bal_start)),
                                len(self.bal_episode_rewards),
                                np.mean(self.bal_episode_rewards), np.median(self.bal_episode_rewards),
                                np.min(self.bal_episode_rewards), np.max(self.bal_episode_rewards),
                                np.mean(self.bal_episode_ratio), np.median(self.bal_episode_ratio),
                                np.min(self.bal_episode_ratio), np.max(self.bal_episode_ratio),
                                np.mean(self.bal_episode_counter), np.median(self.bal_episode_counter),
                                np.min(self.bal_episode_counter), np.max(self.bal_episode_counter),
                                self.bal_ratio_recorder,
                                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                                distance_loss_epoch, normal_loss_epoch, worst_loss_epoch,
                                )
                print(episodes_training_results)
    

    def execute_bpp_policy(self, envs, bpp_obs, batchX, device, box_idx, inv_reward=True):
        num_box, num_next_box, num_candidate_action = \
            self.args.num_box, self.args.num_next_box, self.args.num_candidate_action
        node_dim = self.args.node_dim
        with jt.no_grad():
            _, loc_idx, _, = self.BPP_policy.forward_actor(
                bpp_obs, deterministic=False, normFactor=self.factor,
            )
        tmp_loc = bpp_obs.split([num_box, num_next_box, num_candidate_action], dim=1)[-1][batchX, loc_idx.squeeze(1)][:, :7]
        zero_padding = jt.zeros((tmp_loc.shape[0], 1))
        bpp_act = jt.concat((tmp_loc, box_idx, zero_padding), dim=-1)
        obs, reward, done, infos = envs.step(bpp_act.data)
        reward = -reward if inv_reward else reward

        # Env may return either base_obs ((num_box+num_next_box)*node_dim) or full_obs
        # ((num_box+num_next_box+num_candidate_action)*node_dim). Training expects base_obs.
        if len(obs.shape) != 2:
            obs = obs.reshape(obs.shape[0], -1)
        expected_dim = int((num_box + num_next_box) * node_dim)
        if int(obs.shape[1]) < expected_dim:
            raise ValueError(f"Env returned obs_dim={int(obs.shape[1])} < expected {expected_dim}.")
        if int(obs.shape[1]) != expected_dim:
            obs = obs[:, :expected_dim]

        return obs.reshape(obs.shape[0], num_box + num_next_box, node_dim), reward, done, infos


    def execute_permute_policy(self, envs, pmt_obs, batchX, device):
        num_box, num_next_box, num_candidate_action = \
            self.args.num_box, self.args.num_next_box, self.args.num_candidate_action
        with jt.no_grad():
            _, box_idx, _, = self.bal_policy.forward_actor(pmt_obs, deterministic=False, normFactor=self.factor)
        tmp_box = pmt_obs.split([num_box, num_next_box, ], dim=1)[1][batchX, box_idx.squeeze(1)][:, :7]
        one_padding = jt.ones((tmp_box.shape[0], 1))
        pmt_act = jt.concat((tmp_box, box_idx, one_padding), dim=-1)
        bpp_obs, _, _, _, = envs.step(pmt_act.data)

        return bpp_obs.reshape(bpp_obs.shape[0], num_box + num_next_box + num_candidate_action, -1), box_idx


    def update(self,
               rollouts: PPO_RolloutStorage,
               optimizer,
               policy_model: DRL_GAT,
               dist_loss=False,
               ):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        distance_loss_epoch = 0
        normal_loss_epoch = 0
        worst_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if policy_model.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

                action_log_probs, dist_entropy, = policy_model.evaluate_actions(
                    obs_batch, actions_batch, normFactor=self.factor
                )
                values = policy_model.evaluate_values(
                    obs_batch, normFactor=self.factor
                )

                if dist_loss:
                    # distance between normal transition and balance transition
                    normal_action = jt.zeros((obs_batch.shape[0], 1))
                    normal_action_log_probs, _ = policy_model.evaluate_actions(
                        obs_batch, normal_action, normFactor=self.factor
                    )
                    normal_loss = (-normal_action_log_probs).mean()

                    bal_log_probs = policy_model.action_log_probs(obs_batch, normFactor=self.factor)
                    with jt.no_grad():
                        wor_log_probs = self.adv_policy.action_log_probs(obs_batch, normFactor=self.factor)

                    # KL divergence: sum(exp(wor) * (wor - bal)) / batch_size
                    worst_loss = (jt.exp(wor_log_probs) * (wor_log_probs - bal_log_probs)).sum() / obs_batch.shape[0]

                    distance_loss = normal_loss + self.alpha * worst_loss
                else:
                    normal_loss = jt.zeros(1)
                    worst_loss = jt.zeros(1)
                    distance_loss = jt.zeros(1)


                ratio = jt.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = jt.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -jt.minimum(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).sqr()
                    value_losses_clipped = (value_pred_clipped - return_batch).sqr()
                    value_loss = 0.5 * jt.maximum(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).sqr().mean()

                optimizer.zero_grad()
                total_loss = (value_loss * self.value_loss_coef
                 + action_loss
                 - dist_entropy * self.entropy_coef
                 + distance_loss
                 )
                optimizer.backward(total_loss)
                optimizer.clip_grad_norm(self.max_grad_norm, 2)
                optimizer.step()

                value_loss_epoch += float(value_loss.data)
                action_loss_epoch += float(action_loss.data)
                dist_entropy_epoch += float(dist_entropy.data)

                distance_loss_epoch += float(distance_loss.data)
                normal_loss_epoch += float(normal_loss.data)
                worst_loss_epoch += float(worst_loss.data)

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        distance_loss_epoch /= num_updates
        normal_loss_epoch /= num_updates
        worst_loss_epoch /= num_updates

        # NOTE: In Jittor, jt.Var cannot be used as a Python boolean (will trigger jt.to_bool assertion).
        # Here we must branch on the python flag `dist_loss` (same semantic as torch version: whether
        # distance_loss was enabled for this update call).
        if dist_loss:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
                   distance_loss_epoch, normal_loss_epoch, worst_loss_epoch
        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
