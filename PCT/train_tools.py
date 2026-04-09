import os
import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import tools
import time
from collections import deque
from storage import PCTRolloutStorage
from kfac import KFACOptimizer
import random
np.set_printoptions(threshold=np.inf)

class train_tools(object):
    def __init__(self, writer, timeStr, PCT_policy, args):
        self.writer = writer
        self.timeStr = timeStr
        self.step_counter = 0
        self.PCT_policy = PCT_policy
        self.use_acktr = args.use_acktr
        seed = args.seed

        if self.use_acktr:
            self.policy_optim = KFACOptimizer(self.PCT_policy) # For ACKTR method.
        else:
            self.policy_optim = optim.Adam(self.PCT_policy.parameters(), lr=args.learning_rate) # For naive A2C method.

        if seed is not None:
            jt.set_global_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _policy_update(self, pct_rollout, args, factor):
        obs_shape = pct_rollout.obs.shape[2:]
        action_shape = pct_rollout.actions.shape[-1]
        flat_obs = pct_rollout.obs[:-1].reshape((-1, *obs_shape))
        flat_actions = pct_rollout.actions.reshape((-1, action_shape))

        if self.use_acktr and self.policy_optim.steps % self.policy_optim.Ts == 0:
            # Jittor's jt.grad can consume a large part of the graph when Fisher
            # statistics and the real actor-critic update share one forward pass.
            # Use a dedicated forward graph for ACKTR statistics.
            self.policy_optim._capturing = True
            fisher_value, fisher_log_prob, _ = self.PCT_policy.evaluate_actions(
                flat_obs, flat_actions, normFactor=factor)
            self.policy_optim._capturing = False

            fisher_value = fisher_value.reshape((args.num_steps, args.num_processes, 1))
            fisher_log_prob = fisher_log_prob.reshape((args.num_steps, args.num_processes, 1))

            pg_fisher_loss = -fisher_log_prob.mean()
            value_noise = jt.randn(fisher_value.shape)
            sample_values = fisher_value + value_noise
            vf_fisher_loss = -(fisher_value - sample_values.stop_grad()).sqr().mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss

            self.policy_optim.acc_stats = True
            self.policy_optim.backward_and_step(fisher_loss)
            self.policy_optim.acc_stats = False

        leaf_node_value, selectedlogProb, dist_entropy = self.PCT_policy.evaluate_actions(
            flat_obs, flat_actions, normFactor=factor)
        leaf_node_value = leaf_node_value.reshape((args.num_steps, args.num_processes, 1))
        selectedlogProb = selectedlogProb.reshape((args.num_steps, args.num_processes, 1))

        critic_advantages = pct_rollout.returns[:-1] - leaf_node_value
        actor_advantages = critic_advantages.stop_grad()
        critic_loss = (critic_advantages * critic_advantages).mean()
        actor_loss = -(actor_advantages * selectedlogProb).mean()
        total_loss = args.actor_loss_coef * actor_loss + args.critic_loss_coef * critic_loss

        if self.use_acktr:
            self.policy_optim.backward_and_step(total_loss, max_grad_norm=args.max_grad_norm)
        else:
            self.policy_optim.zero_grad()
            self.policy_optim.backward(total_loss)
            self.policy_optim.clip_grad_norm(args.max_grad_norm, 2)
            self.policy_optim.step()

        return actor_loss.stop_grad(), critic_loss.stop_grad(), dist_entropy.stop_grad()

    def train_n_steps(self, envs, args, device):
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        self.PCT_policy.train()
        factor = args.normFactor # NormFactor controlls the maximum value of the original input of the network to less than 1.0, which helps the training of the network

        obs = envs.reset()
        all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
        pct_rollout = PCTRolloutStorage(args.num_steps,
                                        args.num_processes,
                                        obs_shape=all_nodes.shape[1:],
                                        gamma = args.gamma)
        pct_rollout.to(device)

        start = time.time()
        ratio_recorder = 0
        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        batchX = np.arange(args.num_processes)

        inside_counter = self.step_counter
        num_steps, num_processes = args.num_steps, args.num_processes
        pct_rollout.obs[0] = all_nodes.stop_grad()

        while True:
            ##############################################
            ####### Collect n-step training sample #######
            ##############################################
            self.step_counter += 1
            for step in range(num_steps):
                selectedlogProb, selectedIdx, dist_entropy, value = self.PCT_policy(all_nodes, normFactor=factor)
                selectedlogProb = selectedlogProb.stop_grad()
                selectedIdx = selectedIdx.stop_grad()
                value = value.stop_grad()

                selected_idx_1d = selectedIdx.reshape((-1,))
                selected_leaf_node = leaf_nodes[batchX, selected_idx_1d.numpy()]
                obs, reward, done, infos = envs.step(selected_leaf_node.numpy())
                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                reward_var = reward if isinstance(reward, jt.Var) else jt.array(np.array(reward), dtype=jt.float32)
                reward_var = reward_var.reshape((args.num_processes, 1))
                mask_var = jt.array((1 - np.array(done, dtype=np.float32)).reshape((args.num_processes, 1)), dtype=jt.float32)
                pct_rollout.insert(
                    all_nodes,
                    selected_idx_1d.reshape((-1, 1)),
                    selectedlogProb,
                    value,
                    reward_var,
                    mask_var
                )

            for _ in range(len(infos)):
                if done[_]:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])

            _, _, _, next_value = self.PCT_policy(pct_rollout.obs[-1], normFactor=factor)
            next_value = next_value.stop_grad()

            pct_rollout.compute_returns(next_value)

            ##############################################
            ########### PCT policy optimzation ###########
            ##############################################
            actor_loss, critic_loss, dist_entropy = self._policy_update(pct_rollout, args, factor)
            jt.gc()

            ##############################################
            ############ After optimzation ###############
            ##############################################
            pct_rollout.after_update()

            # Save the trained policy model
            if (self.step_counter % args.model_save_interval == 0) and args.model_save_path != "":
                if self.step_counter % args.model_update_interval == 0:
                    sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                tools.safe_save(self.PCT_policy.state_dict(), os.path.join(model_save_path, 'PCT-' + self.timeStr + '_' + sub_time_str + ".pt"))

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards)>1:
                total_num_steps = (self.step_counter + 1 - inside_counter) * num_processes * num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {}, the ratio threshold is{}\n"
                        .format(self.timeStr,
                                self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards),
                                float(dist_entropy.item()) if hasattr(dist_entropy, 'item') else float(dist_entropy),
                                float(critic_loss.item()) if hasattr(critic_loss, 'item') else float(critic_loss),
                                float(actor_loss.item()) if hasattr(actor_loss, 'item') else float(actor_loss),
                                np.mean(episode_ratio), ratio_recorder))
                self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio in history", ratio_recorder, self.step_counter)
                self.writer.add_scalar("Training/Value loss", float(critic_loss.item()) if hasattr(critic_loss, 'item') else float(critic_loss), self.step_counter)
                self.writer.add_scalar("Training/Action loss", float(actor_loss.item()) if hasattr(actor_loss, 'item') else float(actor_loss), self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', float(dist_entropy.item()) if hasattr(dist_entropy, 'item') else float(dist_entropy), self.step_counter)
