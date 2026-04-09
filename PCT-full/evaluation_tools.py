import os
import numpy as np
import jittor as jt
import tools
import copy
from time import time as clock, time
import cv2


def evaluate_CDRL(PCT_policy, eval_envs, timeStr, args, device, eval_freq=100, factor=1):
    PCT_policy.eval()
    obs = eval_envs.reset()
    obs = jt.array(obs, dtype=jt.float32).unsqueeze(dim=0)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []
    time_dur = []
    episode_constraint_reward = []
    start = clock()
    num_processes = 1

    eval_recurrent_hidden_states = jt.zeros((num_processes, PCT_policy.recurrent_hidden_state_size))
    eval_masks = jt.zeros((num_processes, 1))

    plainum = int(args.container_size[0] * args.container_size[1])
    if args.setting != 2:
        location_masks = obs[:, plainum: (2 + True) * plainum]
    else:
        location_masks = obs[:, plainum: (2 + 5) * plainum]

    while step_counter < eval_freq:
        value, action, _, eval_recurrent_hidden_states = PCT_policy.act(
            obs,
            eval_recurrent_hidden_states,
            eval_masks,
            location_masks,
            deterministic=True)

        action_np = action.numpy() if isinstance(action, jt.Var) else np.array(action)
        obs, _, done, infos = eval_envs.step(action_np[0])
        items = eval_envs.packed

        if args.draw:
            if args.continuous:
                placed_items = eval_envs.space.root_ems.getInternalNodes()
            else:
                placed_items = np.array(eval_envs.packed)
                placed_items = np.concatenate([placed_items[:, 3:6], placed_items[:, 3:6] + placed_items[:, 0:3]], axis=1)
            root_path = os.path.join('res_imgs', args.timeStr)
            img_path = "rec" + str(len(placed_items)) + ".png"
            tools.draw(placed_items, step_counter, img_path, root_path, container_size=args.container_size)

        if done:
            end = clock()
            time_dur.append(end - start)

            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])
            if 'constraint_reward' in infos.keys() and infos['constraint_reward'] < 500:
                episode_constraint_reward.append(infos['constraint_reward'])

            print('Episode {} ends.'.format(step_counter))
            print("Time cost:", end - start)
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            print('Mean time per item: {}'.format((end - start) / infos['counter']))
            print()
            print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
            print('Var ratio:  {}, length: {}'.format(np.var(episode_ratio), np.var(episode_length)))
            if len(episode_constraint_reward) != 0:
                print('Mean contrain:  {}, var constrain: {}'.format(np.mean(episode_constraint_reward), np.var(episode_constraint_reward)))

            print("Mean time: {}".format(np.mean(time_dur)))
            print('----------------------------------------------')
            all_episodes.append(items)
            step_counter += 1
            obs = eval_envs.reset()
            start = clock()

        obs = jt.array(obs, dtype=jt.float32).unsqueeze(dim=0)
        if args.setting != 2:
            location_masks = obs[:, plainum: (2 + True) * plainum]
        else:
            location_masks = obs[:, plainum: (2 + 5) * plainum]

    result = "Evaluation using {} episodes\nMean ratio {:.5f}, var ratio {:.5f}, mean length {:.5f}\n".format(
        len(episode_ratio), np.mean(episode_ratio), np.var(episode_ratio), np.mean(episode_length))
    if len(episode_constraint_reward) != 0:
        result += "Mean contrain {:.5f}, var contrain {:.5f} \n".format(np.mean(episode_constraint_reward), np.var(episode_constraint_reward))
    print(result)
    if not os.path.exists(os.path.join('./logs/evaluation', timeStr)):
        os.makedirs(os.path.join('./logs/evaluation', timeStr))
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()
    return episode_ratio, episode_length


def evaluate_PCT(PCT_policy, eval_envs, timeStr, args, device, eval_freq=100, factor=1):
    PCT_policy.eval()
    obs = eval_envs.reset()
    obs = jt.array(obs, dtype=jt.float32).unsqueeze(dim=0)
    all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                             args.internal_node_holder, args.leaf_node_holder)
    batchX = np.arange(args.num_processes)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []
    time_dur = []
    episode_constraint_reward = []
    start = clock()

    while step_counter < eval_freq:
        if args.drl_method != 'rainbow':
            if args.pred_value_with_heightmap:
                _, selectedIdx, _, _ = PCT_policy(all_nodes, True, normFactor=factor,
                                                  internal_node_holder=args.internal_node_holder,
                                                  leaf_node_holder=args.leaf_node_holder,
                                                  return_v_distill=True)
            else:
                _, selectedIdx, _, _ = PCT_policy(all_nodes, True, normFactor=factor,
                                                  internal_node_holder=args.internal_node_holder,
                                                  leaf_node_holder=args.leaf_node_holder)
        else:
            selectedIdx = PCT_policy.act(obs)

        selected_idx_np = selectedIdx.squeeze().numpy() if isinstance(selectedIdx, jt.Var) else np.array(selectedIdx).squeeze()
        selected_leaf_node = leaf_nodes[batchX, selected_idx_np]
        items = eval_envs.packed
        selected_leaf_np = selected_leaf_node.numpy() if isinstance(selected_leaf_node, jt.Var) else np.array(selected_leaf_node)
        obs, reward, done, infos = eval_envs.step(selected_leaf_np[0][0:6])

        if args.draw:
            if args.continuous:
                placed_items = eval_envs.space.root_ems.getInternalNodes()
            else:
                placed_items = np.array(eval_envs.packed)
                placed_items = np.concatenate([placed_items[:, 3:6], placed_items[:, 3:6] + placed_items[:, 0:3]], axis=1)
            root_path = os.path.join('res_imgs', args.timeStr)
            img_path = "rec" + str(len(placed_items)) + ".png"
            tools.draw(placed_items, step_counter, img_path, root_path, container_size=args.container_size)

        if done:
            end = clock()
            time_dur.append(end - start)

            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])
            if 'constraint_reward' in infos.keys() and infos['constraint_reward'] < 500:
                episode_constraint_reward.append(infos['constraint_reward'])

            print('Episode {} ends.'.format(step_counter))
            print("Time cost:", end - start)
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            print('Mean time per item: {}'.format((end - start) / infos['counter']))
            print()
            print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
            print('Var ratio:  {}, length: {}'.format(np.var(episode_ratio), np.var(episode_length)))
            if len(episode_constraint_reward) != 0:
                print('Mean contrain:  {}, var constrain: {}'.format(np.mean(episode_constraint_reward), np.var(episode_constraint_reward)))

            print("Mean time: {}".format(np.mean(time_dur)))
            print('----------------------------------------------')
            all_episodes.append(items)
            step_counter += 1
            obs = eval_envs.reset()
            start = clock()

        obs = jt.array(obs, dtype=jt.float32).unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                 args.internal_node_holder, args.leaf_node_holder)

    result = "Evaluation using {} episodes\nMean ratio {:.5f}, var ratio {:.5f}, mean length {:.5f}\n".format(
        len(episode_ratio), np.mean(episode_ratio), np.var(episode_ratio), np.mean(episode_length))
    if len(episode_constraint_reward) != 0:
        result += "Mean contrain {:.5f}, var contrain {:.5f} \n".format(np.mean(episode_constraint_reward), np.var(episode_constraint_reward))
    print(result)
    if not os.path.exists(os.path.join('./logs/evaluation', timeStr)):
        os.makedirs(os.path.join('./logs/evaluation', timeStr))
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()
    return episode_ratio, episode_length