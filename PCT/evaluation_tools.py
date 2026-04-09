import os
import numpy as np
import jittor as jt
import tools


def evaluate(PCT_policy, eval_envs, timeStr, args, device, eval_freq = 100, factor = 1):
    PCT_policy.eval()
    obs = eval_envs.reset()
    obs = jt.array(np.array(obs, dtype=np.float32)).unsqueeze(dim=0)
    all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                             args.internal_node_holder, args.leaf_node_holder)
    batchX = jt.arange(args.num_processes)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []

    while step_counter < eval_freq:
        with jt.no_grad():
            selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor = factor)
        selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
        items = eval_envs.packed
        obs, reward, done, infos = eval_envs.step(selected_leaf_node.data[0][0:6])

        if done:
            print('Episode {} ends.'.format(step_counter))
            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])

            print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            all_episodes.append(items)
            step_counter += 1
            obs = eval_envs.reset()

        obs = jt.array(np.array(obs, dtype=np.float32)).unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                 args.internal_node_holder, args.leaf_node_holder)
        # Jittor manages device automatically, no need for .to(device)

    result = "Evaluation using {} episodes\n" \
             "Mean ratio {:.5f}, mean length{:.5f}\n".format(len(episode_ratio), np.mean(episode_ratio), np.mean(episode_length))
    print(result)
    # Save the test trajectories.
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    # Write the test results into local file.
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()
