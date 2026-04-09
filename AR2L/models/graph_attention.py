import numpy as np
import sys
import math
import gym

import jittor as jt
import jittor.nn as nn
sys.path.append('/')

from models.attention_model import AttentionModel, init
from utils import orthogonal_init, constant_init


class DRL_GAT(nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 embedding_size,
                 hidden_size,
                 gat_layer_num,
                 bin_node_num,
                 next_item_num,
                 candidate_pos_num,
                 node_dim,
                 policy_ctg=None,
                 ):
        """
        obs_space <Box> -> shape: (obs_shape, ), low: [low_val]*obs_shape, high: [high_value]*obs_shape
        action_space <Discrete> -> n: action_shape
        """

        super(DRL_GAT, self).__init__()
        self.actor = AttentionModel(obs_space,
                                    action_space,
                                    embedding_size,
                                    hidden_size,
                                    n_encode_layers = gat_layer_num,
                                    n_heads = 1,
                                    bin_node_num = bin_node_num,
                                    next_item_num = next_item_num,
                                    candidate_pos_num = candidate_pos_num,
                                    node_dim = node_dim,
                                    model_ctg='actor',
                                    policy_ctg=policy_ctg,
                                    )

        self.critic_embed = AttentionModel(obs_space,
                                           action_space,
                                           embedding_size,
                                           hidden_size,
                                           n_encode_layers=gat_layer_num,
                                           n_heads=1,
                                           bin_node_num=bin_node_num,
                                           next_item_num=next_item_num,
                                           candidate_pos_num=candidate_pos_num,
                                           node_dim=node_dim,
                                           model_ctg='critic',
                                           policy_ctg=policy_ctg,
                                           )

        init_ = lambda m: init(m, orthogonal_init, lambda x: constant_init(x, 0), np.sqrt(2))
        self.critic = init_(nn.Linear(embedding_size, 1))
        self.is_recurrent = False


    def forward_actor(self, input, deterministic=False, normFactor=1, evaluate=False,):
        action_log_probs, action, entropy, dist, _ = self.actor(input,
                                                                deterministic=deterministic,
                                                                normFactor=normFactor,
                                                                evaluate=evaluate,
                                                                evaluate_action=False,)

        return action_log_probs, action, entropy

    def forward_critic(self, input, deterministic=False, normFactor=1, evaluate=False):
        hidden = self.critic_embed(input,
                                   deterministic=deterministic,
                                   normFactor=normFactor,
                                   evaluate=evaluate,
                                   evaluate_action=False, )

        values = self.critic(hidden)
        return values

    def evaluate_actions(self, input, actions, normFactor=1):
        _, _, entropy, dist, _ = self.actor(input, evaluate_action=True, normFactor=normFactor)
        action_log_probs = dist.log_probs(actions)
        return action_log_probs, entropy.mean()

    def evaluate_values(self, input, normFactor=1):
        hidden = self.critic_embed(input, normFactor=normFactor, evaluate_action=True,)
        values = self.critic(hidden)
        return values

    def action_log_probs(self, input, normFactor=1):
        _, _, _, dist, _ = self.actor(input, evaluate_action=True, normFactor=normFactor)
        return dist.probs.log()


