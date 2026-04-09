import math
import os
from numpy import sqrt

import jittor as jt
from jittor import nn
import numpy as np

from tools import init, AddBias, orthogonal_init, constant_init
from attention_model import AttentionModel
from distributions import FixedNormal, FixedCategorical


class ScalarValueHead(nn.Module):
    def __init__(self, in_features):
        super(ScalarValueHead, self).__init__()
        # Keep torch's affine value-head semantics while avoiding Jittor's
        # unstable Linear(out=1, bias=True) CUDA fusion path.
        self.weight = jt.zeros((1, in_features), dtype=jt.float32)
        self.value_bias = jt.zeros((1, 1), dtype=jt.float32)
        orthogonal_init(self.weight, gain=sqrt(2))

    def execute(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        else:
            x = x.reshape((-1, x.shape[-1]))
        out = (x * self.weight).sum(dim=1, keepdims=True)
        bias_term = jt.ones((x.shape[0], 1), dtype=x.dtype) @ self.value_bias
        return out + bias_term

    forward = execute


class DRL_GAT(nn.Module):
    def __init__(self, args):
        super(DRL_GAT, self).__init__()

        self.actor = AttentionModel(
            args.embedding_size,
            args.hidden_size,
            n_encode_layers=args.gat_layer_num,
            n_heads=args.head_num,
            internal_node_holder=args.internal_node_holder,
            internal_node_length=args.internal_node_length,
            leaf_node_holder=args.leaf_node_holder,
            feature_aggregation=args.feature_aggregation,
            args=args
        )
        init_ = lambda m: init(m, orthogonal_init, lambda x, **kwargs: constant_init(x, 0), sqrt(2))

        self.critic_moduel = ScalarValueHead(args.embedding_size * (2 if args.actor_with_q else 1))
        self.args = args
        self.pred_value_with_heightmap = args.pred_value_with_heightmap
        self.critic_distill = ScalarValueHead(args.embedding_size) if self.pred_value_with_heightmap else None

    def critic(self, hidden, embeddings, log_p, p):
        if self.args.actor_with_q:
            batch_size = hidden.shape[0]
            hidden_expand = hidden.unsqueeze(1).repeat((1, embeddings.shape[1], 1))
            embeddings = jt.concat([hidden_expand, embeddings], dim=-1)
            embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
            full_q = self.critic_moduel(embeddings).reshape((batch_size, -1))
            v = (full_q * log_p).sum(1, keepdims=True)
            q = jt.gather(full_q, 1, p)
            return q, full_q, v, None
        elif self.pred_value_with_heightmap:
            v = self.critic_moduel(hidden[0])
            v_distill = self.critic_distill(hidden[1])
            return None, None, v, v_distill
        else:
            v = self.critic_moduel(hidden)
            return None, None, v, None

    def execute(self, items, deterministic=False, normFactor=1, evaluate=False,
                internal_node_holder=None, leaf_node_holder=None, return_q=False, return_v_distill=False):
        internal_node_holder = internal_node_holder if internal_node_holder is not None else self.args.internal_node_holder
        leaf_node_holder = leaf_node_holder if leaf_node_holder is not None else self.args.leaf_node_holder
        o, p, dist_entropy, hidden, _, embeddings, log_p = self.actor(
            items, deterministic, normFactor=normFactor, evaluate=evaluate,
            internal_node_holder=internal_node_holder, leaf_node_holder=leaf_node_holder)
        embeddings = embeddings.reshape((items.shape[0], -1, self.args.embedding_size))[:, internal_node_holder: internal_node_holder + leaf_node_holder, :]
        q_values, full_q_values, v_values, v_distill = self.critic(hidden, embeddings, log_p, p)

        if return_q:
            return o, p, dist_entropy, (q_values, full_q_values, v_values)
        elif return_v_distill:
            return o, p, dist_entropy, (v_distill, v_values)
        else:
            return o, p, dist_entropy, v_values

    forward = execute

    def evaluate_actions(self, items, actions, normFactor=1, return_q=False, return_v_distill=False):
        _, p, dist_entropy, hidden, dist, embeddings, log_p = self.actor(items, evaluate_action=True, normFactor=normFactor)
        embeddings = embeddings.reshape((items.shape[0], -1, self.args.embedding_size))[:, self.args.internal_node_holder: self.args.internal_node_holder + self.args.leaf_node_holder, :]
        action_log_probs = dist.log_probs(actions)
        q_values, full_q_values, v_values, v_distill = self.critic(hidden, embeddings, log_p, p)

        if return_q:
            return (q_values, full_q_values, v_values), action_log_probs, dist_entropy.mean()
        elif return_v_distill:
            return (v_distill, v_values), action_log_probs, dist_entropy.mean()
        else:
            return v_values, action_log_probs, dist_entropy.mean()

    def get_node_inputs(self, items, normFactor=1):
        return self.actor.get_node_inputs(items, self.args.internal_node_holder, self.args.leaf_node_holder, normFactor)

    def get_action_rank(self, items, deterministic=False, normFactor=1, evaluate=False,
                        internal_node_holder=None, leaf_node_holder=None, return_q=False):
        internal_node_holder = internal_node_holder if internal_node_holder is not None else self.args.internal_node_holder
        leaf_node_holder = leaf_node_holder if leaf_node_holder is not None else self.args.leaf_node_holder
        o, p, dist_entropy, hidden, _, embeddings, log_p = self.actor(
            items, deterministic, normFactor=normFactor, evaluate=evaluate,
            internal_node_holder=internal_node_holder, leaf_node_holder=leaf_node_holder)
        embeddings = embeddings.reshape((items.shape[0], -1, self.args.embedding_size))[:, internal_node_holder: internal_node_holder + leaf_node_holder, :]
        q_values, full_q_values, v_values, v_distill = self.critic(hidden, embeddings, log_p, p)
        if return_q:
            return log_p, p, dist_entropy, (q_values, full_q_values, v_values)
        return log_p, p, dist_entropy, v_values


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m, orthogonal_init, lambda x, **kwargs: constant_init(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(jt.zeros((num_outputs,)))
        self.right_bound = 0.5
        self.left_bound = 0.1
        self.max_logstd = 0
        self.min_logstd = -10

    def execute(self, x):
        x = x.reshape((-1, x.shape[-1]))
        action_mean = self.fc_mean(x)
        zeros = jt.zeros(action_mean.shape)
        action_logstd = self.logstd(zeros)
        action_mean = jt.sigmoid(action_mean) * (self.right_bound - self.left_bound + 0.04) + self.left_bound - 0.02
        return FixedNormal(action_mean, jt.exp(action_logstd)), action_mean, action_logstd

    forward = execute
