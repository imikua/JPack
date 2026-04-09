import jittor.nn as nn
from tools import init, orthogonal_init
from numpy import sqrt
from attention_model import AttentionModel
import jittor as jt

class ScalarValueHead(nn.Module):
    def __init__(self, in_features):
        super(ScalarValueHead, self).__init__()
        # Keep torch's affine value head semantics while avoiding Jittor's
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

class DRL_GAT(nn.Module):
    def __init__(self, args):
        super(DRL_GAT, self).__init__()

        self.actor = AttentionModel(args.embedding_size,
                                    args.hidden_size,
                                    n_encode_layers = args.gat_layer_num,
                                    n_heads = 1,
                                    internal_node_holder = args.internal_node_holder,
                                    internal_node_length = args.internal_node_length,
                                    leaf_node_holder = args.leaf_node_holder,
                                    )
        self.critic = ScalarValueHead(args.embedding_size)

    def execute(self, items, deterministic = False, normFactor = 1, evaluate = False):
        o, p, dist_entropy, hidden, _= self.actor(items, deterministic, normFactor = normFactor, evaluate = evaluate)
        values = self.critic(hidden)
        return o, p, dist_entropy,values

    def evaluate_actions(self, items, actions, normFactor = 1):
        _, p, dist_entropy, hidden, dist = self.actor(items, evaluate_action = True, normFactor = normFactor)
        action_log_probs = dist.log_probs(actions)
        values =  self.critic(hidden)
        return values, action_log_probs, dist_entropy.mean()
