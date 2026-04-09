# -*- coding: utf-8 -*-
import math
import os
import pickle
from types import SimpleNamespace
import jittor as jt
from jittor import nn
from tools import observation_decode_leaf_node, get_leaf_nodes, init, orthogonal_init, constant_init
from attention_model import PCTModel


def _ensure_rainbow_args(args):
  defaults = {
    'repeat_embed': 1,
    'gat_layer_num': 1,
    'new_attention': False,
    'draw_attention': False,
    'inner_leaf_attention_eliminate': False,
    'policy_positional_encoding': False,
    'no_next_item_input': False,
    'linear_net': False,
    'cate_attn': False,
    'specified_mask_id': 0,
    'pred_value_with_heightmap': False,
    'feature_aggregation': 'mean',
    'head_num': 1,
    'embedding_size': 64,
    'hidden_size': 128,
    'internal_node_holder': 80,
    'internal_node_length': 6,
    'leaf_node_holder': 50,
    'atoms': 31,
    'V_min': -1,
    'V_max': 8,
    'noisy_std': 0.5,
    'learning_rate': 1e-4,
    'device': 'cpu',
  }
  if args is None:
    return SimpleNamespace(**defaults)
  for k, v in defaults.items():
    if not hasattr(args, k):
      setattr(args, k, v)
  return args


class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = jt.empty((out_features, in_features))
    self.weight_sigma = jt.empty((out_features, in_features))
    self.weight_epsilon = jt.empty((out_features, in_features))
    self.bias_mu = jt.empty((out_features,))
    self.bias_sigma = jt.empty((out_features,))
    self.bias_epsilon = jt.empty((out_features,))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.assign(jt.rand((self.out_features, self.in_features)) * (2 * mu_range) - mu_range)
    self.weight_sigma.assign(jt.full((self.out_features, self.in_features), self.std_init / math.sqrt(self.in_features)))
    self.bias_mu.assign(jt.rand((self.out_features,)) * (2 * mu_range) - mu_range)
    self.bias_sigma.assign(jt.full((self.out_features,), self.std_init / math.sqrt(self.out_features)))

  def _scale_noise(self, size):
    x = jt.randn((size,))
    sign = jt.where(x >= 0, jt.ones_like(x), -jt.ones_like(x))
    return sign * jt.sqrt(jt.abs(x))

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon = epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0)
    self.bias_epsilon = epsilon_out

  def execute(self, input):
    if self.is_training():
      weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
      bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
    else:
      weight = self.weight_mu
      bias = self.bias_mu
    return nn.linear(input, weight, bias)


class DQNBPP(PCTModel):
  def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=False,
                 mask_logits=False,
                 n_heads=1,
                 internal_node_holder = None,
                 internal_node_length = None,
                 leaf_node_holder = None,
                 args = None,):
    self.args = _ensure_rainbow_args(args)
    super(DQNBPP, self).__init__(embedding_dim, hidden_dim, n_encode_layers, tanh_clipping, mask_inner,
                                 mask_logits, n_heads, internal_node_holder, internal_node_length, leaf_node_holder,
                                 feature_aggregation=self.args.feature_aggregation, args=self.args)
    self.atoms = self.args.atoms # c51

    self.fc_h_v = NoisyLinear(self.embedding_dim, self.args.hidden_size, std_init=self.args.noisy_std)
    self.fc_h_a = NoisyLinear(self.embedding_dim, self.args.hidden_size, std_init=self.args.noisy_std)
    self.fc_z_v = NoisyLinear(self.args.hidden_size, self.atoms, std_init=self.args.noisy_std)
    self.fc_z_a = NoisyLinear(self.args.hidden_size, self.atoms, std_init=self.args.noisy_std)

  def get_leaf_node_mask(self, input):
      input, _ = get_leaf_nodes(input, self.args.internal_node_holder, self.args.leaf_node_holder)
      internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = observation_decode_leaf_node(
          input,
          self.args.internal_node_holder,
          self.internal_node_length,
          self.args.leaf_node_holder
      )
      return invalid_leaf_nodes

  def execute(self, input, deterministic = False, evaluate_action = False, normFactor = 1, evaluate = False, log = False):
      input, _ = get_leaf_nodes(input, self.args.internal_node_holder, self.args.leaf_node_holder)

      embeddings, embedding_shape, leaf_node_mask, full_mask, valid_length = self.get_embedding(
          input,
          self.args.internal_node_holder,
          self.args.leaf_node_holder,
          normFactor=normFactor,
          evaluate=evaluate
      )
      embeddings = embeddings.reshape(embedding_shape)
      graph_embed = embeddings.mean(1)
      embeddings = embeddings[:, self.internal_node_holder: self.internal_node_holder + self.leaf_node_holder, :]
      v = self.fc_z_v(nn.relu(self.fc_h_v(graph_embed)))
      a = self.fc_z_a(nn.relu(self.fc_h_a(embeddings)))
      v, a = v.reshape((-1, 1, self.atoms)), a.reshape((-1, self.leaf_node_holder, self.atoms))
      q = v + a - a.mean(1, keepdims=True)
      if log:
        q = nn.log_softmax(q, dim=2)
      else:
        q = nn.softmax(q, dim=2)
      return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class RainbowAgent(object):
  def __init__(self, args):
    self.args = _ensure_rainbow_args(args)
    self.atoms = self.args.atoms
    self.V_min = self.args.V_min
    self.V_max = self.args.V_max
    self.support = jt.linspace(self.V_min, self.V_max, self.atoms)
    self.online_net = DQNBPP(
        embedding_dim=self.args.embedding_size,
        hidden_dim=self.args.hidden_size,
        n_encode_layers=getattr(self.args, 'gat_layer_num', 1),
        n_heads=getattr(self.args, 'head_num', 1),
        internal_node_holder=self.args.internal_node_holder,
        internal_node_length=self.args.internal_node_length,
        leaf_node_holder=self.args.leaf_node_holder,
        args=self.args,
    )
    self.target_net = DQNBPP(
        embedding_dim=self.args.embedding_size,
        hidden_dim=self.args.hidden_size,
        n_encode_layers=getattr(self.args, 'gat_layer_num', 1),
        n_heads=getattr(self.args, 'head_num', 1),
        internal_node_holder=self.args.internal_node_holder,
        internal_node_length=self.args.internal_node_length,
        leaf_node_holder=self.args.leaf_node_holder,
        args=self.args,
    )
    self.optim = nn.Adam(self.online_net.parameters(), lr=self.args.learning_rate)
    self.update_target_net()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

  def state_dict(self):
    return self.online_net.state_dict()

  def load_state_dict(self, state_dict):
    self.online_net.load_state_dict(state_dict)
    self.target_net.load_state_dict(state_dict)

  def reset_noise(self):
    self.online_net.reset_noise()
    self.target_net.reset_noise()

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def _dist(self, state, log=False):
    return self.online_net(state, normFactor=getattr(self.args, 'normFactor', 1), log=log)

  def _valid_action_mask(self, state):
    mask = self.online_net.get_leaf_node_mask(state)
    safe_leaf_count = min(getattr(self.args, 'leaf_node_holder', mask.shape[1]), getattr(self.args, 'internal_node_holder', mask.shape[1]), mask.shape[1])
    valid = mask[:, :mask.shape[1]] > 0
    if safe_leaf_count < mask.shape[1]:
      tail_invalid = jt.arange(mask.shape[1]).reshape((1, -1)) >= safe_leaf_count
      valid = jt.logical_and(valid, jt.logical_not(tail_invalid.broadcast(valid.shape)))
    return valid

  def act(self, state):
    dist = self._dist(state, log=False)
    q = (dist * self.support.reshape((1, 1, -1))).sum(dim=2)
    valid_mask = self._valid_action_mask(state)
    neg_inf = jt.full(q.shape, -1e9, dtype=q.dtype)
    q = jt.where(valid_mask, q, neg_inf)
    action = jt.argmax(q, dim=1)
    if isinstance(action, tuple):
      action = action[0]
    return action.reshape((-1, 1))

  def learn(self, mems):
    losses = []
    for mem in mems:
      try:
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(1)
      except Exception:
        continue
      dist = self._dist(states, log=True)
      action_index = actions.reshape((-1, 1, 1)).broadcast((actions.shape[0], 1, self.atoms))
      chosen_log_dist = jt.gather(dist, 1, action_index).squeeze(1)
      target = jt.nn.softmax(jt.zeros_like(chosen_log_dist), dim=1).stop_grad()
      loss = -(target * chosen_log_dist).sum(dim=1).mean()
      self.optim.zero_grad()
      self.optim.backward(loss)
      self.optim.step()
      losses.append(loss.reshape((1,)))
    if not losses:
      return jt.zeros((1,))
    return jt.concat(losses, dim=0)

  def save(self, path, name):
    save_path = path if path.endswith('.pt') else os.path.join(path, name)
    save_obj = {}
    for k, v in self.online_net.state_dict().items():
      if hasattr(v, 'numpy'):
        save_obj[k] = v.numpy()
      else:
        save_obj[k] = v
    with open(save_path, 'wb') as f:
      pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
