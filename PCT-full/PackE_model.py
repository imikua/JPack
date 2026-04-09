import numpy as np
import jittor as jt
from jittor import nn

from distributions import FixedCategorical
from tools import init, orthogonal_init, constant_init


def _to_storage_var(src, dtype, shape=None):
    if src is None:
        assert shape is not None
        return jt.zeros(shape, dtype=dtype)
    if isinstance(src, jt.Var):
        out = src.stop_grad()
        if out.dtype != dtype:
            out = out.astype(dtype)
        return out
    return jt.array(np.array(src), dtype=dtype)


def _resize_bilinear(x, target_size):
    return nn.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


class Flatten(nn.Module):
    def execute(self, x):
        return x.reshape((x.shape[0], -1))


class ScalarValueHead(nn.Module):
    def __init__(self, in_features):
        super(ScalarValueHead, self).__init__()
        self.weight = jt.zeros((1, in_features), dtype=jt.float32)
        orthogonal_init(self.weight, gain=np.sqrt(2.0))

    def execute(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        else:
            x = x.reshape((-1, x.shape[-1]))
        return (x * self.weight).sum(dim=1, keepdims=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, container_size):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            orthogonal_init,
            lambda x: constant_init(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs * 2, container_size[0] * container_size[1] * 2))
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, feature, mask):
        latent = self.linear(feature)
        inver_mask = 1 - mask
        latent = jt.where(inver_mask == 1, jt.full(latent.shape, -1e9, dtype=latent.dtype), latent)
        prob = self.softmax(latent)
        return FixedCategorical(probs=prob)

    def get_policy_distribution(self,x):
        x = self.linear(x)
        mx = self.softmax(x)
        return mx


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, container_size, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.recoder = []
        self.recoder_counter = 0
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = CNNBase # 10 * 10

        self.base = base(obs_shape[0], container_size = container_size, **base_kwargs)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, container_size)
        self.container_size = container_size

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def execute(self, inputs, rnn_hxs=None, masks=None):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features, location_masks = self.base(inputs)
        dist= self.dist(actor_features, location_masks)

        if deterministic:  # 这里要有一个解耦动作的机制
            action = dist.mode()
        else:
            action = dist.sample()
        action = action.reshape(-1, 1)
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features, location_masks = self.base(inputs)
        dist = self.dist(actor_features, location_masks)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


def observation_decode(observation, contianer_size):
    frontiers = observation[:, 0: contianer_size[0] * contianer_size[1]]
    maskposition = observation[:, contianer_size[0] * contianer_size[1]: -6]
    next_box = observation[:, -6:]
    return frontiers, maskposition, next_box

class CNNBase(NNBase):
    def __init__(self, num_inputs,  container_size,  recurrent=False,  hidden_size=128):
        recurrent = False,
        super(CNNBase, self).__init__(recurrent, num_inputs, hidden_size)

        activate = nn.LeakyReLU
        self.container_size = container_size
        self.max_axis = max(self.container_size[0], self.container_size[1])
        init_ = lambda m: init(m, orthogonal_init, lambda x: constant_init(x, 0), 1.0)
        self.target_size = (256, 256)
        outchannel = 4

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(1, outchannel, 4, stride=2, padding=1)),
            activate(),
            init_(nn.Conv2d(outchannel, outchannel, 4, stride=4)),
            activate(),
            init_(nn.Conv2d(outchannel, 4, 4, stride=4)),
            activate(),
            Flatten())
        self.image_encoder_linear = nn.Sequential(init_(nn.Linear(256, hidden_size)), activate())


        self.next_box_encoder = nn.Sequential(
            init_(nn.Linear(6, hidden_size)),
            activate(),
            init_(nn.Linear(hidden_size, hidden_size)),
            activate())

        self.critic_linear = ScalarValueHead(hidden_size * 2)
        self.train()

    def execute(self, inputs):
        frontiers, maskposition, next_box = observation_decode(inputs, self.container_size)
        frontiers = frontiers.reshape((-1, 1, self.container_size[0], self.container_size[1]))
        frontiers = _resize_bilinear(frontiers, self.target_size)
        image_vector = self.image_encoder(frontiers)
        image_vector = self.image_encoder_linear(image_vector)
        next_box_vector = self.next_box_encoder(next_box)
        hidden_vector = jt.concat((image_vector, next_box_vector), dim=1)
        return self.critic_linear(hidden_vector), hidden_vector, maskposition


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape((T * N, *_tensor.shape[2:]))


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = jt.zeros((num_steps + 1, num_processes, *obs_shape), dtype=jt.float32)
        self.rewards = jt.zeros((num_steps, num_processes, 1), dtype=jt.float32)
        self.value_preds = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.returns = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.action_log_probs = jt.zeros((num_steps, num_processes, 1), dtype=jt.float32)
        self.actions = jt.zeros((num_steps, num_processes, 1), dtype=jt.int32)
        self.masks = jt.ones((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.bad_masks = jt.ones((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        return self

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1] = _to_storage_var(obs, self.obs.dtype)
        self.actions[self.step] = _to_storage_var(actions.reshape((-1, 1)), self.actions.dtype)
        self.action_log_probs[self.step] = _to_storage_var(
            action_log_probs.reshape((-1, 1)), self.action_log_probs.dtype
        )
        self.value_preds[self.step] = _to_storage_var(
            value_preds.reshape((-1, 1)), self.value_preds.dtype
        )
        self.rewards[self.step] = _to_storage_var(rewards.reshape((-1, 1)), self.rewards.dtype)
        self.masks[self.step + 1] = _to_storage_var(masks.reshape((-1, 1)), self.masks.dtype)
        self.bad_masks[self.step + 1] = _to_storage_var(
            bad_masks.reshape((-1, 1)), self.bad_masks.dtype
        )
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0] = self.obs[-1].stop_grad()
        self.masks[0] = self.masks[-1].stop_grad()
        self.bad_masks[0] = self.bad_masks[-1].stop_grad()

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
        next_val_np = next_value.stop_grad().numpy().reshape((-1, 1))
        rewards_np = self.rewards.stop_grad().numpy()
        masks_np = self.masks.stop_grad().numpy()
        bad_masks_np = self.bad_masks.stop_grad().numpy()
        value_preds_np = self.value_preds.stop_grad().numpy()
        returns_np = np.zeros(self.returns.shape, dtype=np.float32)

        if use_proper_time_limits:
            if use_gae:
                value_preds_np[-1] = next_val_np
                gae = np.zeros_like(next_val_np, dtype=np.float32)
                for step in reversed(range(rewards_np.shape[0])):
                    delta = rewards_np[step] + gamma * value_preds_np[step + 1] * masks_np[step + 1] - value_preds_np[step]
                    gae = delta + gamma * gae_lambda * masks_np[step + 1] * gae
                    gae = gae * bad_masks_np[step + 1]
                    returns_np[step] = gae + value_preds_np[step]
            else:
                returns_np[-1] = next_val_np
                for step in reversed(range(rewards_np.shape[0])):
                    returns_np[step] = (
                        returns_np[step + 1] * gamma * masks_np[step + 1] + rewards_np[step]
                    ) * bad_masks_np[step + 1] + (1 - bad_masks_np[step + 1]) * value_preds_np[step]
        else:
            if use_gae:
                value_preds_np[-1] = next_val_np
                gae = np.zeros_like(next_val_np, dtype=np.float32)
                for step in reversed(range(rewards_np.shape[0])):
                    delta = rewards_np[step] + gamma * value_preds_np[step + 1] * masks_np[step + 1] - value_preds_np[step]
                    gae = delta + gamma * gae_lambda * masks_np[step + 1] * gae
                    returns_np[step] = gae + value_preds_np[step]
            else:
                returns_np[-1] = next_val_np
                for step in reversed(range(rewards_np.shape[0])):
                    returns_np[step] = returns_np[step + 1] * gamma * masks_np[step + 1] + rewards_np[step]

        self.returns = jt.array(returns_np, dtype=jt.float32)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.shape[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        indices = np.random.permutation(batch_size)
        flat_obs = self.obs[:-1].reshape((batch_size, *self.obs.shape[2:]))
        flat_actions = self.actions.reshape((batch_size, self.actions.shape[-1]))
        flat_value_preds = self.value_preds[:-1].reshape((batch_size, 1))
        flat_returns = self.returns[:-1].reshape((batch_size, 1))
        flat_masks = self.masks[:-1].reshape((batch_size, 1))
        flat_action_log_probs = self.action_log_probs.reshape((batch_size, 1))
        flat_adv = None if advantages is None else advantages.reshape((batch_size, 1))

        for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
            batch_indices = indices[start:start + mini_batch_size].tolist()
            obs_batch = flat_obs[batch_indices]
            actions_batch = flat_actions[batch_indices]
            value_preds_batch = flat_value_preds[batch_indices]
            return_batch = flat_returns[batch_indices]
            masks_batch = flat_masks[batch_indices]
            old_action_log_probs_batch = flat_action_log_probs[batch_indices]
            adv_targ = None if flat_adv is None else flat_adv[batch_indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
