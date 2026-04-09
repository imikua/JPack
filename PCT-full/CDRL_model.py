import os
import numpy as np
import jittor as jt
from jittor import nn

from distributions import FixedCategorical
from tools import init, orthogonal_init, constant_init
from kfac import KFACOptimizer


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


class ScalarValueHead(nn.Module):
    def __init__(self, in_features, gain=1.0):
        super(ScalarValueHead, self).__init__()
        self.weight = jt.zeros((1, in_features), dtype=jt.float32)
        orthogonal_init(self.weight, gain=gain)

    def execute(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        else:
            x = x.reshape((-1, x.shape[-1]))
        return (x * self.weight).sum(dim=1, keepdims=True)


class Flatten(nn.Module):
    def execute(self, x):
        return x.reshape((x.shape[0], -1))


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, container_size, enable_rotation = True):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            orthogonal_init,
            lambda x: constant_init(x, 0),
            gain=0.01)

        self.linear_o = init_(nn.Linear(num_inputs, 1 + enable_rotation))
        self.linear_x = init_(nn.Linear(num_inputs, container_size[0]))
        self.linear_y = init_(nn.Linear(num_inputs, container_size[1]))
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, feature, mask, container_size, actiono=None, actionx=None):
        compensation = 1e-20
        prob, latentV = None, None
        enable_rotation = True
        if actiono is None and actionx is None:
            latentV = self.linear_o(feature)
            prob = self.softmax(latentV)
            self.prob_o_before = prob
        elif actiono is not None and actionx is None:
            x_norm = (1 + enable_rotation)
            latentV = self.linear_x(feature + actiono.float32().stop_grad().reshape((-1, 1)) / x_norm)
            prob = self.softmax(latentV)
            self.prob_x_before = prob
        elif actiono is not None and actionx is not None:
            y_norm = (1 + enable_rotation) * container_size[0]
            latentV = self.linear_y(feature + actionx.float32().stop_grad().reshape((-1, 1)) * actiono.float32().stop_grad().reshape((-1, 1)) / y_norm)
            prob = self.softmax(latentV)
            self.prob_y_before = prob

        ones = jt.ones_like(mask)
        inver_mask = ones - mask
        logx = jt.log(1 + self.softmax(latentV))
        bad_prob = logx * inver_mask
        mask_value = 1e-3
        prob = prob * (mask + mask_value * inver_mask) + compensation
        prob = prob / jt.clamp(prob.sum(dim=-1, keepdims=True), min_v=1e-12)
        return FixedCategorical(probs=prob), bad_prob


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, channel=6, container_size = None, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.recoder = []
        self.recoder_counter = 0
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = CNNBase

        self.enable_rotation = True
        self.base = base(obs_shape[0], channel, container_size, **base_kwargs)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, container_size, self.enable_rotation)
        self.o_entropy_coef = np.log(container_size[0]) / np.log(2)
        self.o_prob_coef = container_size[0] / 2
        self.container_size = container_size

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def execute(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_o_mask(self, location_masks):
        oxy_mask = location_masks.reshape((-1, 1 + self.enable_rotation, self.container_size[0] * self.container_size[1]))
        o_mask = jt.sum(oxy_mask, 2)
        o_mask[o_mask > 0] = 1
        return o_mask

    def get_x_mask(self, location_masks, action_o):
        oxy_mask = location_masks.reshape((-1, 1 + self.enable_rotation, self.container_size[0], self.container_size[1]))
        batch_index = jt.arange(oxy_mask.shape[0])
        xy_masks = oxy_mask[batch_index, action_o.reshape((-1,))]
        x_mask = jt.sum(xy_masks, 2)
        x_mask[x_mask > 0] = 1
        return x_mask, xy_masks

    def get_y_mask(self, xy_masks, action_x):
        oxy_mask = xy_masks.reshape((-1, self.container_size[0], self.container_size[1]))
        batch_index = jt.arange(oxy_mask.shape[0])
        y_mask = oxy_mask[batch_index, action_x.reshape((-1,))]
        return y_mask

    def act(self, inputs, rnn_hxs, masks, location_masks, deterministic=False):

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        o_mask = self.get_o_mask(location_masks)
        dist_o, bad_prob_o = self.dist(actor_features, o_mask, container_size = self.container_size)

        if deterministic:  # 这里要有一个解耦动作的机制
            action_o = dist_o.mode()
        else:
            action_o = dist_o.sample()

        x_mask, xy_masks = self.get_x_mask(location_masks, action_o)
        dist_x, bad_prob_x = self.dist(actor_features, x_mask, actiono = action_o, container_size = self.container_size)

        if deterministic:  # 这里要有一个解耦动作的机制
            action_x = dist_x.mode()
        else:
            action_x = dist_x.sample()

        y_mask = self.get_y_mask(xy_masks, action_x)
        dist_y, bad_prob_y = self.dist(actor_features, y_mask, actiono = action_o, actionx = action_x, container_size = self.container_size)

        if deterministic:  # 这里要有一个解耦动作的机制
            action_y = dist_y.mode()
        else:
            action_y = dist_y.sample()

        action_log_probs_o = dist_o.log_probs(action_o)
        action_log_probs_x = dist_x.log_probs(action_x)
        action_log_probs_y = dist_y.log_probs(action_y)

        action_log_probs = jt.concat((action_log_probs_o, action_log_probs_x, action_log_probs_y), dim=1)
        action = jt.concat((action_o, action_x, action_y), dim=1)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, location_masks):

        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        o_mask = self.get_o_mask(location_masks)
        x_mask, xy_masks = self.get_x_mask(location_masks, action_o)
        y_mask = self.get_y_mask(xy_masks, action_x)

        dist_o, bad_prob_o = self.dist(actor_features, o_mask, container_size = self.container_size)
        dist_x, bad_prob_x = self.dist(actor_features, x_mask, actiono = action_o, container_size = self.container_size)
        dist_y, bad_prob_y = self.dist(actor_features, y_mask, actiono = action_o, actionx = action_x, container_size = self.container_size)

        action_log_probs_o = dist_o.log_probs(action_o)
        action_log_probs_x = dist_x.log_probs(action_x)
        action_log_probs_y = dist_y.log_probs(action_y)
        action_log_probs = jt.concat((action_log_probs_o, action_log_probs_x, action_log_probs_y), dim=1)

        dist_entropy_o = dist_o.entropy().mean()
        dist_entropy_x = dist_x.entropy().mean()
        dist_entropy_y = dist_y.entropy().mean()
        dist_entropy = dist_entropy_o + dist_entropy_x + dist_entropy_y # 我有必要

        bad_prob_o = bad_prob_o.mean()
        bad_prob_x = bad_prob_x.mean()
        bad_prob_y = bad_prob_y.mean()
        bad_prob = self.o_prob_coef * bad_prob_o + bad_prob_x + bad_prob_y


        return value, action_log_probs, dist_entropy, rnn_hxs, bad_prob, \
               [dist_entropy_o, dist_entropy_x, dist_entropy_y, bad_prob_o, bad_prob_x, bad_prob_y]


class NNBase(nn.Module):  # 如果是递归的话，就在这里创建递归模型
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    # 返回模型属性
    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size



class CNNBase(NNBase):
    def __init__(self, num_inputs, channel = 6, container_size = None, recurrent=False, hidden_size=128):
        super(CNNBase, self).__init__(recurrent, num_inputs, hidden_size)
        activate = nn.LeakyReLU
        # Match the torch reference initialization for LeakyReLU blocks.
        leaky_relu_gain = float(np.sqrt(2.0 / (1.0 + 0.01 ** 2)))
        init_ = lambda m: init(m, orthogonal_init, lambda x: constant_init(x, 0), leaky_relu_gain)

        self.container_size = container_size
        self.channel = channel
        self.target_size = (256, 256)
        outchannel = 4

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(channel, outchannel, 4, stride=2, padding=1)),
            activate(),
            init_(nn.Conv2d(outchannel, outchannel, 4, stride=4)),
            activate(),
            init_(nn.Conv2d(outchannel, 4, 4, stride=4)),
            activate(),
            Flatten())
        self.image_encoder_linear = nn.Sequential(init_(nn.Linear(256, hidden_size)), activate())

        # Torch used a dedicated `nn.Linear(..., 1)` with orthogonal gain=1 here.
        self.critic_linear = ScalarValueHead(hidden_size, gain=1.0)
        self.train()

    def execute(self, inputs, rnn_hxs, masks):
        x = inputs.reshape((-1, self.channel, self.container_size[0], self.container_size[1]))
        x = _resize_bilinear(x, self.target_size)
        x = self.image_encoder(x)
        x = self.image_encoder_linear(x)
        assert not self.is_recurrent
        return self.critic_linear(x), x, rnn_hxs


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape((T * N, *_tensor.shape[2:]))


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, can_give_up, enable_rotation, pallet_size):
        self.obs = jt.zeros((num_steps + 1, num_processes, *obs_shape), dtype=jt.float32)
        self.recurrent_hidden_states = jt.zeros(
            (num_steps + 1, num_processes, recurrent_hidden_state_size), dtype=jt.float32)
        self.rewards = jt.zeros((num_steps, num_processes, 1), dtype=jt.float32)
        self.value_preds = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.returns = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.action_log_probs = jt.zeros((num_steps, num_processes, 3), dtype=jt.float32)
        self.actions = jt.zeros((num_steps, num_processes, 3), dtype=jt.int32)
        self.masks = jt.ones((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.bad_masks = jt.ones((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.location_masks = jt.zeros(
            (num_steps + 1, num_processes, int(2 * pallet_size[0] * pallet_size[1])), dtype=jt.float32)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        return self

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, location_masks):
        self.obs[self.step + 1] = _to_storage_var(obs, self.obs.dtype)
        self.recurrent_hidden_states[self.step + 1] = _to_storage_var(
            recurrent_hidden_states,
            self.recurrent_hidden_states.dtype,
            shape=self.recurrent_hidden_states[self.step + 1].shape,
        )
        self.actions[self.step] = _to_storage_var(actions.reshape((-1, 3)), self.actions.dtype)
        self.action_log_probs[self.step] = _to_storage_var(
            action_log_probs.reshape((-1, 3)), self.action_log_probs.dtype
        )
        self.value_preds[self.step] = _to_storage_var(
            value_preds.reshape((-1, 1)), self.value_preds.dtype
        )
        self.rewards[self.step] = _to_storage_var(rewards.reshape((-1, 1)), self.rewards.dtype)
        self.masks[self.step + 1] = _to_storage_var(masks.reshape((-1, 1)), self.masks.dtype)
        self.bad_masks[self.step + 1] = _to_storage_var(
            bad_masks.reshape((-1, 1)), self.bad_masks.dtype
        )
        self.location_masks[self.step + 1] = _to_storage_var(
            location_masks, self.location_masks.dtype
        )
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0] = self.obs[-1].stop_grad()
        self.recurrent_hidden_states[0] = self.recurrent_hidden_states[-1].stop_grad()
        self.masks[0] = self.masks[-1].stop_grad()
        self.bad_masks[0] = self.bad_masks[-1].stop_grad()
        self.location_masks[0] = self.location_masks[-1].stop_grad()

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
        next_val_np = next_value.stop_grad().numpy().reshape((-1, 1))
        rewards_np = self.rewards.stop_grad().numpy()
        masks_np = self.masks.stop_grad().numpy()
        bad_masks_np = self.bad_masks.stop_grad().numpy()
        value_preds_np = self.value_preds.stop_grad().numpy()
        num_steps = rewards_np.shape[0]
        returns_np = np.zeros(self.returns.shape, dtype=np.float32)

        if use_proper_time_limits:
            if use_gae:
                value_preds_np[-1] = next_val_np
                gae = np.zeros_like(next_val_np, dtype=np.float32)
                for step in reversed(range(num_steps)):
                    delta = rewards_np[step] + gamma * value_preds_np[step + 1] * masks_np[step + 1] - value_preds_np[step]
                    gae = delta + gamma * gae_lambda * masks_np[step + 1] * gae
                    gae = gae * bad_masks_np[step + 1]
                    returns_np[step] = gae + value_preds_np[step]
            else:
                returns_np[-1] = next_val_np
                for step in reversed(range(num_steps)):
                    returns_np[step] = (
                        returns_np[step + 1] * gamma * masks_np[step + 1] + rewards_np[step]
                    ) * bad_masks_np[step + 1] + (1 - bad_masks_np[step + 1]) * value_preds_np[step]
        else:
            if use_gae:
                value_preds_np[-1] = next_val_np
                gae = np.zeros_like(next_val_np, dtype=np.float32)
                for step in reversed(range(num_steps)):
                    delta = rewards_np[step] + gamma * value_preds_np[step + 1] * masks_np[step + 1] - value_preds_np[step]
                    gae = delta + gamma * gae_lambda * masks_np[step + 1] * gae
                    returns_np[step] = gae + value_preds_np[step]
            else:
                returns_np[-1] = next_val_np
                for step in reversed(range(num_steps)):
                    returns_np[step] = returns_np[step + 1] * gamma * masks_np[step + 1] + rewards_np[step]

        self.returns = jt.array(returns_np, dtype=jt.float32)

class A2C_XY_MASK():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr = None,
                 eps = None,
                 alpha = None,
                 max_grad_norm = None,
                 acktr = False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.loss_func = nn.MSELoss(reduction='none')
        # self.loss_func = nn.NLLLoss(reduce=False, size_average=True) #1205
        self.acktr_optimizer = KFACOptimizer(actor_critic) if acktr else None
        self.optimizer = None if acktr else nn.Adam(
            actor_critic.parameters(),
            lr=1e-3 if lr is None else lr
        )

    def update(self, rollouts):
        # check_nan(self.actor_critic, 1)
        obs_shape = rollouts.obs.shape[2:]
        action_shape = rollouts.actions.shape[-1]
        num_steps, num_processes, _ = rollouts.rewards.shape
        mask_size = rollouts.location_masks.shape[-1]

        flat_obs = rollouts.obs[:-1].reshape((-1, *obs_shape))
        flat_rnn = rollouts.recurrent_hidden_states[0].reshape((-1, self.actor_critic.recurrent_hidden_state_size))
        flat_masks = rollouts.masks[:-1].reshape((-1, 1))
        flat_actions = rollouts.actions.reshape((-1, action_shape))
        flat_loc_masks = rollouts.location_masks[:-1].reshape((-1, mask_size))

        if self.acktr and self.acktr_optimizer.steps % self.acktr_optimizer.Ts == 0:
            self.acktr_optimizer._capturing = True
            fisher_values, fisher_action_log_probs, _, _, _, _ = self.actor_critic.evaluate_actions(
                flat_obs, flat_rnn, flat_masks, flat_actions, flat_loc_masks)
            self.acktr_optimizer._capturing = False
            fisher_values = fisher_values.reshape((num_steps, num_processes, 1))
            fisher_action_log_probs = fisher_action_log_probs.reshape((num_steps, num_processes, 3))
            pg_fisher_loss = -fisher_action_log_probs.mean()
            value_noise = jt.randn(fisher_values.shape)
            sample_values = fisher_values + value_noise
            vf_fisher_loss = -(fisher_values - sample_values.stop_grad()).sqr().mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.acktr_optimizer.acc_stats = True
            self.acktr_optimizer.backward_and_step(fisher_loss)
            self.acktr_optimizer.acc_stats = False

        values, action_log_probs, dist_entropy, _, bad_prob, metrics = self.actor_critic.evaluate_actions(
            flat_obs, flat_rnn, flat_masks, flat_actions, flat_loc_masks)
        values = values.reshape((num_steps, num_processes, 1))
        action_log_probs = action_log_probs.reshape((num_steps, num_processes, 3))

        critic_advantages = rollouts.returns[:-1] - values
        actor_advantages = (rollouts.returns[:-1] - values).stop_grad()
        value_loss = (critic_advantages * critic_advantages).mean()
        action_loss = -(actor_advantages * action_log_probs).mean()
        prob_loss = bad_prob

        prob_loss_coef = 1
        action_loss_coef = 1  #
        total_loss = (
            prob_loss * prob_loss_coef+
            value_loss * self.value_loss_coef
            + action_loss * action_loss_coef
            - dist_entropy * self.entropy_coef
        )

        debug_baselines = os.environ.get('PCT_DEBUG_BASELINES', '0') == '1'
        debug_before = None
        if debug_baselines and not self.acktr and self.actor_critic is not None:
            debug_before = []
            for idx, (_, p) in enumerate(self.actor_critic.named_parameters()):
                if idx >= 3:
                    break
                pd = p.data
                if not isinstance(pd, np.ndarray):
                    pd = np.array(pd)
                debug_before.append(pd.copy())

        if self.acktr:
            self.acktr_optimizer.backward_and_step(total_loss, max_grad_norm=self.max_grad_norm)
        else:
            self.optimizer.zero_grad()
            # Use the official Jittor update entrypoint for complex graphs.
            self.optimizer.step(total_loss)

        if debug_baselines:
            rewards_np = rollouts.rewards.stop_grad().numpy()
            returns_np = rollouts.returns[:-1].stop_grad().numpy()
            values_np = values.stop_grad().numpy()
            print(
                "[debug CDRL] "
                f"reward_mean={rewards_np.mean():.6e} reward_std={rewards_np.std():.6e} "
                f"return_mean={returns_np.mean():.6e} return_std={returns_np.std():.6e} "
                f"value_mean={values_np.mean():.6e} value_std={values_np.std():.6e}"
            )
            if debug_before is not None:
                deltas = []
                for idx, (_, p) in enumerate(self.actor_critic.named_parameters()):
                    if idx >= len(debug_before):
                        break
                    pd = p.data
                    if not isinstance(pd, np.ndarray):
                        pd = np.array(pd)
                    deltas.append(float(np.max(np.abs(pd - debug_before[idx]))))
                print(f"[debug CDRL] first_param_deltas={[f'{d:.3e}' for d in deltas]}")

        return float(value_loss.numpy()), float(action_loss.numpy()), float(dist_entropy.numpy()), metrics