import numpy as np
import jittor as jt


def _copy_into(dst, src):
    """Copy src data into dst, detaching from the computation graph."""
    if isinstance(src, jt.Var):
        dst.assign(src.stop_grad())
    else:
        dst.assign(jt.array(np.array(src), dtype=dst.dtype))


def _randperm_indices(batch_size):
    return np.random.permutation(batch_size)


# Storage for n-step training.
class PCTRolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, gamma):
        self.obs = jt.zeros((num_steps + 1, num_processes, *obs_shape))
        self.rewards = jt.zeros((num_steps, num_processes, 1))
        self.returns = jt.zeros((num_steps + 1, num_processes, 1))
        self.action_log_probs = jt.zeros((num_steps, num_processes, 1))
        self.actions = jt.zeros((num_steps, num_processes, 1), dtype=jt.int32)
        self.masks = jt.ones((num_steps + 1, num_processes, 1))
        self.value_preds = jt.zeros((num_steps + 1, num_processes, 1))

        self.num_steps = num_steps
        self.gamma = gamma
        self.step = 0

    def to(self, device):
        return self

    def cuda(self):
        return self

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs.stop_grad()
        self.actions[self.step] = actions.stop_grad()
        self.action_log_probs[self.step] = action_log_probs.stop_grad()
        self.value_preds[self.step] = value_preds.stop_grad()
        self.rewards[self.step] = rewards.stop_grad()
        self.masks[self.step + 1] = masks.stop_grad()
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0] = self.obs[-1]
        self.masks[0] = self.masks[-1]

    def compute_returns(self, next_value):
        # Compute returns in numpy to avoid chaining computation graphs
        next_val_np = next_value.stop_grad().numpy()
        rewards_np = self.rewards.stop_grad().numpy()
        masks_np = self.masks.stop_grad().numpy()
        num_steps = rewards_np.shape[0]
        returns_np = np.zeros(self.returns.shape, dtype=np.float32)
        returns_np[-1] = next_val_np
        for step in reversed(range(num_steps)):
            returns_np[step] = returns_np[step + 1] * self.gamma * masks_np[step + 1] + rewards_np[step]
        self.returns = jt.array(returns_np)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.shape[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        indices = _randperm_indices(batch_size)
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


class RehearsalRolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, gamma, add_obs_shape, action_shape):
        self.obs = jt.zeros((num_steps + 1, num_processes, *obs_shape))
        self.add_obs = jt.zeros((num_steps + 1, num_processes, *add_obs_shape))

        self.rewards = jt.zeros((num_steps, num_processes, 1))
        self.returns = jt.zeros((num_steps + 1, num_processes, 1))
        self.action_log_probs = jt.zeros((num_steps, num_processes, 1))
        self.actions = jt.zeros((num_steps, num_processes, action_shape))
        self.masks = jt.ones((num_steps + 1, num_processes, 1))
        self.value_preds = jt.zeros((num_steps + 1, num_processes, 1))

        self.num_steps = num_steps
        self.gamma = gamma
        self.step = 0

    def to(self, device):
        return self

    def cuda(self):
        return self

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, add_obs):
        _copy_into(self.obs[self.step + 1], obs)
        _copy_into(self.add_obs[self.step + 1], add_obs)
        _copy_into(self.actions[self.step], actions)
        _copy_into(self.action_log_probs[self.step], action_log_probs)
        _copy_into(self.value_preds[self.step], value_preds)
        _copy_into(self.rewards[self.step], rewards)
        _copy_into(self.masks[self.step + 1], masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        _copy_into(self.obs[0], self.obs[-1])
        _copy_into(self.add_obs[0], self.add_obs[-1])
        _copy_into(self.masks[0], self.masks[-1])

    def compute_returns(self, next_value):
        next_val_np = next_value.stop_grad().numpy()
        rewards_np = self.rewards.stop_grad().numpy()
        masks_np = self.masks.stop_grad().numpy()
        num_steps = rewards_np.shape[0]
        returns_np = np.zeros(self.returns.shape, dtype=np.float32)
        returns_np[-1] = next_val_np
        for step in reversed(range(num_steps)):
            returns_np[step] = returns_np[step + 1] * self.gamma * masks_np[step + 1] + rewards_np[step]
        self.returns = jt.array(returns_np)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.shape[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        indices = _randperm_indices(batch_size)
        flat_obs = self.obs[:-1].reshape((batch_size, *self.obs.shape[2:]))
        flat_add_obs = self.add_obs[:-1].reshape((batch_size, *self.add_obs.shape[2:]))
        flat_actions = self.actions.reshape((batch_size, self.actions.shape[-1]))
        flat_value_preds = self.value_preds[:-1].reshape((batch_size, 1))
        flat_returns = self.returns[:-1].reshape((batch_size, 1))
        flat_masks = self.masks[:-1].reshape((batch_size, 1))
        flat_action_log_probs = self.action_log_probs.reshape((batch_size, 1))
        flat_adv = None if advantages is None else advantages.reshape((batch_size, 1))

        for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
            batch_indices = indices[start:start + mini_batch_size].tolist()
            obs_batch = flat_obs[batch_indices]
            add_obs_batch = flat_add_obs[batch_indices]
            actions_batch = flat_actions[batch_indices]
            value_preds_batch = flat_value_preds[batch_indices]
            return_batch = flat_returns[batch_indices]
            masks_batch = flat_masks[batch_indices]
            old_action_log_probs_batch = flat_action_log_probs[batch_indices]
            adv_targ = None if flat_adv is None else flat_adv[batch_indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, add_obs_batch
