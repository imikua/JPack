import numpy as np
import jittor as jt

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
        # Jittor manages device automatically, this is a no-op for compatibility
        pass

    def cuda(self):
        # Jittor manages device automatically via jt.flags.use_cuda
        pass

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
        next_val_np = next_value.stop_grad().numpy()
        rewards_np = self.rewards.stop_grad().numpy()
        masks_np = self.masks.stop_grad().numpy()
        returns_np = np.zeros(self.returns.shape, dtype=np.float32)
        returns_np[-1] = next_val_np
        for step in reversed(range(self.rewards.shape[0])):
            returns_np[step] = returns_np[step + 1] * self.gamma * masks_np[step + 1] + rewards_np[step]
        self.returns = jt.array(returns_np, dtype=jt.float32)

