import jittor as jt
import numpy as np

class PPO_RolloutStorage(object):
    """Rollout storage backed entirely by numpy arrays.

    In PyTorch the storage uses CUDA tensors and `.copy_()` which implicitly
    detaches from the computation graph.  Jittor has no reliable equivalent:
    `.detach()` is a no-op, `.stop_grad()` flag is lost after slice assignment,
    and even `jt.array(x.numpy())` round-trips still get re-attached when
    stored inside a `jt.Var` via slice indexing.

    The solution is to keep ALL storage as plain **numpy arrays**.  Data is
    converted to `jt.Var` only when yielded by the generator for the PPO
    update, guaranteeing every batch tensor is a fresh leaf variable with no
    history.
    """

    def __init__(self, num_steps, num_processes, obs_shape, action_shape):
        self.obs        = np.zeros((num_steps + 1, num_processes, *obs_shape), dtype=np.float32)
        self.rewards    = np.zeros((num_steps,     num_processes, 1),          dtype=np.float32)
        self.returns    = np.zeros((num_steps + 1, num_processes, 1),          dtype=np.float32)
        self.value_preds = np.zeros((num_steps + 1, num_processes, 1),         dtype=np.float32)
        self.action_log_probs = np.zeros((num_steps, num_processes, *action_shape), dtype=np.float32)
        self.action     = np.zeros((num_steps,     num_processes, *action_shape), dtype=np.float32)
        self.masks      = np.ones((num_steps + 1,  num_processes, 1),          dtype=np.float32)
        self.bad_masks  = np.ones((num_steps + 1,  num_processes, 1),          dtype=np.float32)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

    # ------------------------------------------------------------------
    def to(self, device):
        pass  # numpy arrays live on CPU; Jittor handles device transfer

    # ------------------------------------------------------------------
    @staticmethod
    def _np(x):
        """Convert anything to a numpy array."""
        if isinstance(x, jt.Var):
            return x.numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float32)

    # ------------------------------------------------------------------
    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1]           = self._np(obs)
        self.action[self.step]            = self._np(actions)
        self.action_log_probs[self.step]  = self._np(action_log_probs)
        self.value_preds[self.step]       = self._np(value_preds)
        self.rewards[self.step]           = self._np(rewards)
        self.masks[self.step + 1]         = self._np(masks)
        self.bad_masks[self.step + 1]     = self._np(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    # ------------------------------------------------------------------
    def after_update(self):
        self.obs[0]       = self.obs[-1].copy()
        self.masks[0]     = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    # ------------------------------------------------------------------
    def compute_returns(self, next_value, use_gae, gamma, gae_lambda,
                        use_proper_time_limits=True):
        next_val = self._np(next_value)

        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_val
                gae = 0
                for step in reversed(range(self.num_steps)):
                    delta = (self.rewards[step]
                             + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                             - self.value_preds[step])
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_val
                for step in reversed(range(self.num_steps)):
                    self.returns[step] = (
                        (self.returns[step + 1] * gamma * self.masks[step + 1]
                         + self.rewards[step])
                        * self.bad_masks[step + 1]
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step])
        else:
            if use_gae:
                self.value_preds[-1] = next_val
                gae = 0
                for step in reversed(range(self.num_steps)):
                    delta = (self.rewards[step]
                             + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                             - self.value_preds[step])
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_val
                for step in reversed(range(self.num_steps)):
                    self.returns[step] = (self.returns[step + 1] * gamma
                                          * self.masks[step + 1]
                                          + self.rewards[step])

    # ------------------------------------------------------------------
    def feed_forward_generator(self, advantages, num_mini_batch=None,
                               mini_batch_size=None):
        num_steps  = self.rewards.shape[0]
        num_procs  = self.rewards.shape[1]
        batch_size = num_procs * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be >= the number of PPO mini batches ({})."
                .format(num_procs, num_steps, batch_size, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        # Flatten (num_steps, num_proc, ...) -> (batch, ...)
        obs_flat      = self.obs[:-1].reshape(batch_size, *self.obs.shape[2:])
        actions_flat  = self.action.reshape(batch_size, self.action.shape[-1])
        vpreds_flat   = self.value_preds[:-1].reshape(batch_size, 1)
        returns_flat  = self.returns[:-1].reshape(batch_size, 1)
        masks_flat    = self.masks[:-1].reshape(batch_size, 1)
        old_lp_flat   = self.action_log_probs.reshape(batch_size, 1)

        if advantages is not None:
            if isinstance(advantages, jt.Var):
                adv_flat = advantages.numpy().reshape(batch_size, 1)
            else:
                adv_flat = np.asarray(advantages).reshape(batch_size, 1)
        else:
            adv_flat = None

        perm = np.random.permutation(batch_size)
        for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
            idx = perm[start:start + mini_batch_size]

            # Convert to jt.Var only here — each tensor is a brand-new leaf
            obs_batch       = jt.array(obs_flat[idx])
            actions_batch   = jt.array(actions_flat[idx])
            vpreds_batch    = jt.array(vpreds_flat[idx])
            return_batch    = jt.array(returns_flat[idx])
            masks_batch     = jt.array(masks_flat[idx])
            old_lp_batch    = jt.array(old_lp_flat[idx])
            adv_targ        = jt.array(adv_flat[idx]) if adv_flat is not None else None

            yield (obs_batch, actions_batch, vpreds_batch, return_batch,
                   masks_batch, old_lp_batch, adv_targ)
