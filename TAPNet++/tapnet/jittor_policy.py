import os
import numpy as np
import jittor as jt
from jittor import nn
from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import BasePolicy


class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = max(m2 / total_count, 1e-12)
        self.count = total_count

    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


class JittorCategorical:
    def __init__(self, logits=None, probs=None):
        if logits is not None:
            self.logits = logits
            self.probs = nn.softmax(logits, dim=-1)
        else:
            self.probs = probs
            self.logits = jt.log(probs + 1e-12)

    def sample(self):
        return jt.multinomial(self.probs, 1, replacement=True).reshape(-1)

    def log_prob(self, act):
        act = act.reshape((-1, 1)).astype(jt.int32)
        return jt.gather(self.logits, 1, act).reshape(-1)

    def entropy(self):
        return -(self.probs * self.logits).sum(1)


class JittorPPOPolicy(BasePolicy):
    def __init__(
        self,
        actor,
        critic,
        optim,
        dist_fn=None,
        discount_factor=0.99,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        reward_normalization=0,
        value_clip=0,
        norm_adv=0,
        recompute_adv=0,
        action_scaling=True,
        action_bound_method="clip",
        lr_scheduler=None,
        dual_clip=None,
        value_mini_batch=64,
        **kwargs
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self.dist_fn = dist_fn

        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.action_scaling = action_scaling
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        # mini-batch size used during value inference (process_fn / recompute_adv)
        # Must be small enough to avoid GPU OOM with large step_per_collect.
        self._value_mini_batch = value_mini_batch

        self._norm_adv = norm_adv
        self._recompute_adv = recompute_adv
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._debug = os.environ.get("TAPNET_DEBUG_PPO", "0") == "1"
        self._last_buffer = None
        self._last_indice = None

    def _unwrap_obs(self, batch_like, key="obs"):
        if hasattr(batch_like, key):
            return getattr(batch_like, key)
        return batch_like

    def _value_scale(self):
        return self.ret_rms.std if self._rew_norm else 1.0

    def _normalize_value(self, value):
        value = np.asarray(value, dtype=np.float32)
        return value / self._value_scale()

    def _unnormalize_value(self, value):
        value = np.asarray(value, dtype=np.float32)
        return value * self._value_scale()

    def _critic_raw_values(self, batch_like, key):
        """Compute critic values in small mini-batches to avoid GPU OOM.

        With large step_per_collect (e.g. 1024) and large observation tensors
        (e.g. [1024, 300, 128] intermediate activations), evaluating the whole
        buffer at once exceeds 14 GB VRAM. We therefore split into chunks of
        size self._value_mini_batch (default 64).
        """
        values = []
        with jt.no_grad():
            # Keep the original batch order. Value targets must align with the
            # collector order, otherwise GAE/returns become meaningless.
            for minibatch in batch_like.split(
                self._value_mini_batch, shuffle=False, merge_last=True
            ):
                obs = self._unwrap_obs(minibatch, key)
                pred = to_numpy(self.critic(obs)).reshape(-1)
                values.append(self._unnormalize_value(pred))
        return np.concatenate(values, axis=0).reshape(-1)

    def _postprocess_returns(self, batch, returns_raw, adv_raw, v_s_raw, update_rms):
        if update_rms and self._rew_norm:
            self.ret_rms.update(returns_raw)

        batch.returns = self._normalize_value(returns_raw).reshape(-1)
        batch.v_s = self._normalize_value(v_s_raw).reshape(-1)
        # Keep advantage on its original scale.
        # Tianshou's reward/value normalization is meant for returns / value
        # targets, not for the PPO policy-gradient term itself. Scaling adv by
        # ret_rms changes the clip objective magnitude and drifts far away from
        # the torch baseline.
        batch.adv = np.asarray(adv_raw, dtype=np.float32).reshape(-1)
        return batch

    def _compute_returns_and_advantages(self, batch, update_rms, buffer=None, indice=None):
        v_s_ = self._critic_raw_values(batch, "obs_next")
        v_s = self._critic_raw_values(batch, "obs")
        returns_raw, adv_raw = self.compute_episodic_return(
            batch, buffer, indice, v_s_, v_s, self.gamma, self.gae_lambda
        )
        self._postprocess_returns(batch, returns_raw, adv_raw, v_s, update_rms=update_rms)

    def forward(self, batch, state=None, **kwargs):
        obs = self._unwrap_obs(batch, "obs")
        probs, state = self.actor(
            obs, state=state, info=batch.info if hasattr(batch, "info") else {}
        )
        dist = JittorCategorical(probs=probs)

        if self.training:
            act = dist.sample()
            act_np = np.asarray(to_numpy(act)).reshape(-1).astype(np.int64, copy=False)
        else:
            act_np = np.asarray(to_numpy(probs)).argmax(axis=-1).astype(np.int64, copy=False).reshape(-1)
        return Batch(logits=probs, act=act_np, state=state, dist=dist)

    def process_fn(self, batch, buffer, indice):
        self._last_buffer = buffer
        self._last_indice = indice
        self._compute_returns_and_advantages(
            batch, update_rms=True, buffer=buffer, indice=indice
        )

        batch.act = np.asarray(batch.act, dtype=np.int64).reshape(-1)

        # Compute old log-probs in mini-batches to avoid GPU OOM when
        # step_per_collect is large (e.g. 1024 × [300, 128] tensors).
        all_logp_old = []
        with jt.no_grad():
            # Keep sequential order here as well. If split shuffles, logp_old
            # and returns no longer correspond to the same sample.
            for minibatch in batch.split(
                self._value_mini_batch, shuffle=False, merge_last=True
            ):
                obs = self._unwrap_obs(minibatch, "obs")
                act_chunk = jt.array(np.asarray(minibatch.act).reshape(-1)).astype(jt.int32)
                probs, _ = self.actor(
                    obs, state=None, info=minibatch.info if hasattr(minibatch, "info") else {}
                )
                dist = JittorCategorical(probs=probs)
                logp_old = dist.log_prob(act_chunk)
                all_logp_old.append(np.asarray(to_numpy(logp_old)).reshape(-1).astype(np.float32))

        batch.logp_old = np.concatenate(all_logp_old, axis=0)
        return batch

    def compute_episodic_return(self, batch, buffer, indice, v_s_, v_s, gamma, gae_lambda):
        # Prefer Tianshou's own return computation so episode boundaries,
        # unfinished trajectories, and vectorized buffers follow the same
        # semantics as the original torch PPOPolicy.
        base_compute = getattr(BasePolicy, "compute_episodic_return", None)
        if base_compute is not None:
            for args in (
                (batch, buffer, indice, v_s_, v_s, gamma, gae_lambda),
                (batch, buffer, indice, v_s_, gamma, gae_lambda),
            ):
                try:
                    returns, advs = base_compute(*args)
                    return (
                        np.asarray(returns, dtype=np.float32).reshape(-1),
                        np.asarray(advs, dtype=np.float32).reshape(-1),
                    )
                except TypeError:
                    continue

        rew = np.asarray(batch.rew, dtype=np.float32).reshape(-1)
        done = np.asarray(batch.done, dtype=np.float32).reshape(-1)

        returns = np.zeros_like(rew)
        advs = np.zeros_like(rew)

        last_gae = 0.0
        for t in reversed(range(len(rew))):
            if done[t]:
                last_gae = 0.0
            delta = rew[t] + gamma * v_s_[t] * (1.0 - done[t]) - v_s[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - done[t]) * last_gae
            advs[t] = last_gae
            returns[t] = advs[t] + v_s[t]
        return returns, advs

    def learn(self, batch, batch_size, repeat, **kwargs):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        update_happened = False

        for step in range(repeat):
            if self._recompute_adv and step > 0:
                if self._debug:
                    print(f"[ppo-debug] recompute advantages at repeat {step}", flush=True)
                self._compute_returns_and_advantages(
                    batch,
                    update_rms=False,
                    buffer=self._last_buffer,
                    indice=self._last_indice,
                )

            for mini_idx, minibatch in enumerate(batch.split(batch_size, merge_last=True)):
                obs = self._unwrap_obs(minibatch, "obs")
                act = jt.array(np.asarray(minibatch.act).reshape(-1)).astype(jt.int32)
                returns = jt.array(np.asarray(minibatch.returns, dtype=np.float32).reshape(-1, 1))
                adv = jt.array(np.asarray(minibatch.adv, dtype=np.float32).reshape(-1, 1))
                logp_old = jt.array(np.asarray(minibatch.logp_old, dtype=np.float32).reshape(-1))
                old_v = jt.array(np.asarray(minibatch.v_s, dtype=np.float32).reshape(-1, 1))

                if self._debug:
                    print(
                        f"[ppo-debug] repeat={step} minibatch={mini_idx} size={len(np.asarray(minibatch.act).reshape(-1))}",
                        flush=True,
                    )

                if self._norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                probs, _ = self.actor(obs)
                dist = JittorCategorical(probs=probs)
                logp = dist.log_prob(act)

                ratio = jt.exp(logp - logp_old).reshape((-1, 1))
                surr1 = ratio * adv
                surr2 = jt.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
                clip_obj = jt.minimum(surr1, surr2)
                if self.dual_clip is not None:
                    dual_obj = jt.maximum(clip_obj, self.dual_clip * adv)
                    neg_mask = (adv < 0).astype(jt.float32)
                    clip_obj = neg_mask * dual_obj + (1.0 - neg_mask) * clip_obj
                clip_loss = -clip_obj.mean()

                v = self.critic(obs).reshape((-1, 1))
                if self.value_clip:
                    v_clip = old_v + jt.clamp(v - old_v, -self.eps_clip, self.eps_clip)
                    vf_loss = jt.maximum((returns - v) ** 2, (returns - v_clip) ** 2).mean()
                else:
                    vf_loss = ((returns - v) ** 2).mean()

                ent_loss = dist.entropy().mean()
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optim.zero_grad()
                self.optim.backward(loss)
                if self.max_grad_norm:
                    self.optim.clip_grad_norm(self.max_grad_norm, 2)
                self.optim.step()
                update_happened = True

                if self._debug:
                    ratio_np = np.asarray(to_numpy(ratio)).reshape(-1)
                    adv_np = np.asarray(to_numpy(adv)).reshape(-1)
                    logp_np = np.asarray(to_numpy(logp)).reshape(-1)
                    logp_old_np = np.asarray(to_numpy(logp_old)).reshape(-1)
                    returns_np = np.asarray(to_numpy(returns)).reshape(-1)
                    old_v_np = np.asarray(to_numpy(old_v)).reshape(-1)
                    print(
                        f"[ppo-debug] repeat={step} minibatch={mini_idx} "
                        f"ratio(mean/min/max)={ratio_np.mean():.6f}/{ratio_np.min():.6f}/{ratio_np.max():.6f} "
                        f"adv(mean/std)={adv_np.mean():.6f}/{adv_np.std():.6f} "
                        f"ret(mean/std)={returns_np.mean():.6f}/{returns_np.std():.6f} "
                        f"old_v(mean/std)={old_v_np.mean():.6f}/{old_v_np.std():.6f} "
                        f"ret_rms_std={self.ret_rms.std:.6f} "
                        f"logp(mean)={logp_np.mean():.6f} logp_old(mean)={logp_old_np.mean():.6f} "
                        f"loss={float(loss.item()):.6f} clip={float(clip_loss.item()):.6f} "
                        f"vf={float(vf_loss.item()):.6f} ent={float(ent_loss.item()):.6f}",
                        flush=True,
                    )

                losses.append(float(loss.item()))
                clip_losses.append(float(clip_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ent_losses.append(float(ent_loss.item()))

        if self.lr_scheduler is not None and update_happened:
            self.lr_scheduler.step()
            if self._debug:
                print("[ppo-debug] lr_scheduler.step() once after learn()", flush=True)

        return {
            "loss": float(np.mean(losses)),
            "loss/clip": float(np.mean(clip_losses)),
            "loss/vf": float(np.mean(vf_losses)),
            "loss/ent": float(np.mean(ent_losses)),
        }


class JittorA2CPolicy(JittorPPOPolicy):
    def learn(self, batch, batch_size, repeat, **kwargs):
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        update_happened = False

        for step in range(repeat):
            if self._recompute_adv and step > 0:
                self._compute_returns_and_advantages(
                    batch,
                    update_rms=False,
                    buffer=self._last_buffer,
                    indice=self._last_indice,
                )

            for mini_idx, minibatch in enumerate(batch.split(batch_size, merge_last=True)):
                obs = self._unwrap_obs(minibatch, "obs")
                act = jt.array(np.asarray(minibatch.act).reshape(-1)).astype(jt.int32)
                returns = jt.array(np.asarray(minibatch.returns, dtype=np.float32).reshape(-1, 1))
                adv = jt.array(np.asarray(minibatch.adv, dtype=np.float32).reshape(-1, 1))

                if self._norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                probs, _ = self.actor(obs)
                dist = JittorCategorical(probs=probs)
                logp = dist.log_prob(act).reshape((-1, 1))
                actor_loss = -(logp * adv).mean()

                v = self.critic(obs).reshape((-1, 1))
                vf_loss = ((returns - v) ** 2).mean()

                ent_loss = dist.entropy().mean()
                loss = actor_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optim.zero_grad()
                self.optim.backward(loss)
                if self.max_grad_norm:
                    self.optim.clip_grad_norm(self.max_grad_norm, 2)
                self.optim.step()
                update_happened = True

                if self._debug:
                    print(
                        f"[a2c-debug] repeat={step} minibatch={mini_idx} "
                        f"loss={float(loss.item()):.6f} actor={float(actor_loss.item()):.6f} "
                        f"vf={float(vf_loss.item()):.6f} ent={float(ent_loss.item()):.6f}",
                        flush=True,
                    )

                losses.append(float(loss.item()))
                actor_losses.append(float(actor_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ent_losses.append(float(ent_loss.item()))

        if self.lr_scheduler is not None and update_happened:
            self.lr_scheduler.step()

        return {
            "loss": float(np.mean(losses)),
            "loss/actor": float(np.mean(actor_losses)),
            "loss/vf": float(np.mean(vf_losses)),
            "loss/ent": float(np.mean(ent_losses)),
        }
