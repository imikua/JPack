import jittor as jt
import jittor.nn as nn
from jittor import distributions as jt_dist
import numpy as np

"""
Modify Jittor distributions so they are compatible with this code.
(Migrated from PyTorch distributions)
"""

# Categorical
class FixedCategorical:
    """Torch-like categorical distribution with explicit gradient-safe ops.

    Jittor's built-in Categorical works for sampling, but its `log_prob` path is
    not reliable enough for this project: actor parameters can end up with zero
    gradients during policy optimization/KFAC statistics collection. This custom
    implementation mirrors the torch version using explicit `log`, `gather`, and
    `multinomial`, which keeps the policy-gradient path differentiable.
    """

    def __init__(self, probs=None, logits=None):
        if logits is not None:
            self.logits = nn.log_softmax(logits, dim=-1)
            self.probs = nn.softmax(logits, dim=-1)
        elif probs is not None:
            probs = probs / (probs.sum(dim=-1, keepdims=True) + 1e-8)
            self.probs = probs
            self.logits = jt.log(probs + 1e-8)
        else:
            raise ValueError("Either probs or logits must be provided")

    def sample(self):
        return jt.multinomial(self.probs, 1, replacement=True)

    def log_prob(self, actions):
        if actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.astype(jt.int32).reshape((-1, 1))
        return jt.gather(self.logits, 1, actions).reshape(-1)

    def log_probs(self, actions):
        return self.log_prob(actions).reshape(actions.shape[0], -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return -(self.probs * self.logits).sum(-1)

    def mode(self):
        return jt.argmax(self.probs, dim=-1, keepdims=True)

# Normal
FixedNormal = jt_dist.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdims=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mu

# Bernoulli (Jittor does not have Bernoulli, implement manually)
class FixedBernoulli:
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs
            self.logits = jt.log(probs) - jt.log(1 - probs)
        elif logits is not None:
            self.logits = logits
            self.probs = jt.sigmoid(logits)
        else:
            raise ValueError("Either probs or logits must be provided")

    def sample(self):
        with jt.no_grad():
            return jt.bernoulli(self.probs).unsqueeze(-1)

    def log_prob(self, value):
        return value * jt.log(self.probs + 1e-8) + (1 - value) * jt.log(1 - self.probs + 1e-8)

    def log_probs(self, actions):
        return self.log_prob(actions).reshape(actions.shape[0], -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        p = self.probs
        return -(p * jt.log(p + 1e-8) + (1 - p) * jt.log(1 - p + 1e-8)).sum(-1)

    def mode(self):
        return (self.probs > 0.5).float()

