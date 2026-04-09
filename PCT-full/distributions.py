import jittor as jt
from jittor import distributions as jt_dist
import numpy as np

"""
Modify Jittor distributions so they are compatible with this code.
Direct port from working PCT version.
"""

# Categorical
FixedCategorical = jt_dist.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
def _categorical_log_probs(self, actions):
    if len(actions.shape) > 1 and actions.shape[-1] == 1:
        actions = actions.squeeze(-1)
    return log_prob_cat(self, actions).reshape(actions.shape[0], -1).sum(-1).unsqueeze(-1)

FixedCategorical.log_probs = _categorical_log_probs

FixedCategorical.mode = lambda self: jt.argmax(self.probs, dim=-1, keepdims=True)

# Normal
FixedNormal = jt_dist.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(-1, keepdims=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean
