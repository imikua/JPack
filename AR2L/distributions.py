import jittor as jt
import jittor.nn as nn
import numpy as _np

"""
Modify standard Jittor distributions so they are compatible with this code.
"""

# ========== Categorical ==========
class FixedCategorical:
    """Wrapper around jittor Categorical with AR2L-compatible API."""
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs
            self._dist = jt.distributions.Categorical(probs=probs)
        else:
            self._dist = jt.distributions.Categorical(logits=logits)
            self.probs = self._dist.probs

    def sample(self):
        return self._dist.sample().unsqueeze(-1)

    def log_prob(self, value):
        return self._dist.log_prob(value)

    def log_probs(self, actions):
        return self._dist.log_prob(actions.squeeze(-1)).reshape(actions.shape[0], -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return jt.argmax(self.probs, dim=-1)[0].unsqueeze(-1)


# ========== Normal ==========
class FixedNormal:
    """Wrapper around jittor Normal with AR2L-compatible API."""
    def __init__(self, loc, scale):
        self._dist = jt.distributions.Normal(loc, scale)

    @property
    def mean(self):
        return self._dist.mu

    def sample(self):
        return self._dist.sample()

    def log_prob(self, value):
        return self._dist.log_prob(value)

    def log_probs(self, actions):
        return self._dist.log_prob(actions).sum(-1, keepdims=True)

    def entropy(self):
        return self._dist.entropy().sum(-1)

    def mode(self):
        return self._dist.mu


# ========== Bernoulli (hand-implemented, Jittor has no Bernoulli) ==========
class FixedBernoulli:
    """Hand-implemented Bernoulli distribution for Jittor."""
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs
        else:
            self.probs = jt.sigmoid(logits)

    def sample(self):
        return (jt.rand_like(self.probs) < self.probs).float()

    def log_prob(self, value):
        eps = 1e-8
        return value * jt.log(self.probs + eps) + (1 - value) * jt.log(1 - self.probs + eps)

    def log_probs(self, actions):
        return self.log_prob(actions).reshape(actions.shape[0], -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        eps = 1e-8
        p = self.probs
        return -(p * jt.log(p + eps) + (1 - p) * jt.log(1 - p + eps)).sum(-1)

    def mode(self):
        return (self.probs > 0.5).float()


# ========== SquashedNormal (hand-implemented, Jittor has no TransformedDistribution) ==========
class SquashedNormal:
    """Normal distribution followed by Sigmoid + Affine transform."""
    def __init__(self, loc, scale, high_value):
        self.loc = loc
        self.scale = scale
        self.high_value = high_value
        self._dist = jt.distributions.Normal(loc, scale)

    @property
    def mean(self):
        return jt.sigmoid(self.loc) * self.high_value

    @property
    def mode(self):
        return self.mean

    def sample(self):
        x = self._dist.sample()
        return jt.sigmoid(x) * self.high_value

    def log_prob(self, value):
        # Invert the transform: value = sigmoid(x) * high_value => x = logit(value / high_value)
        eps = 1e-6
        y = jt.clamp(value / self.high_value, eps, 1.0 - eps)
        x = jt.log(y / (1.0 - y))  # logit
        # log_prob = base_log_prob - log|det(J)|
        # J = sigmoid'(x) * high_value = sigmoid(x)*(1-sigmoid(x)) * high_value
        log_det = jt.log(y * (1.0 - y) * self.high_value + eps)
        return self._dist.log_prob(x) - log_det

    def log_probs(self, actions):
        return self.log_prob(actions).sum(-1, keepdims=True)
