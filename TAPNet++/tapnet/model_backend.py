import math

import jittor as jt
from jittor import nn


def ensure_var(x, dtype=None):
    if isinstance(x, jt.Var):
        if dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        return x
    arr = jt.array(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def to_float(x):
    return ensure_var(x).astype(jt.float32)


def arange(*args, dtype=jt.int32):
    return jt.arange(*args, dtype=dtype)


def cat(xs, dim=0):
    return jt.concat(xs, dim=dim)


def bmm(a, b):
    return nn.bmm(a, b)


def masked_fill(x, mask, value):
    return jt.masked_fill(x, mask, value)


def softmax(x, dim=-1):
    return nn.softmax(x, dim=dim)


def leaky_relu(x, scale=0.01):
    return nn.leaky_relu(x, scale=scale)


def sigmoid(x):
    return jt.sigmoid(x)


def zeros(shape, dtype=jt.float32):
    return jt.zeros(shape, dtype=dtype)


def zeros_like(x):
    return jt.zeros_like(x)


def ones(shape, dtype=jt.float32):
    return jt.ones(shape, dtype=dtype)


def ones_like(x):
    return jt.ones_like(x)


def xavier_uniform_(var):
    shape = tuple(var.shape)
    if len(shape) < 2:
        fan_in = shape[0] if len(shape) == 1 else 1
        fan_out = fan_in
    else:
        receptive = 1
        for s in shape[2:]:
            receptive *= s
        fan_in = shape[1] * receptive
        fan_out = shape[0] * receptive
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    var.assign(jt.init.uniform(shape, dtype=var.dtype, low=-bound, high=bound))
    return var


def self_attention_scores(queries, keys):
    # queries: [N, query_len, heads, head_dim]
    # keys: [N, key_len, heads, head_dim]
    q = queries.transpose((0, 2, 1, 3))
    k = keys.transpose((0, 2, 3, 1))
    return jt.matmul(q, k)


def self_attention_context(attention, values):
    # attention: [N, heads, query_len, key_len]
    # values: [N, key_len, heads, head_dim]
    v = values.transpose((0, 2, 1, 3))
    out = jt.matmul(attention, v)
    return out.transpose((0, 2, 1, 3))


def cross_attention_scores(query, keys):
    # query: [N, query_len, hidden], keys: [N, key_len, hidden]
    return jt.matmul(query, keys.transpose((0, 2, 1)))
