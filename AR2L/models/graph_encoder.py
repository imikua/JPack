import sys

import jittor as jt
import jittor.nn as nn
import math
import numpy as _np

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
    def execute(self, input):
        return {'data': input['data'] + self.module(input),
                'mask': input['mask'],
                'graph_size': input['graph_size'],
                'evaluate': input['evaluate']}


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_size, feed_forward_size):
        super(FeedForwardLayer, self).__init__()
        if feed_forward_size > 0:
            self.layer = nn.Sequential(nn.Linear(embedding_size, feed_forward_size),
                                       nn.ReLU(),
                                       nn.Linear(feed_forward_size, embedding_size))
        else:
            self.layer = nn.Linear(embedding_size, embedding_size)


    def execute(self, input):
        module_input = input['data']
        module_input_size = module_input.shape
        if len(module_input_size) != 2:
            module_input = module_input.view(-1, module_input.shape[-1])
        return self.layer(module_input).view(-1, *module_input_size[1:])



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_heads,
                 input_size,
                 embedding_size,
                 value_size=None,
                 key_size=None,):
        super(MultiHeadAttention, self).__init__()

        if value_size is None:
            value_size = embedding_size // n_heads
        if key_size is None:
            key_size = embedding_size // n_heads

        self.n_heads = n_heads
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.value_size = value_size
        self.key_size = key_size

        self.norm_factor = 1 / math.sqrt(key_size)

        # Project to all heads at once (n_heads * head_dim)
        self.W_query = nn.Linear(input_size, n_heads * key_size, bias=False)
        self.W_key = nn.Linear(input_size, n_heads * key_size, bias=False)
        self.W_value = nn.Linear(input_size, n_heads * value_size, bias=False)

        if embedding_size is not None:
            # Output projection from concatenated heads
            self.W_out = nn.Linear(n_heads * value_size, embedding_size)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.shape[-1])
            param.assign(jt.array(_np.random.uniform(-stdv, stdv, param.shape).astype(_np.float32)))

    def execute(self, input, h=None):
        """
        q: query [batch_size, n_query, input_dim]
        h: key and value [batch_size, graph_size, input_dim]
        mask: [batch_size, graph_size] (node mask) OR [batch_size, n_query, graph_size]
        """
        q = input['data']
        mask = input['mask']
        graph_size = input['graph_size']
        evaluate = input['evaluate']

        if h == None:
            h = q  # compute self attention

        batch_size = int(q.shape[0] / graph_size)
        input_size = h.shape[-1]
        n_query = graph_size

        assert input_size == self.input_size

        hflat = h.contiguous().view(-1, input_size)
        qflat = q.contiguous().view(-1, input_size)

        v_shape = (self.n_heads, batch_size, graph_size, self.value_size)
        k_shape = (self.n_heads, batch_size, graph_size, self.key_size)
        q_shape = (self.n_heads, batch_size, n_query, self.key_size)

        # jt.view/reshape needs explicit sizes; use fixed last dim
        Q = self.W_query(qflat).view(self.n_heads, batch_size, n_query, self.key_size)
        K = self.W_key(hflat).view(self.n_heads, batch_size, graph_size, self.key_size)
        V = self.W_value(hflat).view(self.n_heads, batch_size, graph_size, self.value_size)

        compatibility = self.norm_factor * jt.matmul(Q, K.transpose(2, 3))  #[n_heads, batch_size, n_query, graph_size]

        if mask is not None:
            # Torch baseline receives a node mask [B, graph_size] and expands it to
            # [1, B, n_query, graph_size]. Our previous implementation used repeat with
            # wrong dims, which can corrupt shapes and even segfault.
            if len(mask.shape) == 2:
                # [B, graph_size] -> [1, B, 1, graph_size] -> broadcast to [n_heads, B, n_query, graph_size]
                mask_exp = mask.bool().view(1, batch_size, 1, graph_size).expand(self.n_heads, batch_size, n_query, graph_size)
            elif len(mask.shape) == 3:
                # [B, n_query, graph_size] -> [1, B, n_query, graph_size] -> broadcast heads
                mask_exp = mask.bool().view(1, batch_size, n_query, graph_size).expand(self.n_heads, batch_size, n_query, graph_size)
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Use large negative value instead of -inf to avoid NaN in softmax/gradients
            neg_val = -math.inf if evaluate else -1e9
            compatibility = jt.where(mask_exp, jt.full_like(compatibility, neg_val), compatibility)

        attention = nn.softmax(compatibility, dim=-1)  #[n_heads, batch_size, n_query, graph_size]

        if mask is not None:
            attention = jt.where(mask_exp, jt.zeros_like(attention), attention)


        heads = jt.matmul(attention, V)  # [n_heads, batch_size, n_query, value_size]

        out = self.W_out(
            heads.permute(1, 2, 0, 3).contiguous().view(batch_size * n_query, self.n_heads * self.value_size)
        ).view(batch_size * n_query, self.embedding_size)

        return out



class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self,
                 n_heads,
                 embedding_size,
                 feed_forward_hidden=128,
                 ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_size=embedding_size,
                    embedding_size=embedding_size,
                )
            ),
            SkipConnection(
                FeedForwardLayer(embedding_size, feed_forward_hidden)
            )
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(self,
                 n_heads,
                 embedding_size,
                 n_layers,
                 graph_size,
                 node_dim=None,
                 feed_forward_hidden=128,
                 ):
        
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Linear(node_dim, embedding_size) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(
            *(MultiHeadAttentionLayer(n_heads, embedding_size, feed_forward_hidden)
              for _ in range(n_layers))
        )


    def execute(self, x, mask=None, evaluate=None):
        '''
        x can be:
          - [batch, graph_size, feature_size]
          - [batch*graph_size, feature_size] (flattened)
        If x is flattened, we try to infer graph_size from mask.shape when possible.
        '''

        if len(x.shape) == 3:
            batch = int(x.shape[0])
            graph_size = int(x.shape[1])
            x_flat = x.view(batch * graph_size, x.shape[-1])
            mask_flat = mask
        else:
            # Flattened input
            if mask is not None and len(mask.shape) >= 2:
                # In our codebase mask is typically [batch, graph_size] (node mask)
                graph_size = int(mask.shape[1])
                batch = int(mask.shape[0])
                expected0 = batch * graph_size
                assert int(x.shape[0]) == expected0, (
                    f"Flattened input has x.shape[0]={x.shape[0]} but mask implies {batch}*{graph_size}={expected0}"
                )
            else:
                graph_size = int(self.graph_size)
                assert x.shape[0] % graph_size == 0, (
                    f"Input first dim {x.shape[0]} not divisible by graph_size {graph_size}"
                )
                batch = int(x.shape[0] // graph_size)
            x_flat = x
            mask_flat = mask

        if self.init_embed is not None:
            node_embedding = self.init_embed(x_flat)
        else:
            node_embedding = x_flat

        data = {'data': node_embedding, 'mask': mask_flat, 'graph_size': graph_size, 'evaluate': evaluate}
        node_embedding = self.layers(data)['data']

        graph_embedding = node_embedding.view(batch, graph_size, -1).mean(dim=1)
        return node_embedding, graph_embedding
