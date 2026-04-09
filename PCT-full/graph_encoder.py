import math
import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from jittor import nn


def draw_attention(attention_matrix, valid_mask, internal_node_holder, leaf_node_holder):
    internal_node_length = int(np.sum(valid_mask[0:internal_node_holder]))
    leaf_node_length = int(np.sum(valid_mask[internal_node_holder:internal_node_holder + leaf_node_holder]))

    selected_rows = np.concatenate((attention_matrix[0:internal_node_length, :],
                                    attention_matrix[internal_node_holder:internal_node_holder + leaf_node_length, :],
                                    attention_matrix[-1:, :]), axis=0)
    attention_matrix = np.concatenate((selected_rows[:, 0:internal_node_length],
                                       selected_rows[:, internal_node_holder:internal_node_holder + leaf_node_length],
                                       selected_rows[:, -1:]), axis=1)

    labels = ['internal node', 'leaf node', 'next item']
    coords = [internal_node_length / 2, internal_node_length + leaf_node_length / 2, internal_node_length + leaf_node_length + 0.5]
    coords = np.array(coords) - 0.5
    plt.xticks(coords, labels, rotation=0, va='center')
    plt.yticks(coords, labels, rotation=90, va='center')

    plt.imshow(attention_matrix, cmap='viridis')
    plt.title('Attention Matrix')
    plt.xlabel('Item Index')
    plt.ylabel('Query Index')
    plt.colorbar()
    plt.show()


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def execute(self, input):
        return {
            'data': input['data'] + self.module(input),
            'mask': input['mask'],
            'graph_size': input['graph_size'],
            'internal_node_holder': input['internal_node_holder'],
            'leaf_node_holder': input['leaf_node_holder'],
            'specified_mask': input['specified_mask'],
            'evaluate': input.get('evaluate', False)
        }

    forward = execute


class SkipConnection_Linear(nn.Module):
    def __init__(self, module):
        super(SkipConnection_Linear, self).__init__()
        self.module = module

    def execute(self, input):
        return {
            'data': input['data'] + self.module(input['data']),
            'mask': input['mask'],
            'graph_size': input['graph_size'],
            'internal_node_holder': input['internal_node_holder'],
            'leaf_node_holder': input['leaf_node_holder'],
            'specified_mask': input['specified_mask'],
            'evaluate': input.get('evaluate', False)
        }

    forward = execute


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None,
                 draw_attention=False, inner_leaf_attention_eliminate=False, cate_attn=False):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, 'Provide either embed_dim or val_dim'
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.draw_attention = draw_attention
        self.inner_leaf_attention_eliminate = inner_leaf_attention_eliminate
        self.cate_attn = cate_attn
        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Linear(input_dim, key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, val_dim, bias=False)
        if embed_dim is not None:
            self.W_out = nn.Linear(key_dim, embed_dim)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.shape[-1])
            param.assign(jt.init.uniform(param.shape, dtype=param.dtype, low=-stdv, high=stdv))

    def execute(self, data, h=None):
        q = data['data']
        mask = data['mask']
        graph_size = data['graph_size']
        internal_node_holder = data['internal_node_holder']
        leaf_node_holder = data['leaf_node_holder']
        specified_mask = data['specified_mask']
        eliminate = None

        if h is None:
            h = q
        assert len(q.shape) == 2
        batch_size = int(q.shape[0] / graph_size)

        input_dim = h.shape[-1]
        n_query = graph_size
        assert input_dim == self.input_dim, 'Wrong embedding dimension of input'

        hflat = h.reshape((-1, input_dim))
        qflat = q.reshape((-1, input_dim))

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = self.W_query(qflat).reshape(shp_q)
        K = self.W_key(hflat).reshape(shp)
        V = self.W_val(hflat).reshape(shp)

        compatibility = self.norm_factor * jt.matmul(Q, K.transpose((0, 1, 3, 2)))

        if mask is not None:
            eliminate = mask.unsqueeze(1).repeat((1, graph_size, 1))
            if self.inner_leaf_attention_eliminate:
                eye = jt.eye(leaf_node_holder, dtype=eliminate.dtype)
                block = eliminate[:, internal_node_holder:internal_node_holder + leaf_node_holder,
                                  internal_node_holder:internal_node_holder + leaf_node_holder]
                eliminate[:, internal_node_holder:internal_node_holder + leaf_node_holder,
                          internal_node_holder:internal_node_holder + leaf_node_holder] = block * eye
            eliminate = eliminate.bool()
            if specified_mask is not None:
                eliminate = jt.logical_or(eliminate, specified_mask.unsqueeze(0).repeat((batch_size, 1, 1)).bool())
            eliminate = eliminate.reshape((1, batch_size, n_query, graph_size)).broadcast(compatibility.shape)
            fill_value = -math.inf if data['evaluate'] else -30.0
            compatibility = jt.where(eliminate, jt.full(compatibility.shape, fill_value, compatibility.dtype), compatibility)

        if not self.cate_attn:
            attn = nn.softmax(compatibility, dim=-1)
        else:
            attn_internal = nn.softmax(compatibility[:, :, :, 0:internal_node_holder], dim=-1)
            attn_leaf = nn.softmax(compatibility[:, :, :, internal_node_holder:internal_node_holder + leaf_node_holder], dim=-1)
            attn_next = nn.softmax(compatibility[:, :, :, -1:], dim=-1)
            attn = jt.concat((attn_internal, attn_leaf, attn_next), dim=-1) / 3

        if self.draw_attention:
            draw_attention(attn[0][0].numpy(), 1 - mask[0].numpy(), internal_node_holder, leaf_node_holder)

        if mask is not None:
            attn = jt.where(eliminate, jt.zeros(attn.shape, dtype=attn.dtype), attn)

        heads = jt.matmul(attn, V)
        out = self.W_out(heads.permute((1, 2, 0, 3)).reshape((-1, self.n_heads * self.val_dim))).reshape((batch_size * n_query, self.embed_dim))
        return out

    forward = execute


class MultiHeadAttention_new(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None, dropout_rate=0.0, draw_attention=False, **kwargs):
        super(MultiHeadAttention_new, self).__init__()
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = embed_dim
        self.key_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.draw_attention = draw_attention

        self.W_query = nn.Linear(input_dim, self.key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, self.key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, self.val_dim, bias=False)
        if embed_dim is not None:
            self.W_out = nn.Linear(input_dim, embed_dim)

        self.init_parameters()
        self.dropout = nn.Dropout(dropout_rate)

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.shape[-1])
            param.assign(jt.init.uniform(param.shape, dtype=param.dtype, low=-stdv, high=stdv))

    def execute(self, data, h=None):
        q = data['data']
        mask = data['mask']
        graph_size = data['graph_size']
        if h is None:
            h = q
        mask_expanded = None

        batch_size = int(q.shape[0] / graph_size)
        input_dim = h.shape[-1]
        n_query = graph_size
        assert input_dim == self.input_dim, 'Wrong embedding dimension of input'

        hflat = h.reshape((-1, input_dim))
        qflat = q.reshape((-1, input_dim))

        shp = (batch_size, -1, self.n_heads, self.head_dim)
        shp_q = (batch_size, -1, self.n_heads, self.head_dim)
        Q = self.W_query(qflat).reshape(shp_q).transpose((0, 2, 1, 3))
        K = self.W_key(hflat).reshape(shp).transpose((0, 2, 1, 3))
        V = self.W_val(hflat).reshape(shp).transpose((0, 2, 1, 3))

        compatibility = jt.matmul(Q, K.transpose((0, 1, 3, 2))) / math.sqrt(float(self.head_dim))

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).repeat((1, graph_size, 1)).bool()
            mask_expanded = mask_expanded.reshape((batch_size, 1, n_query, graph_size)).broadcast(compatibility.shape)
            fill_value = -math.inf if data['evaluate'] else -30.0
            compatibility = jt.where(mask_expanded, jt.full(compatibility.shape, fill_value, compatibility.dtype), compatibility)

        attn = nn.softmax(compatibility, dim=-1)
        if mask is not None:
            attn = jt.where(mask_expanded, jt.zeros(attn.shape, dtype=attn.dtype), attn)

        weighted_sum = jt.matmul(attn, V)
        output = weighted_sum.transpose((0, 2, 1, 3)).reshape((-1, input_dim))
        output = self.W_out(output).reshape((batch_size * n_query, self.embed_dim))
        return output

    forward = execute


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=128, new_attention=False,
                 draw_attention=False, inner_leaf_attention_eliminate=False,
                 linear_net=False, cate_attn=False):
        super(MultiHeadAttentionLayer, self).__init__()
        attention_func = MultiHeadAttention_new if new_attention else MultiHeadAttention
        self.skip_attention = SkipConnection(attention_func(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim,
            draw_attention=draw_attention,
            inner_leaf_attention_eliminate=inner_leaf_attention_eliminate,
            cate_attn=cate_attn
        ))
        self.skip_linear = SkipConnection_Linear(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim)
            ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        ) if not linear_net else SkipConnection_Linear(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden, bias=False),
                nn.Linear(feed_forward_hidden, embed_dim, bias=False)
            ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        )

    def execute(self, data):
        data = self.skip_attention(data)
        data = self.skip_linear(data)
        return data

    forward = execute


class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, node_dim=None, feed_forward_hidden=128,
                 graph_size=None, new_attention=False, draw_attention=False,
                 inner_leaf_attention_eliminate=False, linear_net=False, cate_attn=False):
        super(GraphAttentionEncoder, self).__init__()
        n_layers = 1
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, new_attention, draw_attention,
                                    inner_leaf_attention_eliminate, linear_net, cate_attn)
            for _ in range(n_layers)
        ))

    def execute(self, x, mask=None, limited=False, given_graph_size=None, evaluate=False,
                internal_node_holder=None, leaf_node_holder=None, specified_mask=None):
        graph_size = given_graph_size if given_graph_size is not None else self.graph_size
        h = self.init_embed(x.reshape((-1, x.shape[-1]))).reshape((*x.shape[:2], -1)) if self.init_embed is not None else x
        data = {
            'data': h,
            'mask': mask,
            'graph_size': graph_size,
            'evaluate': evaluate,
            'internal_node_holder': internal_node_holder,
            'leaf_node_holder': leaf_node_holder,
            'specified_mask': specified_mask
        }
        h = self.layers(data)['data']
        return h, h.reshape((int(h.shape[0] / graph_size), graph_size, -1)).mean(dim=1)

    forward = execute
