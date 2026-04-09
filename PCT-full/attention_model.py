import math
from dataclasses import dataclass

import numpy as np
import jittor as jt
from jittor import nn

from graph_encoder import GraphAttentionEncoder
from distributions import FixedCategorical
from tools import observation_decode_leaf_node, init, orthogonal_init, constant_init


@dataclass
class AttentionModelFixed():
    node_embeddings: jt.Var
    context_node_projected: jt.Var
    glimpse_key: jt.Var
    glimpse_val: jt.Var
    logit_key: jt.Var

    def __getitem__(self, key):
        if isinstance(key, (slice, int)) or hasattr(key, 'shape'):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],
                glimpse_val=self.glimpse_val[:, key],
                logit_key=self.logit_key[key]
            )
        return self[key]


class PCTModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=False,
                 mask_logits=False,
                 n_heads=1,
                 internal_node_holder=None,
                 internal_node_length=None,
                 leaf_node_holder=None,
                 leaf_node_length=8,
                 next_node_length=6,
                 feature_aggregation='mean',
                 step_counter=0,
                 args=None):
        super(PCTModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.step_counter = step_counter
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = 1
        self.internal_node_holder = internal_node_holder
        self.internal_node_length = internal_node_length
        self.next_holder = 1

        self.leaf_node_holder = leaf_node_holder
        self.leaf_node_length = leaf_node_length
        self.next_node_length = next_node_length

        self.repeat_embed = args.repeat_embed
        self.gat_layer_num = args.gat_layer_num
        self.new_attention = args.new_attention
        self.draw_attention = args.draw_attention
        self.inner_leaf_attention_eliminate = args.inner_leaf_attention_eliminate
        self.policy_positional_encoding = args.policy_positional_encoding
        self.similarity = None
        self.no_next_item_input = args.no_next_item_input

        graph_size = internal_node_holder + leaf_node_holder + self.next_holder

        activate = nn.LeakyReLU
        init_ = lambda m: init(m, orthogonal_init, lambda x, **kwargs: constant_init(x, 0), 1.0)

        self.init_internal_node_embed = nn.Sequential(
            init_(nn.Linear(self.internal_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        self.init_leaf_node_embed = nn.Sequential(
            init_(nn.Linear(self.leaf_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        self.init_next_embed = nn.Sequential(
            init_(nn.Linear(self.next_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        d_model = embedding_dim
        max_len = 2000
        if self.policy_positional_encoding:
            position = jt.arange(max_len).unsqueeze(1)
            div_term = jt.exp(jt.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = jt.zeros((max_len, 1, d_model))
            pe[:, 0, 0::2] = jt.sin(position * div_term)
            pe[:, 0, 1::2] = jt.cos(position * div_term)
            self._pe = pe.transpose((1, 0, 2))
        else:
            self._pe = jt.zeros((1, max_len, d_model))

        self.embedder = nn.ModuleList([GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            graph_size=graph_size,
            new_attention=self.new_attention,
            draw_attention=self.draw_attention,
            inner_leaf_attention_eliminate=self.inner_leaf_attention_eliminate,
            linear_net=args.linear_net,
            cate_attn=args.cate_attn
        ) for _ in range(self.gat_layer_num)])

        self.feature_aggregation = feature_aggregation
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False) if feature_aggregation != 'rehearsal' else None
        if feature_aggregation == 'no_leaf' or feature_aggregation == 'item_attention':
            self.project_dim = embedding_dim * 2
        elif feature_aggregation in ('mean_norm', 'mean_sep', 'mean_running_norm', 'rehearsal'):
            self.project_dim = embedding_dim * 3
        else:
            self.project_dim = embedding_dim
        self.project_fixed_context = nn.Linear(self.project_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

        self.W_final_query = nn.Linear(embedding_dim, embedding_dim, bias=False) if feature_aggregation == 'item_attention' else None
        self.W_final_key = nn.Linear(embedding_dim, embedding_dim, bias=False) if feature_aggregation == 'item_attention' else None
        self.W_final_val = nn.Linear(embedding_dim, embedding_dim, bias=False) if feature_aggregation == 'item_attention' else None

        self.specified_mask_id = args.specified_mask_id
        self._specified_mask = jt.array(self.generate_eliminate_matrix(args.specified_mask_id)).bool() if self.specified_mask_id != 0 else None

    def PositionalEncoding(self, x):
        return x + self._pe[:, :x.shape[1]]

    def set_step_counter(self, step_counter):
        self.step_counter = step_counter

    def linear_decay(self, time_step, initial_value=1, final_value=1e-3, final_step=100000):
        return final_value + (initial_value - final_value) * max(0, (1 - time_step / final_step))

    def batch_norm(self, x, epsilon=1e-5):
        mean = jt.mean(x, dim=0)
        variance = jt.var(x, dim=0)
        return (x - mean) / jt.sqrt(variance + epsilon)

    def Min_Max_Scaling(self, x):
        min_val = jt.min(x, dim=0)
        max_val = jt.max(x, dim=0)
        if isinstance(min_val, tuple):
            min_val = min_val[0]
        if isinstance(max_val, tuple):
            max_val = max_val[0]
        return (x - min_val) / (max_val - min_val + 1e-8)

    def Robust_Scaling(self, x):
        # Compute statistics from detached values (no grad needed for stats)
        x_np = x.stop_grad().numpy()
        median = np.median(x_np, axis=0)
        q1 = np.quantile(x_np, 0.25, axis=0)
        q3 = np.quantile(x_np, 0.75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0.0] = 1.0
        # Apply transformation using Jittor ops to maintain gradient
        return (x - jt.array(median.astype(np.float32))) / jt.array(iqr.astype(np.float32))

    def layer_norm(self, x, epsilon=1e-5):
        mean = jt.mean(x, dim=-1).unsqueeze(-1)
        variance = jt.var(x, dim=-1).unsqueeze(-1)
        return (x - mean) / jt.sqrt(variance + epsilon)

    def generate_eliminate_matrix(self, specified_mask_id):
        self.interval = [0, self.internal_node_holder, self.internal_node_holder + self.leaf_node_holder,
                         self.internal_node_holder + self.leaf_node_holder + self.next_holder]
        mask = np.zeros((self.internal_node_holder + self.leaf_node_holder + self.next_holder,
                         self.internal_node_holder + self.leaf_node_holder + self.next_holder))
        if specified_mask_id == 0:
            return None
        all_masks = []
        for i in range(3):
            for j in range(3):
                x_left = self.interval[i]
                x_right = self.interval[i + 1]
                y_left = self.interval[j]
                y_right = self.interval[j + 1]
                target_mask = mask.copy()
                target_mask[x_left:x_right, y_left:y_right] = 1
                all_masks.append(target_mask)
        all_masks = np.array(all_masks)
        binary_string = bin(specified_mask_id)[2:]
        binary_list = np.array([int(bit) for bit in binary_string])
        real_index = np.where(binary_list == 1)[0]
        matrices_list = all_masks[real_index]
        return np.logical_or.reduce(matrices_list)

    def get_node_inputs(self, input, internal_node_holder, leaf_node_holder, normFactor=1):
        internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = observation_decode_leaf_node(
            input, internal_node_holder, self.internal_node_length, leaf_node_holder)
        batch_size = input.shape[0]
        internal_nodes_size = internal_nodes.shape[1]
        leaf_node_size = leaf_nodes.shape[1]
        next_size = next_item.shape[1]
        internal_inputs = internal_nodes.reshape((batch_size * internal_nodes_size, self.internal_node_length)) * normFactor
        leaf_inputs = leaf_nodes.reshape((batch_size * leaf_node_size, self.leaf_node_length)) * normFactor
        current_inputs = next_item.reshape((batch_size * next_size, 6)) * normFactor
        return internal_inputs, leaf_inputs, current_inputs

    def get_embedding(self, input, internal_node_holder, leaf_node_holder, normFactor=1, evaluate=False, decode_func=None):
        decode_func = observation_decode_leaf_node if decode_func is None else decode_func
        internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = decode_func(
            input, internal_node_holder, self.internal_node_length, leaf_node_holder)
        leaf_node_mask = 1 - invalid_leaf_nodes
        valid_length = full_mask.sum(1)
        valid_length = jt.maximum(valid_length, jt.ones_like(valid_length))
        eliminate_mask = 1 - full_mask

        batch_size = input.shape[0]
        graph_size = input.shape[1]
        internal_nodes_size = internal_nodes.shape[1]
        leaf_node_size = leaf_nodes.shape[1]
        next_size = next_item.shape[1]

        internal_inputs = internal_nodes.reshape((batch_size * internal_nodes_size, self.internal_node_length)) * normFactor
        leaf_inputs = leaf_nodes.reshape((batch_size * leaf_node_size, self.leaf_node_length)) * normFactor
        current_inputs = next_item.reshape((batch_size * next_size, 6)) * normFactor
        if self.no_next_item_input:
            current_inputs = jt.zeros(current_inputs.shape, dtype=current_inputs.dtype)

        internal_embedded_inputs = self.init_internal_node_embed(internal_inputs).reshape((batch_size, -1, self.embedding_dim))
        if self.policy_positional_encoding:
            internal_embedded_inputs = self.PositionalEncoding(internal_embedded_inputs)
        leaf_embedded_inputs = self.init_leaf_node_embed(leaf_inputs).reshape((batch_size, -1, self.embedding_dim))
        next_embedded_inputs = self.init_next_embed(current_inputs).reshape((batch_size, -1, self.embedding_dim))

        embeddings = jt.concat((internal_embedded_inputs, leaf_embedded_inputs, next_embedded_inputs), dim=1).reshape((batch_size * graph_size, self.embedding_dim))
        for layer in self.embedder:
            for _ in range(self.repeat_embed):
                embeddings, _ = layer(
                    embeddings,
                    given_graph_size=internal_node_holder + leaf_node_holder + 1,
                    mask=eliminate_mask,
                    evaluate=evaluate,
                    internal_node_holder=internal_node_holder,
                    leaf_node_holder=leaf_node_holder,
                    specified_mask=self._specified_mask
                )
        embedding_shape = (batch_size, graph_size, embeddings.shape[-1])
        return embeddings, embedding_shape, leaf_node_mask, eliminate_mask, valid_length


class AttentionModel(PCTModel):
    def __init__(self, embedding_dim, hidden_dim, n_encode_layers=2, tanh_clipping=10., mask_inner=False,
                 mask_logits=False, n_heads=1, internal_node_holder=None, internal_node_length=None,
                 leaf_node_holder=None, leaf_node_length=8, next_node_length=6, feature_aggregation='mean', args=None):
        super(AttentionModel, self).__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            n_heads=n_heads,
            internal_node_holder=internal_node_holder,
            internal_node_length=internal_node_length,
            leaf_node_holder=leaf_node_holder,
            leaf_node_length=leaf_node_length,
            feature_aggregation=feature_aggregation,
            next_node_length=next_node_length,
            args=args)
        self.pred_value_with_heightmap = args.pred_value_with_heightmap

    def execute(self, input, deterministic=False, evaluate_action=False, normFactor=1, evaluate=False,
                internal_node_holder=None, leaf_node_holder=None):
        internal_node_holder = internal_node_holder if internal_node_holder is not None else self.internal_node_holder
        leaf_node_holder = leaf_node_holder if leaf_node_holder is not None else self.leaf_node_holder
        embeddings, embedding_shape, leaf_node_mask, eliminate_mask, valid_length = self.get_embedding(
            input, internal_node_holder, leaf_node_holder, normFactor=normFactor, evaluate=evaluate)
        log_p, action_log_prob, pointers, dist_entropy, dist, hidden = self._inner(
            embeddings,
            deterministic=deterministic,
            evaluate_action=evaluate_action,
            shape=embedding_shape,
            mask=leaf_node_mask,
            eliminate_mask=eliminate_mask,
            valid_length=valid_length,
            internal_node_holder=internal_node_holder,
            leaf_node_holder=leaf_node_holder)
        return action_log_prob, pointers, dist_entropy, hidden, dist, embeddings, log_p

    forward = execute

    def _inner(self, embeddings, mask=None, deterministic=False, evaluate_action=False,
               shape=None, eliminate_mask=None, valid_length=None, internal_node_holder=None, leaf_node_holder=None):
        fixed, internal_graph_embed = self._precompute(
            embeddings,
            shape=shape,
            eliminate_mask=eliminate_mask,
            valid_length=valid_length,
            internal_node_holder=internal_node_holder,
            leaf_node_holder=leaf_node_holder,
            evaluate_action=evaluate_action)
        log_p, mask = self._get_log_p(fixed, mask, internal_node_holder=internal_node_holder, leaf_node_holder=leaf_node_holder)

        masked_outs = log_p * (1 - mask) + 1e-20
        if deterministic:
            masked_outs = log_p * (1 - mask) + 1e-20
        log_p = masked_outs / jt.sum(masked_outs, dim=1).unsqueeze(1)
        dist = FixedCategorical(probs=log_p)
        dist_entropy = dist.entropy()
        selected = dist.mode() if deterministic else dist.sample()
        action_log_probs = None if evaluate_action else dist.log_probs(selected)

        if self.pred_value_with_heightmap:
            return log_p, action_log_probs, selected, dist_entropy, dist, (fixed.context_node_projected, internal_graph_embed)
        return log_p, action_log_probs, selected, dist_entropy, dist, fixed.context_node_projected

    def _precompute(self, embeddings, num_steps=1, shape=None, eliminate_mask=None, valid_length=None,
                    internal_node_holder=None, leaf_node_holder=None, evaluate_action=False):
        transEmbedding = embeddings.reshape(shape)
        valid_mask = 1 - eliminate_mask.int32()
        origin_eliminate_mask = eliminate_mask.copy()
        eliminate_mask_expand = eliminate_mask.reshape((shape[0], shape[1], 1)).broadcast(shape).bool()
        transEmbedding = jt.where(eliminate_mask_expand, jt.zeros(transEmbedding.shape, dtype=transEmbedding.dtype), transEmbedding)
        transEmbedding_flat = transEmbedding.reshape(embeddings.shape)
        internal_graph_embed = None

        if self.feature_aggregation == 'no_leaf':
            internal_graph_embed = transEmbedding[:, 0:internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0:internal_node_holder].sum(1)
            next_graph_embed = transEmbedding[:, -1]
            safe_internal_length = jt.maximum(internal_graph_length.reshape((-1, 1)), jt.ones_like(internal_graph_length.reshape((-1, 1))))
            graph_embed = internal_graph_embed / safe_internal_length
            fixed_context = self.project_fixed_context(jt.concat((graph_embed, next_graph_embed), dim=1))
        elif self.feature_aggregation == 'mean':
            safe_valid_length = jt.maximum(valid_length.reshape((-1, 1)), jt.ones_like(valid_length.reshape((-1, 1))))
            graph_embed = transEmbedding.sum(1) / safe_valid_length
            internal_graph_embed = transEmbedding[:, 0:internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0:internal_node_holder].sum(1)
            safe_internal_length = jt.maximum(internal_graph_length.reshape((-1, 1)), jt.ones_like(internal_graph_length.reshape((-1, 1))))
            internal_graph_embed = internal_graph_embed / safe_internal_length
            fixed_context = self.project_fixed_context(graph_embed)
        elif self.feature_aggregation == 'item_attention':
            norm_factor = 1 / math.sqrt(self.embedding_dim)
            next_graph_embed = transEmbedding[:, -1].reshape((-1, shape[-1]))
            internal_graph_embed = transEmbedding[:, 0:internal_node_holder].reshape((-1, shape[-1]))
            next_graph_embed_query = self.W_final_query(next_graph_embed).reshape((shape[0], -1, shape[2]))
            internal_graph_embed_key = self.W_final_key(internal_graph_embed).reshape((shape[0], -1, shape[2]))
            internal_graph_embed_val = self.W_final_val(internal_graph_embed).reshape((shape[0], -1, shape[2]))
            internal_graph_mask = origin_eliminate_mask[:, 0:internal_node_holder].unsqueeze(1).bool()
            compatibility = norm_factor * jt.matmul(next_graph_embed_query, internal_graph_embed_key.transpose((0, 2, 1)))
            fill_value = -1e9
            compatibility = jt.where(internal_graph_mask, jt.full(compatibility.shape, fill_value, compatibility.dtype), compatibility)
            attn = nn.softmax(compatibility, dim=-1)
            internal_graph_embed = jt.matmul(attn, internal_graph_embed_val).squeeze(1)
            fixed_context = self.project_fixed_context(jt.concat((internal_graph_embed, next_graph_embed_query.squeeze(1)), dim=1))
        elif self.feature_aggregation == 'mean_norm':
            internal_graph_embed = transEmbedding[:, 0:internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0:internal_node_holder].sum(1)
            internal_graph_embed = internal_graph_embed / jt.maximum(internal_graph_length.reshape((-1, 1)), jt.ones_like(internal_graph_length.reshape((-1, 1))))
            internal_graph_embed = self.Robust_Scaling(internal_graph_embed)
            leaf_graph_embed = transEmbedding[:, internal_node_holder: internal_node_holder + leaf_node_holder].sum(1)
            leaf_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            leaf_graph_embed = leaf_graph_embed / jt.maximum(leaf_graph_length.reshape((-1, 1)), jt.ones_like(leaf_graph_length.reshape((-1, 1))))
            leaf_graph_embed = self.Robust_Scaling(leaf_graph_embed)
            next_graph_embed = self.Robust_Scaling(transEmbedding[:, -1])
            graph_embed = jt.concat((internal_graph_embed, leaf_graph_embed, next_graph_embed), dim=1)
            fixed_context = self.project_fixed_context(graph_embed)
        elif self.feature_aggregation == 'mean_sep':
            internal_graph_embed = transEmbedding[:, 0:internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0:internal_node_holder].sum(1)
            internal_graph_embed = internal_graph_embed / jt.maximum(internal_graph_length.reshape((-1, 1)), jt.ones_like(internal_graph_length.reshape((-1, 1))))
            leaf_graph_embed = transEmbedding[:, internal_node_holder: internal_node_holder + leaf_node_holder].sum(1)
            leaf_graph_length = valid_mask[:, 0:internal_node_holder].sum(1)
            leaf_graph_embed = leaf_graph_embed / jt.maximum(leaf_graph_length.reshape((-1, 1)), jt.ones_like(leaf_graph_length.reshape((-1, 1))))
            next_graph_embed = transEmbedding[:, -1]
            graph_embed = jt.concat((internal_graph_embed, leaf_graph_embed, next_graph_embed), dim=1)
            fixed_context = self.project_fixed_context(graph_embed)
        else:
            assert self.feature_aggregation == 'item'
            graph_embed = transEmbedding[:, -1]
            fixed_context = self.project_fixed_context(graph_embed)

        self.similarity = (0, 0, 0)
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_node_embeddings(transEmbedding_flat).reshape((shape[0], 1, shape[1], -1)).chunk(3, dim=-1)
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed
        )
        return AttentionModelFixed(transEmbedding_flat, fixed_context, *fixed_attention_node_data), internal_graph_embed

    def _get_log_p(self, fixed, mask=None, normalize=True, internal_node_holder=None, leaf_node_holder=None):
        query = fixed.context_node_projected[:, None, :]
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask,
                                                  internal_node_holder=internal_node_holder,
                                                  leaf_node_holder=leaf_node_holder)
        if normalize:
            log_p = nn.log_softmax(log_p / self.temp, dim=-1)
        return jt.exp(log_p), mask

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, internal_node_holder=None, leaf_node_holder=None):
        batch_size, num_steps, embed_dim = query.shape
        key_size = embed_dim // self.n_heads
        glimpse_Q = query.reshape((batch_size, num_steps, self.n_heads, 1, key_size)).permute((2, 0, 1, 3, 4))
        compatibility = jt.matmul(glimpse_Q, glimpse_K.transpose((0, 1, 2, 4, 3))) / math.sqrt(glimpse_Q.shape[-1])
        logits = compatibility.reshape((-1, 1, compatibility.shape[-1]))
        if self.tanh_clipping > 0:
            logits = jt.tanh(logits) * self.tanh_clipping
        logits = logits[:, 0, internal_node_holder: internal_node_holder + leaf_node_holder]
        if self.mask_logits:
            mask_bool = mask.bool()
            logits = jt.where(mask_bool, jt.full(logits.shape, -1e9, logits.dtype), logits)
        return logits, None

    def _one_to_many_logits_n_head(self, query, glimpse_K, glimpse_V, logit_K, mask, internal_node_holder=None, leaf_node_holder=None):
        return self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, internal_node_holder, leaf_node_holder)

    def _get_attention_node_data(self, fixed):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.shape[1] == 1 or v.shape[1] == num_steps
        target_steps = v.shape[1] if num_steps is None else num_steps
        return v.reshape((v.shape[0], v.shape[1], v.shape[2], self.n_heads, -1)).broadcast((v.shape[0], target_steps, v.shape[2], self.n_heads, v.shape[-1] // self.n_heads)).permute((3, 0, 1, 2, 4))
