#!/usr/bin/env python3

import math
import numpy as np
import jittor as jt
from jittor import nn
from distributions import FixedCategorical


device='cuda'


def _to_storage_var(src, dtype, shape=None):
    if src is None:
        assert shape is not None
        return jt.zeros(shape, dtype=dtype)
    if isinstance(src, jt.Var):
        out = src.stop_grad()
        if out.dtype != dtype:
            out = out.astype(dtype)
        return out
    return jt.array(np.array(src), dtype=dtype)

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    """

    def __init__(self, hidden_size, enable_mem, attn_span):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.enable_mem = enable_mem

    def execute(self, query, key, value):
        attn = jt.matmul(query, key.transpose(-1, -2))
        attn = attn / math.sqrt(self.hidden_size)
        attn = nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, value)
        return out

    def get_cache_size(self):
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, enable_mem, nb_heads, attn_span):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, enable_mem=enable_mem, attn_span=attn_span)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

        # note that the linear layer initialization in current Pytorch is kaiming uniform init

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def execute(self, query, key, value):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value)
        out = out.view(B, K, M, D)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, M, -1)
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)

    def execute(self, h):
        h1 = nn.relu(self.fc1(h))
        h2 = self.fc2(h1)
        return h2

class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size, enable_mem, nb_heads, attn_span, inner_hidden_size):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(
            hidden_size=hidden_size, enable_mem=enable_mem, nb_heads=nb_heads, attn_span=attn_span)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, inner_hidden_size=inner_hidden_size)
        self.norm1=nn.BatchNorm1d(hidden_size)
        self.norm2=nn.BatchNorm1d(hidden_size)
        self.enable_mem = enable_mem

    def execute(self, h, h_cache):
        # h = B x M x H
        # h_cache = B x L x H
        if self.enable_mem:
            h_all = jt.concat([h_cache, h], dim=1)
        else:
            h_all = h_cache
        attn_out = self.attn(h, h_all, h_all)
        h = self.norm1((h + attn_out).view(-1, h.size(-1))).view(*h.size()) # B x M x H
        ff_out = self.ff(h)
        out = self.norm2((h+ ff_out).view(-1, h.size(-1))).view(*h.size()) # B x M x H
        return out


class EncoderSeq(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, encoder_nb_layers,
                 attn_span, inner_hidden_size):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList([
            TransformerSeqLayer(
                hidden_size=hidden_size, enable_mem=True, nb_heads=nb_heads,
                attn_span=attn_span, inner_hidden_size=inner_hidden_size)
            for _ in range(encoder_nb_layers)
        ])

    def execute(self, x, h_cache):
        # x size = B x M
        block_size = x.size(1)
        h = self.init_embed(x)  # B x M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()

            # B x L x H
            h_cache_next_l = jt.concat(
                [h_cache[l][:, -cache_size + 1:, :], h[:, 0:1, :]],
                dim=1).stop_grad()

            h_cache_next.append(h_cache_next_l)

            h = layer(h, h_cache[l])  # B x M x H

        return h, h_cache_next


class QDecoder(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, decoder_nb_layers,
                 attn_span, inner_hidden_size):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList([
            TransformerSeqLayer(
                hidden_size=hidden_size, enable_mem=False, nb_heads=nb_heads,
                attn_span=attn_span, inner_hidden_size=inner_hidden_size)
            for _ in range(decoder_nb_layers)
        ])

    def execute(self, x, embedding):
        # x size = B x Q_M
        block_size = x.size(1)
        h = self.init_embed(x)  # B x Q_M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):

            h = layer(h, embedding)  # B x Q_M x H

        return h

class PackDecoder(nn.Module):
    def __init__(self, head_hidden_size, res_size, state_size, hidden_size,nb_heads, decoder_layers,attn_span, inner_hidden_size):
        nn.Module.__init__(self)

        self.att_decoder = QDecoder(state_size, hidden_size,nb_heads=nb_heads,decoder_nb_layers=decoder_layers,attn_span=attn_span, inner_hidden_size=inner_hidden_size)

        self.head  = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, res_size)
                            )
        self.critic = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, 1)
                            )

    def execute(self, x, embedding):
        h = self.att_decoder(x, embedding)
        out = self.head(h)
        value = self.critic(h)
        return out, value

def orientation_transform(l,w,h,num):
    if(num==0):
        return l,w,h
    elif(num==1):
        return l,h,w
    elif(num==2):
        return w,l,h
    elif(num==3):
        return w,h,l
    elif(num==4):
        return h,l,w
    else:
        return h,w,l

def observation_decode(observation, args = None):
    batch_size = observation.size(0)
    observation = observation.reshape(batch_size, -1, 9)
    packed_state = observation[:, 0: args.internal_node_holder, 0:6]
    next_box = observation[:,args.internal_node_holder:, 3:6]
    return packed_state, next_box

class RCQL(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, encoder_nb_layers, attn_span, inner_hidden_size,src_head_hidden_size,pos_head_hidden_size, 
                 s_res_size, r_res_size,x_res_size,y_res_size,decoder_nb_layers,item_state_size):
        nn.Module.__init__(self)

        self.encoder = nn.Linear(state_size, hidden_size)


        self.encoder=EncoderSeq(state_size=state_size,hidden_size=hidden_size,nb_heads=nb_heads,encoder_nb_layers=encoder_nb_layers,attn_span=attn_span,
                                inner_hidden_size=inner_hidden_size)
        # self.s_decoder=PackDecoder(head_hidden_size=src_head_hidden_size,res_size=s_res_size,state_size=item_state_size,hidden_size=hidden_size,
        #                            decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.r_decoder=PackDecoder(head_hidden_size=src_head_hidden_size,res_size=r_res_size,state_size=item_state_size,hidden_size=hidden_size,
                                   decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.x_decoder=PackDecoder(head_hidden_size=pos_head_hidden_size,res_size=x_res_size,state_size=item_state_size,hidden_size=hidden_size,
                                   decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.y_decoder=PackDecoder(head_hidden_size=pos_head_hidden_size,res_size=y_res_size,state_size=item_state_size,hidden_size=hidden_size,
                                   decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.softmax=nn.Softmax(dim=1)
        self.alpha=jt.array(1.0)

    def calc_ori_idx(self,actor_encoder_out,select_seq_idx,isGreedy):
        r_out, r_value = self.r_decoder(select_seq_idx, actor_encoder_out)
        batch_size=actor_encoder_out.size()[0]
        r_out=self._flatten_action_scores(r_out)
        r_out=self.softmax(r_out)
        if(isGreedy):
            oriidx=jt.argmax(r_out, dim=1)
            if isinstance(oriidx, tuple):
                oriidx = oriidx[0]
        else:
            oriidx=jt.multinomial(r_out,1)
            oriidx=jt.squeeze(oriidx)
        oripro=jt.gather(r_out, 1, oriidx.reshape((-1,1))).squeeze(1)
        return oriidx, oripro, r_out, r_value
        
    def calc_x_idx(self,actor_encoder_out,select_seq_idx_ori,isGreedy):
        x_out, x_value = self.x_decoder(select_seq_idx_ori, actor_encoder_out)
        batch_size=actor_encoder_out.size()[0]
        x_out=self._flatten_action_scores(x_out)
        x_out=self.softmax(x_out)
        if(isGreedy):
            xidx=jt.argmax(x_out, dim=1)
            if isinstance(xidx, tuple):
                xidx = xidx[0]
        else:
            xidx=jt.multinomial(x_out,1)
            xidx=jt.squeeze(xidx)
        xpro=jt.gather(x_out, 1, xidx.reshape((-1,1))).squeeze(1)
        return xidx, xpro, x_out, x_value
    
    def calc_y_idx(self,actor_encoder_out,select_seq_idx_ori,isGreedy):
        y_out, y_value = self.y_decoder(select_seq_idx_ori, actor_encoder_out)
        batch_size=actor_encoder_out.size()[0]
        y_out=self._flatten_action_scores(y_out)
        y_out=self.softmax(y_out)
        if(isGreedy):
            yidx=jt.argmax(y_out, dim=1)
            if isinstance(yidx, tuple):
                yidx = yidx[0]
        else:
            yidx=jt.multinomial(y_out,1)
            yidx=jt.squeeze(yidx)
        ypro=jt.gather(y_out, 1, yidx.reshape((-1,1))).squeeze(1)
        return yidx, ypro, y_out, y_value
    
    def set_device(self,device):
        return self

    def set_eval(self):
        self.encoder=self.encoder.eval()
        # self.s_decoder=self.s_decoder.eval()
        self.r_decoder=self.r_decoder.eval()
        self.x_decoder=self.x_decoder.eval()
        self.y_decoder=self.y_decoder.eval()

    def execute(self, observation, h_caches, IsGreedy, args):
        batch_size = observation.size(0)
        packed_state, select_item = observation_decode(observation, args)

        actor_encoder_out, h_caches = self.encoder(packed_state, h_caches)

        oriidx, oripro, oriout, orivalue = self.calc_ori_idx(actor_encoder_out, select_item, IsGreedy)

        select_ori_item=jt.zeros(select_item.size())
        for jk in range(batch_size):
            select_ori_item[jk,:,0],select_ori_item[jk,:,1],select_ori_item[jk,:,2]=orientation_transform(select_item[jk,:,0],select_item[jk,:,1],select_item[jk,:,2],oriidx[jk])

        xidx, xpro, xout, xvalue = self.calc_x_idx(actor_encoder_out, select_ori_item, IsGreedy)
        yidx, ypro, yout, yvalue = self.calc_y_idx(actor_encoder_out, select_ori_item, IsGreedy)
        value = jt.stack([
            self._value_to_batch_scalar(orivalue),
            self._value_to_batch_scalar(xvalue),
            self._value_to_batch_scalar(yvalue)
        ], dim=1).mean(dim=1)
        return jt.concat([oriidx.unsqueeze(1), xidx.unsqueeze(1), yidx.unsqueeze(1)], dim = 1), \
               jt.concat([oripro.unsqueeze(1), xpro.unsqueeze(1), ypro.unsqueeze(1)], dim = 1), \
               value

    def evaluate_actions(self, observation, h_caches, action, args):
        batch_size = observation.size(0)
        packed_state, select_item = observation_decode(observation, args)

        actor_encoder_out, h_caches = self.encoder(packed_state, h_caches)

        oriidx = action[:, 0]
        _, _, oriout, orivalue = self.calc_ori_idx(actor_encoder_out, select_item, False)

        select_ori_item=jt.zeros(select_item.size())
        for jk in range(batch_size):
            select_ori_item[jk,:,0],select_ori_item[jk,:,1],select_ori_item[jk,:,2]=orientation_transform(select_item[jk,:,0],select_item[jk,:,1],select_item[jk,:,2],oriidx[jk])

        xidx = action[:, 1]
        _, _, xout, xvalue = self.calc_x_idx(actor_encoder_out, select_ori_item, False)
        yidx = action[:, 2]
        _, _, yout, yvalue = self.calc_y_idx(actor_encoder_out, select_ori_item, False)
        dist_ori = FixedCategorical(probs=oriout)
        dist_x = FixedCategorical(probs=xout)
        dist_y = FixedCategorical(probs=yout)
        action_log_probs_ori = dist_ori.log_prob(oriidx)
        entropy_ori = dist_ori.entropy().mean()
        action_log_probs_x = dist_x.log_prob(xidx)
        entropy_x = dist_x.entropy().mean()
        action_log_probs_y = dist_y.log_prob(yidx)
        entropy_y = dist_y.entropy().mean()
        value = jt.stack([
            self._value_to_batch_scalar(orivalue),
            self._value_to_batch_scalar(xvalue),
            self._value_to_batch_scalar(yvalue)
        ], dim=1).mean(dim=1)
        return value, \
               jt.concat([action_log_probs_ori.unsqueeze(1), action_log_probs_x.unsqueeze(1), action_log_probs_y.unsqueeze(1)], dim = 1), \
               entropy_ori + entropy_x + entropy_y

    def _squeeze_if_singleton_dim1(self, x):
        if len(x.shape) > 1 and x.shape[1] == 1:
            return jt.squeeze(x, dim=1)
        return x

    def _reduce_value(self, x):
        while len(x.shape) > 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if len(x.shape) > 1:
            reduce_dims = tuple(range(1, len(x.shape)))
            x = jt.mean(x, dim=reduce_dims)
        return x

    def _flatten_action_scores(self, x):
        x = self._squeeze_if_singleton_dim1(x)
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], -1))
        return x

    def _value_to_batch_scalar(self, x):
        while len(x.shape) > 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if len(x.shape) == 1:
            return x
        if len(x.shape) >= 2:
            return x.reshape((x.shape[0], -1)).mean(dim=1)
        return x

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size = 128):
        self.obs = jt.zeros((num_steps + 1, num_processes, *obs_shape), dtype=jt.float32)
        self.recurrent_hidden_states = jt.zeros(
            (num_steps + 1, num_processes, recurrent_hidden_state_size), dtype=jt.float32)
        self.rewards = jt.zeros((num_steps, num_processes, 1), dtype=jt.float32)
        self.value_preds = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.returns = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.action_log_probs = jt.zeros((num_steps, num_processes, 3), dtype=jt.float32)
        self.actions = jt.zeros((num_steps, num_processes, 3), dtype=jt.int32)
        self.masks = jt.ones((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        return self

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks):
        self.obs[self.step + 1] = _to_storage_var(obs, self.obs.dtype)
        self.recurrent_hidden_states[self.step + 1] = _to_storage_var(
            recurrent_hidden_states,
            self.recurrent_hidden_states.dtype,
            shape=self.recurrent_hidden_states[self.step + 1].shape,
        )
        self.actions[self.step] = _to_storage_var(actions.reshape((-1, 3)), self.actions.dtype)
        self.action_log_probs[self.step] = _to_storage_var(
            action_log_probs.reshape((-1, 3)), self.action_log_probs.dtype
        )
        self.value_preds[self.step] = _to_storage_var(
            value_preds.reshape((-1, 1)), self.value_preds.dtype
        )
        self.rewards[self.step] = _to_storage_var(rewards.reshape((-1, 1)), self.rewards.dtype)
        self.masks[self.step + 1] = _to_storage_var(masks.reshape((-1, 1)), self.masks.dtype)
        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self.obs[0] = self.obs[-1].stop_grad()
        self.recurrent_hidden_states[0] = self.recurrent_hidden_states[-1].stop_grad()
        self.masks[0] = self.masks[-1].stop_grad()


    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
            next_val_np = next_value.stop_grad().numpy().reshape((-1, 1))
            rewards_np = self.rewards.stop_grad().numpy()
            masks_np = self.masks.stop_grad().numpy()
            returns_np = np.zeros(self.returns.shape, dtype=np.float32)
            returns_np[-1] = next_val_np
            for step in reversed(range(rewards_np.shape[0])):
                returns_np[step] = returns_np[step + 1] * gamma * masks_np[step + 1] + rewards_np[step]
            self.returns = jt.array(returns_np, dtype=jt.float32)

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.shape[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        indices = np.random.permutation(batch_size)
        flat_obs = self.obs[:-1].reshape((batch_size, *self.obs.shape[2:]))
        flat_rnn = self.recurrent_hidden_states[:-1].reshape((batch_size, self.recurrent_hidden_states.shape[-1]))
        flat_actions = self.actions.reshape((batch_size, self.actions.shape[-1]))
        flat_value_preds = self.value_preds[:-1].reshape((batch_size, 1))
        flat_returns = self.returns[:-1].reshape((batch_size, 1))
        flat_masks = self.masks[:-1].reshape((batch_size, 1))
        flat_action_log_probs = self.action_log_probs.reshape((batch_size, self.action_log_probs.shape[-1]))
        flat_adv = None if advantages is None else advantages.reshape((batch_size, 1))

        for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
            batch_indices = indices[start:start + mini_batch_size].tolist()
            obs_batch = flat_obs[batch_indices]
            recurrent_hidden_states_batch = flat_rnn[batch_indices]
            actions_batch = flat_actions[batch_indices]
            value_preds_batch = flat_value_preds[batch_indices]
            return_batch = flat_returns[batch_indices]
            masks_batch = flat_masks[batch_indices]
            old_action_log_probs_batch = flat_action_log_probs[batch_indices]
            adv_targ = None if flat_adv is None else flat_adv[batch_indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


