import jittor as jt
from jittor import nn
import numpy as np
import math
from distributions import FixedCategorical


def _resize_bilinear(x, target_size):
    return nn.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


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


def observation_decode(observation, contianer_size):
    frontiers = observation[:, 0: contianer_size[0] * contianer_size[2] * 2]
    next_box  = observation[:, contianer_size[0] * contianer_size[2] * 2:contianer_size[0] * contianer_size[2] * 2+3]
    next_box  = next_box.unsqueeze(1)
    maskposition = observation[:, contianer_size[0] * contianer_size[2] * 2 + 3:]
    maskseq = []
    seqidx = jt.zeros((frontiers.shape[0],), dtype=jt.int32)
    return next_box, frontiers, maskposition, maskseq, seqidx

class Attend2Pack(nn.Module):
    def __init__(self,originaldim,embedingdim,num_head,FFNdim,AFFNnum,inchannel,outchannel,C,outdim,L,W, scale, target_size = (256, 256)):
        super().__init__()
        self.AFFNnum=AFFNnum
        self.AFFNlist=nn.ModuleList()
        self.embeding=nn.Linear(originaldim,embedingdim)
        for i in range(AFFNnum):
            self.AFFNlist.append(ResidualFFN(embedingdim,num_head,FFNdim))
        self.FrontierEmbedding=FrontierEmbedding(inchannel,outchannel,embedingdim,L,W)
        self.DecodingPosition=DecodingPosition(embedingdim,outdim)
        self.scale = scale
        self.target_size = target_size

    def clacb(self,input,scale):
        x=self.embeding(input)
        for i in range(self.AFFNnum):
            x=self.AFFNlist[i](x,scale)
        return x

    def execute(self, observation, IsGreedy, contianer_size):
        inputb, inputf, maskposition, maskseq, seqidx = observation_decode(observation, contianer_size)
        inputb = self.clacb(inputb, self.scale)
        inputf = inputf.reshape(inputf.shape[0], 2, contianer_size[0], contianer_size[2])
        inputf = _resize_bilinear(inputf, self.target_size)
        inputf = self.FrontierEmbedding(inputf)
        posidx, policypro, policy, value = self.DecodingPosition(seqidx, inputb, inputf, maskseq, maskposition,IsGreedy)
        return policypro, posidx, value

    def evaluate_actions(self, observation, action, contianer_size):
        inputb, inputf, maskposition, maskseq, seqidx = observation_decode(observation, contianer_size)
        inputb = self.clacb(inputb, self.scale)
        inputf = inputf.reshape(inputf.shape[0], 2, contianer_size[0], contianer_size[2])
        inputf = _resize_bilinear(inputf, self.target_size)
        inputf = self.FrontierEmbedding(inputf)
        _, _, policy, value = self.DecodingPosition(seqidx, inputb, inputf, maskseq, maskposition,IsGreedy = False)
        dist = FixedCategorical(probs=policy)
        action_log_probs = dist.log_probs(action)
        return value, action_log_probs, dist.entropy().mean()

class MHA(nn.Module):
    def __init__(self,embedingdim,num_head):
        super().__init__()
        self.embedingdim=embedingdim
        self.num_head=num_head
        self.edn=embedingdim//num_head
        self.Q=nn.Linear(embedingdim,embedingdim,bias=False)
        self.K=nn.Linear(embedingdim,embedingdim,bias=False)
        self.V=nn.Linear(embedingdim,embedingdim,bias=False)
        self.softmax=nn.Softmax(dim=-1)
        self.O=nn.Linear(embedingdim,embedingdim,bias=False)
    
    def execute(self,input,scale):
        batch,n,dim=input.shape
        nh=self.num_head
        edn=self.edn
        q=self.Q(input).reshape(batch,n,nh,edn).transpose(1,2)
        k=self.K(input).reshape(batch,n,nh,edn).transpose(1,2)
        v=self.V(input).reshape(batch,n,nh,edn).transpose(1,2)
        dist=jt.matmul(q,k.transpose(0,1,3,2))*scale
        dist=self.softmax(dist)
        att=jt.matmul(dist,v)
        att=att.transpose(1,2).reshape(batch,n,dim)
        att=self.O(att)
        return att

class AttentionFFN(nn.Module):
    def __init__(self,embedingdim,num_head,lineardim):
        super().__init__()
        self.mha=MHA(embedingdim,num_head)
        self.norm=nn.LayerNorm(embedingdim)
        self.linear1=nn.Linear(embedingdim,lineardim)
        self.linear2=nn.Linear(lineardim,embedingdim)
        
    def execute(self,input,scale):
        x=self.norm(input)
        x=self.mha(x,scale)+x
        x=self.norm(x)
        x2=nn.relu(self.linear1(x))
        x=nn.relu(self.linear2(x2))+x
        return x


class ResidualFFN(nn.Module):
    def __init__(self, embedding_dim, num_head, linear_dim):
        super().__init__()
        # 移除 MHA 模块
        self.norm1 = nn.LayerNorm(embedding_dim)  # 第一个 LayerNorm
        self.norm2 = nn.LayerNorm(embedding_dim)  # 第二个 LayerNorm

        # 定义残差网络中的线性层
        self.linear1 = nn.Linear(embedding_dim, linear_dim)  # 第一个全连接层
        self.linear2 = nn.Linear(linear_dim, embedding_dim)  # 第二个全连接层

    def execute(self, input, scale=None):
        x = self.norm1(input)
        x = self.linear2(nn.relu(self.linear1(x))) + x
        x = self.norm2(x)
        x = self.linear2(nn.relu(self.linear1(x))) + x
        return x


class FrontierEmbedding(nn.Module):
    def __init__(self,inchannel,outchannel,embedingdim,L,W):
        super().__init__()
        self.conv1=nn.Conv2d(inchannel,  outchannel,kernel_size=4, stride=2, padding=1)
        self.conv2=nn.Conv2d(outchannel, outchannel,kernel_size=4, stride=4)
        self.conv3=nn.Conv2d(outchannel, outchannel,kernel_size=4, stride=4)
        L = 8
        W = 8
        self.norm1          = nn.LayerNorm([outchannel,L,W])
        self.linear1        = nn.Linear(outchannel*L*W,embedingdim)
        self.LinearFrontier = nn.Linear(embedingdim,embedingdim,bias=False)

    def execute(self,input):
        x=self.conv1(input)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.norm1(x)
        x=nn.relu(x)
        x=jt.flatten(x,1)
        x=self.linear1(x)
        x=self.LinearFrontier(x)
        return x


class DecodingPosition(nn.Module):
    def __init__(self,embedingdim,out_dim):
        super().__init__()
        self.embedingdim=embedingdim
        self.LinearW_selected=nn.Linear(embedingdim,embedingdim,bias=False)
        self.LinearP_leftover=nn.Linear(embedingdim,embedingdim,bias=False)
        self.LinearOut=nn.Linear(embedingdim,out_dim,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.critic_moduel = nn.Linear(embedingdim, 1)

    def execute(self,selectb,inputb,inputf,maskseq,maskposition,IsGreedy):
        batch_size=inputb.size()[0]

        bst=self.LinearW_selected(inputb)
        bst=jt.squeeze(bst,1)
        inputq=(bst+inputf)/2
        ct2=self.LinearOut(inputq)

        ct2[maskposition==0] = -1e6

        policy=self.softmax(ct2)
        batchX = jt.arange(batch_size)
        value = self.critic_moduel(inputq)

        if(IsGreedy):
            posidx = jt.argmax(policy, dim=1)
            if isinstance(posidx, tuple):
                posidx = posidx[0]
        else:
            posidx=jt.multinomial(policy,1)
            posidx=jt.squeeze(posidx)
        policypro=policy[batchX,posidx]
        return posidx, policypro, policy, value


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size=128):
        self.obs = jt.zeros((num_steps + 1, num_processes, *obs_shape), dtype=jt.float32)
        self.recurrent_hidden_states = jt.zeros(
            (num_steps + 1, num_processes, recurrent_hidden_state_size), dtype=jt.float32)
        self.rewards = jt.zeros((num_steps, num_processes, 1), dtype=jt.float32)
        self.value_preds = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.returns = jt.zeros((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.action_log_probs = jt.zeros((num_steps, num_processes, 1), dtype=jt.float32)
        self.actions = jt.zeros((num_steps, num_processes, 1), dtype=jt.int32)
        self.masks = jt.ones((num_steps + 1, num_processes, 1), dtype=jt.float32)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        return self

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks):
        self.obs[self.step + 1] = _to_storage_var(obs, self.obs.dtype)
        self.actions[self.step] = _to_storage_var(actions.reshape((-1, 1)), self.actions.dtype)
        self.action_log_probs[self.step] = _to_storage_var(
            action_log_probs.reshape((-1, 1)), self.action_log_probs.dtype
        )
        self.value_preds[self.step] = _to_storage_var(
            value_preds.reshape((-1, 1)), self.value_preds.dtype
        )
        self.rewards[self.step] = _to_storage_var(rewards.reshape((-1, 1)), self.rewards.dtype)
        self.masks[self.step + 1] = _to_storage_var(masks.reshape((-1, 1)), self.masks.dtype)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0] = self.obs[-1].stop_grad()
        self.masks[0] = self.masks[-1].stop_grad()

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
        next_val_np = next_value.stop_grad().numpy().reshape((-1, 1))
        rewards_np = self.rewards.stop_grad().numpy()
        masks_np = self.masks.stop_grad().numpy()
        returns_np = np.zeros(self.returns.shape, dtype=np.float32)
        returns_np[-1] = next_val_np
        for step in reversed(range(rewards_np.shape[0])):
            returns_np[step] = returns_np[step + 1] * gamma * masks_np[step + 1] + rewards_np[step]
        self.returns = jt.array(returns_np, dtype=jt.float32)
