import torch
import torch.nn as nn
import torch.nn.functional as F
from tapnet.models.attention import CrossTransformer
from tapnet.models.network import ObjectEncoder, SpaceEncoder, obs_to_tensor

class Encoder(nn.Module):
    def __init__(self, encoder_type, box_dim, ems_dim, hidden_dim, corner_num, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        self.box_dim = box_dim
        self.device = device

        self.corner_num = corner_num

        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, encoder_type, device)
        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim, corner_num)

    def forward(self, box, precedences, ems):
        
        box_vecs = self.obj_encoder(box, precedences)
        ems_vecs = self.space_encoder(ems)
        return box_vecs, ems_vecs
    
class StrategyAttention(nn.Module):
    def __init__(self, encoder: Encoder, hidden_dim, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(StrategyAttention, self).__init__()

        self.encoder = encoder
        self.transformer = CrossTransformer(hidden_dim, device=device)

    def forward(self, box, precedences, ems, ems_mask, ems_to_box_mask, box_valid_mask=None):
        
        box_vecs, ems_vecs = self.encoder(box, precedences, ems)
        attn_vecs = self.transformer(ems_vecs, box_vecs, ems_mask, box_valid_mask, ems_to_box_mask)

        return attn_vecs


class Policy(nn.Module):
    def __init__(self, encoder: Encoder, hidden_dim, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()

        self.strategy = StrategyAttention( encoder, hidden_dim, device )

        self.device = device
        
    def forward(self, obs, state=None, info={} ):
        
        ems_num, box_state_num, box_states, valid_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask = obs_to_tensor(obs, self.device)
        precedences = None

        batch_size = len(box_states)

        attn_vecs = self.strategy(box_states, precedences, ems, ems_mask, ems_to_box_mask, valid_mask)

        # mask the attnetion score
        if attn_vecs.shape[1] == ems_num:
            attn_score = attn_vecs + valid_mask.unsqueeze(1).float().log()
        else:
            attn_vecs = attn_vecs.reshape(batch_size, -1, ems_num, box_state_num)
            attn_score = attn_vecs.clone()
            for i in range(attn_vecs.shape[1]):
                attn_score[:,i,:,:] = attn_vecs[:,i,:,:] + valid_mask.unsqueeze(1).float().log()

        attn_score = attn_score.reshape(batch_size, -1)
        probs = F.softmax(attn_score, dim=1)
        
        return probs, state
    


class Critic(nn.Module):

    def __init__(self, encoder: Encoder, box_num, ems_num, hidden_dim, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Critic, self).__init__()
        
        self.box_num = box_num
        self.ems_num = ems_num

        self.encoder = encoder

        self.box_combine_mlp = nn.Sequential(
            nn.Linear(box_num, box_num),
            nn.ReLU(),
            nn.Linear(box_num, 1)
        )
        
        self.ems_combine_mlp = nn.Sequential(
            nn.Linear(ems_num, ems_num),
            nn.ReLU(),
            nn.Linear(ems_num, 1)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.device = device

    def forward(self, obs, act=None, info={} ):
        
        # encoder_input, decoder_input = observation
        ems_num, box_state_num, box_states, valid_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask = obs_to_tensor(obs, self.device)
        
        valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)

        box_vecs, ems_vecs = self.encoder(box_states, None, ems)
        ems_vecs = ems_vecs[0]

        box_feats = box_vecs.masked_fill(valid_mask.squeeze(1).squeeze(1).unsqueeze(-1) == 0, float("0"))
        ems_feats = ems_vecs.masked_fill(ems_mask.squeeze(1).squeeze(1).unsqueeze(-1) == 0, float("0"))

        box_vec = self.box_combine_mlp(box_feats.transpose(2,1)).squeeze(2)
        ems_vec = self.ems_combine_mlp(ems_feats.transpose(2,1)).squeeze(2)
        
        output_value = self.output_layer( torch.cat([box_vec, ems_vec], dim=-1) )

        return output_value
    


class Actor(nn.Module):
    def __init__(self, box_dim, ems_dim, hidden_dim, encoder_type, device) -> None:
        super(Actor, self).__init__()
        encoder = Encoder(encoder_type, box_dim, ems_dim, hidden_dim, 1, device)
        self.actor = Policy(encoder, hidden_dim, device)
    
    def forward(self, data):
        ret = self.actor(data)
        return ret