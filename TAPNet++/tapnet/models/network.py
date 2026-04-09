import jittor as jt
from jittor import nn
from tapnet.models.attention import CrossTransformer, CrossLayer
from tapnet.models.encoder import obs_to_tensor, ObjectEncoder, SpaceEncoder
from tapnet.model_backend import cat, masked_fill, softmax, xavier_uniform_

class StrategyAttention(nn.Module):
    def __init__(self, encoder_type, box_dim, ems_dim, hidden_dim, prec_dim, corner_num, stable_predict=False, device="cuda:0"):
        super(StrategyAttention, self).__init__()
        self.encoder_type = encoder_type
        self.box_dim = box_dim
        self.device = device

        self.corner_num = corner_num

        # prec_dim = 2
        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, prec_dim, encoder_type, device)
        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim, corner_num)

        self.transformer = CrossTransformer(hidden_dim, mask_predict=stable_predict, device=device, corner_num=corner_num)

    def execute(self, box, precedences, ems, ems_mask, ems_to_box_mask, box_valid_mask=None):
        
        box_vecs = self.obj_encoder(box, precedences, box_valid_mask)
        ems_vecs = self.space_encoder(ems)
        attn_vecs = self.transformer(ems_vecs, box_vecs, ems_mask, box_valid_mask, ems_to_box_mask)

        return attn_vecs

    def forward(self, box, precedences, ems, ems_mask, ems_to_box_mask, box_valid_mask=None):
        return self.execute(box, precedences, ems, ems_mask, ems_to_box_mask, box_valid_mask)

class Net(nn.Module):

    def __init__(self, box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict=False,
                corner_num = 1,
                device="cuda:0"):
        super(Net, self).__init__()

        self.corner_num = corner_num

        self.strategy = StrategyAttention( encoder_type, box_dim, ems_dim, hidden_dim, prec_dim, corner_num, stable_predict, device )

        for p in self.parameters():
            if len(p.shape) > 1:
                xavier_uniform_(p)


        self.device = device
        self.box_dim = box_dim
        
        self.encoder_type = encoder_type

    def execute(self, obs, state=None, info={} ):
        
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)
        action_mask = (valid_mask * access_mask).unsqueeze(1)

        attn_vecs = self.strategy(box_states, prec_states, ems, ems_mask, ems_to_box_mask, valid_mask)

        # mask the attnetion score
        if attn_vecs.shape[1] == ems_num:
            # Match the torch behavior exactly: invalid actions must become
            # -inf before softmax, not merely a very small finite log-value.
            attn_score = masked_fill(attn_vecs, action_mask == 0, float("-inf"))
        else:
            attn_vecs = attn_vecs.reshape(batch_size, -1, ems_num, box_state_num)
            attn_score = attn_vecs.clone()
            for i in range(attn_vecs.shape[1]):
                attn_score[:, i, :, :] = masked_fill(
                    attn_vecs[:, i, :, :], action_mask == 0, float("-inf")
                )

        attn_score = attn_score.reshape(batch_size, -1)
        probs = softmax(attn_score, dim=1)
        
        # if self.training == False:
        #     prob_max = probs.max(dim=1, keepdim=True)[0]
        #     probs[ probs != prob_max ] = 0
        #     probs[ probs == prob_max ] = 1
        #     probs /= probs.sum(dim=1, keepdim=True)
        
        return probs, state

    def forward(self, obs, state=None, info={} ):
        return self.execute(obs, state, info)

class Critic(nn.Module):

    def __init__(self, box_dim, ems_dim, box_num, ems_num, hidden_dim, prec_dim=2, heads = 4, output_dim=1,
                 prec_type = 'none',
                device="cuda:0"):
        super(Critic, self).__init__()
        
        self.box_num = box_num
        self.ems_num = ems_num

        # prec_dim = 2
        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, prec_dim, prec_type, device=device)
        self.space_encoder = SpaceEncoder(ems_dim, hidden_dim)

        # self.box_attn = AttnModule(hidden_dim, heads, 0, device, 50)
        # self.ems_attn = AttnModule(hidden_dim, heads, 0, device, 50)
        
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

        # self.output_attn = SelfAttention(hidden_dim, heads)
        self.output_layer = nn.Sequential(
            # nn.Linear(hidden_dim*2, 1),
            # nn.ReLU()
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.device = device

    def execute(self, obs, act=None, info={} ):
        
        # encoder_input, decoder_input = observation
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, _, _ = obs_to_tensor(obs, self.device)
        
        # TODO what

        valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        # ems_mask = ems_mask.unsqueeze(1).unsqueeze(2)

        box_vecs = self.obj_encoder(box_states, prec_states, valid_mask)
        ems_vecs = self.space_encoder(ems)[0]

        # box_vecs = self.box_attn( box_vecs, valid_mask, pos_embed=False)
        # ems_vecs = self.ems_attn( ems_vecs, ems_mask, pos_embed=False)

        box_feats = masked_fill(box_vecs, valid_mask.squeeze(1).squeeze(1).unsqueeze(-1) == 0, float("0"))
        ems_feats = masked_fill(ems_vecs, ems_mask.squeeze(1).squeeze(1).unsqueeze(-1) == 0, float("0"))

        box_vec = self.box_combine_mlp(box_feats.transpose((0, 2, 1))).squeeze(2)
        ems_vec = self.ems_combine_mlp(ems_feats.transpose((0, 2, 1))).squeeze(2)
        
        output_value = self.output_layer(cat([box_vec, ems_vec], dim=-1))
        # output_value = F.relu(output_value)
        return output_value

    def forward(self, obs, act=None, info={} ):
        return self.execute(obs, act, info)

class Actor(nn.Module):
    def __init__(self, box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict, corner_num, device) -> None:
        super(Actor, self).__init__()
        self.actor = Net(box_dim, ems_dim, hidden_dim, prec_dim, encoder_type, stable_predict, corner_num, device)
    
    def execute(self, obs, state=None, info={} ):
        probs, state = self.actor(obs, state=None, info={})
        return probs, state

    def forward(self, obs, state=None, info={} ):
        return self.execute(obs, state, info)