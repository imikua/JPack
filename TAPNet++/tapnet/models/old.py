import torch
import torch.nn as nn
import torch.nn.functional as F
from tapnet.models.attention import SelfAttention
from tapnet.models.encoder import obs_to_tensor, ObjectEncoder, HeightmapEncoder

class PackDecoder(nn.Module):
    def __init__(self, hidden_dim, heightmap_width, heightmap_length, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(PackDecoder, self).__init__()
        
        half_dim = int(hidden_dim / 2)

        self.height_encoder = HeightmapEncoder(2, half_dim, [heightmap_width, heightmap_length])
        self.prebox_encoder = nn.Conv1d(3, half_dim, kernel_size=1)

        drop = 0.1
        
        self.rnn = nn.GRU( hidden_dim, hidden_dim, 1, batch_first=True, dropout=drop)
        
    def forward(self, pre_box, heightmap, last_hh):
        '''
            pre_box: [batch_size x 3]
        '''
        pre_vec = self.prebox_encoder(pre_box.unsqueeze(2)) #.squeeze(2)
        height_vec = self.height_encoder(heightmap) #.squeeze(2)
        decoder_vec = torch.cat([pre_vec, height_vec], dim=1)

        rnn_out, last_hh = self.rnn(decoder_vec.transpose(2,1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        return rnn_out, last_hh

class StrategyAttention(nn.Module):
    def __init__(self, encoder_type, box_dim, state_num, hidden_dim, heightmap_width, heightmap_length, max_length=200, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(StrategyAttention, self).__init__()
        self.encoder_type = encoder_type
        self.box_dim = box_dim
        self.device = device
        self.hidden_dim = hidden_dim

        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, state_num, encoder_type, device)
        self.pack_decoder = PackDecoder(hidden_dim, heightmap_width, heightmap_length, device)
        
        
        self.v1 = nn.Parameter(torch.zeros((1, 1, hidden_dim), requires_grad=True))
        self.W1 = nn.Parameter(torch.zeros((1, hidden_dim, hidden_dim + hidden_dim), requires_grad=True))

        self.v2 = nn.Parameter(torch.zeros((1, 1, hidden_dim), requires_grad=True))
        self.W2 = nn.Parameter(torch.zeros((1, hidden_dim, hidden_dim + hidden_dim), requires_grad=True))

        nn.init.xavier_uniform_(self.v1)
        nn.init.xavier_uniform_(self.v2)

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        # self.position_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(self, box, precedences, pre_box, heightmap, last_hh):
        
        batch_size, state_num, _ = box.shape

        encoder_vecs = self.obj_encoder(box, precedences)

        # positions = torch.arange(0, state_num).expand(batch_size, state_num).to(self.device)
        # emb_pos = self.position_embedding(positions)
        # encoder_vecs = encoder_vecs + emb_pos
        
        decoder_vec, last_hh = self.pack_decoder(pre_box, heightmap, last_hh)
        decoder_vec = decoder_vec.unsqueeze(1)

        # attention
        query = decoder_vec
        keys = encoder_vecs
        values = keys

        key_len = keys.shape[1]
        value_len = key_len
        query_len = query.shape[1] # 1

        if True:
            # step 1
            query = query.expand(-1, key_len, -1) # expand_as(keys)
            hidden = torch.cat((keys, query), 2) # N, key_len, hiddem_dim*2

            v1 = self.v1.expand(batch_size, -1, -1) # N, 1, hidden
            W1 = self.W1.expand(batch_size, -1, -1) # N, hidden, hidden*2
            scores = torch.bmm(v1, torch.tanh(torch.bmm(W1, hidden.transpose(2,1)))) # N, 1, key_len

            attns = F.softmax(scores, dim=2)  # (N, 1, key_len)
            new_query = attns.bmm( values )  # (N, 1, hidden)

            # step 2
            new_query = new_query.expand(-1, key_len, -1) # expand_as(keys)
            hidden2 = torch.cat((keys, new_query), 2) # N, key_len, hiddem_dim*2

            v2 = self.v2.expand(batch_size, -1, -1) # N, 1, hidden
            W2 = self.W2.expand(batch_size, -1, -1) # N, hidden, hidden*2
            attn_vecs = torch.bmm(v2, torch.tanh(torch.bmm(W2, hidden2.transpose(2,1)))) # N, 1, key_len
            # attn_vecs = torch.bmm(v1, torch.tanh(torch.bmm(W1, hidden2.transpose(2,1)))) # N, 1, key_len

        # query: N, query_len, hiddem_dim
        # values: N, key_len, hiddem_dim
        # keys: N, key_len, hiddem_dim
        # energy: N, query_len, key_len
        else:
            energy = torch.einsum("nqd,nkd->nqk", [query, keys]) / self.hidden_dim ** (1/2.)

            attn_weight = torch.softmax(energy, dim=2)
            new_query = torch.einsum("nqk,nkd->nqd", [attn_weight, values])

            energy = torch.einsum("nqd,nkd->nqk", [new_query, keys])
            attn_vecs = energy / (self.hidden_dim ** (1/2.))

        return attn_vecs, last_hh



class StrategyAttention_old(nn.Module):
    def __init__(self, encoder_type, box_dim, state_num, hidden_dim, heightmap_width, heightmap_length, max_length=200, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(StrategyAttention, self).__init__()
        self.encoder_type = encoder_type
        self.box_dim = box_dim
        self.device = device
        self.hidden_dim = hidden_dim

        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, state_num, encoder_type, device)
        self.pack_decoder = PackDecoder(hidden_dim, heightmap_width, heightmap_length, device)
        
        # self.position_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(self, box, precedences, pre_box, heightmap, last_hh):
        
        batch_size, state_num, _ = box.shape

        encoder_vecs = self.obj_encoder(box, precedences)

        # positions = torch.arange(0, state_num).expand(batch_size, state_num).to(self.device)
        # emb_pos = self.position_embedding(positions)
        # encoder_vecs = encoder_vecs + emb_pos
        
        decoder_vec, last_hh = self.pack_decoder(pre_box, heightmap, last_hh)
        decoder_vec = decoder_vec.unsqueeze(1)

        # attention
        keys = encoder_vecs
        query = decoder_vec
 
        # query: N, query_len, hiddem_dim
        # values: N, key_len, hiddem_dim
        # keys: N, key_len, hiddem_dim
        # energy: N, query_len, key_len
        energy = torch.einsum("nqd,nkd->nqk", [query, keys])

        # values = encoder_vecs
        # attn_weight = torch.softmax(energy / (self.hidden_dim ** (1/2.)), dim=2)
        # new_query = torch.einsum("nqk,nkd->nqd", [attn_weight, values])
        # energy = torch.einsum("nqd,nkd->nqk", [new_query, keys])

        attn_vecs = energy / (self.hidden_dim ** (1/2.))

        return attn_vecs, last_hh



class Net(nn.Module):

    def __init__(self, encoder_type, box_dim, prec_dim, hidden_dim, heightmap_width, heightmap_length, max_length,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        # prec_dim = box_num * 2
        super(Net, self).__init__()

        self.strategy = StrategyAttention( encoder_type, box_dim, prec_dim, hidden_dim, heightmap_width, heightmap_length, max_length, device )

        self.device = device
        self.box_dim = box_dim
        
        self.encoder_type = encoder_type

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, obs, state=None, info={} ):
        
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, pre_box, heightmap = obs_to_tensor(obs, self.device)

        batch_size = len(box_states)

        attn_vecs, state = self.strategy(box_states, prec_states, pre_box, heightmap, state)

        attn_score = attn_vecs + (valid_mask * access_mask).unsqueeze(1).float().log()
        
        attn_score = attn_score.reshape(batch_size, -1)
        probs = F.softmax(attn_score, dim=1)

        if self.training == False:
            prob_max = probs.max(dim=1, keepdim=True)[0]
            probs[ probs != prob_max ] = 0
            probs[ probs == prob_max ] = 1
            probs /= probs.sum(dim=1, keepdim=True)
        
        return probs, state

class Critic(nn.Module):

    def __init__(self, box_dim, box_state_num, prec_dim, heightmap_width, heightmap_length, hidden_dim, output_dim=1,
                 prec_type='cnn',
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Critic, self).__init__()
        
        # self.box_num = box_num
        # prec_dim = box_num * 2

        self.box_state_num = box_state_num
        max_length = 200

        self.obj_encoder = ObjectEncoder(box_dim, hidden_dim, prec_dim, prec_type, device=device)
        self.pack_decoder = PackDecoder(hidden_dim, heightmap_width, heightmap_length)
                
        # self.position_embedding = nn.Embedding(max_length, hidden_dim)

        # self.attn = SelfAttention(hidden_dim, 1)
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_dim), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_dim, hidden_dim + hidden_dim), requires_grad=True))

        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.W)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.device = device

    def forward(self, obs, act=None, info={} ):
        
        # encoder_input, decoder_input = observation
        ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, pre_box, heightmap = obs_to_tensor(obs, self.device)
        
        # TODO
        # valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)

        batch_size, state_num, _ = box_states.shape

        encoder_vecs = self.obj_encoder(box_states, prec_states)

        # positions = torch.arange(0, state_num).expand(batch_size, state_num).to(self.device)
        # emb_pos = self.position_embedding(positions)
        # encoder_vecs = encoder_vecs + emb_pos
        
        decoder_vec, last_hh = self.pack_decoder(pre_box, heightmap, None)
        decoder_vec = decoder_vec.unsqueeze(1)
        
        # batch_size x 1 x hidden_dim
        # attn_vec = self.attn(encoder_vecs, encoder_vecs, decoder_vec, None)

        # attention
        query = decoder_vec
        keys = encoder_vecs
        values = keys

        key_len = keys.shape[1]
        value_len = key_len
        query_len = query.shape[1] # 1

        v = self.v.expand(batch_size, -1, -1) # N, 1, hidden
        W = self.W.expand(batch_size, -1, -1) # N, hidden, hidden*2

        for _ in range(3):
            query = query.expand(-1, key_len, -1) # expand_as(keys)
            hidden = torch.cat((keys, query), 2) # N, key_len, hiddem_dim*2
            scores = torch.bmm(v, torch.tanh(torch.bmm(W, hidden.transpose(2,1)))) # N, 1, key_len
            attns = F.softmax(scores, dim=2)  # (N, 1, key_len)
            query = attns.bmm( values )  # (N, 1, hidden)

        
        output_value = self.output_layer( query ).squeeze(1)
        return output_value

class Actor(nn.Module):
    def __init__(self, encoder_type, box_dim, box_num, hidden_dim, heightmap_width, heightmap_length, max_length, device):
        super(Actor, self).__init__()
        self.actor = Net(encoder_type, box_dim, box_num, hidden_dim, heightmap_width, heightmap_length, max_length, device)
    
    def forward(self, obs, state=None, info={} ):
        probs, state = self.actor(obs, state=None, info={})
        return probs, state