import jittor as jt
from jittor import nn
import math
from tapnet.models.attention import TransformerBlock
from tapnet.model_backend import arange, cat, ensure_var, leaky_relu, ones, to_float, zeros


def _scalar_value(x):
    if isinstance(x, jt.Var):
        data = x.numpy()
        if hasattr(data, "reshape"):
            return int(data.reshape(-1)[0])
        return int(data)
    return int(x)


def obs_to_tensor(obs, device):
    box_num = _scalar_value(obs.box_num[0])
    ems_num = _scalar_value(obs.ems_num[0])
    state_num = _scalar_value(obs.state_num[0])
    container_width = _scalar_value(obs.container_width[0])
    container_length = _scalar_value(obs.container_length[0])

    corner_num = _scalar_value(obs.corner_num[0])

    box_state_num = box_num * state_num

    batch_size = len(obs.box_num)

    if len(obs.box_states.shape) == 3:
        box_num = _scalar_value(box_num)
        ems_num = _scalar_value(ems_num)
        state_num = _scalar_value(state_num)
        box_state_num = _scalar_value(box_state_num)

    box_states = obs.box_states.reshape(batch_size, box_state_num, 3)
    
    pre_box = obs.pre_box.reshape(batch_size, 3)
    heightmap = obs.heightmap.reshape(batch_size, 2, container_width, container_length)

    valid_mask = obs.valid_mask.reshape(batch_size, -1)
    access_mask = obs.access_mask.reshape(batch_size, -1)
    
    ems = obs.ems.reshape(batch_size, ems_num, -1)
    ems_mask = obs.ems_mask.reshape(batch_size, ems_num)
    
    ems_size_mask = obs.ems_size_mask.reshape(batch_size, ems_num, box_state_num)
    ems_to_box_mask = obs.ems_to_box_mask.reshape(batch_size, corner_num, ems_num, box_state_num)

    if len(obs.precedence[0]) > 1:
        # NOTE prec
        prec_dim = 2
        prec_states = obs.precedence.reshape(batch_size, box_state_num, box_num, prec_dim)
        prec_states = to_float(prec_states)
    else:
        prec_states = None
    
    pre_box = to_float(pre_box)
    heightmap = to_float(heightmap)
        
    box_states = to_float(box_states)
    valid_mask = ensure_var(valid_mask, dtype=jt.bool)
    access_mask = ensure_var(access_mask, dtype=jt.bool)
    ems = to_float(ems)
    ems_mask = ensure_var(ems_mask, dtype=jt.bool)
    ems_size_mask = ensure_var(ems_size_mask, dtype=jt.bool)
    ems_to_box_mask = ensure_var(ems_to_box_mask, dtype=jt.bool)
    
    ems_mask = ems_mask.unsqueeze(1).unsqueeze(2)

    return ems_num, box_state_num, box_states, prec_states, valid_mask, access_mask, ems, ems_mask, ems_size_mask, ems_to_box_mask, pre_box, heightmap


class RnnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RnnEncoder, self).__init__()

        self.gru = nn.GRU( input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        
        self.drop_hh = nn.Dropout(p=dropout)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

    def execute(self, data):
        # batch_size x data_num x dim_num
        # output  batch_size x (hidden_size) x dim_num

        # encoder_input  batch_size x data_num x dim_num
        batch_size = data.shape[0]
        data_num = data.shape[1]
        dim_num = data.shape[2]

        outputs = []
        
        for dim_index in range(dim_num):            
            # dim_input  batch_size x data_num x input_size(1)
            dim_input = data[:,:,dim_index:dim_index+1]
            last_hh = None
            rnn_out, last_hh = self.gru(dim_input, last_hh)
            
            if self.num_layers == 1:
                # If > 1 layer dropout is already applied
                last_hh = self.drop_hh(last_hh)
            outputs.append(last_hh[-1].unsqueeze(-1))
            
        # output  batch_size x hidden_size x dim_num
        return cat(outputs, dim=2)

    def forward(self, data):
        return self.execute(data)
        


class PrecedenceModule(nn.Module):
    def __init__(self, prec_dim, embed_size, prec_type, device, heads=4, dropout=0 ):
        super(PrecedenceModule, self).__init__()
        self.prec_type = prec_type
        self.prec_dim = prec_dim
        self.device = device

        if prec_type == 'attn':
            # self.position_embedding = nn.Embedding(1000, embed_size)
            self.prec_embedding = nn.Linear(prec_dim, embed_size)
            self.attn = TransformerBlock(embed_size, heads, dropout, 4)
        elif prec_type == 'cnn':
            self.prec_embedding = nn.Conv1d(prec_dim, embed_size, kernel_size=1)
        elif prec_type == 'rnn':
            self.prec_embedding = RnnEncoder(1, embed_size, 1, 0.1, device)

    def execute(self, precedence, top_mask):
        '''
            prec_states: [batch x (rot * axis * box_num)  x box_num x 2]
            top_mask: [batch x box_num]  a valid mask, if box *already* packed, no compute in atten
            
            ret: [batch x (rot * axis * box_num) x embed_size]
        '''
        batch_size, state_num, box_num, _ = precedence.shape
        mask = top_mask.unsqueeze(1).unsqueeze(2)

        if self.prec_type == 'attn':
            prec_vecs = self.prec_embedding(precedence)
            
            bs_mask = mask.expand(batch_size, state_num, 1, box_num).reshape(batch_size * state_num, 1, 1, box_num)
            values = prec_vecs.reshape(batch_size * state_num, box_num, -1)
            query_list = []
            for i in range(state_num):
                query_list.append(prec_vecs[:, i:i + 1, i % box_num, :])
            query = cat(query_list, dim=1).reshape(batch_size * state_num, 1, -1)

            ret = self.attn(values, values, query, bs_mask)
            ret = ret.reshape(batch_size, state_num, -1)
            
            # NOTE old way to compute >> 

            # ret = []
            # for i in range(state_num):
            #     prec_vec = prec_vecs[:,i]
            #     # positions = torch.arange(0, box_num).expand(batch_size, box_num).to(self.device)
            #     # emb_pos = self.position_embedding(positions)
            #     # prec_vec = prec_vec + emb_pos
            #     prec_i = i % box_num
            #     prec_vec = self.attn(prec_vec, prec_vec, prec_vec[:, prec_i:prec_i+1], mask )
            #     ret.append(prec_vec)
            # ret = torch.cat(ret, dim=1)

        elif self.prec_type == 'cnn' or self.prec_type == 'rnn':
            prec_vecs = precedence.reshape(batch_size, state_num, -1)
            prec_vecs = prec_vecs.transpose((0, 2, 1))
            ret = self.prec_embedding( prec_vecs ).transpose((0, 2, 1))

        return ret

    def forward(self, precedence, top_mask):
        return self.execute(precedence, top_mask)


class HeightmapEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, map_size):
        super(HeightmapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, int(hidden_size/4), stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(int(hidden_size/4), int(hidden_size/2), stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(int(hidden_size/2), int(hidden_size), kernel_size=( math.ceil(map_size[0]/4), math.ceil(map_size[1]/4) ) )

    def execute(self, input):
        output = leaky_relu(self.conv1(input))
        output = leaky_relu(self.conv2(output))
        output = self.conv3(output).squeeze(-1)
        return output  # (batch, hidden_size, seq_len)

    def forward(self, input):
        return self.execute(input)

class SpaceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, corner_num=1) -> None:
        super(SpaceEncoder, self).__init__()

        self.corner_num = corner_num

        if input_dim == 100:
            input_dim -= 1

            type_dim = int(hidden_dim / 4)
            self.ems_type_embedding = nn.Linear(1, type_dim)
            self.ems_merge = nn.Linear(hidden_dim + type_dim, hidden_dim)

            if corner_num > 1:
                input_dim += 1
            
            self.ems_embedding = nn.Sequential(
                # nn.Linear(input_dim, input_dim-1),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.ems_type_embedding = None

            if corner_num > 1:
                input_dim += 1
            
            self.ems_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def embed(self, ems_data):
        if self.ems_type_embedding is None:
            ret = self.ems_embedding(ems_data)
        else:
            type_vec = self.ems_type_embedding(ems_data[:, :, 6:7])
            ems_vec = self.ems_embedding(ems_data[:, :, :6])
            ret = self.ems_merge(cat([ems_vec, type_vec], dim=-1))
        return ret

    def execute(self, ems):
        ems_vecs = []
        for ems_i in range(self.corner_num):
            # pos | size
            ems_in = ems.clone()
            # ems_size = ems_in[:, :, 3:6]
            
            
            if self.corner_num == 1:
                ems_vec = self.embed(ems_in)
            else:
                ems_ids = jt.zeros_like(ems[:, :, :1]) + ems_i
                # if ems_i == 1:
                #     ems_in[:,:,0] += ems_size[:,:,0]
                # elif ems_i == 2:
                #     ems_in[:,:,1] += ems_size[:,:,1]
                # elif ems_i == 3:
                #     ems_in[:,:,0] += ems_size[:,:,0]
                #     ems_in[:,:,1] += ems_size[:,:,1]
                # elif ems_i == 2:
                #     ems_in[:,:,2] += (ems_size[:,:,2] - 1)
                # elif ems_i == 3:
                #     ems_in[:,:,0] += (ems_size[:,:,0] - 1)
                #     ems_in[:,:,2] += (ems_size[:,:,2] - 1)
                ems_vec = self.embed(cat([ems_in, ems_ids], dim=-1))
            ems_vecs.append(ems_vec)

        return ems_vecs

    def forward(self, ems):
        return self.execute(ems)

class ObjectEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, prec_dim=2, prec_type='none', \
                 device="cuda:0"):
        super(ObjectEncoder, self).__init__()

        self.prec_type = prec_type
        self.device = device

        if self.prec_type == 'none':
            self.box_embedding = nn.Linear(input_dim, hidden_dim)
            self.prec_embeding = None
        else:
            half_dim = int(hidden_dim/2)
            self.box_embedding = nn.Linear(input_dim, half_dim)
            self.prec_embeding = PrecedenceModule(prec_dim, half_dim, prec_type, device=device)
            # elif self.prec_type == 'rnn':
            #     self.prec_embeding = RnnEncoder(1, half_size, 1, 0.1)
            # elif self.prec_type == 'cnn':
            #     self.prec_embeding = Encoder(box_num * 3, half_size)

    def execute(self, box_states, prec_states=None, valid_mask=None):
        '''
        box_states:  [batch x (rot * axis * box_num)  x 3]
        prec_states: [batch x (rot * axis * box_num)  x box_num x 2]
        valid_mask:  [batch x (rot * axis * box_num)]
        '''

        if self.prec_embeding is None:
            box_vecs = self.box_embedding(box_states)
        else:
            # precedences = precedences.clone()
            batch_size, state_num, box_num, _ = prec_states.shape
            if valid_mask is None:
                prec_mask = ones((batch_size, box_num), dtype=jt.float32)
            else:
                # NOTE prec_mask, the object which are not in remove_list, we only need the move prec, for attn
                prec_mask = valid_mask.reshape(batch_size, 2, -1, box_num )[:,0, 0]

            box_vec = self.box_embedding( box_states )
            prec_vec = self.prec_embeding( prec_states, prec_mask )
            # prec_vec = self.prec_embeding( prec_states ).transpose(1,2) # cnn, rnn
            box_vecs = cat([box_vec, prec_vec], dim=2)

        return box_vecs

    def forward(self, box_states, prec_states=None, valid_mask=None):
        return self.execute(box_states, prec_states, valid_mask)


