import jittor as jt
from jittor import nn

from tapnet.model_backend import (
    arange,
    bmm,
    cat,
    masked_fill,
    self_attention_context,
    self_attention_scores,
    sigmoid,
    softmax,
    xavier_uniform_,
)


#----------------------------------------------
# Normal Transformer
#----------------------------------------------


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def execute(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = self_attention_scores(queries, keys)
        # query: N, query_len, heads, heads_dim
        # keys: N, key_len, heads, heads_dim
        # energy: N, heads, query_len, key_len

        if mask is not None:
            energy = masked_fill(energy, mask == 0, float("-inf"))
        
        attention = softmax(energy / (self.embed_size ** (1/2.)), dim=3)
        out = self_attention_context(attention, values).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention: N, heads, query_len, key_len
        # value: N, value_len, heads, heads_dim
        # out: N, query_len, heads, heads_dim
        out = self.fc_out(out)
        return out

    def forward(self, values, keys, query, mask):
        return self.execute(values, keys, query, mask)

class SelfAttention_cat(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention_cat, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # TODO multi head
        self.v = nn.Parameter(jt.zeros((1, 1, embed_size), dtype=jt.float32), requires_grad=True)
        self.W = nn.Parameter(jt.zeros((1, embed_size, embed_size + embed_size), dtype=jt.float32), requires_grad=True)
        
        # NOTE must init !! or i will be zero
        xavier_uniform_(self.v)
        xavier_uniform_(self.W)
        # self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def execute(self, values, keys, query, mask):
        
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        if query_len == 1:
            query = query.expand(-1, key_len, -1) # expand_as(keys)
            
        # query = query.expand(-1, key_len, -1) # expand_as(keys)
        hidden = cat((keys, query), dim=2) # N, key_len, hiddem_dim*2

        v = self.v.expand(N, query_len, -1) # N, 1, hidden
        W = self.W.expand(N, -1, -1) # N, hidden, hidden*2
        energy = bmm(v, jt.tanh(bmm(W, hidden.transpose((0, 2, 1))))) # N, query_len, key_len

        if mask is not None:
            # mask: N, 1, 1, key_len
            mask = mask.squeeze(1)
        
            energy = masked_fill(energy, mask == 0, float("-inf"))

        attns = softmax(energy, dim=2)  # (N, query_len, key_len)
        out = bmm(attns, values)  # (N, query_len, hidden)

        # query: N, query_len, hidden_dim
        # keys: N, key_len, hidden_dim
        # values: N, key_len, hidden_dim
        # energy: N, query_len, key_len
        
        # out: N, query_len, hidden_dim
        return out

    def forward(self, values, keys, query, mask):
        return self.execute(values, keys, query, mask)

class AttnModule(nn.Module):
    def __init__(self, embed_size, heads, dropout, device, max_length, merge_flag=False ):
        super(AttnModule, self).__init__()
        self.device = device
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)
        
        self.attention = SelfAttention(embed_size, heads)
        self.merge_flag = merge_flag
        if merge_flag:
            self.ff = nn.Linear(embed_size+1, embed_size)

    def execute(self, x, mask=None, pos_embed=True, query_index=None):
        # x: batch x data_len x embed_size
        if self.merge_flag:
            x = self.ff(x)
        
        if pos_embed:
            N, seq_length, _ = x.shape

            # seq_length = 6 * box_num

            # rot_num = 6
            # box_num = int(seq_length / rot_num)
            
            # positions = torch.arange(0, box_num).expand(N, box_num).to(self.device)
            # emb_pos = self.position_embedding(positions)

            # x = x.reshape(N, rot_num, box_num, -1)
            # for rot in range(rot_num):
            #     x[:,rot] += emb_pos[:, rot:rot+1]
            # x = x.reshape(N, seq_length, -1)
            # x = self.dropout( x )

            positions = arange(0, seq_length).unsqueeze(0).expand(N, seq_length)
            emb_pos = self.position_embedding(positions)
            x = self.dropout( x + emb_pos )

        # x_vecs:   batch x ems_len x embed_size
        if query_index is not None:
            x_vecs = self.attention(x, x, x[:,query_index:query_index+1], mask)
            # x_vecs = self.norm(x_vecs)
            x_vecs = self.dropout(self.norm(x_vecs + x[:,query_index:query_index+1]))
        else:
            x_vecs = self.attention(x, x, x, mask)
            x_vecs = self.dropout(self.norm(x_vecs + x))
        return x_vecs

    def forward(self, x, mask=None, pos_embed=True, query_index=None):
        return self.execute(x, mask, pos_embed, query_index)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def execute(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

    def forward(self, value, key, query, mask):
        return self.execute(value, key, query, mask)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_size, 
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        merge_flag = False
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.merge_flag = merge_flag
        if merge_flag:
            self.ff = nn.Linear(embed_size+1, embed_size)
    
    def execute(self, x, mask, extra_len=None, pos_embed=True):
        # x: batch x seq_length x embed_size
        if self.merge_flag:
            x = self.ff(x)

        N, seq_length, embed_size = x.shape
        
        if pos_embed:
            if extra_len is not None:
                positions = arange(0, extra_len + 1).unsqueeze(0).expand(N, extra_len + 1)
                pos = self.position_embedding(positions)
                prefix = x[:, :extra_len] + pos[:, :extra_len]
                suffix = x[:, extra_len:] + pos[:, extra_len:extra_len + 1].expand(N, seq_length - extra_len, embed_size)
                x = cat([prefix, suffix], dim=1)

            else:
                positions = arange(0, seq_length).unsqueeze(0).expand(N, seq_length)
                pos = self.position_embedding(positions)
                x = x + pos

        out = self.dropout( x )
        # out = x
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out 

    def forward(self, x, mask, extra_len=None, pos_embed=True):
        return self.execute(x, mask, extra_len, pos_embed)

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

    def forward(self, x, value, key, src_mask, trg_mask):
        return self.execute(x, value, key, src_mask, trg_mask)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length ):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
         for _ in range(num_layers) ])

        self.dropout = nn.Dropout(dropout)
    
    def execute(self, x, enc_out, src_mask, trg_mask):
        N, seq_length, _ = x.shape
        positions = arange(0, seq_length).unsqueeze(0).expand(N, seq_length)
        emb_pos = self.position_embedding(positions)
        x = self.dropout( x + emb_pos )

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        return x

    def forward(self, x, enc_out, src_mask, trg_mask):
        return self.execute(x, enc_out, src_mask, trg_mask)

class Transformer(nn.Module):
    def __init__(
        self,
        embed_size=256,
        num_layers=4,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device='cuda:0',
        max_length=300
    ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = TransformerDecoder(embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)

        self.device = device

    def make_trg_mask(self, N, trg_len):
        # trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        trg_mask = jt.ones((trg_len, trg_len), dtype=jt.float32).expand(N, 1, trg_len, trg_len)

        return trg_mask

    def execute(self, extra_src, src, trg, extra_src_mask, src_mask, trg_mask=None):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        if trg_mask is None:
            trg_mask = self.make_trg_mask(batch_size, trg_len)
        else:
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)

        extra_len = extra_src.shape[1]
        concat_src = cat([extra_src, src], dim=1)
        concat_src_mask = cat([extra_src_mask, src_mask], dim=-1)

        enc_src = self.encoder(concat_src, concat_src_mask, extra_len)
        enc_src = enc_src[:, extra_len:] # we don't need extra data for next input

        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

    def forward(self, extra_src, src, trg, extra_src_mask, src_mask, trg_mask=None):
        return self.execute(extra_src, src, trg, extra_src_mask, src_mask, trg_mask)


#----------------------------------------------
# Spectial for TAP-Net
#----------------------------------------------

class CrossLayer(nn.Module):
    def __init__(self, embed_size=0 ):
        super(CrossLayer, self).__init__()
        self.key_encoder = None
        self.query_encoder = None    
        
        if embed_size > 0:
            self.key_encoder = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
            )
            self.query_encoder = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
            )
        
    def execute(self, keys, queries, queries_to_keys_mask = None):
        # keys:     batch x box_len x embed_size
        # queries:  batch x ems_len x embed_size
        if self.key_encoder is not None:
            keys = self.key_encoder(keys)
            qs = []
            for q in queries:
                qs.append(self.query_encoder(q))
        else:
            qs = queries
            
        energy = []
        for query in qs:
            score = jt.matmul(query, keys.transpose((0, 2, 1)))
            energy.append(score.unsqueeze(1))
        energy = cat(energy, dim=1)
        
        if queries_to_keys_mask is not None:
            for i in range(energy.shape[1]):
                energy[:, i, :, :] = masked_fill(energy[:, i, :, :], queries_to_keys_mask == 0, float("-inf"))
        return energy

    def forward(self, keys, queries, queries_to_keys_mask = None):
        return self.execute(keys, queries, queries_to_keys_mask)

class CrossTransformer(nn.Module):
    def __init__(
        self,
        embed_size=256,
        heads=8,
        dropout=0,
        mask_predict=False,
        device="cuda:0",
        max_length=300,
        corner_num=1
    ):
        super(CrossTransformer, self).__init__()

        # self.encoder = TransformerEncoder(embed_size, 2, heads, device, 4, dropout, max_length)
        # self.decoder = TransformerEncoder(embed_size, 1, heads, device, 4, dropout, max_length)

        # self.decoder = AttnModule(embed_size, heads, dropout, device, max_length, merge_flag = False)

        # self.attn = TransformerEncoder(embed_size, 2, heads, device, 4, dropout, max_length, merge_flag = True)
        self.corner_num = corner_num
        self.corner_num = 1
        self.attn = None
        # self.attn = AttnModule(embed_size, heads, dropout, device, max_length, True)

        if self.attn is None:
            if self.corner_num > 1:
                self.id_encoder = nn.Linear(embed_size + 1, embed_size)

            self.encoder = AttnModule(embed_size, heads, dropout, device, max_length, False)
            self.decoder = AttnModule(embed_size, heads, dropout, device, max_length, False)

        self.cross = CrossLayer()

        if mask_predict:
            self.mask_predictor = CrossLayer(embed_size)
        else:
            self.mask_predictor = None

        self.device = device

    def make_trg_mask(self, N, trg_len):
        # trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        trg_mask = jt.ones((trg_len, trg_len), dtype=jt.float32).expand(N, 1, trg_len, trg_len)
        return trg_mask

    def execute(self, srcs, trg, src_mask, trg_mask, src_to_trg_mask):
        """
        srcs: List[tensor], ems features
        trg: tensor, box features
        src_mask: tensor-bool, mask for ems
        trg_mask: tensor-bool, mask for box
        src_to_trg_mask: tensor-bool, mask of <ems, box>
        """
        
        batch_size = trg.shape[0]
        
        trg_len = trg.shape[1]

        # batch x 1 x 1 x trg_len
        if trg_mask is None:
            trg_mask = self.make_trg_mask(batch_size, trg_len)
        else:
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)

        key_id = jt.zeros_like(trg[:, :, :1])
        query_id = jt.ones_like(srcs[0][:, :, :1])
        # if len(srcs) > 1:
        # else:
        #     query_id = torch.zeros_like(srcs[0][:,:,:1])
        #     key_id = torch.ones_like(trg[:,:,:1])

        queries = []
        i = 0
        for src in srcs:
            if self.attn is None:
                if self.corner_num > 1:
                    input = cat([src, query_id + i], dim=-1)
                    input_src = self.id_encoder(input)
                else:
                    input_src = src
                query = self.encoder( input_src, src_mask, pos_embed=False)
            else:
                input_src = cat([src, query_id + i], dim=-1)
                query = self.attn( input_src, src_mask, pos_embed=False)

            queries.append(query)
            i += 1

        if self.attn is None:
            keys = self.decoder(trg, trg_mask, pos_embed=False)
        else:
            keys = self.attn(cat([trg, key_id], dim=-1), trg_mask, pos_embed=True)

        energy = self.cross(keys, queries)

        if self.mask_predictor is not None:
            predict_mask = self.mask_predictor(keys, queries)
            predict_mask = sigmoid(predict_mask) # to 0~1
            energy = energy * predict_mask

        if src_to_trg_mask is not None:
            for i in range(energy.shape[1]):
                energy[:, i, :, :] = masked_fill(energy[:, i, :, :], src_to_trg_mask[:, i] == 0, float("-inf"))

        return energy

    def forward(self, srcs, trg, src_mask, trg_mask, src_to_trg_mask):
        return self.execute(srcs, trg, src_mask, trg_mask, src_to_trg_mask)
