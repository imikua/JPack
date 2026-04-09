import sys
import math

import jittor as jt
import jittor.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, sequence_len, embedding_size):
        super(PositionalEncoder, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size

        pos = jt.arange(0, sequence_len, 1).float32().unsqueeze(dim=1)
        div_term = jt.exp(jt.arange(0, embedding_size, 2).float32() / embedding_size * (-math.log(10000.0)))

        PE = jt.zeros((sequence_len, embedding_size))
        PE[:, 0::2] = jt.sin(pos*div_term)
        PE[:, 1::2] = jt.cos(pos*div_term)
        PE = PE.unsqueeze(dim=0)

        # register_buffer replacement: store as a non-parameter member
        self.PE = PE.stop_grad()

    def execute(self, x,):
        x = x + self.PE[:, :x.shape[1]]
        return x
