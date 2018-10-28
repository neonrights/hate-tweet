import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import *
from custom import *


class Transformer(nn.Module):
    def __init__(self, layers, heads, embed_dim, dropout=None):
        super().__init__()
        self.pe = PositionalEncoder()
        self.encoding_layers = [EncoderLayer(embed_dim, heads, dropout) for _ in range(layers)]
    
    def forward(self, inputs):
        outputs = self.pe(inputs)
        for encoding_layer in self.encoding_layers:
            outputs = encoding_layer(outputs)

        return outputs


if __name__ == '__main__':
    print("performing smoke test for model")
    test_input = torch.rand((3,4,5))
    ilugp = ILUGP(12, 1, 5)
    test_output = ilugp(test_input)
    assert test_output.shape == (3,4,5)
    print("smoke test passed")


