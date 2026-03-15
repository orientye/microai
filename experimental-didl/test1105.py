# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html#batch-matrix-multiplication

import torch

# https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html#sec-rnn-scratch
def check_len(a, n):
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'


Q = torch.ones((2, 3, 4))
K = torch.ones((2, 4, 6))
check_shape(torch.bmm(Q, K), (2, 3, 6))