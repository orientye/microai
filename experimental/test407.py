import numpy as np
import microai.layers as L

rnn = L.RNN(10)
x = np.random.rand(1 , 1)
h = rnn(x)
print(h.shape)
