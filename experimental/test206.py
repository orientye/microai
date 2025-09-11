import numpy as np

from microai import Variable
import microai.funcs as F

x = Variable(np.random.randn(2, 3))
w = Variable(np.random.randn(3, 4))
print(x)
print(w)
y = F.matmul(x, w)
print(y)
y.backward()
print(x.grad.shape)
print(w.grad.shape)
