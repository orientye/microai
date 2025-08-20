import numpy as np
from microai import Variable
import microai.funcs as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
print(y)
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)  # y = x.T
print(y)
y.backward()
print(x.grad)
