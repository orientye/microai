import numpy as np
import microai.funcs as F
from microai.models import MLP

model = MLP((10, 3))

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
p = F.softmax(y)
print(y)
print(p)

loss = F.softmax_cross_entropy(y, t)
loss.backward()
print(loss)
