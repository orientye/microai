if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from microai import Variable


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)