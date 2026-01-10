import numpy as np
from microai import test_mode
import microai.funcs as F

x = np.ones(5)
print(x)

# When training
y = F.dropout(x)
print(y)

# When testing
with test_mode():
    y = F.dropout(x)
    print(y)

