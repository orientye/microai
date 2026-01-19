import numpy as np
import microai.conv as conv


# im2col
x1 = np.random.rand(1, 3, 7, 7)
col1 = conv.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)  # (9, 75)

