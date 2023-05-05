import numpy as np
l=100
print(np.arange(0, l, 1))
print(np.random.randint(0, 1, 1))
a = np.setdiff1d(np.arange(0, 10, 1), [1,5,6])
print(np.tensordot(a,a, axes=0))