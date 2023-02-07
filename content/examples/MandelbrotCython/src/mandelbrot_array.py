# %%
from array import array
import numpy as np


# %%
def linspace(start, stop, num):
    step = (stop - start) / (num - 1) if num > 1 else 0
    return [start + i * step for i in range(num)]


# %%
assert linspace(-2, 2, 5) == list(np.linspace(-2, 2, 5))


# %%
assert linspace(0, 1, 10) == list(np.linspace(0, 1, 10))

# %%
