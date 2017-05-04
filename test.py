from stableState import findStableState as fss
from tensionOptimizer import findOptimalTension as fot

import numpy as np
import sys

# test on a simple five-vertex chain
L = np.array(
	[[1, -1, 0, 0, 0],
	[-1, 2, -1, 0, 0],
	[0, -1, 2, -1, 0],
	[0, 0, -1, 2, -1],
	[0, 0, 0, -1, 1]], dtype = float)
boundaries = [(0, np.array([2, 0, 4])), (2, np.array([4, 5, 0])), (4, np.array([0, 0, 3]))]
stableState = fss(L, boundaries)
print(stableState)

# test on a 3x3 grid
L = np.array(
	[[2, -1, 0, -1, 0, 0, 0, 0, 0],
	[-1, 3, -1, 0, -1, 0, 0, 0, 0],
	[0, -1, 2, 0, 0, -1, 0, 0, 0],
	[-1, 0, 0, 3, -1, 0, -1, 0, 0],
	[0, -1, 0, -1, 4, -1, 0, -1, 0],
	[0, 0, -1, 0, -1, 3, 0, 0, -1],
	[0, 0, 0, -1, 0, 0, 2, -1, 0],
	[0, 0, 0, 0, -1, 0, -1, 3, -1],
	[0, 0, 0, 0, 0, -1, 0, -1, 2]], dtype = float)
boundaries = [(0, np.array([0, 0, 0])), (2, np.array([2, 0, 0])), (6, np.array([0, 2, 0])), (8, np.array([2, 2, 0]))]
stableState = fss(L, boundaries)
print(stableState)
sys.stdout.flush()
optTension = fot(L, boundaries, 4, 5, .5)
print(optTension)

