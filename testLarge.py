from stableState import findStableState as fss
from tensionOptimizer import findOptimalTension as fot
from tensionOptimizer import getObjectiveFn as gof
from spectral import estimateOptimalTension as eot
from helpers import createGrid

import numpy as np
import sys
import time


# make a 25 x 25 grid
L, boundaries = createGrid(25)
start = time.time()
stableState = fss(L, boundaries)
end = time.time()
print(stableState[301])
print("Stable State found in time " + str(end - start) + "\n")
sys.stdout.flush()
start = time.time()
optTension = fot(L, boundaries, 301, 300, 1.0)
end = time.time()
print(optTension)
print("Tension found in time " + str(end - start))
sys.stdout.flush()
tempB = boundaries[:]
tempB.append((301, optTension))
obVal = gof(L, tempB, 300)
print("Value of tension: " + str(obVal) + "\n")
sys.stdout.flush()
start = time.time()
estOptTension = eot(L, boundaries, 301, 300, 1.0)
end = time.time()
print("Estimated tension in time " + str(end - start) + "\n")
sys.stdout.flush()
tempB = boundaries[:]
tempB.append((301, estOptTension))
obVal = gof(L, tempB, 300)
print("Value of tension: " + str(obVal) + "\n")
tempB = boundaries[:]
tempB.append((301, stableState[301]))
obVal = gof(L, tempB, 300)
print("Value of tension: " + str(obVal) + "\n")

sys.exit(0)

# do a 10 x 10 grid
L, boundaries = createGrid(10)
start = time.time()
stableState = fss(L, boundaries)
end = time.time()
print(stableState)
print("Found in time " + str(end - start) + "\n")
sys.stdout.flush()
start = time.time()
optTension = fot(L, boundaries, 55, 50, 1.0)
end = time.time()
print(optTension)
print("Found in time " + str(end - start) + "\n")
sys.stdout.flush()

"""
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
start = time.time()
stableState = fss(L, boundaries)
end = time.time()
print(stableState)
print("Found in time " + str(end - start) + "\n")
sys.stdout.flush()
start = time.time()
optTension = fot(L, boundaries, 4, 5, .5)
end = time.time()
print(optTension)
print("Found in time " + str(end - start) + "\n")
"""
