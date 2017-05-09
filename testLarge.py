from stableState import findStableState as fss
from tensionOptimizer import findOptimalTension as fot
from tensionOptimizer import getObjectiveFn as gof
from spectral import estimateOptimalTension as eot

import numpy as np
import sys
import time

def createGrid(n):
	"""Creates an nxn grid to run testing on
	returns a tuple containing the Laplacian and the boundary conditions
	assumes that the four corners are held in place at (0, 0), (0, n-1), (n-1, 0), and (n-1, n-1)
	n should be larger than 3--for smaller graphs, make it manually"""
	L = np.identity(n * n, dtype = float) * 4
	# deal with the corners
	L[0][0] = 2
	L[0][1] = -1
	L[0][n] = -1
	L[n-1][n-1] = 2
	L[n-1][n-2] = -1
	L[n-1][(2 * n) - 1] = -1
	L[(n * n) - n][(n * n) - n] = 2
	L[(n * n) - n][(n * n) - n + 1] = -1
	L[(n * n) - n][(n * n) - (2 * n)] = -1
	L[(n * n) - 1][(n * n) - 1] = 2
	L[(n * n) - 1][(n * n) - 2] = -1
	L[(n * n) - 1][(n * n) - n - 1] = -1
	# deal with remainder of top row
	for v in range(1, n - 1):
		L[v][v] = 3
		L[v][v + 1] = -1
		L[v][v - 1] = -1
		L[v][v + n] = -1
	# deal with remainder of bottom row
	for v in range((n * n) - n + 1, (n * n) - 1):
		L[v][v] = 3
		L[v][v + 1] = -1
		L[v][v - 1] = -1
		L[v][v - n] = -1
	# deal with all remaining vertices
	for v in range(n, (n * n) - n):
		if v % n == 0:
			# on the left edge
			L[v][v] = 3
			L[v][v + 1] = -1
		elif v % n == n - 1:
			# on right edge
			L[v][v] = 3
			L[v][v - 1] = -1
		else:
			L[v][v + 1] = -1
			L[v][v - 1] = -1
		L[v][v + n] = -1
		L[v][v - n] = -1
	boundaries = [(0, np.array([0, 0, 0])), (n - 1, np.array([n - 1, 0, 0])), ((n * n) - n, np.array([0, n - 1, 0])), ((n * n) - 1, np.array([n, n, 0]))]
	return (L, boundaries)


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
