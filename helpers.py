import numpy as np

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