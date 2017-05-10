import numpy as np

def findPotential(L, boundaryConditions, Minv = None):
	"""Returns the potential stored in the system given a Laplacian and boundary conditions
	Note that boundaryConditions should also include the tensioning point"""
	X = findStableState(L, boundaryConditions, Minv)
	return np.trace(X.T.dot(L).dot(X))

def findMinv(L, boundaryConditions):
	"""Given the Laplacian of a graph and its boundary conditions, returns the inverse of the M matrix
	Calling this function once then using Minv many times should be faster than repeated calculations"""
	n = L.shape[0]
	m = len(boundaryConditions)
	Vb = np.zeros(m)
	for i in range(m):
		condition = boundaryConditions[i]
		Vb[i] = condition[0]
	Vb = np.sort(Vb)
	BPrime = np.zeros((m, n))
	for i in range(m):
		BPrime[i][int(Vb[i])] = 1

	zeroCorner = np.zeros((m, m))
	M = np.array(np.bmat([[L, -BPrime.T], [BPrime, zeroCorner]]))
	Minv = np.linalg.inv(M)
	return Minv

def findStableState(L, boundaryConditions, Minv = None):
	"""Given the Laplacian of a graph and boundary conditions, returns an nx3 matrix of stable state positions
		L should be an the nxn Laplacian of the graph (a numpy array)
		boundaryConditions should be a list of tuples, where each tuple contains the index of the vertex 
			to be constrained (0 indexed) and a length-3 numpy array for the position it should be in
		Minv should be the inverse of the M matrix (for saving on computation time)
			if it is left as None, this function will compute Minv itself."""
	n = L.shape[0]
	m = len(boundaryConditions)
	Vb = np.zeros(m)
	positions = {}
	for i in range(m):
		condition = boundaryConditions[i]
		Vb[i] = condition[0]
		positions[condition[0]] = condition[1]
	Vb = np.sort(Vb)
	BPrime = np.zeros((m, n))
	YPrime = np.zeros((m, 3))
	for i in range(m):
		BPrime[i][int(Vb[i])] = 1
		YPrime[i] = positions[Vb[i]]

	if Minv is None:
		zeroCorner = np.zeros((m, m))
		M = np.array(np.bmat([[L, -BPrime.T], [BPrime, zeroCorner]]))
		Minv = np.linalg.inv(M)

	XT = np.zeros((3, n))
	# find x coordinates
	y = np.zeros(n + m)
	y[n:] = YPrime.T[0]
	x = np.dot(Minv, y)
	XT[0] = x[:n]
	# find y coordinates
	y = np.zeros(n + m)
	y[n:] = YPrime.T[1]
	x = np.dot(Minv, y)
	XT[1] = x[:n]
	# find z coordinates
	y = np.zeros(n + m)
	y[n:] = YPrime.T[2]
	x = np.dot(Minv, y)
	XT[2] = x[:n]

	return XT.T
