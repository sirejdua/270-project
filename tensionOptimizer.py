import numpy as np
from stableState import findStableState as fss
from stableState import findMinv as fmi

import matplotlib.pyplot as plt

def findOptimalTension(L, boundaryConditions, tensionPoint, cutPoint, tensionRadius):
	"""Uses gradient descent to find the best way of tensioning the given point
		L should be an the nxn Laplacian of the graph (a numpy array)
		boundaryConditions should be a list of tuples, where each tuple contains the index of the vertex 
			to be constrained (0 indexed) and a length-3 numpy array for the position it should be in
		tensionPoint is the index of the point that we're allowed to tension
		cutPoint is the index of the point that will be cut
		tensionRadius is the radius of the ball of possible tensioning locations

	Returns a length-3 numpy array with the x, y, and z coordinates of the optimal position for the tensioning point."""

	def findLPrime():
		"""Returns L', which is the Laplacian for the post-cut graph"""
		LPrime = np.copy(L)
		LPrime += np.diag(LPrime[cutPoint]) # get rid of any edges from the cut point to any other point
		LPrime = np.delete(LPrime, cutPoint, axis = 0) # get rid of the cut point's row
		LPrime = np.delete(LPrime, cutPoint, axis = 1) # get rid of the cut point's column
		return LPrime

	def objectiveFn(tensionLocation, Minv = None, MinvPrime = None):
		"""Given a location to tension the tension point to, calculates the objective function Tr(X^TLX) - Tr(X'^TL'X')"""
		newBoundaries = boundaryConditions[:] # only need a shallow copy
		newBoundaries.append((tensionPoint, tensionLocation))
		X = fss(L, newBoundaries, Minv)
		LPrime = findLPrime()
		# all indices in L' larger than cut point are decreased by 1 b/c we removed cut point
		for i in range(len(newBoundaries)):
			if newBoundaries[i][0] > cutPoint:
				newBoundaries[i] = (newBoundaries[i][0] - 1, newBoundaries[i][1])
		XPrime = fss(LPrime, newBoundaries, MinvPrime)
		return np.trace(X.T.dot(L).dot(X)) - np.trace(XPrime.T.dot(LPrime).dot(XPrime))

	def estimateGradient(tensionLocation, Minv = None, MinvPrime = None, epsilon = 10**(-3)):
		"""Finds a numeric approximation for the gradient with respect to where we move our tensioning point
			tensionLocation is the current position we put our tensioning point at
			Minv is the inverse M matrix for the pre-cut graph
			MinvPrime is the inverse M matrix for the post-cut graph
			epsilon is the distance on either side of the current position that we look at to estimate the gradient

		Returns a length-3 numpy array with the estimates for the gradient in the x, y, and z directions"""
		# estimate gradient in x direction
		tensionLocation[0] += epsilon
		upper = objectiveFn(tensionLocation, Minv, MinvPrime)
		tensionLocation[0] -= 2 * epsilon
		lower = objectiveFn(tensionLocation, Minv, MinvPrime)
		tensionLocation[0] += epsilon
		gradx = (upper - lower) / (2.0 * epsilon)

		# estimate gradient in y direction
		tensionLocation[1] += epsilon
		upper = objectiveFn(tensionLocation, Minv, MinvPrime)
		tensionLocation[1] -= 2 * epsilon
		lower = objectiveFn(tensionLocation, Minv, MinvPrime)
		tensionLocation[1] += epsilon
		grady = (upper - lower) / (2.0 * epsilon)

		# estimate gradient in z direction
		tensionLocation[2] += epsilon
		upper = objectiveFn(tensionLocation, Minv, MinvPrime)
		tensionLocation[2] -= 2 * epsilon
		lower = objectiveFn(tensionLocation, Minv, MinvPrime)
		tensionLocation[2] += epsilon
		gradz = (upper - lower) / (2.0 * epsilon)

		return np.array([gradx, grady, gradz])

	restingPositions = fss(L, boundaryConditions)
	tensionRest = restingPositions[tensionPoint]

	tempBC = boundaryConditions[:]
	tempBC.append((tensionPoint, tensionRest))
	Minv = fmi(L, tempBC)
	LPrime = findLPrime()
	# all indices in L' larger than cut point are decreased by 1 b/c we removed cut point
	for i in range(len(tempBC)):
			if tempBC[i][0] > cutPoint:
				tempBC[i] = (tempBC[i][0] - 1, tempBC[i][1])
	MinvPrime = fmi(LPrime, tempBC)

	gradient = estimateGradient(tensionRest, Minv, MinvPrime)
	tensionLocation = np.copy(tensionRest)
	numIterations = 0
	while numIterations < 3000 and np.linalg.norm(gradient) > 10**(-7):
		tensionLocation += (10**(-1)) * gradient # gradient ascent, not descent
		if np.linalg.norm(tensionLocation - tensionRest) > tensionRadius:
			# might have to rescale the tensioning so it doesn't tension too far
			diff = tensionLocation - tensionRest
			diff = diff * (tensionRadius / np.linalg.norm(diff))
			tensionLocation = tensionRest + diff
		gradient = estimateGradient(tensionLocation, Minv, MinvPrime)
		numIterations += 1

	return tensionLocation

