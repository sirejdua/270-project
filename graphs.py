import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from stableState import findStableState as fss
from tensionOptimizer import findOptimalTension as fot
from tensionOptimizer import getObjectiveFn as gof
from spectral import estimateOptimalTension as eot
from helpers import createGrid

cutPoints = [25, 25, 25, 300, 300, 300, 575, 575, 575] # cuts near the top, middle, and bottom
tensionPoints = [26, 30, 75, 301, 305, 350, 576, 580, 525] # try tensioning right next to the point, somewhat to the right, or somewhat up / down
clusterNums = [10, 25, 50, 100, 200, 300] # the number of clusters to use

L, boundaries = createGrid(25)
stableState = fss(L, boundaries)

speedups = [] # time taken for reduced graph / time taken for original graph
badnessRatios = [] # (objective fn for exact - objective fn for estimate) / (objective fn for exact - objective fn for no tension)

for cutPoint, tensionPoint in zip(cutPoints, tensionPoints):
	print("Starting on cut point " + str(cutPoint) + " and tension point " + str(tensionPoint))
	sys.stdout.flush()
	start = time.time()
	exactTension = fot(L, boundaries, tensionPoint, cutPoint, 1.0)
	end = time.time()
	exactTime = end - start
	tempB = boundaries[:]
	tempB.append((tensionPoint, exactTension))
	exactVal = gof(L, tempB, cutPoint)
	noTensionVal = gof(L, boundaries, cutPoint)
	for numClusters in clusterNums:
		start = time.time()
		estTension = eot(L, boundaries, tensionPoint, cutPoint, 1.0, numClusters)
		end = time.time()
		speedups.append((end - start) / exactTime)
		tempB = boundaries[:]
		tempB.append((tensionPoint, estTension))
		estVal = gof(L, tempB, cutPoint)
		badnessRatios.append((exactVal - estVal) / (exactVal - noTensionVal))

xVals = clusterNums * len(cutPoints)

plt.figure()
plt.scatter(xVals, speedups)
plt.xlabel("Number of clusters")
plt.ylabel("Speedup")
plt.title("Speedups for Spectral Clustering")
plt.show()

plt.figure()
plt.scatter(xVals, badnessRatios)
plt.xlabel("Number of clusters")
plt.ylabel("Decrease in performance")
plt.title("Decreases in performance for Spectral Clustering")
plt.show()