import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from collections import defaultdict

def spectral_cluster(L,k):
	normalized_L = normalize_L(L)
	spectral = SpectralClustering(n_clusters=k,
		eigen_solver='arpack',
		affinity='precomputed')
	A = adj_mat(L)
	spectral.fit(A)
	return spectral.labels_

def reduce_graph(L, Vb, num_clusters, cluster_algorithm=spectral_cluster):
	small_L,mapping_new_old,Vb_edge_list = remove_vertices(L,Vb)
	cluster_assignments = cluster_algorithm(small_L, num_clusters)
	num_clusters = max(cluster_assignments) + 1
	i = 0
	for v in sorted(Vb):
		cluster_assignments.insert(v, num_clusters + i)
		i += 1
	clustered_graph = redrawGraph(L,cluster_assignments)
	return clustered_graph

def normalize_L(L):
	d = np.array([L[i][i] for i in range(len(L))])
	d_neg_half = np.reciprocal(np.sqrt(d))
	D_neg_half = np.diag(d_neg_half)
	normalized_L = D_neg_half.dot(L).dot(D_neg_half)
	return normalized_L

def adj_mat(L):
	d = np.array([L[i][i] for i in range(len(L))])
	D = np.diag(d)
	return D - L

def redrawGraph(L,new_labels):
	"""
	takes in a graph, and the clusters that each vertex belongs to.
	Returns a new clustered graph.
	Ex: line graph a-b-c-d
	if new_labels = [0,0,1,1]
	new graph is e-f
	"""
	num_labels = len(set(new_labels))
	adjacency_list = defaultdict(list)
	new_adjacency_list = defaultdict(lambda:defaultdict(int))
	new_L = np.zeros((num_labels,num_labels))
	for i in range(len(L)):
		for j in range(len(L)):
			if (i != j and L[i][j] != 0):
				new_adjacency_list[new_labels[i]][new_labels[j]] += 1
	for i in range(num_labels):
		for j in range(num_labels):
			new_L[i][j] = new_adjacency_list[new_labels[i]][new_labels[j]]
	return new_L

def remove_vertices(L,Vb):
	"""
	Removes the vertices in Vb from L.
	Returns laplacian of the new graph, a mapping from new indices to old
	indices, and an adjacent list of Vb nodes to other nodes (represented
	as a ddict)
	"""
	Vb_edge_list = defaultdict(list)
	mapping = {}
	encountered = 0
	for i in range(len(L)):
		if i in Vb:
			encountered += 1
		else:
			mapping[i - encountered] = i

	for v in Vb:
		for i in range(len(L)):
			if (v !=i):
				Vb_edge_list[v].append(i)
	new_L = np.delete(np.delete(L,Vb,axis=0),Vb,axis=1)
	return new_L, mapping, Vb_edge_list

# def add_vertices(L,mapping,Vb,Vb_edge_list):

# 	new_L = np.zeros(len(L) + len(Vb))
# 	for v in sorted(Vb_edge_list.keys()):
# 		index = mapping[Vb_edge_list[v]]
# 		new_row = 
# 		new_col = 

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



# def disconnect_from_Vb(L, Vb):

# 	temp = np.delete(L,Vb,axis=0)
# 	reduced_L = np.delete(L,Vb,axis=1)



# 	new_L = np.zeros((L.shape[0]-len(Vb),L.shape[1]-len(Vb)))
# 	encountered = 0
# 	old_to_new_mapping = {}
# 	new_to_old_mapping = {}
# 	for i in range(len(L)):
# 		if i == Vb:
# 			encountered += 1
# 		else:
# 			old_to_new_mapping[i] = i - encountered
# 			new_to_old_mapping[i-encountered] = i
# 	for i in range(len(new_L)):



# def sweep_cut(components):
# 	for c in sorted(components):

if __name__ == '__main__':
	L = np.array([[1,-1,0,0],
				[-1,1,0,0],
				[0,0,1,-1],
				[0,0,-1,1]], dtype='float')
	L1 = createGrid(25)
	print(spectral_cluster(L1[0],25))

