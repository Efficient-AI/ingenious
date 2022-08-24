import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import (
    euclidean_distances, 
    cosine_similarity
)
from numba import jit, njit
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

def create_cluster_kernels(X, metric, cluster_lab=None, num_cluster=None, onlyClusters=False):
    # Here cluster_lab is a list which specifies custom cluster mapping of a datapoint to a cluster
    lab=[]
    if cluster_lab==None:
        obj=Birch(n_clusters=num_cluster)
        obj=obj.fit(X)
        lab=obj.predict(X).tolist()
        if num_cluster==None:
            num_cluster=len(obj.subcluster_labels_)
    else:
        if num_cluster==None:
            raise Exception("ERROR: num_cluster needs to be specified if cluster_lab is provided")
        lab=cluster_lab
    
    l_cluster=[set() for _ in range(num_cluster)]
    l_ind=[0]*np.shape(X)[0]
    l_count=[0]*num_cluster

    for i, el in enumerate(lab):
        # For any cluster ID(el), smallest datapoint(i) is filled first
        # Therefore, the set l_cluster will always be sorted
        l_cluster[el].add(i)
        l_ind[i]=l_count[el]
        l_count[el]=l_count[el]+1

    if onlyClusters==True:
        return l_cluster, None, None
    
    l_kernel=[]
    for el in l_cluster:
        k=len(el)
        l_kernel.append(np.zeros((k,k)))
    
    M=None
    if metric=="euclidean":
        D=euclidean_distances(X)
        gamma=1/np.shape(X)[1]
        M=np.exp(-D * gamma)
    elif metric=="cosine":
        M=cosine_similarity(X)
    else:
        raise Exception("ERROR: unsupported metric")
    
    # Create kernel for each cluster using the bigger kernel
    for ind, val in np.ndenumerate(M):
        if lab[ind[0]]==lab[ind[1]]: # if a pair of datapoints is in same cluster then update the kernel corresponding to that cluster
            c_ID=lab[ind[0]]
            i=l_ind[ind[0]]
            j=l_ind[ind[1]]
            l_kernel[c_ID][i,j]=val
    
    return l_cluster, l_kernel, l_ind

@jit(nopython=True, parallel=True)
def euc_dis_numba(A, B):
    M=A.shape[0]
    N=B.shape[0]
    A_dots=(A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1, N))
    B_dots=(B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared=A_dots+B_dots-2*A.dot(B.T)
    D_squared=np.where(D_squared<0.0, 0, D_squared)
    return np.sqrt(D_squared)

@jit(nopython=True, parallel=True)
def cos_sim_square_numba(A):
    similarity=np.dot(A, A.T)
    square_mag=np.diag(similarity)
    inv_square_mag=1/square_mag
    inv_square_mag[np.isinf(inv_square_mag)]=0
    inv_mag=np.sqrt(inv_square_mag)
    cosine=similarity*inv_mag
    cosine=cosine.T*inv_mag
    return cosine
    
@jit(nopython=True, parallel=True)
def cos_sim_rectangle_numba(A, B):
    num=np.dot(A, B.T)
    p1=np.sqrt(np.sum(A**2, axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2, axis=1))[np.newaxis,:]
    return num/(p1*p2)

@jit(nopython=True, parallel=True)
def create_kernel_dense_np_numba(X, metric):
    dense=None
    D=None
    if metric=="euclidean":
        D=euc_dis_numba(X, X)
        gamma=1/np.shape(X)[1]
        dense=np.exp(-D*gamma)
    elif metric=="cosine":
        dense=cos_sim_square_numba(X)
    else:
        raise Exception("ERROR: unsupported metric")
    
    assert (dense.shape==(X.shape[0], X.shape[0]))
    return dense

@jit(nopython=True, cache=True, parallel=True)
def create_kernel_dense_np_numba_rectangular(X, X_rep, metric):
    dense=None
    D=None
    if metric=="euclidean":
        D=euc_dis_numba(X_rep, X)
        gamma=1/np.shape(X)[1]
        dense=np.exp(-D*gamma)
    elif metric=="cosine":
        dense=cos_sim_rectangle_numba(X_rep, X)
    elif metric=="dot":
        dense=np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")
    
    if type(X_rep)!=type(None):
        assert (dense.shape==(X_rep.shape[0], X.shape[0]))
    else:
        assert (dense.shape==(X.shape[0], X.shape[0]))

    return dense

# Not implemented efficiently. First gets the dense matrix and then forms sparse matrix
def create_kernel_sparse_np_numba(X, metric, num_neighbors):
    if num_neighbors>np.shape(X)[0]:
        raise Exception("ERROR: num_neighbors can't be more than the number of datapoints")
    dense=create_kernel_dense_np_numba(X, metric)
    dense_=None
    if num_neighbors==-1:
        num_neighbors=np.shape(X)[0]
    nbrs=NearestNeighbors(n_neighbors=num_neighbors, metric=metric, n_jobs=-1).fit(X)
    _, ind=nbrs.kneighbors(X)
    ind_l=[(index[0], x) for index, x in np.ndenumerate(ind)]
    row, col=zip(*ind_l)
    mat=np.zeros(np.shape(dense))
    mat[row, col]=1
    dense_=dense*mat
    sparse_csr=sparse.csr_matrix(dense_)
    return sparse_csr

def get_max_sim_dense(datapoint_ind, dataset_ind, dense_kernel):
    # datapoint_ind is int, dataset_ind is list, dense_kernel is np.ndarray
    return np.max(dense_kernel[datapoint_ind, dataset_ind])

def get_max_sim_sparse(datapoint_ind, dataset_ind, sparse_kernel):
    # datapoint_ind is int, dataset_ind is list, dense_kernel is scipy.sparse.csr_matrix
    row=sparse_kernel.getrow(datapoint_ind).toarray()
    return np.max(row[dataset_ind])

def get_max_sim_cluster(datapoint_ind, dataset_ind, cluster_map, cluster_sijs, cluster_id):
    datapoint_ind_=cluster_map[datapoint_ind]
    dataset_ind=[cluster_map[elem] for elem in dataset_ind]
    return np.max(cluster_sijs[cluster_id][datapoint_ind, dataset_ind])

@jit(nopython=True, parallel=True)
def numba_sum(X):
    return np.sum(X)