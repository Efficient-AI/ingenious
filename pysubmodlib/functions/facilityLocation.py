import numpy as np
import scipy
from pysubmodlib.helper import (
    create_cluster_kernels,
    create_kernel_dense_np_numba,
    create_kernel_dense_np_numba_rectangular,
    create_kernel_sparse_np_numba,
    get_max_sim_dense,
    get_max_sim_sparse,
    get_max_sim_cluster,
)
from .setFunction import SetFunction

class FacilityLocationFunction(SetFunction):
    def __init__(self, n, mode, separate_rep=None, n_rep=None, sijs=None, data=None, data_rep=None, num_clusters=None, cluster_labels=None, metric="cosine", num_neighbors=None):
        self.n=n                                    # size of ground set
        self.n_rep=n_rep                            # size of master set
        self.mode=mode                              # can be dense, sparse or clustered
        # self.partial=None                           # if masked implementation is desired, relavant to be used in ClusteredFunction
        self.separate_rep=separate_rep              # if master set is separate from ground set
        self.effective_ground=None                  # effective ground set considering mask if partial=True
        self.master_set=None                        # set of items in master set
        self.num_effective_ground=None              # size of effective ground 
        self.original_to_partial_idx_map=None       # 
        self.sijs=sijs
        self.metric=metric
        self.data=data
        self.data_rep=data_rep
        #specific to cluster mode only
        self.num_clusters=num_clusters
        self.clusters=None                          # list of clusters where each cluster is taken as a set of datapoint indices in that cluster. size=num_clusters
        self.cluster_labels=cluster_labels          # maps index of a datapoint to the ID of cluster to which belongs to, size=n
        self.cluster_sijs=None                      # list which contains dense similarity matrices for each cluster, size=n
        self.cluster_map=None                       # mapping from datapoint index to index in cluster kernel, size=n
        # memoized statistics
        self.sim_nearest_in_effectiveX=None         # for each i in master set, this contains max(i, effectiveX), size=n_rep
        self.relevant_X=None                        # (list of sets)for each cluster C_i, this contains X \cap C_i, size=num_clusters
        self.clus_sim_nearest_in_relevant_X=None    # for every element in ground set, this list contains max similarity with items in X \cap C_i where C_i is the cluster which this element belongs to, size=n
        self.num_neighbors=num_neighbors
        # specific to sparse mode only
        self.indices_sparse=None
        self.data_sparse=None

        if self.n<=0:
            raise Exception("ERROR: Number of elements in ground set must be positive")
        
        if self.mode not in ["dense", "sparse", "clustered"]:
            raise Exception("ERROR: Incorrect Mode. Must be one of 'dense', 'sparse' or 'clustered'")
        
        if self.separate_rep==True:
            if self.n_rep is None or self.n_rep<=0:
                raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")
            if self.mode!="dense":
                raise Exception("Only dense mode supported if separate_rep=True")
        
        if self.mode=="clustered":
            if type(self.cluster_labels)!=type(None) and (self.num_clusters is None or self.num_clusters<=0):
                raise Exception("ERROR: Positive number of clusters must be provided in clustered mode when cluster_labels is provided")
            if type(self.cluster_labels)==type(None) and self.num_clusters is not None and self.num_clusters<=0:
                raise Exception("Invalid number of clusters provided")
            if type(self.cluster_labels)!=type(None) and len(self.cluster_labels)!=self.n:
                raise Exception("ERROR: cluster_label's size is NOT same as ground set size")
            if type(self.cluster_labels)!=type(None) and not all(ele>=0 and ele<self.num_clusters-1 for ele in self.cluster_labels):
                raise Exception("Cluster IDs/labels contain invalid values")

        if type(self.sijs)!=type(None):
            # User has provided similarity kernel
            if type(self.sijs)==scipy.sparse.csr_matrix:
                if num_neighbors is None or num_neighbors<=0:
                    raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
                if mode!="sparse":
                    raise Exception("ERROR: Sparse kernel provided, but mode is not set to sparse")
            elif type(self.sijs)==np.ndarray:
                if self.separate_rep is None:
                    raise Exception("ERROR: separate_rep bool must be specified with custom dense kernel")
                if mode!="dense":
                    raise Exception("ERROR: Dense kernel provided, but mode is not dense")
            else:
                raise Exception("Invalid kernel provided")
            # TODO: is the below dimensionality check valid for both dense and sparse kernels?
            if self.separate_rep==True:
                if np.shape(self.sijs)[1]!=self.n or np.shape(self.sijs)[0]!=self.n_rep:
                    raise Exception("ERROR: Inconsistency between n_rep, n and no. of rows, columns of given kernel")
            else:
                if np.shape(self.sijs)[0]!=self.n or np.shape(self.sijs)[1]!=self.n:
                    raise Exception("ERROR: Inconsistency between n and dimensionality of given similarity kernel")
            if type(self.data)!=type(None) or type(self.data_rep)!=type(None):
                print("WARNING: Similarity kernel found. Provided data matrix will be ignored!")
        else:
            # User hasn't provided similarity matrix
            if type(self.data)!=type(None): 
                if self.separate_rep == True:
                    if type(self.data_rep) == type(None):
                        raise Exception("Represented data matrix not given")
                    if np.shape(self.data)[0]!=self.n or np.shape(self.data_rep)[0]!=self.n_rep:
                        raise Exception("ERROR: Inconsistentcy between n, n_rep and no of examples in the given ground data matrix and represented data matrix")
                    else:
                        if type(self.data_rep) != type(None):
                            print("WARNING: Represented data matrix not required but given, will be ignored.")
                        if np.shape(self.data)[0]!=self.n:
                            raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
        
                if self.mode=="clustered":
                    self.clusters, self.cluster_sijs, self.cluster_map=create_cluster_kernels(self.data, self.metric, self.cluster_labels, self.num_clusters)
                else:
                    if self.separate_rep==True: # mode in this case will always be 'dense'
                        self.sijs=create_kernel_dense_np_numba_rectangular(self.data, self.data_rep, self.metric)
                    else:
                        if self.mode=="dense":
                            if self.num_neighbors is not None:
                                raise Exception("ERROR: num_neighbors wrongly provided for dense mode")
                            self.sijs=create_kernel_dense_np_numba(self.data, self.metric)
                        else:
                            self.sijs=create_kernel_sparse_np_numba(self.data, self.metric, self.num_neighbors)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel is provided")
        
        if separate_rep==None:
            separate_rep=False
        
        self.effective_ground=set(list(range(n)))
        self.num_effective_ground=n
        if self.mode=="dense":
            if separate_rep==True:
                self.master_set=set(list(range(n_rep)))
            else:
                self.n_rep=n
                self.master_set=self.effective_ground
            self.sim_nearest_in_effectiveX=np.zeros(self.n_rep)
        elif self.mode=="sparse":
            self.n_rep=n
            self.master_set=self.effective_ground
            self.sim_nearest_in_effectiveX=np.zeros(self.n_rep)
            self.indices_sparse=self.sijs.indices.reshape(self.n_rep, self.num_neighbors)
            self.data_sparse=self.sijs.data.reshape(self.n_rep, self.num_neighbors)
        elif self.mode=="clustered":
            self.n_rep=n
            self.master_set=self.effective_ground
            self.clus_sim_nearest_in_relevant_X=np.zeros(self.n_rep)
        
    def evaluate(self, X):
        # TODO: Can make it super efficient using numba
        if type(X)!=set:
            raise Exception("ERROR: X should be a set")
        
        if not X.issusbet(self.effective_ground):
            raise Exception("ERROR: X should be a subset of effective ground set")

        effectiveX=list(X)
        if len(effectiveX)==0:
            return 0.0
        
        result=0.0
        if self.mode=="dense":
            for ind in self.master_set:
                result+=get_max_sim_dense(ind, effectiveX, self.sijs)
        elif self.mode=="sparse":
            for ind in self.master_set:
                result+=get_max_sim_sparse(ind, effectiveX, self.sijs)
        else:
            for i in range(self.num_clusters):
                relevant_subset=list(X.intersection(self.clusters[i]))
                if len(relevant_subset)==0:
                    continue
                for ind in self.clusters[i]:
                    result+=get_max_sim_cluster(ind, relevant_subset, self.cluster_map, self.cluster_sijs, i)
        result=float(result)
        return result
        
    def marginalGain(self, X, element):
        if type(X)!=set:
            raise Exception("ERROR: X should be a set")
        
        if type(element)!=int:
            raise Exception("ERROR: element should be an int")
        
        if not X.issubset(self.effective_ground)==False:
            raise Exception("ERROR: X is not a subset of effective ground set")
        
        if element not in self.effective_ground:
            raise Exception("ERROR: element must be in the effective ground set")
        
        if element in X:
            return 0.0
        
        gain=0.0
        effectiveX=list(X)
        
        if self.mode=="dense":
            for ind in self.master_set:
                m=get_max_sim_dense(ind, effectiveX, self.sijs)
                M=self.sijs[ind, element]
                if M>m:
                    gain+=(M-m)
        elif self.mode=="sparse":
            for ind in self.master_set:
                m=get_max_sim_sparse(ind, effectiveX, self.sijs)
                M=self.sijs[ind, element]
                if M>m:
                    gain+=(M-m)
        else:
            i=self.cluster_labels[element]
            item_=self.cluster_map[element]
            relevant_subset=list(X.intersection(self.clusters[i]))
            if len(relevant_subset)==0:
                for ind in self.clusters[i]:
                    gain+=self.cluster_sijs[i][self.cluster_map[ind]][item_]
            else:
                for ind in self.clusters[i]:
                    m=get_max_sim_cluster(ind, relevant_subset, self.cluster_map, self.cluster_sijs, i)
                    M=self.cluster_sijs[i][self.cluster_map[ind]][item_]
                    if M>m:
                        gain+=(M-m)
        gain=float(gain)
        return gain

    def marginalGainWithMemoization(self, X, element):
        # if type(X)!=set:
        #     raise Exception("ERROR: X should be a set")
        
        # if type(element)!=int:
        #     raise Exception("ERROR: element should be an int")
        
        # if not X.issubset(self.effective_ground):
        #     raise Exception("ERROR: X is not a subset of effective ground set")
        
        # if element not in self.effective_ground:
        #     raise Exception("ERROR: element must be in the effective ground set")
        
        # if element in X:
        #     return 0
        
        # effectiveX=list(X)
        gain=0.0
        if self.mode=="dense":
            for ind in self.master_set:
                M=self.sijs[ind, element]
                m=self.sim_nearest_in_effectiveX[ind]
                if M>m:
                    gain+=(M-m)
        elif self.mode=="sparse":
            neighbors=self.indices_sparse[element]
            M=self.data_sparse[element]
            m=self.sim_nearest_in_effectiveX[neighbors]
            gain=np.sum(M[M>m])
        else:
            i=self.cluster_labels[element]
            item_=self.cluster_map[element]
            relevant_subset=list(self.relevant_X[i])
            if len(relevant_subset)==0:
                for ind in self.clusters[i]:
                    gain+=self.cluster_sijs[i][self.cluster_map[ind]][item_]
            else:
                for ind in self.clusters[i]:
                    M=self.cluster_sijs[i][self.cluster_map[ind]][item_]
                    m=self.clus_sim_nearest_in_relevant_X[ind]
                    if M>m:
                        gain+=(M-m)
        gain=float(gain)
        return gain
        
    def evaluateWithMemoization(self, X):
        if type(X)!=set:
            raise Exception("ERROR: X should be a set")
        
        if not X.issubset(self.effective_ground):
            raise Exception("ERROR: X should be a subset of effective ground set")
        
        if len(X)==0:
            return 0
        
        # effectiveX=list(X)
        result=0.0
        if self.mode=="dense":
            for ind in self.master_set:
                result+=self.sim_nearest_in_effectiveX[ind]
        elif self.mode=="sparse":
            for ind in self.master_set:
                result+=self.sim_nearest_in_effectiveX[ind]
        else:
            for i in range(self.num_clusters):
                if len(self.relevant_X[i])==0:
                    continue
                for ind in self.clusters[i]:
                    result+=self.clus_sim_nearest_in_relevant_X[ind]
        result=float(result)
        return result
        
    def updateMemoization(self, X, element):
        # if type(X)!=set:
        #     raise Exception("EEROR X should be a set")
        
        # if type(element)!=int:
        #     raise Exception("ERROR: element should be an int")
        
        # if not X.issubset(self.effective_ground):
        #     raise Exception("ERROR: X is not a subset of effective ground set")
        
        # if element not in self.effective_ground:
        #     raise Exception("ERROR: element must be in the effective ground set")
        
        # if element in X:
        #     return
        
        # effectiveX=list(X)
        if self.mode=="dense":
            for ind in self.master_set:
                M=self.sijs[ind, element]
                m=self.sim_nearest_in_effectiveX[ind]
                if M>m:
                    self.sim_nearest_in_effectiveX[ind]=M
        elif self.mode=="sparse":
            neighbors=self.indices_sparse[element]
            M=self.data_sparse[element]
            m=self.sim_nearest_in_effectiveX[neighbors]
            self.sim_nearest_in_effectiveX[neighbors]=np.maximum(M, m)
        else:
            i=self.cluster_labels[element]
            item_=self.cluster_map[element]
            for ind in self.clusters[i]:
                M=self.sijs[i][self.cluster_map[ind]][item_]
                m=self.clus_sim_nearest_in_relevant_X[ind]
                if M>m:
                    self.clus_sim_nearest_in_relevant_X[ind]=M
            self.relevant_X[i].add(element)
            

    def clearMemoization(self):
        if self.mode=="dense" or self.mode=="sparse":
            for i in range(self.n_rep):
                self.sim_nearest_in_effectiveX[i]=0
        else:
            for i in range(self.num_clusters):
                self.relevant_X[i].clear()
            for i in range(self.n):
                self.clus_sim_nearest_in_relevant_X[i]=0
        
    def setMemoization(self, X):
        if type(X)!=set:
            raise Exception("ERROR: X should be a set")
        
        if not X.issubset(self.effective_ground):
            raise Exception("ERROR: X should a subset of effective ground set")
        
        self.clearMemoization()
        temp=set()
        for element in X:
            self.updateMemoization(temp, element)
            temp.add(element)

    def getEffectiveGroundSet(self):
        return self.effective_ground