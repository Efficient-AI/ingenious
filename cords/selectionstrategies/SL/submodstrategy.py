import math
import random
import time
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import submodlib
import faiss
import pickle

def query_generator(representations, partition_indices, partition_budgets, smi_func_type, metric, sparse_rep):
    for i, partition in enumerate(partition_indices):
        yield (representations[partition], partition_budgets[i], partition, smi_func_type, metric, sparse_rep)

def partition_subset_strat(args):
        return partition_subset_selection(*args)
    
def partition_subset_selection(representations, partition_budget, partition_ind, smi_func_type, metric, sparse_rep):
    kernel_time=time.time()
    if smi_func_type in ["fl", "logdet", "gc"]:
        if sparse_rep:
            data_sijs=cosine_similarity(representations)
        else:
            data_sijs=submodlib.helper.create_kernel(X=representations, metric=metric, method="sklearn")
    else:
        raise Exception(f"{smi_func_type} not yet supported by this script")
    
    greedy_selection_start_time=time.time()
    if smi_func_type=="fl":
        obj = submodlib.FacilityLocationFunction(n = representations.shape[0],
                                                separate_rep=False,
                                                mode = 'dense',
                                                sijs = data_sijs)
    if smi_func_type == 'gc':
        obj = submodlib.GraphCutFunction(n = representations.shape[0],
                                        mode = 'dense',
                                        lambdaVal = 1,
                                        separate_rep=False,
                                        ggsijs = data_sijs)     
    if smi_func_type == 'logdet':
        obj = submodlib.LogDeterminantFunction(n = representations.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 1,
                                                sijs = data_sijs)
    
    greedyList=obj.maximize(budget=partition_budget, optimizer="LazierThanLazyGreedy", stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, show_progress=True)

    del representations
    del obj
    if smi_func_type in ["fl", "logdet", "gc"]:
        del data_sijs
    #Converting selected indices to global indices
    return [partition_ind[x[0]] for x in greedyList]

class SubmodStrategy():
    def __init__(self, logger, smi_func_type, 
                num_partitions=1000, partition_strategy="random",
                optimizer="LazierThanLazyGreedy", similarity_criterion="feature",
                metric="cosine", eta=1, stopIfZeroGain=False,
                stopIfNegativeGain=False, verbose=False, lambdaVal=1, sparse_rep=False):
        self.logger=logger
        self.optimizer=optimizer
        self.smi_func_type=smi_func_type
        self.num_partitions=num_partitions
        self.partition_strategy=partition_strategy
        self.metric=metric
        self.eta=eta
        self.stopIfZeroGain=stopIfZeroGain
        self.stopIfNegativeGain=stopIfNegativeGain
        self.verbose=verbose
        self.lambdaVal=lambdaVal
        self.similarity_criterion=similarity_criterion
        self.sparse_rep=sparse_rep

    def random_partition(self, num_partitions, indices):
        partition_indices = []
        partition_size = int(math.ceil(len(indices)/num_partitions))
        random_indices = list(range(len(indices)))
        random.shuffle(random_indices)
        for i in range(num_partitions):
            partition_indices.append(random_indices[i*partition_size:(i+1)*partition_size])
        return partition_indices
    
    def kmeans_partition(self, num_partitions, representations, indices):
        self.logger.info("Started KMeans clustering routine")
        kmeans_start_time=time.time()
        n=representations.shape[0]
        d=representations.shape[1]
        kmeans=faiss.Kmeans(d, num_partitions, spherical=True, max_points_per_centroid=math.ceil(n/num_partitions), niter=20, verbose=True, gpu=True)
        # kmeans=faiss.Kmeans(d, num_partitions, spherical=True, niter=20, nredo=5, verbose=True, gpu=True)
        self.logger.info("Starting training")
        kmeans.train(representations)
        D, I=kmeans.index.search(representations, 1)
        partition_indices=[[] for i in range(num_partitions)]
        for i, lab in enumerate(I.reshape((-1,)).tolist()):
            partition_indices[lab].append(indices[i])
        kmeans_end_time=time.time()
        self.logger.info("Kmeans routine took %.4f of time", kmeans_end_time-kmeans_start_time)
        return partition_indices
    
    def select(self, budget, indices, representations, parallel_processes=96):
        smi_start_time=time.time()
        
        # return partitions of the data for subset selection
        if self.partition_strategy=="random":
            partition_indices=self.random_partition(self.num_partitions, indices)
            partition_budgets=[math.ceil((len(partition)/len(indices)) * budget) for partition in partition_indices]
        elif self.partition_strategy=="kmeans_clustering":
            partition_indices=self.kmeans_partition(self.num_partitions, representations, indices)
            partition_budgets=[min(math.ceil((len(partition)/len(indices)) * budget), len(partition)-1) for partition in partition_indices]
        else:
            partition_indices=[list(range(len(indices)))]
            partition_budgets=[math.ceil(budget)]
        
        if self.partition_strategy not in ["random", "kmeans_clustering"]:
            assert self.num_partitions == 1, "Partition strategy {} not implemented for {} partitions".format(self.partition_strategy, self.num_partitions)
        
        greedyIdxs=[]

        # Parallel computation of subsets
        with Pool(parallel_processes) as pool:
            greedyIdx_list=list(tqdm.tqdm(pool.imap_unordered(partition_subset_strat, 
                                                    query_generator(representations, partition_indices, partition_budgets, self.smi_func_type, self.metric, self.sparse_rep)), total=len(partition_indices)))
        
        # with ProcessingPool(nodes=parallel_processes) as pool:
        #     greedyIdx_list=list(tqdm.tqdm(pool.uimap(partition_subset_strat, query_generator(representations, partition_indices, partition_budgets, self.smi_func_type, self.metric, self.sparse_rep)), total=len(partition_indices)))
        greedyIdxs=[]
        for idxs in greedyIdx_list:
            greedyIdxs.extend(idxs)
        
        originalIdxs=[indices[x] for x in greedyIdxs]
        # assert len(set(originalIdxs))==sum(partition_budgets), "Selected subset must be equal to the budget"
        smi_end_time=time.time()
        self.logger.info("SMI algorithm subset selection time is %.4f", smi_end_time-smi_start_time)
        return partition_indices, originalIdxs