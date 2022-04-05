import math
import numpy as np
import time
import random
from scipy.sparse import csr_matrix
from torch.utils.data.sampler import SubsetRandomSampler
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import submodlib
# from submodlib import FacilityLocationMutualInformationFunction, FacilityLocationVariantMutualInformationFunction

class SMIStrategy():
    def __init__(self, train_representations, query_representations,
                 original_indices, logger, smi_func_type, 
                 num_partitions=20, partition_strategy='random',
                 optimizer='LazyGreedy', similarity_criterion='feature', 
                 metric='cosine', eta=1, stopIfZeroGain=False, 
                 stopIfNegativeGain=False, verbose=False, lambdaVal=1):
        """
        Constructer method
        """
        # super().__init__(train_representations, query_representations, original_indices, smi_func_type, logger)
        self.train_rep = train_representations
        self.query_rep = query_representations
        self.indices = original_indices
        self.logger = logger
        self.optimizer = optimizer
        self.smi_func_type = smi_func_type
        self.num_partitions = num_partitions
        self.partition_strategy = partition_strategy
        self.metric = metric
        self.eta = eta
        self.stopIfZeroGain = stopIfZeroGain
        self.stopIfNegativeGain = stopIfNegativeGain
        self.verbose = verbose
        self.lambdaVal = lambdaVal
        self.similarity_criterion = similarity_criterion
    
    def update_representations(self, train_representations, query_representations, indices):
        self.train_rep = train_representations
        self.query_rep = query_representations
        self.indices = indices
        assert len(self.indices) == self.train_rep.shape[0], "Indices and representations must have same length"

    def random_partition(self, num_partitions, indices):
        """
        Randomly partition the data into num_partitions
        Parameters
        ----------
        num_partitions : int
            Number of partitions
        indices : list
            List of indices to partition
        Returns
        -------
        partition_indices : list
            List of lists of indices
        """
        partition_indices = []
        partition_size = int(math.ceil(len(indices)/num_partitions))
        random_indices = list(range(len(indices)))
        random.shuffle(random_indices)
        for i in range(num_partitions):
            partition_indices.append(random_indices[i*partition_size:(i+1)*partition_size])
        return partition_indices


    def select(self, budget):
        """

        Parameters
        ----------
        budget :
        model_params :

        Returns
        -------

        """
        smi_start_time = time.time()
        if self.partition_strategy == 'random':
            partition_indices = self.random_partition(self.num_partitions, self.indices) 
        elif self.partition_strategy == 'dbscan':
            partition_indices = self.dbscan(self.num_partitions, self.train_rep, self.indices)
        else:
            partition_indices = [list(range(len(self.indices)))]
        
        if self.partition_strategy not in ['random', 'dbscan']:
            assert self.num_partitions == 1, "Partition strategy {} not implemented for {} partitions".format(self.partition_strategy, self.num_partitions)
    
        partition_budget_split = math.ceil(budget/self.num_partitions)
        greedyIdxs = []
        for partition in partition_indices:
            partition_train_rep = self.train_rep[partition]
            if self.query_rep is None:
                partition_query_rep = self.train_rep[partition]
            else:
                partition_query_rep = self.query_rep

            query_sijs = submodlib.helper.create_kernel(X=partition_query_rep, X_rep=partition_train_rep, 
                                                        metric=self.metric, method='sklearn')

            if self.smi_func_type in ['fl1mi', 'logdetmi']:
                data_sijs = submodlib.helper.create_kernel(X=partition_train_rep, metric=self.metric,
                                                            method='sklearn')
            if self.smi_func_type in ['logdetmi']:
                query_query_sijs = submodlib.helper.create_kernel(X=partition_query_rep,
                                                                    metric=self.metric,
                                                                    method='sklearn')
            
            if self.smi_func_type == 'fl1mi':
                obj = submodlib.FacilityLocationMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                num_queries=partition_query_rep.shape[0],
                                                                data_sijs=data_sijs,
                                                                query_sijs=query_sijs,
                                                                magnificationEta=self.eta)
            if self.smi_func_type == 'fl2mi':
                obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                num_queries=partition_query_rep.shape[0],
                                                                query_sijs=query_sijs,
                                                                queryDiversityEta=self.eta)
            if self.smi_func_type == 'logdetmi':
                obj = submodlib.LogDeterminantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                        num_queries=partition_query_rep.shape[0],
                                                                        data_sijs=data_sijs,
                                                                        lambdaVal=self.lambdaVal,
                                                                        query_sijs=query_sijs,
                                                                        query_query_sijs=query_query_sijs,
                                                                        magnificationEta=self.eta
                                                                        )
            if self.smi_func_type == 'gcmi':
                obj = submodlib.GraphCutMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                    num_queries=partition_query_rep.shape[0],
                                                                    query_sijs=query_sijs)

            greedyList = obj.maximize(budget=partition_budget_split, optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                        stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)

            greedyIdxs.extend([x[0] for x in greedyList])
            
        originalIdxs = [self.indices[x] for x in greedyIdxs]
        smi_end_time = time.time()
        self.logger.info("SMI algorithm Subset Selection time is: %.4f", smi_end_time - smi_start_time)
        return originalIdxs
