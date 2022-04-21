import math
import numpy as np
import torch
import time
import random
from scipy.sparse import csr_matrix
from torch.utils.data.sampler import SubsetRandomSampler
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import submodlib
# from cuml.cluster import KMeans
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
    
    # def update_representations(self, train_representations, query_representations, indices):
    #     self.train_rep = train_representations
    #     self.query_rep = query_representations
    #     self.indices = indices
    #     assert len(self.indices) == self.train_rep.shape[0], "Indices and representations must have same length"

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

    def kmeans(self, num_partitions, indices, partition_budget_split):
        # partition_indices=[[] for i in range(num_partitions)]
        # kmeans=KMeans(n_clusters=num_partitions)
        # kmeans.fit(self.train_rep)
        # for i, lab in enumerate(kmeans.labels_):
        #     partition_indices[lab].append(indices[i])
        # for l in partition_indices:
        #     assert len(l)>=partition_budget_split, "Budget must be less than effective ground set size"
        # return partition_indices
        partition_indices = []
        partition_size = int(math.ceil(len(indices)/num_partitions))
        random_indices = list(range(len(indices)))
        random.shuffle(random_indices)
        for i in range(num_partitions):
            partition_indices.append(random_indices[i*partition_size:(i+1)*partition_size])
        return partition_indices

    def select(self, budget, indices, representations, query_representations=None):
        """

        Parameters
        ----------
        budget :
        model_params :

        Returns
        -------

        """
        partition_budget_split = math.ceil(budget/self.num_partitions)
        smi_start_time = time.time()
        if self.partition_strategy == 'random':
            partition_indices = self.random_partition(self.num_partitions, indices) 
        elif self.partition_strategy == 'kmeans':
            partition_indices = self.kmeans(self.num_partitions, indices, partition_budget_split)
        else:
            partition_indices = [list(range(len(indices)))]
        
        if self.partition_strategy not in ['random', 'kmeans']:
            assert self.num_partitions == 1, "Partition strategy {} not implemented for {} partitions".format(self.partition_strategy, self.num_partitions)
    
        greedyIdxs = []
        partition_cnt = 1
        if self.smi_func_type in ['logdetmi']:
            if query_representations is not None:
                query_query_sijs = submodlib.helper.create_kernel(X=query_representations,
                                                                metric=self.metric,
                                                                method='sklearn')                
            
        for partition in partition_indices:
            partition_train_rep = representations[partition]
            
            if query_representations is None:
                partition_query_rep = representations[partition]    
                query_sijs = submodlib.helper.create_kernel(X=partition_query_rep, X_rep=partition_train_rep, 
                                                            metric=self.metric, method='sklearn')
            else:
                query_sijs = submodlib.helper.create_kernel(X=query_representations, X_rep=partition_train_rep, 
                                                            metric=self.metric, method='sklearn')
            
            if self.smi_func_type in ['fl1mi', 'logdetmi']:
                if query_representations is None:
                    data_sijs = query_sijs
                else:
                    data_sijs = submodlib.helper.create_kernel(X=partition_train_rep, metric=self.metric,
                                                            method='sklearn')
            
            if self.smi_func_type in ['logdetmi']:
                if query_representations is None:
                    query_query_sijs = query_sijs
                
            
            if self.smi_func_type == 'fl1mi':
                if query_representations is None:
                    obj = submodlib.FacilityLocationMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                num_queries=partition_query_rep.shape[0],
                                                                data_sijs=data_sijs,
                                                                query_sijs=query_sijs,
                                                                magnificationEta=self.eta)
                else:
                    obj = submodlib.FacilityLocationMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                num_queries=query_representations.shape[0],
                                                                data_sijs=data_sijs,
                                                                query_sijs=query_sijs,
                                                                magnificationEta=self.eta)               
            if self.smi_func_type == 'fl2mi':
                if query_representations is None:
                    obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                num_queries=partition_query_rep.shape[0],
                                                                query_sijs=query_sijs,
                                                                queryDiversityEta=self.eta)
                else:
                    obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                num_queries=query_representations.shape[0],
                                                                query_sijs=query_sijs,
                                                                queryDiversityEta=self.eta)

            if self.smi_func_type == 'logdetmi':
                if query_representations is None:
                    obj = submodlib.LogDeterminantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                            num_queries=partition_query_rep.shape[0],
                                                                            data_sijs=data_sijs,
                                                                            lambdaVal=self.lambdaVal,
                                                                            query_sijs=query_sijs,
                                                                            query_query_sijs=query_query_sijs,
                                                                            magnificationEta=self.eta
                                                                            )
                else:
                    obj = submodlib.LogDeterminantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                        num_queries=query_representations.shape[0],
                                                                        data_sijs=data_sijs,
                                                                        lambdaVal=self.lambdaVal,
                                                                        query_sijs=query_sijs,
                                                                        query_query_sijs=query_query_sijs,
                                                                        magnificationEta=self.eta
                                                                        )
            if self.smi_func_type == 'gcmi':
                if query_representations is None:
                    obj = submodlib.GraphCutMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                        num_queries=partition_query_rep.shape[0],
                                                                        query_sijs=query_sijs)
                else:
                    obj = submodlib.GraphCutMutualInformationFunction(n=partition_train_rep.shape[0],
                                                                        num_queries=query_representations.shape[0],
                                                                        query_sijs=query_sijs)

            greedyList = obj.maximize(budget=partition_budget_split, optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                        stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)
            del partition_train_rep
            if query_representations is None:
                del partition_query_rep
            del obj
            del query_sijs
            if self.smi_func_type in ['fl1mi', 'logdetmi']:
                del data_sijs
            if self.smi_func_type in ['logdetmi']:
                if query_representations is None:
                    del query_query_sijs
            self.logger.info("Partition {}: {} greedy queries".format(partition_cnt, len(greedyList)))
            partition_cnt += 1
            greedyIdxs.extend([x[0] for x in greedyList])
            
        originalIdxs = [self.indices[x] for x in greedyIdxs]
        smi_end_time = time.time()
        self.logger.info("SMI algorithm Subset Selection time is: %.4f", smi_end_time - smi_start_time)
        return originalIdxs
