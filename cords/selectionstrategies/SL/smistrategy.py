import math
from operator import gt
import re
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

    def select(self, budget, indices, representations, query_representations=None, 
                private_representations=None, private_partitions=5):
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
            
            if self.smi_func_type in ['fl1mi', 'logdetmi', 'flcg', 'fl', 'logdetcg', 'logdet', 'gc', 'gccg']:
                if query_representations is None:
                    data_sijs = query_sijs
                else:
                    data_sijs = submodlib.helper.create_kernel(X=partition_train_rep, metric=self.metric,
                                                            method='sklearn')
            
            
            if self.smi_func_type in ['flcg', 'logdetcg', 'gccg']:
                if private_representations is not None:
                    private_sijs = submodlib.helper.create_kernel(X=private_representations, X_rep=partition_train_rep, 
                                                            metric=self.metric, method='sklearn')
                    if self.smi_func_type in ['logdetcg']:
                        private_private_sijs = submodlib.helper.create_kernel(X=private_representations,
                                                            metric=self.metric, method='sklearn')

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

            if self.smi_func_type == 'fl':
                obj = submodlib.FacilityLocationFunction(n = partition_train_rep.shape[0],
                                                        separate_rep=False,
                                                        mode = 'dense',
                                                        sijs = data_sijs)

            if self.smi_func_type == 'logdet':
                obj = submodlib.LogDeterminantFunction(n = partition_train_rep.shape[0],
                                                        mode = 'dense',
                                                        lambdaVal = 1,
                                                        sijs = data_sijs)
            
            if self.smi_func_type == 'gc':
                obj = submodlib.GraphCutFunction(n = partition_train_rep.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 1,
                                                seperate_rep=False,
                                                ggsijs = data_sijs)
            
            if self.smi_func_type == 'flcg':
                if private_representations is not None:
                    obj = submodlib.FacilityLocationConditionalGainFunction(n=partition_train_rep.shape[0], 
                                                                            num_privates=private_representations.shape[0],
                                                                            data_sijs=data_sijs,
                                                                            private_sijs=private_sijs)
                                                    

            if self.smi_func_type == 'logdetcg':
                if private_representations is not None:
                    obj = submodlib.LogDeterminantConditionalGainFunction(n=partition_train_rep.shape[0], 
                                                                          num_privates=private_representations.shape[0],
                                                                          lambdaVal=1,
                                                                          data_sijs=data_sijs,
                                                                          private_sijs=private_sijs)

            if self.smi_func_type == 'gccg':
                if private_representations is not None:
                    obj = submodlib.GraphCutConditionalGainFunction(n=partition_train_rep.shape[0], 
                                                                    num_privates=private_representations.shape[0],
                                                                    lambdaVal=1,
                                                                    data_sijs=data_sijs,
                                                                    private_sijs=private_sijs)


            if (self.smi_func_type in ['flcg', 'gccg', 'logdetcg']) and (private_representations is None):
                greedyList = []
                private_idxs = []
                private_partition_budget_split = [math.floor(partition_budget_split/private_partitions) for _ in range(private_partitions)]
                rem_budget = partition_budget_split - sum(private_partition_budget_split)
                for i in range(rem_budget):
                    private_partition_budget_split[i] += 1

                if self.smi_func_type == 'flcg':
                    obj = submodlib.FacilityLocationFunction(n = partition_train_rep.shape[0],
                                                            mode = 'dense',
                                                            separate_rep=False,
                                                            sijs = data_sijs)
                elif self.smi_func_type == 'gccg':
                    obj = submodlib.GraphCutFunction(n = partition_train_rep.shape[0],
                                                    mode = 'dense',
                                                    lambdaVal=1,
                                                    separate_rep=False,
                                                    ggsijs = data_sijs)
                else: 
                    obj = submodlib.LogDeterminantFunction(n = partition_train_rep.shape[0],
                                                        mode = 'dense',
                                                        lambdaVal = 1,
                                                        sijs = data_sijs)
                
                temp = obj.maximize(budget=private_partition_budget_split[0], optimizer=self.optimizer, 
                                    stopIfZeroGain=self.stopIfZeroGain, stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)

                [greedyList.append(temp[i]) for i in range(len(temp))]
                [private_idxs.append(temp[i][0]) for i in range(len(temp))]
                for i in range(1, private_partitions):
                    rem_idxs = [idx for idx in range(partition_train_rep.shape[0]) if idx not in private_idxs]
                    private_sijs = data_sijs[rem_idxs]
                    private_sijs = private_sijs[:, private_idxs]
                    gt_data_sijs = data_sijs[rem_idxs]
                    gt_data_sijs = gt_data_sijs[:, rem_idxs]
                    
                    if self.smi_func_type == 'flcg':
                        obj = submodlib.FacilityLocationConditionalGainFunction(n=gt_data_sijs.shape[0], 
                                                                            num_privates=private_sijs.shape[1],
                                                                            data_sijs=gt_data_sijs,
                                                                            private_sijs=private_sijs)
                    elif self.smi_func_type == 'gccg':
                        obj = submodlib.GraphCutConditionalGainFunction(n=gt_data_sijs.shape[0], 
                                                                        num_privates=private_sijs.shape[1],
                                                                        lambdaVal=1,
                                                                        data_sijs=gt_data_sijs,
                                                                        private_sijs=private_sijs)
                    else:
                        private_private_sijs = data_sijs[private_idxs]
                        private_private_sijs = private_private_sijs[:, private_idxs]
                        obj = submodlib.LogDeterminantConditionalGainFunction(n=gt_data_sijs.shape[0], 
                                                                        num_privates=private_sijs.shape[1],
                                                                        lambdaVal=1,
                                                                        data_sijs=gt_data_sijs,
                                                                        private_sijs=private_sijs,
                                                                        private_private_sijs=private_private_sijs)

                    temp = obj.maximize(budget=private_partition_budget_split[i], optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                    stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)
                    temp =list(temp)
                    [greedyList.append((rem_idxs[temp[i][0]], temp[i][1])) for i in range(len(temp))]
                    [private_idxs.append(rem_idxs[temp[i][0]]) for i in range(len(temp))]
                    del private_sijs
                    del gt_data_sijs
                    del obj
                    if self.smi_func_type == 'logdetcg':
                        del private_private_sijs
                    obj = []
            else:
                greedyList = obj.maximize(budget=partition_budget_split, optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                        stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)
            del partition_train_rep
            if query_representations is None:
                del partition_query_rep
            del obj
            del query_sijs
            if self.smi_func_type in ['fl1mi', 'logdetmi', 'flcg']:
                del data_sijs
            if self.smi_func_type in ['logdetmi']:
                if query_representations is None:
                    del query_query_sijs
            self.logger.info("Partition {}: {} greedy queries".format(partition_cnt, len(greedyList)))
            partition_cnt += 1
            greedyIdxs.extend([partition[x[0]] for x in greedyList])  
        originalIdxs = [indices[x] for x in greedyIdxs]
        assert len(set(originalIdxs)) == (partition_budget_split * self.num_partitions), "Selected subset must be equal to the budget"
        smi_end_time = time.time()
        self.logger.info("SMI algorithm Subset Selection time is: %.4f", smi_end_time - smi_start_time)
        return originalIdxs
