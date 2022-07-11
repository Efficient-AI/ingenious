import submodlib
import time

class KnnSMIStrategy():
    def __init__(self, logger, smi_func_type, 
                optimizer="LazyGreedy", similarity_criterion='feature',
                metric="cosine", eta=1, stopIfZeroGain=False,
                stopIfNegativeGain=False, verbose=False, lambdaVal=1,
        ):
        """
        Constructor method
        """
        self.train_rep=None
        self.query_rep=None
        self.private_rep=None
        self.indices=None
        self.logger=logger
        self.optimizer=optimizer
        self.smi_func_type=smi_func_type
        self.metric=metric
        self.eta=eta
        self.stopIfZeroGain=stopIfZeroGain
        self.stopIfNegativeGain=stopIfNegativeGain
        self.verbose=verbose
        self.lambdaVal=lambdaVal
        self.similarity_criterion=similarity_criterion
    
    def select(self, budget, representations,
                index_key, ngpu=-1, tempmem=-1, altadd=False,
                use_float16=True, use_precomputed_tables=True,
                replicas=1, max_add=-1, add_batch_size=32768,
                query_batch_size=16384, nprobe=128, nnn=10):
        smi_start_time=time.time()
        if self.smi_func_type not in ["fl", "gc", "logdet"]:
            assert False, f"{self.smi_func_type} not yet supported"
        kernel_time=time.time()
        data_sijs=submodlib.helper.create_sparse_kernel_faiss_innerproduct(
            X=representations, index_key=index_key, logger=self.logger, ngpu=ngpu, 
            tempmem=tempmem, altadd=altadd, 
            use_float16=use_float16, use_precomputed_tables=use_precomputed_tables, 
            replicas=replicas, max_add=max_add, add_batch_size=add_batch_size, 
            query_batch_size=query_batch_size, nprobe=nprobe, nnn=nnn,
        )
        n=representations.shape[0]
        del representations
        self.logger.info(f"Kernel Computation Time: {time.time()-kernel_time}")
        greedy_selection_start_time=time.time()
        if self.smi_func_type=="fl":
            obj=submodlib.FacilityLocationFunction(n=n,
                                                    separate_rep="False",
                                                    mode="sparse",
                                                    sijs=data_sijs,
                                                    num_neighbors=nnn)
        if self.smi_func_type=="gc":
            obj=submodlib.GraphCutFunction(n=n,
                                            mode="sparse",
                                            lambdaVal=self.lambdaVal,
                                            separate_rep=False,
                                            ggsijs=data_sijs,
                                            num_neighbors=nnn)
        if self.smi_func_type=="logdet":
            obj=submodlib.LogDeterminantFunction(n=n,
                                                    mode="sparse",
                                                    lambdaVal=self.lambdaVal,
                                                    sijs=data_sijs,
                                                    num_neighbors=nnn)
        greedyList=obj.maximize(budget=budget, optimizer=self.optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=True)
        self.logger.info(f"Greedy Selection Time: {time.time()-greedy_selection_start_time}")
        self.logger.info(f"Total Subset Selection time: {time.time()-smi_start_time}")
        return greedyList