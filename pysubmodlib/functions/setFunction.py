import math
import random
from functools import partial
# from sortedcontainers import SortedSet
from tqdm.auto import tqdm as tq
from multiprocessing import Pool
import tqdm
from numba import jit, njit
import numpy as np

class SetFunction():
    def __init__(self):
        pass

    def evaluate(self, X):
        pass

    def marginalGain(self, X, element):
        pass

    def marginalGainWithMemoization(self, X, element):
        pass

    def evaluateWithMemoization(self, X):
        pass

    def updateMemoization(self, X, element):
        pass

    def clearMemoization(self):
        pass

    def setMemoization(self, X):
        pass

    def getEffectiveGroundSet(self):
        pass

    def maximize(self, budget, optimizer="NaiveGreedy", stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        if budget>=len(self.effective_ground):
            raise Exception("Budget must be less than effective ground set size")
        
        if type(costs)==type(None):
            costs=[]
        else:
            if len(costs)!=self.n:
                raise Exception("ERROR: Mismatch between length of costs and number of elements in the ground set")

        if optimizer=="NaiveGreedy":
            progressbar=tq(range(budget))
            greedyVector=[]
            greedySet=set()
            rem_budget=budget
            groundSet=self.effective_ground
            if verbose:
                print(f"Num elements in Ground Set = {len(self.effective_ground)}")
                print("Starting the Naive Greedy Algorithm")
            self.clearMemoization()
            best_id=0
            best_val=0.0
            step=1
            displayNext=step
            percent=0
            N=rem_budget
            iter=0
            while rem_budget>0:
                best_id=-1
                best_val=-math.inf
                remaining_set=list(groundSet.difference(greedySet))
                with Pool(96) as p:
                    gains=list(tqdm.tqdm(p.map(partial(self.marginalGainWithMemoization, greedySet), remaining_set), total=len(remaining_set)))
                idx=np.argmax(gains)
                best_id=remaining_set[idx]
                best_val=gains[idx]
                if verbose:
                    if best_id==-1:
                        raise Exception("Nobody has greater gain than minus infinity")
                    print(f"Next best item to add is {best_id} and its value addition is {best_val}")
                if (best_val<0 and stopIfNegativeGain) or (abs(best_val)<1e-5 and stopIfZeroGain):
                    break
                else:
                    self.updateMemoization(greedySet, best_id)
                    greedySet.add(best_id)
                    greedyVector.append((best_id, best_val))
                    rem_budget-=1
                    if verbose:
                        print(f"Added element {best_id} and the gain is {best_val}")
                    progressbar.update(1)
            return greedyVector
        elif optimizer=="LazyGreedy":
            pass
        elif optimizer=="StochasticGreedy":
            progressbar=tq(range(budget))
            greedyVector=[]
            greedySet=set()
            rem_budget=budget
            remaining_set=self.effective_ground.copy()
            n=len(remaining_set)
            random_set_size=int((n*1.0/budget )* math.log(1/epsilon))
            if verbose:
                print(f"Epsilon = {epsilon}")
                print(f"Random set size = {random_set_size}")
                print("Ground Set: ")
                for i in self.effective_ground:
                    print(i, end=" ")
                print(f"\nNum elements in Ground Set = {len(self.effective_ground)}")
                print("Starting the stochastic greedy algorithm")
                print("Initial Greedy Set: ")
                for i in greedySet:
                    print(i, end=" ")
                print("\n")
            self.clearMemoization()
            random.seed(23)
            best_id=0
            best_val=0.0

            i=0
            step=1
            displayNext=step
            percent=0
            N=rem_budget
            iter=0

            while rem_budget>0:
                random_set=set()
                while len(random_set)<random_set_size:
                    elem=random.randint(0, self.n)   
                    if (elem in remaining_set) and (elem not in random_set):
                        random_set.add(elem)
                # random_set=set(list(random.sample(list(remaining_set), random_set_size)))
                if verbose:
                    print(f"Iteration {i}")
                    print("Random Set: ")
                    for elem in random_set:
                        print(elem, end=" ")
                    print("\n")
                    print("Now running naive greedy on the random set")
                best_id=-1
                best_val=-math.inf
                # random_set=list(random_set)
                # gains=map(partial(self.marginalGainWithMemoization, greedySet), random_set)
                # gains=np.array(list(gains))
                # argmin_idx=np.argmin(gains)
                # best_id=random_set[argmin_idx]
                # best_val=gains[argmin_idx]
                for i in random_set:
                    gain=self.marginalGainWithMemoization(greedySet, i)
                    if gain>best_val:
                        best_id=i
                        best_val=gain
                if verbose:
                    if best_id==-1:
                        raise Exception("Nobody had greater gain than minus infinity")
                    print(f"Next best item to add is {best_id} and its value addition is {best_val}")
                if (best_val<0 and stopIfNegativeGain) or (abs(best_val)<1e-5 and stopIfZeroGain):
                    break
                else:
                    self.updateMemoization(greedySet, best_id)
                    greedySet.add(best_id)
                    progressbar.update(1)
                    greedyVector.append((best_id, best_val))
                    rem_budget-=1
                    remaining_set.remove(best_id)
                    if verbose:
                        print(f"Added element {best_id} and the gain is {best_val}")
                        print("Updated GreedySet: ")
                        for i in greedySet:
                            print(i, end=" ")
                        print("\n")
                i+=1
            return greedyVector
        elif optimizer=="LazierThanLazyGreedy":
            pass
            # progressbar=tq(range(budget))
            # greedyVector=[]
            # greedySet=set()
            # rem_budget=budget
            # remaining_set=self.effective_ground.copy()
            # n=len(remaining_set)
            # random_set_size=int((n*1.0 / budget) * math.log(1/epsilon))
            # if verbose:
            #     print(f"Epsilon = {epsilon}")
            #     print(f"Random set size = {random_set_size}")
            #     # print("Ground Set: ")
            #     # for i in self.effective_ground:
            #     #     print(i, end=" ")
            #     print(f"\nNum elements in Ground Set = {len(self.effective_ground)}")
            #     print("Starting the LazierThanLazy Greedy Algorithm")
            #     # print("Initial Greedy Set:")
            #     # for i in greedySet:
            #     #     print(i, end=" ")
            #     # print("\n")
            # self.clearMemoization()
            # random.seed(23)
            # best_id=0
            # best_val=0.0

            # sorted_gains=SortedSet(key=lambda x:(-x[0],-x[1]))
            # with Pool(96) as p:
            #     gains=list(tqdm.tqdm(p.map(partial(self.marginalGainWithMemoization, greedySet), remaining_set), total=len(remaining_set)))
            # for gain, element in zip(gains, list(remaining_set)):
            #     sorted_gains.add((gain, element))
            # # pbar=tqdm(range(len(remaining_set)))
            # # for element in remaining_set:
            # #     sorted_gains.add((self.marginalGainWithMemoization(greedySet, element), element))
            # #     pbar.update(1)
            # if verbose:
            #     # print("Initial Sorted Set =")
            #     # print(sorted_gains)
            #     pass
            
            # i=0
            # step=1
            # displayNext=step
            # percent=0
            # N=rem_budget
            # iter=0
            # while rem_budget>0:
            #     random_set=set(list(random.sample(list(remaining_set), random_set_size)))
            #     if verbose:
            #         print(f"Iteration {i}")
            #         # print("Random set = ")
            #         # for element in random_set:
            #         #     print(element, end=" ")
            #         print("\nNow running lazy greedy on the random set")
            #     best_id=0
            #     best_val=0.0
            #     candidate_val=0.0
            #     candidate_id=0
            #     newCandidateBound=0.0

            #     it=0
            #     while True:
            #         if it==len(sorted_gains):
            #             break
            #         if verbose:
            #             # print("Current sorted gains=")
            #             # print(sorted_gains)
            #             pass
            #         p=sorted_gains[it]
            #         if p[1] in random_set:
            #             if verbose:
            #                 print(p)
            #                 print("...present in random set...")
            #             candidate_id=p[1]
            #             candidate_val=p[0]
            #             newCandidateBound=self.marginalGainWithMemoization(greedySet, candidate_id)
            #             if verbose:
            #                 print(f"Updated gain as per updated greedy set = {newCandidateBound}")
            #             next_p=sorted_gains[it+1]
            #             if verbose:
            #                 print(f"Next element is: {next_p[0]},{next_p[1]}")
            #             if newCandidateBound >= next_p[0]:
            #                 if verbose:
            #                     print("...better than next best upper bound, selecting...")
            #                 best_id=candidate_id
            #                 best_val=newCandidateBound
            #                 break
            #             else:
            #                 if verbose:
            #                     print("...NOT better than next best upper bound, updating...")
            #                     print("...updating its value in sorted set")
            #                 sorted_gains.pop(it)
            #                 sorted_gains.add((newCandidateBound, candidate_id))
            #         else:
            #             it+=1
            #     if verbose:
            #         print(f"Next best item to add is {best_id} and its value addition is {best_val}")
            #     sorted_gains.discard((candidate_val, candidate_id))

            #     if (best_val<0 and stopIfNegativeGain) or (abs(best_val)<1e-5 and stopIfZeroGain):
            #         break
            #     else:
            #         self.updateMemoization(greedySet, best_id)
            #         greedySet.add(best_id)
            #         progressbar.update(1)
            #         greedyVector.append((best_id, best_val))
            #         rem_budget-=1
            #         remaining_set.remove(best_id)
            #         if verbose:
            #             print(f"Added element {best_id} and the gain is {best_val}")
            #             # print("Updated greedy set:")
            #             # for i in greedySet:
            #             #     print(i, end=" ")
            #             # print("\n")
            #     i+=1
            # return greedyVector