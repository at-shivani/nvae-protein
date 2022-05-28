
import numpy as np 

class BaseMetric:
    def __init__(self):
        self.collect = []

    def __call__(self, ytrue, ypred):
        raise NotImplementedError

    def aggr(self):
        raise NotImplementedError
    

class RankMetric(BaseMetric):

    def __call__(self, ytrue, ypred):
        dists = np.array([np.abs(ytrue - yp).mean(axis=-1) for yp in ypred])
        sorted_dist = dists.argsort(axis=-1)
        ranks = [np.where(d==i)[0][0] for i, d in enumerate(sorted_dist)]
        self.collect.extend(ranks)
        return ranks

    def aggr(self, c=None):
        c = c or self.collect
        if not c: print('Empty Collection, invalid aggr')
        # print(c)
        return sum([1/(i+1) for i in c])/len(c)

class AvgRank(BaseMetric):
    def __init__(self):
        super().__init__()
        self.rankmetric = RankMetric() 
    
    def __call__(self, ytrue, ypred):
        self.rankmetric(ytrue, ypred)
        
    def aggr(self):
        c = self.rankmetric.collect
        if not c: print('Empty Collection, invalid aggr')
        return sum([i for i in c])/len(c)

        
    

class RecallAtN(BaseMetric):
    def __init__(self, n=5):
        super().__init__()
        self.n = n 
        self.rankmetric = RankMetric() 


    def __call__(self, ytrue, ypred):
        ranks = self.rankmetric(ytrue, ypred)
        recall_at_n = [r<self.n for r in ranks]
        self.collect.extend(recall_at_n)
        return recall_at_n
    
    def aggr(self, c=None):
        c = c or self.collect
        if not c: print('Empty Collection, invalid aggr')
        return sum([i for i in c])/len(c)

        

        


