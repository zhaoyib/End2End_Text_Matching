'''
@File    :   Basic_LSH.py
@Time    :   2024/03/11 09:28:56
@Author  :   YiboZhao 
@Version :   1.0
@Site    :   https://github.com/zhaoyib
'''

import numpy as np

class Basic_LSH:
    def __init__(self,dim:int,l:int,m:int,w:int,seed=1) -> None:
        self.dim = dim
        self.l = l
        self.m = m
        self.w = w

        np.random.seed(seed)
        #generate the normal distribution with shape l,m,dim
        self.a = np.random.randn(l,m,dim)
        #randomly generate the b
        self.b = np.random.rand(l,m)

        self.hash_tables = []
        for i in range (l):# l looks like 1 but not 1
            self.hash_tables.append({})
        pass

    def hash(self,point):
        '''
        hash function:  h = (a · v + b) / w 向下取整
        '''
        hash_values = np.floor((np.dot(self.a,point) + self.b) / self.w)# hash_values with the shape [l,m]
        hash_values.astype(np.int16)
        return hash_values
    
    def insert(self, point, label):
        '''
        point is the vector, and label is the cv_id
        '''
        hash_values = self.hash(point)
        for i in range(hash_values.shape[0]):#equal to i in range self.l
            self.hash_tables[i][tuple(hash_values[i])] = label

    def query(self, point):
        results = set()
        hash_values = self.hash(point)
        for i in range(self.l):
            key = tuple(hash_values[i])
            target = self.hash_tables[i].get(key)
            if(target):
                results.add(target)
        return results
    