#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

class Ldict(object):
    '''
    Ldict use a list for a Gather 

    example:
    >>> records = Ldict()
    >>> records.append({'loss': 0.1, 'time': 0.4})
    >>> records.data
    {'loss': [0.1], 'time': [0.4]}
    >>> records.extend({'loss': [0.15, 0.13], 'time': [0.41, 0.39]})
    >>> records.data
    {'loss': [0.1, 0.15, 0.13], 'time': [0.4, 0.41, 0.39]}
    '''

    def __init__(self):
        self._data = defaultdict(lambda:[])


    def append(self, dict_Gathers):
        
        for k, v in dict_Gathers.items():
            self._data[k].append(v)


    def extend(self, dict_datas):
        
        for k, v in dict_datas.items():
            self._data[k].extend(v)


    @property
    def data(self):
        return {k:v for k, v in self._data.items()}


class SmoothGather(object):
    """
    存储参数当前值, 并计算延迟平滑值
    store current value and 
    Compute the delay smooth value of all updated values

    example:
    >>> gather = SmoothGather()
    >>> gather.update(1, 4)
    >>> gather.update(2, 3)
    >>> 'sum: %d, count: %d, avg: %.2f' % (gather.sum, gather.count, gather.avg)
    'sum: 10, count: 7, avg: 1.43'
    """

    def __init__(self, delay=0.99):
        
        self.reset()
        self.delay = delay


    def reset(self):
        
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0


    def update(self, val, n=1):
        
        self.val = val
        self.count += n
        if(self.count > 20):
            delay = self.delay**n
            self.avg = self.avg*delay + val*(1-delay)
        else:
            self.sum += val * n
            self.avg = self.sum / self.count


class AverageGather(object):
    """
    存储参数当前值, 并计算平均值
    store current value and 
    Compute the average of all updated values

    example:
    >>> gather = AverageGather()
    >>> gather.update(1, 4)
    >>> gather.update(2, 3)
    >>> 'sum: %d, count: %d, avg: %.2f' % (gather.sum, gather.count, gather.avg)
    'sum: 10, count: 7, avg: 1.43'
    """

    def __init__(self):
        self.reset()


    def reset(self):
        
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0


    def update(self, val, n=1):
        
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


if __name__ == '__main__':

    import doctest
    doctest.testmod()
    
    
