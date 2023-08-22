"""
@file   annealers.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Scalar hyperparameter annealling tools.
"""

import numpy as np
from typing import Any, List

from nr3d_lib.config import ConfigDict

def get_anneal_val(type: str, **params) -> float:
    if type == 'linear':
        return get_anneal_val_linear(**params)
    elif type == 'logspace':
        return get_anneal_val_logspace(**params)
    elif type == 'milestones':
        return get_anneal_val_milestones(**params)
    else:
        raise RuntimeError(f"Invalid type={type}")

def get_anneal_val_linear(it: int, *, stop_it: int, start_it: int = 0, start_val: float = 0.0, stop_val: float = 1.0, update_every: int=1) -> float:
    cur_stage, total_stages = (it - start_it)//update_every, (stop_it-start_it)//update_every
    alpha = min(1.0, max(0.0, cur_stage / total_stages))
    return (1-alpha) * start_val + alpha * stop_val

def get_anneal_val_logspace(it: int, *, stop_it: int, start_it: int = 0, start_val: float = 1.0, stop_val: float = 10.0, update_every: int=1) -> float:
    cur_stage, total_stages = (it - start_it)//update_every, (stop_it-start_it)//update_every
    alpha = min(1.0, max(0.0, cur_stage / total_stages))
    logv0, logv1 = np.log(start_val), np.log(stop_val)
    return np.exp(logv0 * (1-alpha) + logv1 * alpha)

def get_anneal_val_milestones(it: int, milestones: List[int], vals: List[Any]) -> float:
    assert (len(milestones)+1) == len(vals), '`vals` should have one more element than `milestones`'
    # NOTE: a[i-1] <= v < a[i]
    i = np.searchsorted(milestones, it, side='right')
    v = vals[i]
    return v

def get_annealer(type: str = None, **params):
    if type == 'linear':
        a = AnnealerLinear(**params)
    elif type == 'logspace':
        a = AnnealerLogSpace(**params)
    elif type == 'constant':
        a = AnnealerConstant(**params)
    elif type == 'milestones':
        a = AnnealerMilestones(**params)
    elif type == 'partitions':
        a = AnnealerPartitions(**params)
    else:
        raise RuntimeError(f"Invalid type={type}")
    a.type = type
    return a
class AnnealerBase(object):
    def __init__(self) -> None:
        self._it = None
        self._bypass_val = None
    def __call__(self, it: int):
        raise NotImplementedError
    def set_iter(self, it: int):
        self._it = it
    def set_alpha(self, alpha: float):
        assert alpha >= 0 and alpha <= 1
    def set_val(self, val: float):
        self._bypass_val = val
    def get_val(self):
        raise NotImplementedError

class AnnealerPartitions(object):
    # Annealers with arbitary partitions
    def __init__(self, partition_cfgs: List[ConfigDict]) -> None:
        self._it = None
        self._bypass_val = None
        
        partitions: List[AnnealerBase] = []
        partition_stops: List[int] = []
        prev_stop_it = 0
        for i, cfg in enumerate(partition_cfgs):
            
            cfg.setdefault('start_it', prev_stop_it)
            a = get_annealer(**cfg)
            partitions.append(a)
            
            prev_stop_it = cfg.stop_it
            partition_stops.append(prev_stop_it)
        
        self.partitions = partitions
        self.partition_stops = np.array(partition_stops)
    def __call__(self, it: int):
        if self._bypass_val is not None:
            return self._bypass_val
        else:
            # NOTE: a[i-1] <= v < a[i]
            i = np.searchsorted(self.partition_stops, it, side='right')
            p = self.partitions[i]
            p.set_iter(it)
            return p.get_val()
    def set_iter(self, it: int):
        self._it = it
    def set_val(self, val: float):
        self._bypass_val = val
    def get_val(self):
        return self(self._it)

class AnnealerConstant(AnnealerBase):
    def __init__(self, val: float) -> None:
        super().__init__()
        self.val = val
    def __call__(self, it: int):
        return self.val
    def set_iter(self, it: int):
        pass
    def set_val(self, val: float):
        self.val = val
    def get_val(self):
        return self.val

class AnnealerMilestones(AnnealerBase):
    def __init__(self, milestones: List[int], vals: List[Any]) -> None:
        """
        e.g.
        milestones: [100, 300]
        vals: [0.1, 1.0, 10.0]
        >>> val=0.1,  if it < 100
        >>> val=1.0,  if 100 <= it < 300
        >>> val=10.0, if it >= 300
        """
        super().__init__()
        assert (len(milestones)+1) == len(vals), '`vals` should have one more element than `milestones`'
        self.milestones = milestones
        self.vals = vals
    def __call__(self, it: int):
        if self._bypass_val is not None:
            return self._bypass_val
        else:
            # NOTE: a[i-1] <= v < a[i]
            i = np.searchsorted(self.milestones, it, side='right')
            v = self.vals[i]
            return v

class AnnealerLinear(AnnealerBase):
    def __init__(self, stop_it: int, start_it: int = 0, start_val: float = 0.0, stop_val: float = 1.0, update_every: int=1) -> None:
        super().__init__()
        self.start_it = int(start_it)
        self.stop_it = int(stop_it)
        self.update_every = int(update_every)
        self.total_stages = (self.stop_it - self.start_it) // self.update_every
        self.start_val = start_val
        self.stop_val = stop_val
    def __call__(self, it: int):
        if self._bypass_val is not None:
            return self._bypass_val
        else:
            if it < self.start_it:
                return self.start_val
            elif it >= self.stop_it:
                return self.stop_val
            else:
                cur_stage = (it - self.start_it) // self.update_every
                alpha = min(1.0, max(0.0, cur_stage / self.total_stages))
                return (1-alpha) * self.start_val + alpha * self.stop_val
    def get_val(self):
        assert self._it is not None, "Please call `set_iter` first"
        return self(self._it)

class AnnealerLogSpace(AnnealerBase):
    def __init__(self, stop_it: int, start_it: int = 0, start_val: float = 1.0, stop_val: float = 10.0, update_every: int=1) -> None:
        super().__init__()
        assert start_val > 0, "Invalid log(start_val)"
        self.start_it = int(start_it)
        self.stop_it = int(stop_it)
        self.update_every = int(update_every)
        self.total_stages = (self.stop_it - self.start_it) // self.update_every
        self.start_val = start_val
        self.stop_val = stop_val
    def __call__(self, it: int):
        if self._bypass_val is not None:
            return self._bypass_val
        else:
            if it < self.start_it:
                return self.start_val
            elif it >= self.stop_it:
                return self.stop_val
            else:
                cur_stage = (it - self.start_it) // self.update_every
                alpha = min(1.0, max(0.0, cur_stage / self.total_stages))
                logv0, logv1 = np.log(self.start_val), np.log(self.stop_val)
                return np.exp(logv0 * (1-alpha) + logv1 * alpha)
    def get_val(self):
        assert self._it is not None, "Please call `set_iter` first"
        return self(self._it)

class AnnealerCosine():
    pass

class AnnealerLambda():
    pass

if __name__ == "__main__":
    def unit_test():
        import matplotlib.pyplot as plt
        
        an = AnnealerMilestones([100,300,600], [0.1, 0.2, 0.3, 0.4])
        vs = []
        its = list(range(1000))
        for it in its:
            v = an(it)
            vs.append(v)
        plt.plot(its, vs)
        plt.show()
        
        common = dict(
            start_val = 10., 
            start_it = 10, 
            stop_val = 100., 
            stop_it = 1000, 
            update_every = 150
        )
        
        cfgs = [
            {'type': 'linear', **common}, 
            {'type': 'logspace', **common}
        ]
        annealers = [get_annealer(**c) for c in cfgs]
        vals = {c['type']:[] for c in cfgs}
        for it in range(1001):
            for a in annealers:
                a.set_iter(it)
                vals[a.type].append(a.get_val())
        
        for i, (k, v) in enumerate(vals.items()):
            plt.subplot(1, 2, i+1)
            plt.plot(np.arange(1001), np.array(v), label='v')
        plt.show()
    unit_test()