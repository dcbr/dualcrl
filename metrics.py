import numpy as np
from typing import Optional

import utils


class Metric:
    __metrics = {}
    metric_id = None
    keys = tuple()

    def __init_subclass__(cls, **kwargs):
        metric_id = kwargs.pop("metric_id")
        super().__init_subclass__(**kwargs)
        cls.__metrics[metric_id] = cls
        cls.metric_id = metric_id

    @classmethod
    def from_id(cls, metric_id, *metric_args, **metric_kwargs):
        return cls.__metrics[metric_id](*metric_args, **metric_kwargs)

    def reset(self):
        pass

    def process(self, batches):
        pass

    def values(self):
        return tuple()

    def results(self):
        return dict(zip(self.keys, self.values()))


class ReturnMetric(Metric, metric_id="return"):
    keys = ("Rmin", "Rmean", "Rmax")

    def __init__(self, run):
        super().__init__()
        self.Rs = []
        self.gamma = run.method.params.gamma

    def reset(self):
        self.Rs.clear()

    def process(self, batches):
        for batch in batches:
            rewards = [exp.r for exp in batch]
            L = len(batch)
            self.Rs.append(np.sum(self.gamma ** np.arange(L) * rewards))

    def values(self):
        return np.min(self.Rs), np.mean(self.Rs), np.max(self.Rs)


class LengthMetric(Metric, metric_id="length"):
    keys = ("Lmin", "Lmean", "Lmax")

    def __init__(self, run):
        super().__init__()
        self.Ls = []

    def reset(self):
        self.Ls.clear()

    def process(self, batches):
        for batch in batches:
            self.Ls.append(len(batch))

    def values(self):
        return np.min(self.Ls), np.mean(self.Ls), np.max(self.Ls)


class StateMetric(Metric, metric_id="state"):
    keys = ("state",)

    def __init__(self, run):
        super().__init__()
        self.states = []

    def reset(self):
        self.states.clear()

    def process(self, batches):
        for batch in batches:
            self.states.append([exp.s0 for exp in batch] + [batch[-1].s1])

    def values(self):
        return np.array(self.states),


class RoughnessMetric(Metric, metric_id="roughness"):
    keys = ("state_roughness", "action_roughness")

    def __init__(self, run):
        super().__init__()
        self.s_range = run.env.observation_space.high - run.env.observation_space.low
        self.a_range = run.env.action_space.high - run.env.action_space.low
        self.s_rhos = []
        self.a_rhos = []

    def reset(self):
        self.s_rhos.clear()
        self.a_rhos.clear()

    def process(self, batches):
        for batch in batches:
            ds = np.array([exp.s1 - exp.s0 for exp in batch])
            actions = [exp.a0 for exp in batch]
            da = np.diff(actions, axis=0)
            s_rho = np.sum((ds / self.s_range) ** 2, axis=1)
            a_rho = np.sum((da / self.a_range) ** 2, axis=1)
            self.s_rhos.append(s_rho)
            self.a_rhos.append(a_rho)

    def values(self):
        return np.mean(np.stack(self.s_rhos, axis=0)), np.mean(np.stack(self.a_rhos, axis=0))


class ValueMetric(Metric, metric_id="value"):
    keys = ("value",)

    def __init__(self, run):
        super().__init__()
        self.gamma = run.method.params.gamma
        _, self.S = utils.dims_from_space(run.env.observation_space)
        self.rsum = np.zeros((self.S,))
        self.visitations = np.zeros((self.S,))

    def reset(self):
        self.rsum = np.zeros((self.S,))
        self.visitations = np.zeros((self.S,))

    def process(self, batches):
        for batch in batches:
            states = [exp.s0 for exp in batch]
            rewards = [exp.r for exp in batch]
            L = len(batch)
            discounts = self.gamma ** np.arange(L)
            returns = np.flip(np.cumsum(np.flip(discounts * rewards))) / discounts
            np.add.at(self.rsum, states, returns)
            np.add.at(self.visitations, states, 1)

    def values(self):
        return np.where(self.visitations == 0, np.nan, self.rsum / self.visitations),


class DensityMetric(Metric, metric_id="density"):
    keys = ("density",)

    def __init__(self, run):
        super().__init__()
        self.ds = []
        self.gamma = run.method.params.gamma
        _, self.S = utils.dims_from_space(run.env.observation_space)

    def reset(self):
        self.ds.clear()

    def process(self, batches):
        for batch in batches:
            states = [exp.s0 for exp in batch] + [batch[-1].s1]
            L = len(batch) + 1
            d = np.zeros((self.S,))
            np.add.at(d, states, self.gamma ** np.arange(L))
            self.ds.append(d / np.sum(d))  # Normalize

    def values(self):
        return np.mean(self.ds, axis=0),
