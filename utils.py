import dataclasses
from typing import Mapping, Optional, Sequence
from weakref import WeakSet

import gymnasium as gym
import numpy as np
import torch

rng = np.random.default_rng()


def set_seed(seed: Optional[int]):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)


class Schedule:
    __schedules = {}

    def __init__(self, v0, t=None):
        """Base schedule. Initial value v0 and trigger name t."""
        super().__init__()

        # Initialize this instance:
        self.v0 = v0
        self.v = v0
        self._t = None
        self.c = 0  # Trigger counter
        self.t = t

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.t = self._t  # Ensure the copied schedule is properly registered

    @classmethod
    def trigger(cls, t=None):
        for schedule in cls.__schedules.get(t, []):
            schedule.update()

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, nt):
        # Remove from previous trigger collection
        if self in self.__schedules.get(self._t, []):
            self.__schedules[self._t].remove(self)
        self._t = nt
        # Add to new trigger collection
        if nt not in self.__schedules:
            self.__schedules[nt] = WeakSet()
        self.__schedules[nt].add(self)

    def update(self):
        self.c += 1
        self._update()

    def _update(self):
        pass

    def __float__(self):
        # Return current float value
        return self.v

    def __int__(self):
        # Return current float value as a rounded integer
        return int(round(self.v))

    def __bool__(self):
        # Return current float value as a boolean (derived from the current rounded integer)
        return bool(int(self))


class ConstantSchedule(Schedule):

    def __init__(self, v, t=None):
        """Schedule with a constant value."""
        super().__init__(v, t)


class LinearSchedule(Schedule):

    def __init__(self, v0, v1, s, t=None):
        """Linear schedule: starting from v0 and going to v1 in s triggers."""
        super().__init__(v0, t)
        self.vm = min(v0, v1)
        self.vM = max(v0, v1)
        self.dv = (v1 - v0) / s

    def _update(self):
        self.v += self.dv
        self.v = min(self.v, self.vM)
        self.v = max(self.v, self.vm)


class ExponentialSchedule(Schedule):

    def __init__(self, v0, v1, s, t=None):
        """Exponential schedule: starting from v0 and going to v1 in s triggers."""
        super().__init__(v0, t)
        self.v1 = v1
        eps = 0.0001
        self.vf = np.power(eps, 1.0 / s)  # Exponential multiplication factor of v, required to change its value from v0 to v1+-eps*(v1-v0) in s steps

    def _update(self):
        self.v = self.v1 + (self.v - self.v1) * self.vf


class PiecewiseSchedule(Schedule):

    def __init__(self, schedules, t=None):
        """Piecewise schedule: connecting the given schedules after the specified trigger counts."""
        durations = [*schedules.keys()]
        assert all(d > 0 for d in durations)
        self.counts = np.cumsum(schedules.keys())
        self.schedules = [*schedules.values()]
        self.si = 0  # Active schedule index
        super().__init__(self.schedules[self.si].v, t)

        for schedule in self.schedules:
            schedule.t = "piecewise_trigger"  # Unused trigger, we manually update the schedules when they become active
        self.counts[-1] = np.infty  # Let last schedule run infinitely

    def _update(self):
        self.schedules[self.si].update()
        self.v = self.schedules[self.si].v

        if self.c >= self.counts[self.si]:
            self.si += 1


def dims_from_space(space: gym.spaces.Space):
    continuous_dim = None
    discrete_dim = None
    if isinstance(space, gym.spaces.Discrete):
        discrete_dim = space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        discrete_dim = space.nvec
    elif isinstance(space, gym.spaces.Box):
        # Note: can still be discrete if the space's dtype is an integer type
        continuous_dim = space.shape
    return continuous_dim, discrete_dim


class RavelTransform:
    """Torch implementation of np.ravel_multi_index and np.unravel_index.
    Code adopted from francois-rozet/torchist and pytorch/pytorch#66687."""

    def __init__(self, shape, dtype=torch.int64, device=None):
        """shape: int, sequence or Tensor of shape (D,)"""
        # Convert to a tensor, with the same properties as that of indices
        if isinstance(shape, Sequence):
            self.shape_tensor = torch.tensor(shape, dtype=dtype, device=device)
        elif isinstance(shape, int) or (isinstance(shape, torch.Tensor) and shape.ndim == 0):
            self.shape_tensor = torch.tensor((shape,), dtype=dtype, device=device)
        else:
            self.shape_tensor = shape

        # By this time, shape tensor will have dim = 1 if it was passed as scalar (see if-elif above)
        assert self.shape_tensor.ndim == 1, "Expected dimension of shape tensor to be <= 1, "
        f"but got the tensor with dim: {self.shape_tensor.ndim}."

        # Calculate coefficients
        coefs = self.shape_tensor[1:].flipud().cumprod(dim=0).flipud()
        self.coefs = torch.cat((coefs, coefs.new_tensor((1,))), dim=0)

    def ravel(self, indices: torch.Tensor) -> torch.Tensor:
        """Torch implementation of np.ravel_multi_index.

        indices: Tensor of shape (*B, D)

        returns Tensor of shape (*B,)
        """
        # if torch.any(indices >= self.shape_tensor):
        #     raise ValueError("Target shape should cover all source indices")

        return torch.sum(indices * self.coefs, dim=-1)

    def unravel(self, indices: torch.Tensor) -> torch.Tensor:
        """Torch implementation of np.unravel_index.

        indices: Tensor of shape (*B,)

        returns Tensor of shape (*B, D)
        """
        # if torch.max(indices) >= torch.prod(self.shape_tensor):
        #     raise ValueError("Target shape should cover all source indices.")

        return torch.div(torch.unsqueeze(indices, dim=-1), self.coefs, rounding_mode='trunc') % self.shape_tensor


def make_tensor(data):
    """Convert data to tensor while omitting UserWarning."""
    if isinstance(data, torch.Tensor):
        return data.clone().detach()
    else:
        return torch.tensor(data)


@dataclasses.dataclass
class BaseDataClass:

    @classmethod
    def create(cls, obj):
        fields = {}
        if isinstance(obj, cls):
            # fields = dataclasses.asdict(obj)  # This does a deep copy, causing nested BaseDataClasses to be converted to dict
            fields = dict((field.name, getattr(obj, field.name)) for field in dataclasses.fields(obj))
        elif isinstance(obj, Mapping):
            fields = obj
        return cls(**fields)

    def copy(self, **changes):
        return dataclasses.replace(self, **changes)
