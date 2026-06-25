from abc import ABC, abstractmethod
from numpy.typing import NDArray

__all__ = [
    "ConstantSpline",
    "LinearSpline",
    "SplineBase",
    "SplineNodeBase",
]

class SplineNodeBase(ABC):
    def __init__(self, f_i: NDArray, m_i: NDArray, x_i: NDArray):
        self._f_i = f_i
        self._m_i = m_i
        self._h = x_i[1] - x_i[0]
        self._x_i = x_i

    @property
    def h(self) -> float:
        return self._h

    @property
    @abstractmethod
    def momentum_0(self) -> float: ...

    @property
    @abstractmethod
    def momentum_1(self) -> float: ...

    @abstractmethod
    def __call__(self, t: float) -> float: ...


class SplineBase(ABC):
    def __init__(self, x: NDArray, f: NDArray):
        assert x.shape[0] == f.shape[0]
        assert x.shape[0] > 1
        self._x = x
        self._f = f

    @abstractmethod
    def fit(self) -> None: ...

    @abstractmethod
    def __call__(self, t: float) -> float: ...


class ConstantSpline(SplineBase):
    def fit(self) -> None: ...

    def __call__(self, t: float) -> float:
        if self._x[0] <= t <= self._x[-1]:
            return self._f[0]
        return 0


class LinearSpline(SplineBase):
    def fit(self) -> None: ...

    def __call__(self, t: float) -> float:
        if self._x[0] <= t <= self._x[-1]:
            return self._f[0] + (self._f[1] - self._f[0]) / (self._x[1] - self._x[0]) * (t - self._x[0])
        return 0
