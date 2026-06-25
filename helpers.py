from functools import cache

import numpy as np
from numpy.typing import NDArray

__all__ = (
    "BoundConditionBase",
    "BoundConditionFirstType",
    "BoundConditionSecondType",
    "ThomasMethod",
    "binary_search_index",
    "DivDiff"
)

class DivDiff:

    def __init__(self, x: NDArray, f: NDArray):
        assert x.shape[0] == f.shape[0]
        self._x = x
        self._f = f

    @cache
    def __call__(self, n: int, i: int):
        x = self._x
        f = self._f
        assert i <= x.shape[0] - n - 1
        if n == 0:
            return f[i]

        return (self(n - 1, i + 1) - self(n - 1, i)) / (x[i + n] - x[i])


class ThomasMethod:
    def __init__(self, a: NDArray, b: NDArray, c: NDArray, d: NDArray):
        assert a.shape[0] == b.shape[0] == c.shape[0] == d.shape[0]
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    @property
    def solution(self):
        return self._d

    def solve(self):
        a = self._a
        b = self._b
        c = self._c
        d = self._d
        N = a.shape[0]

        c[0] = c[0] / b[0]
        d[0] = d[0] / b[0]
        for i in range(1, N - 1):
            c[i] = c[i] / (b[i] - a[i] * c[i - 1])
            d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1])

        d[N - 1] = (d[N - 1] - a[N - 1] * d[N - 2]) / (b[N - 1] - a[N - 1] * c[N - 2])

        for i in range(N - 2, -1, -1):
            d[i] = -c[i] * d[i + 1] + d[i]


class BoundConditionBase:
    def __init__(self, f_a, f_b):
        self.f_a = f_a
        self.f_b = f_b


class BoundConditionFirstType(BoundConditionBase):
    pass


class BoundConditionSecondType(BoundConditionBase):
    pass


def binary_search_index(arr: NDArray, t: float):
    n = arr.shape[0]

    if arr[0] > t:
        return -1
    elif arr[n - 1] < t:
        return n

    l = 0
    r = n
    while r - l > 1:
        m = (r - l) // 2
        if arr[l + m] < t:
            l = l + m
        else:
            r = l + m

    return l


