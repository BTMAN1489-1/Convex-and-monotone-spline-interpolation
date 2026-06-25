import math
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from .base import *
from helpers import ThomasMethod, BoundConditionBase, BoundConditionFirstType, BoundConditionSecondType, \
    binary_search_index, DivDiff

__all__ = (
    "WeightMomentumNode",
    "WeightGradientNode",
    "WeightSplineBase",
    "WeightMomentumSpline",
    "WeightGradientSpline",
    "MonotoneWeightSpline",
    "ConvexWeightSpline"
)


class WeightSplineNode(SplineNodeBase, ABC):
    def __init__(self, f_i: NDArray, m_i: NDArray, x_i: NDArray, w: float = 1):
        super().__init__(f_i, m_i, x_i)
        self._w_max = 1000000
        self.w = w

    def __call__(self, t: float) -> float:
        tau = (t - self._x_i[0]) / self._h
        if 0 <= tau <= 1:
            return (self._f_i[0] * (1 - tau) + self._f_i[1] * tau - tau * (1 - tau) * (
                    self.momentum_0 * (2 - tau) + self.momentum_1 * (1 + tau)) * self.h ** 2 / (6 * self.w))
        return 0

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, w: float):
        assert w > 0
        self._w = min(w, self._w_max)


class WeightMomentumNode(WeightSplineNode):

    @property
    def momentum_0(self):
        return self._m_i[0]

    @property
    def momentum_1(self):
        return self._m_i[1]


class WeightGradientNode(WeightSplineNode):

    @property
    def momentum_0(self):
        return 2 * self.w / self.h * (-2 * self._m_i[0] - self._m_i[1] + 3 * (self._f_i[1] - self._f_i[0]) / self.h)

    @property
    def momentum_1(self):
        return 2 * self.w / self.h * (2 * self._m_i[1] + self._m_i[0] - 3 * (self._f_i[1] - self._f_i[0]) / self.h)


class WeightSplineBase(SplineBase, ABC):
    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase):
        super().__init__(x, f)
        self._bound_cond = bound_condition
        self._nodes: list[WeightSplineNode]

    def __getitem__(self, tup):
        assert hasattr(self, '_nodes')
        t, i = tup
        if isinstance(i, int):
            return self._nodes[i](t)
        elif isinstance(i, slice):
            return sum(node(t) for node in self._nodes[i])
        else:
            raise TypeError("Невалидный индекс или срез индексов.")

    def __call__(self, t: float, *args, **kwargs):
        if self._x[0]> t or self._x[-1] < t:
            return 0
        i = binary_search_index(self._x, t)
        return self[t, i]

    @abstractmethod
    def _init_nodes(self, *args, **kwargs):
        ...


class WeightMomentumSpline(WeightSplineBase):

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        N = x.shape[0]
        vec_m = np.zeros(N, dtype=float)
        self._nodes = [WeightMomentumNode(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k + 2]) for k in range(N - 1)]
        return vec_m

    def fit(self):
        vec_m = self._init_nodes()
        nodes = self._nodes
        x = self._x
        f = self._f
        dd = DivDiff(x, f)
        bound_cond = self._bound_cond
        N = vec_m.shape[0]
        a = np.zeros(N, dtype=float)
        b = np.ones(N, dtype=float) * 2
        c = np.zeros(N, dtype=float)
        d = vec_m

        get_lambda = lambda k: nodes[k - 1].w * nodes[k].h / (nodes[k].w * nodes[k - 1].h + nodes[k - 1].w * nodes[k].h)
        get_mu = lambda k: 1 - get_lambda(k)
        get_d = lambda k: 6 * get_lambda(k) * nodes[k].w / nodes[k].h * (dd(1, k) - dd(1, k - 1))

        if isinstance(bound_cond, BoundConditionSecondType):
            c[0] = a[N - 1] = 0
            d[0] = 2 * bound_cond.f_a * nodes[0].w
            d[N - 1] = 2 * bound_cond.f_b * nodes[N - 2].w
        elif isinstance(bound_cond, BoundConditionFirstType):
            c[0] = 1
            a[N - 1] = 1

            d[0] = (dd(1, 0) - bound_cond.f_a) * 6 * nodes[0].w / nodes[0].h
            d[N - 1] = (bound_cond.f_b - dd(1, N - 2)) * 6 * nodes[N - 2].w / nodes[N - 2].h

        else:
            raise TypeError("Невалидные краевые условия")

        for i in range(1, N - 1):
            a[i] = get_mu(i)
            c[i] = get_lambda(i)
            d[i] = get_d(i)

        t_m = ThomasMethod(a, b, c, d)
        t_m.solve()


class WeightGradientSpline(WeightSplineBase):

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        N = x.shape[0]
        vec_m = np.zeros(N, dtype=float)
        self._nodes = [WeightGradientNode(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k + 2]) for k in range(N - 1)]
        return vec_m

    def fit(self):
        vec_m = self._init_nodes()
        nodes = self._nodes

        x = self._x
        f = self._f
        dd = DivDiff(x, f)
        bound_cond = self._bound_cond
        N = vec_m.shape[0]
        a = np.zeros(N, dtype=float)
        b = np.ones(N, dtype=float) * 2
        c = np.zeros(N, dtype=float)
        d = vec_m

        get_lambda = lambda k: nodes[k - 1].w * nodes[k].h / (nodes[k].w * nodes[k - 1].h + nodes[k - 1].w * nodes[k].h)
        get_mu = lambda k: 1 - get_lambda(k)
        get_d = lambda k: 3 * (get_mu(k) * dd(1, k) + get_lambda(k) * dd(1, k - 1))

        if isinstance(bound_cond, BoundConditionFirstType):
            c[0] = a[N - 1] = 0
            d[0] = 2 * bound_cond.f_a
            d[N - 1] = 2 * bound_cond.f_b
        elif isinstance(bound_cond, BoundConditionSecondType):
            c[0] = 1
            a[N - 1] = 1

            d[0] = 3 * dd(1, 0) - nodes[0].h * nodes[0].w * bound_cond.f_a / 2
            d[N - 1] = 3 * dd(1, N - 2) + bound_cond.f_b * nodes[N - 2].w * nodes[N - 2].h / 2

        else:
            raise TypeError("Невалидные краевые условия")

        for i in range(1, N - 1):
            c[i] = get_mu(i)
            a[i] = get_lambda(i)
            d[i] = get_d(i)

        t_m = ThomasMethod(a, b, c, d)
        t_m.solve()


class MonotoneWeightSpline(WeightGradientSpline):
    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase, eps: float = 1e-16):
        super().__init__(x, f, bound_condition)
        self._eps = eps

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        dd = DivDiff(x, f)
        N = x.shape[0]
        vec_m = np.zeros(N)
        nodes = [WeightGradientNode(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k + 2]) for k in range(N - 1)]
        for i in range(1, N - 1):
            a = (dd(1, i - 1) / dd(1, i) - 2) * nodes[i - 1].w * nodes[i].h / nodes[i - 1].h
            b = (dd(1, i) / dd(1, i - 1) - 2) * nodes[i - 1].h / (nodes[i - 1].w * nodes[i].h)
            if nodes[i].w < a:
                nodes[i].w = 1.1 * a
            elif 1 / nodes[i].w < b:
                nodes[i].w = 1.1 / b

        self._nodes = nodes
        return vec_m


class ConvexWeightSpline(WeightMomentumSpline):
    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase, eps: float = 1e-16):
        assert x.shape[0] > 2
        super().__init__(x, f, bound_condition)
        self._eps = eps

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        dd = DivDiff(x, f)
        N = x.shape[0]
        eps = self._eps
        vec_m = np.zeros(N)
        nodes = [WeightMomentumNode(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k + 2]) for k in range(N - 1)]
        delta_f = lambda k: dd(1, k) - dd(1, k - 1)
        if N > 3:
            nodes[1].w = max((delta_f(1) / delta_f(2) - 1) * nodes[1].h * nodes[0].w / nodes[0].h, nodes[1].w)

        get_lambda = lambda k: nodes[k - 1].w * nodes[k].h / (nodes[k].w * nodes[k - 1].h + nodes[k - 1].w * nodes[k].h)
        for i in range(2, N - 1):

            a = nodes[i - 1].h / (nodes[i - 1].w * nodes[i].h) * (
                        1 / get_lambda(i - 1) * delta_f(i) / delta_f(i - 1) - 1)
            if a > eps:
                nodes[i].w = 1 / a
            elif i < N - 2:
                nodes[i].w = max((delta_f(i) / delta_f(i + 1) - 1) * nodes[i].h * nodes[i - 1].w / nodes[i - 1].h,
                                 nodes[i - 1].w)

        self._nodes = nodes
        return vec_m
