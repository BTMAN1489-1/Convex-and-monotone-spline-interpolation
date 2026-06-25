from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

from .base import *
from helpers import ThomasMethod, BoundConditionBase, BoundConditionFirstType, BoundConditionSecondType, \
    binary_search_index, DivDiff

__all__ = (
    "GeneralSplineNode",
    "GeneralMomentumNode",
    "GeneralGradientNode",
    "GeneralMomentumSpline",
    "StandardSplineNode",
    "MonotoneGeneralSpline",
    "ConvexGeneralSpline",
    "MomentumPowNode",
    "GradientPowNode",
    "GeneralGradientSpline",
    "GeneralSplineBase",
    "MomentumExpNode",
    "GradientExpNode"
)



class GeneralSplineNode(SplineNodeBase, ABC):
    def __init__(self, f_i: NDArray, m_i: NDArray, x_i: NDArray, q: float = 0):
        super().__init__(f_i, m_i, x_i)
        self.q = q
        self.Phi = lambda tau: self.h ** 2 * self._zeta(1 - tau)
        self.Psi = lambda tau: self.h ** 2 * self._zeta(tau)
        self.dPhi = lambda tau: -self.h * self._diff_zeta(1 - tau)
        self.dPsi = lambda tau: self.h * self._diff_zeta(tau)

    def __call__(self, t: float) -> float:
        tau = (t - self._x_i[0]) / self._h
        if 0 <= tau <= 1:
            return ((self._f_i[0] - self.momentum_0 * self.Phi(0)) * (1 - tau) + (
                    self._f_i[1] - self.momentum_1 * self.Psi(1)) * tau +
                    self.momentum_0 * self.Phi(tau) + self.momentum_1 * self.Psi(tau))
        return 0

    @property
    def q(self) -> int | float:
        return self._q

    @q.setter
    def q(self, value: float | int):
        self._q = value

    @abstractmethod
    def _diff_zeta(self, tau: float, *args, **kwargs) -> float: ...

    @abstractmethod
    def _zeta(self, tau: float, *args, **kwargs) -> float: ...


class StandardSplineNode(GeneralSplineNode, ABC):
    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value: float | int):
        self._q = max(value - 3, 0)


class GeneralSplineBase(SplineBase, ABC):
    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase, node_cls: type[GeneralSplineNode]):
        super().__init__(x, f)
        self._bound_cond = bound_condition
        self._node_cls = node_cls
        self._nodes: list[GeneralSplineNode]

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


class GeneralMomentumNode(GeneralSplineNode, ABC):

    @property
    def momentum_0(self):
        return self._m_i[0]

    @property
    def momentum_1(self):
        return self._m_i[1]


class GeneralGradientNode(GeneralSplineNode, ABC):
    @property
    def momentum_0(self):
        psi = -self.Psi(1) + self.h * self.dPsi(1)
        phi = -self.Phi(0) - self.h * self.dPhi(0)
        T = psi * phi - self.Phi(0) * self.Psi(1)
        divdiff = (self._f_i[1] - self._f_i[0]) / self.h
        return self.h / T * ((psi + self.Psi(1)) * divdiff - psi * self._m_i[0] - self.Psi(1) * self._m_i[1])

    @property
    def momentum_1(self):
        psi = -self.Psi(1) + self.h * self.dPsi(1)
        phi = -self.Phi(0) - self.h * self.dPhi(0)
        T = psi * phi - self.Phi(0) * self.Psi(1)
        divdiff = (self._f_i[1] - self._f_i[0]) / self.h
        return self.h / T * (-(phi + self.Phi(0)) * divdiff + phi * self._m_i[1] + self.Phi(0) * self._m_i[0])


class GeneralMomentumSpline(GeneralSplineBase):
    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase,
                 node_cls: type[GeneralMomentumNode]):
        assert issubclass(node_cls, GeneralMomentumNode)
        super().__init__(x, f, bound_condition, node_cls)

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        N = x.shape[0]
        vec_m = np.zeros(N, dtype=float)
        self._nodes = [self._node_cls(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k+2]) for k in range(N - 1)]
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
        b = np.ones(N, dtype=float)
        c = np.zeros(N, dtype=float)
        d = vec_m

        psi = lambda k: -nodes[k].Psi(1) + nodes[k].h * nodes[k].dPsi(1)
        phi = lambda k: -nodes[k].Phi(0) - nodes[k].h * nodes[k].dPhi(0)

        get_a = lambda k: nodes[k - 1].Phi(0) / nodes[k-1].h
        get_b = lambda k: psi(k - 1) / nodes[k-1].h + phi(k) / nodes[k].h
        get_c = lambda k: nodes[k].Psi(1) / nodes[k].h

        if isinstance(bound_cond, BoundConditionSecondType):
            c[0] = a[N - 1] = 0
            d[0] = bound_cond.f_a
            d[N - 1] = bound_cond.f_b
        elif isinstance(bound_cond, BoundConditionFirstType):
            c[0] = nodes[0].Psi(1) / nodes[0].h
            b[0] = phi(0) / nodes[0].h
            a[N - 1] = nodes[N - 2].Phi(0) / nodes[N-2].h
            b[N - 1] = psi(N - 2) / nodes[N-2].h

            d[0] = dd(1, 0) - bound_cond.f_a
            d[N - 1] = bound_cond.f_b - dd(1, N - 2)

        else:
            raise TypeError("Невалидные краевые условия")

        for i in range(1, N - 1):
            a[i] = get_a(i)
            b[i] = get_b(i)
            c[i] = get_c(i)
            d[i] = dd(1, i) - dd(1, i - 1)

        t_m = ThomasMethod(a, b, c, d)
        t_m.solve()


class GeneralGradientSpline(GeneralSplineBase):
    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase,
                 node_cls: type[GeneralGradientNode]):
        assert issubclass(node_cls, GeneralGradientNode)
        super().__init__(x, f, bound_condition, node_cls)

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        N = x.shape[0]
        vec_m = np.zeros(N, dtype=float)
        self._nodes = [self._node_cls(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k+2]) for k in range(N - 1)]
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
        b = np.ones(N, dtype=float)
        c = np.zeros(N, dtype=float)
        d = vec_m
        psi = lambda k: -nodes[k].Psi(1) + nodes[k].h * nodes[k].dPsi(1)
        phi = lambda k: -nodes[k].Phi(0) - nodes[k].h * nodes[k].dPhi(0)
        T = lambda k: psi(k) * phi(k) - nodes[k].Psi(1) * nodes[k].Phi(0)

        get_a = lambda k: nodes[k - 1].Phi(0) * nodes[k-1].h/ T(k - 1)
        get_b = lambda k: phi(k - 1) * nodes[k-1].h / T(k - 1) + psi(k) * nodes[k].h / T(k)
        get_c = lambda k: nodes[k].Psi(1) * nodes[k].h / T(k)

        if isinstance(bound_cond, BoundConditionFirstType):
            c[0] = a[N - 1] = 0
            d[0] = bound_cond.f_a
            d[N - 1] = bound_cond.f_b
        elif isinstance(bound_cond, BoundConditionSecondType):
            c[0] = nodes[0].Psi(1)
            b[0] = psi(0)
            a[N - 1] = nodes[N - 2].Phi(0)
            b[N - 1] = phi(N - 2)

            d[0] = nodes[0].h * nodes[0].dPsi(1) * dd(1, 0) - bound_cond.f_a * T(0) / nodes[0].h
            d[N - 1] = bound_cond.f_b * T(N - 2) / nodes[N-2].h - nodes[N-2].h * nodes[N - 2].dPhi(0) * dd(1, N - 2)

        else:
            raise TypeError("Невалидные краевые условия")

        for i in range(1, N - 1):
            a[i] = get_a(i)
            b[i] = get_b(i)
            c[i] = get_c(i)
            d[i] = nodes[i].dPsi(1) * dd(1, i) * nodes[i].h ** 2 / T(i) - nodes[i - 1].dPhi(0) * dd(1, i - 1) * nodes[i-1].h** 2 / T(i - 1)

        t_m = ThomasMethod(a, b, c, d)
        t_m.solve()


class ConvexGeneralSpline(GeneralMomentumSpline):

    def __init__(self, x: NDArray, f: NDArray, bound_condition: BoundConditionBase,
                 node_cls: type[GeneralMomentumNode]):
        assert x.shape[0] > 2
        super().__init__(x, f, bound_condition, node_cls)

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        dd = DivDiff(x, f)
        delta_f = lambda k: dd(1, k) - dd(1, k - 1)
        N = x.shape[0]
        vec_m = np.zeros(N)
        nodes = [self._node_cls(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k+2]) for k in range(N - 1)]
        for i in range(1, N - 2):
            q = max(0, 2 * delta_f(i) / delta_f(i + 1) + 1, 2 * delta_f(i + 1) / delta_f(i) + 1)
            nodes[i].q = q

        if isinstance(self._bound_cond, BoundConditionFirstType):
            delta_0 = dd(1, 0) - self._bound_cond.f_a
            delta_N = self._bound_cond.f_b - dd(1, N - 2)

        elif isinstance(self._bound_cond, BoundConditionSecondType):
            delta_0 = self._bound_cond.f_a
            delta_N = self._bound_cond.f_b
        else:
            raise TypeError("Невалидные краевые условия")

        nodes[0].q=max(0, 2 * delta_0 / delta_f(1) + 1)

        nodes[N-2].q = max(0, 2 * delta_N / delta_f(N - 2) + 1)
        self._nodes = nodes
        return vec_m


class MonotoneGeneralSpline(GeneralGradientSpline):

    def _init_nodes(self) -> NDArray:
        x = self._x
        f = self._f
        dd = DivDiff(x, f)
        N = x.shape[0]
        vec_m = np.zeros(N)
        nodes = [self._node_cls(f_i=f[k:k + 2], m_i=vec_m[k:k + 2], x_i=x[k:k+2]) for k in range(N - 1)]
        psi = lambda k: -nodes[k].Psi(1) + nodes[k].h * nodes[k].dPsi(1)
        phi = lambda k: -nodes[k].Phi(0) - nodes[k].h * nodes[k].dPhi(0)
        T = lambda k: psi(k) * phi(k) - nodes[k].Psi(1) * nodes[k].Phi(0)

        for i in range(1, N - 2):
            q = 2*dd(1, i-1)/dd(1, i) + 2
            if i < N-2:
                q = max(0, q, 2*dd(1, i+1)/dd(1, i) + 2)
            nodes[i].q = q

        if isinstance(self._bound_cond, BoundConditionFirstType):
            dd_0 = self._bound_cond.f_a
            dd_N = self._bound_cond.f_b
        elif isinstance(self._bound_cond, BoundConditionSecondType):
            dd_0 = (nodes[0].h * nodes[0].dPsi(1) * dd(1, 0) - self._bound_cond.f_a * T(0) / nodes[0].h) / psi(0)
            dd_N = (self._bound_cond.f_b * T(N - 2) / nodes[N-2].h -
                    nodes[N-2].h * nodes[N - 2].dPhi(0) * dd(1, N - 2)) / phi(N - 2)
        else:
            raise TypeError("Невалидные краевые условия")

        q_0 = 2 * dd_0 / dd(1, 0)
        q_N = 2 * dd_N / dd(1, N - 2)
        if N > 2:
            q_0 = max(0, q_0, 2 * dd(1, 1) / dd(1, 0) + 2)
            q_N = max(0, q_N, 2 * dd(1, N-3) / dd(1, N-2) + 2)
        nodes[0].q = q_0
        nodes[N-2].q = q_N
        self._nodes = nodes
        return vec_m


class PowNode(StandardSplineNode, ABC):

    def _diff_zeta(self, tau: float, *args, **kwargs) -> float:
        return tau ** (self.q + 2) / (self.q + 2)

    def _zeta(self, tau: float, *args, **kwargs) -> float:
        return tau ** (self.q + 3) / ((self.q + 3) * (self.q + 2))


class MomentumPowNode(GeneralMomentumNode, PowNode): ...


class GradientPowNode(GeneralGradientNode, PowNode): ...

class ExpNode(StandardSplineNode, ABC):

    def _diff_zeta(self, tau: float, *args, **kwargs) -> float:
        q = self.q
        one_minus_tau = 1 - tau
        exp_arg = -q * one_minus_tau - 0.5 * q ** 2 * one_minus_tau ** 2
        exp_val = np.exp(exp_arg)
        denominator = 6 + 6 * q

        d_exp_arg = q + q ** 2 * one_minus_tau  # производная exp_arg по tau

        d_numerator = (3 * tau ** 2 + tau ** 3 * d_exp_arg) * exp_val

        return d_numerator / denominator

    def _zeta(self, tau: float, *args, **kwargs) -> float:
        q = self.q
        exponent = -q * (1 - tau) - 0.5 * q ** 2 * (1 - tau) ** 2
        numerator = tau ** 3 * np.exp(exponent)
        denominator = 6 + 6 * q
        return numerator / denominator


class MomentumExpNode(GeneralMomentumNode, PowNode): ...


class GradientExpNode(GeneralGradientNode, PowNode): ...