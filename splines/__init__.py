import numpy as np

from .weight import *
from .gen import *
from .base import *
from helpers import *
from typing import Literal
from numpy.typing import NDArray

__all__ = (
    "MonotoneInterpolation",
    "ConvexInterpolation",
)

SPLINE = Literal["gen", "weight"]


class MonotoneInterpolation(SplineBase):
    def __init__(self, x: NDArray, f: NDArray, method: SPLINE, gen_node_cls: type[GeneralGradientNode] | None = None,
                 eps: float = 1e-16):
        super().__init__(x, f)
        parms = {}
        match method:
            case "gen":
                if gen_node_cls is None:
                    raise ValueError("gen_node_cls cannot be None for method: gen")
                parms["node_cls"] = gen_node_cls
                self._sp_cls = MonotoneGeneralSpline
            case "weight":
                parms["eps"] = eps
                self._sp_cls = MonotoneWeightSpline
            case _:
                raise ValueError(f"Unknown method: {method}")
        self._parms = parms
        self._splines = []
        self._eps = eps

    def fit(self):
        x = self._x
        f = self._f
        eps = self._eps
        dd = DivDiff(x, f)
        N = x.shape[0]
        splines = self._splines
        k = 0
        while k < N - 1:
            if abs(dd(1, k)) <= eps:
                m = k + 1
                while m < N - 1 and abs(dd(1, m)) <= eps:
                    m += 1
                splines.append(ConstantSpline(x[k:m + 1], f[k:m + 1]))
                k = m
            else:
                sgn = dd(1, k)
                m = k + 1
                while m < N - 1 and sgn * dd(1, m) > 0:
                    m += 1
                splines.append(
                    self._sp_cls(x[k:m + 1], f[k:m + 1], bound_condition=BoundConditionFirstType(0, 0), **self._parms))
                k = m

        for sp in splines:
            sp.fit()

    def __call__(self, t: float) -> float:
        return sum(sp(t) for sp in self._splines)


class ConvexInterpolation(SplineBase):
    def __init__(self, x: NDArray, f: NDArray, method: SPLINE, gen_node_cls: type[GeneralMomentumNode] | None = None,
                 rho: float = 0.5, eps: float = 1e-16):
        assert eps < rho < 1 - eps
        super().__init__(x, f)
        parms = {}
        match method:
            case "gen":
                if gen_node_cls is None:
                    raise ValueError("gen_node_cls cannot be None for method: gen")
                parms["node_cls"] = gen_node_cls
                self._sp_cls = ConvexGeneralSpline
            case "weight":
                parms["eps"] = eps
                self._sp_cls = ConvexWeightSpline
            case _:
                raise ValueError(f"Unknown method: {method}")
        self._parms = parms
        self._splines = []
        self._eps = eps
        self._rho = rho

    def fit(self):
        x = self._x
        f = self._f
        eps = self._eps
        rho = self._rho
        dd = DivDiff(x, f)
        delta_f = lambda k: dd(1, k) - dd(1, k - 1)
        h = lambda k: x[k] - x[k - 1]
        N = x.shape[0]
        splines = self._splines
        i = 1
        j = 0
        x_lst = [x[0], x[1]]
        f_lst = [f[0], f[1]]
        while i < N - 1:
            if abs(delta_f(i)) <= eps:
                m = i + 1
                while m < N - 1 and abs(delta_f(m)) <= eps:
                    x_lst.append(x[m])
                    f_lst.append(f[m])
                    m += 1
                splines.append(LinearSpline(np.array(x_lst[j:], dtype=float), np.array(f_lst[j:], dtype=float)))
                j = len(x_lst) - 1
                x_lst.append(x[m])
                f_lst.append(f[m])
                i = m
            else:
                sgn = delta_f(i)
                m = i + 1
                while m < N - 1 and sgn * delta_f(m) > 0:
                    x_lst.append(x[m])
                    f_lst.append(f[m])
                    m += 1
                if m < N - 1:
                    x_m = x[m - 1] + rho * h(m - 1)
                    f_m = f[m - 1] + rho * h(m - 1) * dd(1, m - 1)
                    x_lst.append(x_m)
                    f_lst.append(f_m)
                else:
                    x_lst.append(x[m])
                    f_lst.append(f[m])

                splines.append(
                    self._sp_cls(np.array(x_lst[j:], dtype=float), np.array(f_lst[j:], dtype=float),
                                 bound_condition=BoundConditionFirstType(dd(1, i - 1), dd(1, m - 1)), **self._parms))
                j = len(x_lst) - 1
                x_lst.append(x[m])
                f_lst.append(f[m])
                i = m

        for sp in splines:
            sp.fit()

    def __call__(self, t: float) -> float:
        return sum(sp(t) for sp in self._splines)
