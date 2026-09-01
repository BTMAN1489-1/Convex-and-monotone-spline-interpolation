import numpy as np
from numpy.typing import NDArray
from splines.weight import *
from helpers import *
from splines.gen import *
from typing import Protocol


class FuriesProtocol(Protocol):
    def __call__(self,*args, **kwargs): ...
    def fit(self): ...


class SimpleFuries:
    def __init__(self, t: NDArray[np.float64], f: NDArray[np.float64]):
        assert t.shape == f.shape
        self._t = t
        self._f = f

    def __call__(self, x: NDArray[np.float64]):
        E = self._f * np.exp(-1j * np.outer(x, self._t))
        dt = np.diff(self._t)
        w = np.zeros_like(self._t)
        w[0] = dt[0] / 2
        w[-1] = dt[-1] / 2
        for i in range(1, len(self._t) - 1):
            w[i] = (dt[i - 1] + dt[i]) / 2

        return E @ w

    def fit(self): ...


class WeightSplineFuries:
    def __init__(self, t: NDArray[np.float64], f: NDArray[np.float64], spline_cls: type[WeightSplineBase], bound_cond:BoundConditionBase):
        self._t = t
        self._f = f
        self._sp = spline_cls(t, f, bound_cond)

    def fit(self):
        self._sp.fit()

    def __call__(self, x: NDArray[np.float64]):
        t = self._t
        f = self._f
        sp = self._sp
        res = np.zeros_like(x, dtype=complex)
        index_zero = x == 0
        res_zero=0
        x[index_zero] = 1
        for i in range(t.shape[0] - 1):
            w_node = sp.get_node(i)
            h = w_node.h
            M_0 = w_node.momentum_0
            M_1 = w_node.momentum_1
            z = 1j*x*h

            J_0 = (1-np.exp(-z))/z
            J_1 = (1-(1+z)*np.exp(-z))/z**2
            J_2 = (2 - (2+2*z + z**2)*np.exp(-z))/z**3
            J_3 = (6 - (6+6*z + 3*z**2 + z**3)*np.exp(-z))/z**4
            res+=h*np.exp(-1j*x*t[i])*(f[i]*J_0 + (f[i+1] - f[i])*J_1 - h**2/6*(M_0*(2*J_1 - 3*J_2 + J_3)+M_1*(J_1-J_3)))

            if any(index_zero):
                J_0 = 1
                J_1 = 1/2
                J_2 = 1/3
                J_3 =1/4

                res_zero+=h*(f[i]*J_0 + (f[i+1] - f[i])*J_1 - h**2/6*(M_0*(2*J_1 - 3*J_2 + J_3)+M_1*(J_1-J_3)))


        res[index_zero] = res_zero

        return res


class GeneralExpSplineFuries:
    def __init__(self, t: NDArray[np.float64], f: NDArray[np.float64], spline_cls: type[GeneralSplineBase], bound_cond:BoundConditionBase, node_cls:type[ExpNode]):
        self._t = t
        self._f = f
        self._sp = spline_cls(t, f, bound_cond, node_cls)

    def fit(self):
        self._sp.fit()

    def __call__(self, x: NDArray[np.float64]):
        t = self._t
        f = self._f
        sp = self._sp
        index_zero = x == 0
        x[index_zero] = 1
        res_zero = 0
        res = np.zeros_like(x, dtype=complex)
        for i in range(t.shape[0] - 1):
            exp_node:ExpNode = sp.get_node(i)
            q = exp_node.q
            h = exp_node.h
            A = f[i+1] - exp_node.momentum_1*exp_node.Psi(1)
            B = f[i] - exp_node.momentum_0*exp_node.Phi(0)
            res+=1j/x *(A*np.exp(-1j*t[i+1]*x) - B*np.exp(-1j*t[i]*x)) + (A-B)/(h * x**2) *(np.exp(-1j*t[i+1]*x) - np.exp(-1j*t[i]*x))

            A_i = h**3/(6+6*q + q**2)
            b1 = q - 1j*x*h

            res += A_i*((np.exp(-1j*x*t[i+1])*(b1**3 - 3*b1**2 + 6*b1 - 6) + 6*np.exp(-q - 1j*x*t[i]))/b1**4)

            b2 = b1.conjugate()
            res += A_i * ((np.exp(-1j * x * t[i]) * (b2 ** 3 - 3 * b2 ** 2 + 6 * b2 - 6) + 6 * np.exp(
                -q - 1j * x * t[i+1])) / b2 ** 4)

            if any(index_zero):
                J_0 = 1/24 if q==0 else ((q**3 - 3*q**2 + 6*q - 6) - 6*np.exp(-q))/q**4
                res_zero += (f[i+1] + f[i])*h/2 + A_i*(J_0 - 1/2)*(exp_node.momentum_0 + exp_node.momentum_1)

        res[index_zero] = res_zero

        return res


if __name__ == '__main__':
    H = 10
    h = 0.05
    t = np.linspace(-np.pi, 0, round(np.pi / h * H))
    f =  1/(1+500*np.exp(-15*np.cos(t)))

    t_sp = t[::H]
    f_sp = f[::H]
    bound_cond = BoundConditionFirstType(0, 0)

    sf: FuriesProtocol
    sf = WeightSplineFuries(t_sp, f_sp, MonotoneWeightSpline, bound_cond)
    # sf= GeneralExpSplineFuries(t_sp, f_sp, MonotoneGeneralSpline, bound_cond, GradientExpNode)
    # sf = SimpleFuries(t_sp, f_sp)
    sf.fit()

    x = np.linspace(-20, 20, 700)

    sol_sp = sf(x)
    sol_d = SimpleFuries(t, f)(x)
    print(f"err={np.max(np.abs(sol_d - sol_sp)):.2e}")
