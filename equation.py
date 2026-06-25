import warnings
from abc import ABC, abstractmethod, ABCMeta
import numpy as np
from numpy import sin, cos
from numpy.linalg import solve as lin_solve, vector_norm, matrix_rank, LinAlgError, det
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import IntEnum
import seaborn as sns

__all__=(
    "BaseSolver",
    "ManyStepMetaSolver",
    "AmpersandEquation",
    "CaspidEquation",
    "SinEquation",
    "FlowerEquation",
    "ModifiedEulerSolver",
    "RungeKuttSolver",
    "OrientationDecision",
    "vector_norm"
)

class DecisionNotContinueWarning(UserWarning):
    """Предупреждение о невозможности продолжения решения"""


class OrientationDecision(IntEnum):
    RIGHT = 1
    LEFT = -1


class BaseSolver(ABC):

    def __init__(self, h_min: float, h_max: float, eps: float, orientation: OrientationDecision):
        assert abs(h_min) <= abs(h_max)
        assert 1e-16 <= h_min <= 1e-1
        assert 1e-16 <= h_max <= 1e-1
        assert 1e-32 <= eps <= 1e-1
        self._h_min = h_min
        self._h_max = h_max
        self._eps = eps
        self._orientation = orientation

    def change_orientation(self, new_orientation: OrientationDecision):
        self._orientation = new_orientation

    def _set_correction(self, fac_min, fac_max, m):
        tol = self._eps

        def wrapped(err):
            if err == 0:
                return fac_max
            return min((fac_max, max(fac_min, (tol / err) ** (1 / (m + 1)))))

        return wrapped

    def _new_tau(self, eq):
        n = eq.n
        diff = eq.differentiate
        _norm = vector_norm
        _solve = lin_solve
        b = np.zeros(n, dtype="float64")
        b[n - 1] = 1
        orientation = self._orientation
        sign = np.sign

        def wrapped(x, tau):
            new_tau = _solve(np.vstack((diff(x), tau)), b)
            new_tau /= _norm(new_tau)
            sgn = orientation * sign(det(np.vstack((diff(x), new_tau))))

            return sgn * new_tau

        return wrapped

    @staticmethod
    def _assert_start_approx(equation, x_0, eps):
        assert vector_norm(equation.vector_func(x_0)) <= eps, f"Точка {x_0} не является решением уравнения."
        assert matrix_rank(equation.differentiate(x_0)) == equation.n - 1, f"Точка {x_0} не является регулярной."

    @staticmethod
    def _find_start_vector(equation, x_0):
        n = equation.n
        tau = np.zeros(n, dtype="float64")
        tau[n - 1] = 1
        jacoby_matrix = equation.differentiate(x_0)
        if not matrix_rank(np.vstack((jacoby_matrix, tau))) == n:
            tau[n - 1] = 0
            tau[n - 2] = 1
        return tau

    @abstractmethod
    def __call__(self, equation, x_0, s_max):
        raise NotImplementedError()


class ManyStepMetaSolver(ABCMeta):
    def __new__(cls, name, base, attrs):
        accuracy_key = "ACCURACY"
        if accuracy_key not in attrs:
            raise AttributeError(f"Attribute of class {accuracy_key} must be defined.")
        accuracy = attrs[accuracy_key]
        assert isinstance(accuracy, int)
        assert accuracy > 0

        simple_solver_key = "SIMPLE_SOlVER"
        if simple_solver_key not in attrs:
            raise AttributeError(f"Attribute of class {simple_solver_key} must be defined.")
        simple_solver = attrs[simple_solver_key]
        assert issubclass(simple_solver, BaseSolver)

        beta_p, beta_c = cls._find_beta(accuracy)
        attrs["beta_c"] = beta_c
        attrs["beta_p"] = beta_p
        return super().__new__(cls, name, base, attrs)

    @staticmethod
    def _find_beta(accuracy):
        b = [1 / q for q in range(1, accuracy + 1)]
        b = np.array(b, dtype="float64")
        matrix = [[1] * accuracy]
        for k in range(1, accuracy):
            matrix.append([0] + [j ** k for j in range(1, accuracy)])

        matrix = np.array(matrix, dtype="float64")
        for i in range(accuracy):
            a = vector_norm(matrix[i])
            matrix[i] /= a
            b[i] /= a

        beta_c = lin_solve(matrix, b)
        beta_p = lin_solve(matrix[:(accuracy - 1), 1:], b[:(accuracy - 1)])

        return beta_p, beta_c


class BaseEquation(ABC):
    @property
    @abstractmethod
    def n(self):
        raise NotImplementedError()

    @abstractmethod
    def differentiate(self, x):
        raise NotImplementedError()

    @abstractmethod
    def vector_func(self, x):
        raise NotImplementedError()

    def solve(self, solver: BaseSolver, x_0, s_max):
        return solver(self, x_0, s_max)


class SimpleEulerSolver(BaseSolver):
    _MAX_ITERS = 10

    def __call__(self, eq: BaseEquation, x_0, s_max):
        h_min = self._h_min
        h_max = self._h_max
        h = (h_max + h_min) / 2
        eps = self._eps
        correcting = self._set_correction(0.5, 2, 1)
        new_tau = self._new_tau(eq)
        func = eq.vector_func
        _norm = vector_norm
        BaseSolver._assert_start_approx(eq, x_0, eps)
        tau = BaseSolver._find_start_vector(eq, x_0)
        max_iters = self._MAX_ITERS
        x = x_0
        s = 0
        cords = []
        new_h = h
        while abs(s) < s_max:
            try:
                tau_p = new_tau(x, tau)
                cords.append((x, s))
                x_p = x + tau_p * h

                for _ in range(max_iters):
                    err = _norm(func(x_p))
                    gamma = correcting(err)
                    if err < eps:
                        new_h = min(h * gamma, h_max)
                        break
                    else:
                        new_h = max(h * gamma, h_min)
                    h = new_h
                    x_p = x + tau_p * h

                tau = tau_p
                x = x_p
            except (LinAlgError, ZeroDivisionError):
                warnings.warn("Дальнейшее продолжение решения невозможно.", DecisionNotContinueWarning)
                break
            s += h
            h = new_h

        return cords


class ModifiedEulerSolver(BaseSolver):
    _MAX_ITERS = 10

    def __call__(self, eq: BaseEquation, x_0, s_max):
        h_min = self._h_min
        h_max = self._h_max
        h = (h_max + h_min) / 2
        eps = self._eps
        correcting = self._set_correction(0.5, 2, 2)
        new_tau = self._new_tau(eq)
        _norm = vector_norm
        BaseSolver._assert_start_approx(eq, x_0, eps)
        tau = BaseSolver._find_start_vector(eq, x_0)
        max_iters = self._MAX_ITERS
        x = x_0
        s = 0
        cords = []
        new_h = h
        while abs(s) < s_max:
            try:
                tau_p = new_tau(x, tau)
                cords.append((x, s))
                x_p = x + tau_p * h
                tau_c = new_tau(x_p, tau_p)

                for _ in range(max_iters):
                    err = _norm(tau_p - tau_c)
                    gamma = correcting(err)
                    if err < eps:
                        new_h = min(h * gamma, h_max)
                        break
                    else:
                        new_h = max(h * gamma, h_min)

                    h = new_h
                    x_p = x + tau_p * h
                    tau_c = new_tau(x_p, tau_p)

                x = x + (tau_p + tau_c) / 2 * h
                tau = tau_c

            except (LinAlgError, ZeroDivisionError):
                warnings.warn("Дальнейшее продолжение решения невозможно.", DecisionNotContinueWarning)
                break
            s += h
            h = new_h

        return cords


class RungeKuttSolver(BaseSolver):
    _MAX_ITERS = 10

    def __call__(self, eq: BaseEquation, x_0, s_max):
        h_min = self._h_min
        h_max = self._h_max
        h = (h_max + h_min) / 2
        eps = self._eps
        correcting = self._set_correction(0.5, 2, 4)
        new_tau = self._new_tau(eq)
        _norm = vector_norm
        BaseSolver._assert_start_approx(eq, x_0, eps)
        tau = BaseSolver._find_start_vector(eq, x_0)
        max_iters = self._MAX_ITERS
        x = x_0
        s = 0
        cords = []
        new_h = h
        while abs(s) < s_max:
            try:
                k1 = new_tau(x, tau)
                cords.append((x, s))
                x_1 = x + h / 2 * k1

                k2 = new_tau(x_1, k1)
                x_2 = x + h / 2 * k2

                k3 = new_tau(x_2, k2)
                x_3 = x + h * k3

                k4 = new_tau(x_3, k3)
                for _ in range(max_iters):

                    err = _norm(k1 - k4)
                    gamma = correcting(err)
                    if err < eps:
                        new_h = min(h * gamma, h_max)
                        break
                    else:
                        new_h = max(h * gamma, h_min)

                    h = new_h

                    x_1 = x + h / 2 * k1

                    k2 = new_tau(x_1, k1)
                    x_2 = x + h / 2 * k2

                    k3 = new_tau(x_2, k2)
                    x_3 = x + h * k3

                    k4 = new_tau(x_3, k3)

                x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                tau = k4

            except (LinAlgError, ZeroDivisionError):
                warnings.warn("Дальнейшее продолжение решения невозможно.", DecisionNotContinueWarning)
                break
            s += h
            h = new_h

        return cords


class AdamsModifiedEulerSolver(BaseSolver, metaclass=ManyStepMetaSolver):
    ACCURACY = 2
    SIMPLE_SOlVER = ModifiedEulerSolver
    _MAX_ITERS = 10

    def __call__(self, eq: BaseEquation, x_0, s_max):
        h = h_min = self._h_min
        h_max = self._h_max
        eps = self._eps
        m = self.ACCURACY
        correcting = self._set_correction(0.5, 2, 2)
        new_tau = self._new_tau(eq)
        _norm = vector_norm
        func = eq.vector_func
        beta_p = self.beta_p
        beta_c = self.beta_c
        max_iters = self._MAX_ITERS
        s = h_min * (m - 1)
        simple_solver = self.SIMPLE_SOlVER(h_min, h_min, eps, self._orientation)
        cords = simple_solver(eq, x_0, s)

        new_h = h
        while abs(s) < s_max:
            try:
                tau = cords[-1][1]
                x = cords[-1][0]
                F_p = np.sum([beta_p[i - 1] * cords[-i][1] for i in range(1, m)], axis=0)
                F_c = np.sum([beta_c[i] * cords[-i][1] for i in range(1, m)], axis=0)

                x_p = x + h * F_p
                tau_c = new_tau(x_p, tau)
                x_c = x + h * (F_c[0] * tau_c + F_c)
                for _ in range(max_iters):
                    err = _norm(x_p - x_c)
                    gamma = correcting(err)
                    if err < eps:
                        new_h = min(h * gamma, h_max)
                        break
                    else:
                        new_h = max(h * gamma, h_min)

                    h = new_h
                    tau_c = new_tau(x_c, tau_c)
                    x_c = x + h * (F_c[0] * tau_c + F_c)

                cords.append((x_c, tau_c))

            except (LinAlgError, ZeroDivisionError):
                warnings.warn("Дальнейшее продолжение решения невозможно.", DecisionNotContinueWarning)
                break
            s += h
            h = new_h

        return cords


class SomeEquation1(BaseEquation):
    n = 2  # Указывается число неизвестных в уравнении

    def vector_func(self, x):
        return np.array([(x[1] ** 2 + x[0] ** 2) ** 2 - 2 * (-x[1] ** 2 + x[0] ** 2) + x[0]], dtype="float64")

    def differentiate(self, x):
        return np.array(
            [4 * (x[1] ** 2 + x[0] ** 2) * x[0] - 4 * x[0] + 1, 4 * (x[1] ** 2 + x[0] ** 2) * x[1] + 4 * x[1]],
            dtype="float64")


class AmpersandEquation(BaseEquation):
    n = 2  # Указывается число неизвестных в уравнении

    def vector_func(self, z):  # Вычисляет значение отображения в точке x
        x, y = z
        return np.array([(y**2 - x**2)*(x-1)*(2*x - 3) - 4*(x**2 + y**2 - 2*x)**2], dtype="float64")

    def differentiate(self, z):  # Вычисляет матрицу Якоби в точке x
        x, y = z
        return np.array([-2*x*(x-1)*(2*x - 3) + (y**2 - x**2)*(4*x-5) - 16*(x-1)*(x**2 + y**2 - 2*x),
                         2*y*(x-1)*(2*x - 3) - 16*y*(x**2 + y**2 - 2*x)], dtype="float64")


class CaspidEquation(BaseEquation):
    n = 2  # Указывается число неизвестных в уравнении

    def vector_func(self, z):
        x, y = z
        return np.array([(x**2-1)*(x-1)**2 + (y**2 -1)**2], dtype="float64")

    def differentiate(self, z):
        x, y = z
        return np.array([2*x*(x-1)**2 + 2*(x-1)*(x**2-1), 4*y*(y**2 -1)], dtype="float64")

class FlowerEquation(BaseEquation):
    n = 2  # Указывается число неизвестных в уравнении

    def vector_func(self, z):
        x, y = z
        return np.array([x**4 + 2*x**2 * y**2 + y**4 - x**3 +3*x*y**2], dtype="float64")

    def differentiate(self, z):
        x, y = z
        return np.array([4*x**3 +4*x*y**2 -3*x**2 +3*y**2, 4*y*x**2 + 4*y**3 + 6*x*y], dtype="float64")


class SinEquation(BaseEquation):
    n = 2  # Указывается число неизвестных в уравнении

    def vector_func(self, z):
        x, y = z
        return np.array([sin(x)*cos(y) / (sin(15*x*y) + 2*y**2 + 2) - y**2], dtype="float64")

    def differentiate(self, z):
        x, y = z
        return np.array([(cos(x)*cos(y)*(sin(15*x*y) + 2*y**2 + 2) - 15*y*cos(15*x*y)* sin(x)*cos(y))/(sin(15*x*y) + 2*y**2 + 2)**2,
                        (-sin(x)*sin(y)*(sin(15*x*y) + 2*y**2 + 2) - (15*x*cos(15*x*y) + 4*y)* sin(x)*cos(y))/(sin(15*x*y) + 2*y**2 + 2)**2 - 2*y], dtype="float64")



if __name__ == '__main__':
    # h_min - минимальный размер шага
    # h_max - максимальный размер шага
    # eps - допустимая погрешность
    # orientation - ориентация кривой
    m_euler_solver = ModifiedEulerSolver(h_min=0.01, h_max=0.05, eps=1e-4, orientation=OrientationDecision.RIGHT)
    # s_euler_solver = ModifiedEulerSolver(h_min=0.001, h_max=0.01, eps=1e-4, orientation=OrientationDecision.LEFT)
    # runge_solver = RungeKuttSolver(h_min=0.001, h_max=0.01, eps=1e-4, orientation=OrientationDecision.LEFT)
    eq = SomeEquation1()
    # x_0 - начальное условие задачи Коши
    # s_max - максимальная длина кривой
    res1 = eq.solve(solver=m_euler_solver, x_0=np.array([0,np.sqrt(3)/2], dtype="float64"), s_max=9.2)
    # res2 = eq.solve(solver=s_euler_solver, x_0=np.array([4*np.pi,0], dtype="float64"), s_max=20)
    # res3 = eq.solve(solver=runge_solver, x_0=np.array([4*np.pi,0], dtype="float64"), s_max=20)
    f1 = [vector_norm(eq.vector_func(x)) for x, _ in res1]
    # f2 = [vector_norm(eq.vector_func(x)) for x, _ in res2]
    # f3 = [vector_norm(eq.vector_func(x)) for x, _ in res3]
    print(max(f1))
    # print(max(f2))
    # print(max(f3))


    x1 = [a[0] for a, _ in res1]
    y1 = [a[1] for a, _ in res1]
    # x2 = [a[0] for a, _ in res2]
    # y2 = [a[1] for a, _ in res2]
    # x3 = [a[0] for a, _ in res3]
    # y3 = [a[1] for a, _ in res3]
    fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(projection="3d")
    sns.lineplot(x=x1, y=y1, linestyle="--", marker='o', markersize=4, markevery=250, label="Модифицированный Эйлер", sort=False)
    # sns.lineplot(x=x2, y=y2,linestyle="-.", marker='s', markersize=4, markevery=300, label="Простой Эйлер", sort=False)
    # sns.lineplot(x=x3, y=y3,linestyle=":", marker='p', markersize=4, markevery=200, label="Рунге-Кутт", sort=False)
    # plt.plot(x1, y1, "--", label="Модифицированный Эйлер", color="blue")
    # plt.plot(x1[::30], y1[::30], "o", markersize=5, color="blue")
    # plt.plot(x2, y2, "-.", label="Простой Эйлер", color="orange")
    # plt.plot(x2[::20], y2[::20], "p", markersize=5, color="orange")
    # plt.plot(x3, y3, ":", label="Рунге-Кутт", color="green")
    # plt.plot(x3[::30], y3[::30], "s", markersize=5, color="green")
    # plt.set_xlabel("x")
    # plt.set_ylabel("y")
    plt.grid()
    plt.legend()
    # plt.xlim((-0.5, 2))
    # plt.ylim((-1.5, 1.5))
    plt.show()
