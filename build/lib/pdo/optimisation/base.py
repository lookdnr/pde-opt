from problem import PoissonControlProblem
from components import Variable
from callback import OptimisationMonitor

import numpy as np


class Optimiser:
    def __init__(
        self,
        problem: PoissonControlProblem,
        q0: Variable,
        c: float = 1e-4,
        tau: float = 0.5,
        trial: float | None = None,
        tol: float = 1e-8,
        max_iter: int = 100,
        min_alpha: float = 1e-14,
        max_backtracks: int = 60,
    ) -> None:
        self.problem = problem  # problem encapsulation (target, functional, operator)
        self.q0 = q0  # initial guess
        self.c = c  # Armijo factor
        self.tau = tau  # Armijo step decrease
        self.monitor = OptimisationMonitor()

        # reduced gradient scales with hx*hy, so use an hx*hy-aware default trial step.
        if trial is None:
            self.trial = 1.0 / max(self.problem.functional.hxhy, 1e-16)
        else:
            self.trial = trial

        self.tol = tol  # tolerance for convergence acceptance
        self.max_iter = max_iter  # iteration cap for optimisation loop
        self.min_alpha = min_alpha  # lower bound on line-search step
        self.max_backtracks = max_backtracks  # safeguard against endless backtracking

    # inner product a dot b
    def _inner(self, a: Variable, b: Variable) -> float:
        return float(np.sum(a.data * b.data))

    # Armijo backtracking line search
    def _line_search(
        self, control: Variable, grad: Variable, direction: Variable
    ) -> float:
        alpha = self.trial
        J0 = self.problem.J(control)
        slope = self._inner(grad, direction)  # nablaJ dot dk

        if slope >= 0:
            raise ValueError("Direction is not a descent direction.")

        for _ in range(self.max_backtracks):
            if (
                self.problem.J(control + direction * alpha)
                <= J0 + self.c * alpha * slope
            ):
                return alpha

            alpha *= self.tau
            if alpha < self.min_alpha:
                return 0.0

        return 0.0

    def converged(self, grad: Variable) -> bool:
        return bool(np.linalg.norm(grad.data.ravel()) < self.tol)

    def optimise(self):
        raise NotImplementedError("Subclasses implement optimise()")
