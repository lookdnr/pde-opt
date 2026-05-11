from base import Optimiser
import numpy as np


class CGDescent(Optimiser):

    # conjugate gradient descent optimisation loop
    def optimise(self):
        # ensure fresh monitor and work on a copy of the initial guess
        self.monitor = self.monitor.__class__()
        q = self.q0.copy()
        g_prev = None
        d_prev = None

        for k in range(self.max_iter):
            state = self.problem._solve_state(q)  # solve state equation A phi = q
            grad = self.problem.grad(
                state, q
            )  # compute the gradient of the functional J

            gnorm = np.linalg.norm(grad.data.ravel())  # compute norm

            Jq = self.problem.J(q)

            # write to history list
            details = {"iter": k, "J": Jq, "grad_norm": gnorm}
            self.monitor.record(details)

            # break if converged (checks norm of gradient below tol)
            if self.converged(grad):
                self.monitor.exit_reason = "Gradient check"
                break

            if g_prev is None:
                beta = 0.0
                direction = grad * -1
            else:
                # Polak–Ribière
                dg = grad - g_prev
                beta_pr = self._inner(grad, dg) / self._inner(g_prev, g_prev)
                beta = max(0.0, beta_pr)  # non‑negative PR

                direction = grad * -1 + d_prev * beta

                if beta < 1e-8:  # heuristic restart: effectively steepest
                    direction = grad * -1
                    beta = 0.0

            # update:
            alpha = self._line_search(
                q, grad, direction
            )  # perform line search, update alpha
            if alpha == 0.0:
                self.monitor.exit_reason = "Line search failed"
                break

            q = q + direction * alpha  # update the control variable
            g_prev = grad
            d_prev = direction

        if not self.monitor.exit_reason:
            if k == self.max_iter - 1:
                self.monitor.exit_reason = "max_iter reached"
            else:
                self.monitor.exit_reason = "Succesful convergence"
        self.has_run = True

        return q, self.monitor
