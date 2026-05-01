from base import Optimiser
import numpy as np

class SteepestDescent(Optimiser):
    # steepest descent optimisation loop
    def optimise(self):
        q = self.q0

        for k in range(self.max_iter):
            state = self.problem._solve_state(q) # solve state equation A phi = q
            grad = self.problem.grad(state, q) # compute the gradient of the functional J
            
            gnorm = np.linalg.norm(grad.data.ravel()) # compute norm

            Jq = self.problem.J(q)

            # write to history list
            details = {"iter": k, "J": Jq, "grad_norm": gnorm}
            self.monitor.record(details)

            # break if converged (checks norm of gradient below tol)
            if self.converged(grad):
                self.monitor.exit_reason = "Gradient check"
                break
            
            # update:
            direction = grad * (-1.0) # descent direction is negative grad
            alpha = self._line_search(q, grad, direction) # perform line search, update alpha
            if alpha == 0.0:
                self.monitor.exit_reason = "Line search failed"
                break

            q = q + direction * alpha # update the control variable

        if k == self.max_iter: self.monitor.exit_reason = "max_iter reached"
        else: self.monitor.exit_reason = "Succesful convergence"

        return q, self.monitor