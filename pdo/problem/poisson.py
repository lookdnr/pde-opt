from components import Variable, PoissonOperator, Functional


class PoissonControlProblem:
    target: Variable # target variable q
    operator: PoissonOperator  # discrete operator
    functional: Functional  # functional evaluation
    _hxhy: float  # product of grid spacings
    _beta: float
    
    def __init__(self, target: Variable, operator: PoissonOperator, functional: Functional) -> None:
        self.target = target
        self.operator = operator
        self.functional = functional
        self._hxhy = self.functional.hxhy
        self._beta = self.functional.beta

    # solve the the discretised PDE A phi = q
    def _solve_state(self, control: Variable) -> Variable:
        return self.operator.solve(control)
    
    # solve the adjoint problem A lambda = hx*hy*r
    def _solve_adjoint(self, state: Variable) -> Variable:
        misfit = state - self.target  # residual vector
        rhs = misfit * self._hxhy # hx*hy*(phi - phihat)
        return self.operator.solve(rhs)  # A lambda = rhs
    
    # evaluate J(varphi(q), q) => J(q) (the reduced functional)
    def J(self, control: Variable) -> float:
        state = self._solve_state(control) # obtain phi(q) from A phi(q) = q
        return self.functional.evaluate(state, control, self.target)

    def grad(self, state: Variable, control: Variable):
        lam = self._solve_adjoint(state)
        return control * (self._beta * self._hxhy) + lam