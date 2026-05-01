from dataclasses import dataclass, field
from domain import Domain2D
from variable import Variable


@dataclass
class Functional:
    domain: Domain2D
    beta: float  # regularisation parameter

    hx: float = field(init=False, repr=False)  # x grid spacing
    hy: float = field(init=False, repr=False)  # y gird spacing
    _hxhy: float = field(
        init=False, repr=False
    )  # grid spacing product for reisdual scaling

    def __post_init__(self) -> None:
        self.hx = self.domain.hx
        self.hy = self.domain.hy
        self._hxhy = self.hx * self.hy

    # evaluate J(phi, q) = hxhy/2 ||phi - phi hat||^2_2 + b hxhy / 2 ||q||^2_2
    def evaluate(self, state: Variable, control: Variable, target: Variable) -> float:
        residual_term = (state - target) ** 2
        control_term = control**2
        return self._hxhy * (residual_term.sum() + self.beta * control_term.sum()) / 2.0

    @property
    def hxhy(self):
        return self._hxhy
