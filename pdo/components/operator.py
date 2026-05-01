import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from domain import Domain2D
from variable import Variable

class PoissonOperator:
    _A: sp.spmatrix # 2D block tridiagonal discrete operator matrix
    domain: Domain2D 

    def __init__(self, domain: Domain2D) -> None:
        self.domain = domain
        self._assemble(domain) # assemble discrete operator matrix over the domain

    def _assemble(self, domain: Domain2D) -> None:

        # extract shape grid spacings
        Nx, Ny = domain.Nx, domain.Ny
        hx, hy = domain.hx, domain.hy

        # build intermediate T and identity matrices
        Tx, Ix = self._build_components(Nx)
        Ty, Iy = self._build_components(Ny)

        # assemble using Kronecker products
        self._A = (
            (sp.kron(Iy, Tx, format="csr") / hx**2) + 
            (sp.kron(Ty, Ix, format="csr") / hy**2)
            )

    # build the intermediate components of the discrete operator assembly
    def _build_components(self, N: int) -> Tuple[sp.spmatrix, sp.dia_matrix]:
        T_diag = 2.0 * np.ones(N) # diagonals of T
        T_offdiag = -1.0 * np.ones(N - 1) # off diagonals of T

        # construct intermediate T matrix
        T = sp.diags(
            [T_offdiag, T_diag, T_offdiag],
            offsets=[-1, 0, 1],
            format="csr"
        )

        # construct identity
        I = sp.eye(N, format="csr")

        return T, I
    
    # spy the discrete operator structure
    def show(self) -> None:

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.spy(self._A, markersize=0.1)
        ax.set_title("Discrete Poisson operator matrix structure")
        plt.show()

    # solve the linear system Au = rhs
    def solve(self, rhs: Variable) -> Variable:
        soln = sp.linalg.spsolve(self._A, rhs.data)
        return Variable(self.domain, init=soln)

    # read only accessor - A will be reused in the adjoint solve
    @property
    def A(self):
        return self._A