from dataclasses import dataclass, field


@dataclass
class Domain2D:

    Lx: float = 1.0  # X length
    Ly: float = 1.0  # Y length
    Nx: int = 101  # number of X nodes
    Ny: int = 101  # number of Y nodes
    hx: float = field(init=False)  # x grid spacing
    hy: float = field(init=False)  # y grid spacing

    # input validation
    def __post_init__(self) -> None:
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx and Ly must be positive.")
        if self.Nx <= 0 or self.Ny <= 0:
            raise ValueError("Nx and Ny must be positive integers.")

        self.hx = self.Lx / (self.Nx + 1)
        self.hy = self.Ly / (self.Ny + 1)

    @property
    def shape(self):
        return self.Nx, self.Ny
