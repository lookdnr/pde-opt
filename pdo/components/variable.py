import numbers
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple

from domain import Domain2D

@dataclass
class Variable:
    domain: Domain2D  # domain which variable belongs to
    init: np.ndarray | None = None  # optionally pass data to initialise
    _data: np.ndarray = field(init=False, repr=False)  # private variable data
    _shape: Tuple[int, int] = field(init=False, repr=False) # shape of domain

    def __post_init__(self):
        self._shape = self.domain.shape

        if self.init is None:
            self._data = np.zeros(self._shape, dtype=float).flatten() # init with zeros if data not provided
        else:
            arr = np.asarray(self.init, dtype=float) # cast to array
            if (len(arr.shape) == 1) and (arr.shape[0] != self._shape[0]*self._shape[1]): # handle flat init data
                raise ValueError(f"Expected shape {np.prod(self._shape)}, got {arr.shape[0]}")
            elif (len(arr.shape) == 2) and (arr.shape != self._shape):
                raise ValueError(f"Expected shape {self._shape}, got {arr.shape}") # throw error for shape mismatch

            arr = arr.flatten() # flatten
            self._data = arr.copy()

    # overloaded subtraction operator
    def __sub__(self, other) -> "Variable":
        if isinstance(other, Variable):
            if self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                raise ValueError("Variables must have the same grid dimensions.")

            return Variable(self.domain, init=self.data - other.data)
        if isinstance(other, numbers.Number):
            return Variable(self.domain, init=self.data - other)
        return NotImplemented
    
    # overloaded addition operator
    def __add__(self, other) -> "Variable":
        if isinstance(other, Variable):
            if self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                raise ValueError("Variables must have the same grid dimensions.")
            return Variable(self.domain, init=self.data + other.data)
        
        if isinstance(other, numbers.Number):
            return Variable(self.domain, init=self.data + other)
        return NotImplemented
    
    # overloaded multiplication operator
    def __mul__(self, other)-> "Variable":
        if isinstance(other, Variable):
            raise NotImplementedError # we won't be needing this, so leave it for now
        
        if isinstance(other, numbers.Number):
            return Variable(self.domain, init=self.data * other)
        return NotImplemented
        
    def __pow__(self, exp) -> "Variable":
        if isinstance(exp, numbers.Number):
            return Variable(self.domain, init=self.data**exp)
        return NotImplemented
        
    # plot the field 
    def show(self, title="Field", interpolate: bool=True) -> None:
        data = self._data.reshape(self._shape) # unflatten

        vmin = data.min()
        vmax = data.max()

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(data, 
                       origin="lower", 
                       extent=(0.0, self.domain.Lx, 0.0, self.domain.Ly), 
                       interpolation="bicubic" if interpolate else "none",
                       vmin=vmin,
                       vmax=vmax
                    )

        # formatting
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$y$", fontsize=12)

        if interpolate: title += " (interpolated)"

        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, label="Field value", orientation="horizontal")
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
    
    # read only accessor
    @property
    def shape(self):
        return self._shape
    
    @property
    def data(self):
        return self._data
    
    # sum functionality for functional evaluation
    def sum(self):
        return self._data.sum()
