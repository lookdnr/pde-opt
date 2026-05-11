#pragma once
#include <array>

namespace pdo::domain
{
    struct Domain2D
    {
        std::size_t Nx, Ny;
        double Lx, Ly;
        Domain2D() noexcept;
        Domain2D(std::size_t nx, std::size_t ny, double lx, double ly);

        // getters
        std::array<std::size_t,2> shape() const noexcept { return {Nx, Ny}; }
        double hx() { return Lx / static_cast<double>(Nx); }
        double hy() { return Ly / static_cast<double>(Ny); }
    };
}