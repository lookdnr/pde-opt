#include "pdo/domain/twod.h"
#include <assert.h>

namespace pdo::domain
{   
    // default constructor
    Domain2D::Domain2D() {
        // set mesh parameters
        Nx = 101; Ny = 101;
        Lx = 1.0; Ly = 1.0;
    }

    // constructor with args
    Domain2D::Domain2D(std::size_t nx, std::size_t ny, double lx, double ly) {
        assert(nx >= 0);
        assert(nx >= 0);
        assert(lx >= 0);
        assert(ly >= 0);

        // set mesh parameters
        Nx = nx; Ny = ny;
        Lx = lx; Ly = ly;
    }
}