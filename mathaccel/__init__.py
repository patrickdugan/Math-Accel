"""Math-Accel: exact modular linear algebra for degree-bounded closures.

The public objects:
- `CompressedBasis`, an exact row-reduced basis mod p that stores only the non-pivot columns
  (float32, or packed at 2.5 bytes per entry);
- `PackedBlock`, the packed storage;
- `submul`, the exact update V <- (V - A @ B) mod p;
- `HAVE_NUMBA`, whether the parallel kernels are active.

`mathaccel.reference` holds the original implementations, verbatim. They are the ground truth for
the equivalence tests.
"""

from .basis import CompressedBasis, PackedBlock, submul
from .kernels import HAVE_NUMBA

__version__ = "0.2.0"
__all__ = ["CompressedBasis", "PackedBlock", "submul", "HAVE_NUMBA", "__version__"]
