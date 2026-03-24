from dataclasses import dataclass
from typing import Callable, Any

from graphblas.core.matrix import Matrix
from graphblas.core.operator import Semiring
# from cfpq_matrix.optimized_matrix import OptimizedMatrix

# SubOp = Callable[[OptimizedMatrix, OptimizedMatrix], OptimizedMatrix]


@dataclass
class SubtractableSemiring:
    one: Any
    semiring: Semiring
    sub_op: Any
