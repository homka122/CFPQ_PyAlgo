from typing import Tuple, Callable

from graphblas.core.dtypes import DataType
from graphblas.core.matrix import Matrix
from graphblas.core.operator import Monoid, Semiring

from cfpq_matrix.optimized_matrix import OptimizedMatrix, MatrixFormat
# from cfpq_matrix.subtractable_semiring import SubOp


class MatrixToOptimizedAdapter(OptimizedMatrix):

    def __init__(self, base: Matrix):
        assert isinstance(base, Matrix)
        self.base = base

    @property
    def nvals(self) -> int:
        return self.base.nvals

    @property
    def shape(self) -> Tuple[int, int]:
        return self.base.shape

    @property
    def format(self) -> MatrixFormat:
        return self.base.ss.config["format"]

    @property
    def dtype(self) -> DataType:
        return self.base.dtype

    def to_unoptimized(self) -> Matrix:
        return self.base

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        return other.mxm(self, op) if swap_operands else MatrixToOptimizedAdapter(self.base.mxm(other.to_unoptimized(), op).new(self.dtype))

    def rsub(self, other: OptimizedMatrix, op: Callable[["OptimizedMatrix", "OptimizedMatrix"], "OptimizedMatrix"]) -> OptimizedMatrix:
        return op(other, self)

    def iadd(self, other: OptimizedMatrix, op: Monoid):
        self.base << self.base.ewise_add(other.to_unoptimized(), op=op)

    def optimize_similarly(self, other: OptimizedMatrix) -> OptimizedMatrix:
        return MatrixToOptimizedAdapter(other.to_unoptimized())

    def __sizeof__(self):
        return self.base.__sizeof__()
