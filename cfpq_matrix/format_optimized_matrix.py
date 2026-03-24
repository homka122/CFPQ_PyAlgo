from typing import Callable
import warnings

from graphblas.core.matrix import Matrix
from graphblas.core.operator import Monoid, Semiring

from cfpq_matrix.abstract_optimized_matrix_decorator import AbstractOptimizedMatrixDecorator
from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter

# from cfpq_matrix.subtractable_semiring import SubOp


class FormatOptimizedMatrix(AbstractOptimizedMatrixDecorator):
    def __new__(cls, base: OptimizedMatrix, discard_base_on_reformat: bool = True, reformat_threshold: float = 3.0):
        if base.format is None:
            warnings.warn("EnhancedMatrix format is attempted to be optimized twice, " f"ignoring outer reformat_threshold: {reformat_threshold}.")
            return base
        self = object.__new__(cls)
        return self

    def __init__(self, base: OptimizedMatrix, discard_base_on_reformat: bool = True, reformat_threshold: float = 3.0):
        self.reformat_threshold = reformat_threshold
        self.discard_base_on_reformat = discard_base_on_reformat
        self._base = base
        self.matrices = {base.format: base}

    @property
    def base(self) -> OptimizedMatrix:
        return self._base

    def _force_init_format(self, desired_format: str) -> OptimizedMatrix:
        if desired_format not in self.matrices:
            base_matrix = self.base.to_unoptimized().dup()
            base_matrix.ss.config["format"] = desired_format
            self.matrices[desired_format] = self.base.optimize_similarly(MatrixToOptimizedAdapter(base_matrix))
            if self.discard_base_on_reformat:
                del self.matrices[self.base.format]
                self._base = self.matrices[desired_format]
                self.discard_base_on_reformat = False
        res = self.matrices[desired_format]
        return res

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        left_nvals = other.nvals if swap_operands else self.nvals
        right_nvals = self.nvals if swap_operands else other.nvals
        desired_format = "by_row" if left_nvals < right_nvals else "by_col"

        base = other.to_unoptimized()
        if desired_format in self.matrices or other.nvals < self.nvals / self.reformat_threshold:
            base.ss.config["format"] = desired_format
            reformatted_self = self._force_init_format(desired_format)
            return reformatted_self.mxm(MatrixToOptimizedAdapter(base), op, swap_operands=swap_operands)
        return self.base.mxm(other, op, swap_operands=swap_operands)

    def rsub(self, other: OptimizedMatrix, op: Callable[["OptimizedMatrix", "OptimizedMatrix"], "OptimizedMatrix"]) -> OptimizedMatrix:
        return self.matrices.get(other.to_unoptimized().ss.config["format"], self.base).rsub(other, op)

    def iadd(self, other: OptimizedMatrix, op: Monoid):
        for m in self.matrices.values():
            m.iadd(other, op)

    def optimize_similarly(self, other: OptimizedMatrix) -> OptimizedMatrix:
        return FormatOptimizedMatrix(self.base.optimize_similarly(other), reformat_threshold=self.reformat_threshold)

    def __sizeof__(self):
        return sum(m.__sizeof__() for m in self.matrices.values())
