from typing import Callable
from graphblas.core.matrix import Matrix
from graphblas.core.operator import Monoid, Semiring

from cfpq_matrix.abstract_optimized_matrix_decorator import AbstractOptimizedMatrixDecorator
from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter
# from cfpq_matrix.subtractable_semiring import SubOp


class LazyAddOptimizedMatrix(AbstractOptimizedMatrixDecorator):
    def __init__(self, base: OptimizedMatrix, nvals_factor: int = 10, min_nvals: int = 10):
        assert min_nvals > 0
        assert nvals_factor > 0
        self.matrices: list[OptimizedMatrix] = [base]
        self.size_factor = nvals_factor
        self.min_size = min_nvals
        self.last_used_monoid: Monoid | None = None

    @property
    def base(self) -> OptimizedMatrix:
        return self.matrices[0]

    @property
    def nvals(self) -> int:
        return sum(m.nvals for m in self.matrices)

    def _map_and_fold_mxm(
        self,
        mapper: Callable[[OptimizedMatrix], Matrix],
        nvals_combine_threshold: int,
        combiner: Callable[[Matrix, Matrix], Matrix],
        acc: Matrix | None = None
    ) -> Matrix:
        self.force_combine_small_matrices(nvals_combine_threshold)

        for cur in sorted(
                (mapper(m) for m in self.matrices if m.nvals != 0),
                key=lambda m: m.nvals, reverse=False
        ):
            acc = cur if acc is None else combiner(acc, cur)
        return mapper(self.base) if acc is None else acc

    def _map_and_fold_rsub(
            self,
            mapper: Callable[[OptimizedMatrix], OptimizedMatrix],
            nvals_combine_threshold: int,
            combiner: Callable[[Matrix, OptimizedMatrix], Matrix],
            acc: Matrix,
            reverse_sort=False,
    ) -> Matrix:
        self.force_combine_small_matrices(nvals_combine_threshold)

        for cur in sorted(
                (mapper(m) for m in self.matrices if m.nvals != 0),
                key=lambda m: m.nvals, reverse=reverse_sort
        ):
            acc = cur if acc is None else combiner(acc, cur)
        return mapper(self.base) if acc is None else acc

    def force_combine_small_matrices(self, nvals_combine_threshold):
        new_matrices: list[OptimizedMatrix] = []
        for m in sorted(self.matrices, key=lambda m: m.nvals):
            if m.nvals <= nvals_combine_threshold and len(new_matrices) > 0:
                assert self.last_used_monoid is not None
                new_matrices[-1].iadd(MatrixToOptimizedAdapter(m.to_unoptimized()), op=self.last_used_monoid)
            else:
                new_matrices.append(m)
        self.matrices = new_matrices

    def to_unoptimized(self) -> Matrix:
        self.force_combine_small_matrices(nvals_combine_threshold=float("inf"))
        return self.base.to_unoptimized()

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        self.update_monoid(op.monoid)
        return MatrixToOptimizedAdapter(self._map_and_fold_mxm(
            mapper=lambda m: m.mxm(other, op=op, swap_operands=swap_operands).to_unoptimized(),
            combiner=lambda acc, cur: acc.ewise_add(cur, op=op.monoid).new(),
            nvals_combine_threshold=other.nvals
        ))

    def rsub(self, other: OptimizedMatrix, op: Callable[["OptimizedMatrix", "OptimizedMatrix"], "OptimizedMatrix"]) -> OptimizedMatrix:
        return MatrixToOptimizedAdapter(self._map_and_fold_rsub(
            acc=other.to_unoptimized(),
            reverse_sort=True,
            mapper=lambda m: m,
            combiner=lambda acc, cur: cur.rsub(MatrixToOptimizedAdapter(acc), op).to_unoptimized(),
            nvals_combine_threshold=other.nvals
        ))

    def iadd(self, other: OptimizedMatrix, op: Monoid):
        self.update_monoid(op)
        other_base = other.to_unoptimized().dup()
        if self.format is not None:
            other_base.ss.config["format"] = self.format
        base = self.base
        while True:
            other_nvals = max(other_base.nvals, self.min_size)
            i = next(
                (i for i in range(len(self.matrices))
                 if other_nvals / self.size_factor <=
                 max(self.min_size, self.matrices[i].nvals)
                 <= other_nvals * self.size_factor),
                None
            )

            if i is None:
                self.matrices.append(base.optimize_similarly(MatrixToOptimizedAdapter(other_base)))
                return self
            other_base << other_base.ewise_add(
                self.matrices[i].to_unoptimized(),
                op=op
            )
            del self.matrices[i]

    def update_monoid(self, op: Monoid):
        if self.last_used_monoid is not op:
            self.force_combine_small_matrices(nvals_combine_threshold=float("inf"))
            self.last_used_monoid = op

    def optimize_similarly(self, other: OptimizedMatrix) -> OptimizedMatrix:
        return LazyAddOptimizedMatrix(
            self.base.optimize_similarly(other),
            nvals_factor=self.size_factor,
            min_nvals=self.min_size
        )

    def __sizeof__(self):
        return sum(m.__sizeof__() for m in self.matrices)
