from typing import Callable
from abc import ABC

from graphblas.core.matrix import Matrix
from graphblas.core.operator import Monoid, Semiring

from cfpq_matrix.abstract_optimized_matrix_decorator import AbstractOptimizedMatrixDecorator
from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter
from cfpq_matrix.block.block_matrix_space import BlockMatrixSpace, BlockMatrixOrientation

# from cfpq_matrix.subtractable_semiring import SubOp


class BlockMatrix(AbstractOptimizedMatrixDecorator, ABC):
    def __init__(self, base: OptimizedMatrix, hyper_space: BlockMatrixSpace):
        self._base = base
        self.block_matrix_space = hyper_space

    @property
    def base(self) -> OptimizedMatrix:
        return self._base

    def optimize_similarly_with_block(self, other: OptimizedMatrix, other_block: BlockMatrixSpace) -> OptimizedMatrix:
        result = other_block.automize_block_operations(self.base.optimize_similarly(other))
        if isinstance(result, CellBlockMatrix):
            assert result.block_matrix_space.cell_shape == result.shape
        return result

    def optimize_similarly(self, other: OptimizedMatrix) -> OptimizedMatrix:
        return self.block_matrix_space.automize_block_operations(self.base.optimize_similarly(other))


class CellBlockMatrix(BlockMatrix):
    def __init__(self, base: OptimizedMatrix, block_matrix_space: BlockMatrixSpace):
        assert block_matrix_space.is_single_cell(base.shape)
        super().__init__(base, block_matrix_space)

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        assert isinstance(other, BlockMatrix)
        assert (
            self.block_matrix_space.cell_shape[1] == other.block_matrix_space.cell_shape[0]
            if not swap_operands
            else self.block_matrix_space.cell_shape[0] == other.block_matrix_space.cell_shape[1]
        )
        if other.block_matrix_space.is_single_cell(other.shape):
            return self.base.mxm(other.base, op, swap_operands=swap_operands)
        return self.base.mxm(
            MatrixToOptimizedAdapter(
                other.block_matrix_space.hyper_rotate(
                    other.to_unoptimized(), BlockMatrixOrientation.VERTICAL if swap_operands else BlockMatrixOrientation.HORIZONTAL
                )
            ),
            op=op,
            swap_operands=swap_operands,
        )

    def rsub(self, other: OptimizedMatrix, op: Callable[["OptimizedMatrix", "OptimizedMatrix"], "OptimizedMatrix"]) -> OptimizedMatrix:
        assert isinstance(other, BlockMatrix)
        assert self.block_matrix_space.cell_shape == other.block_matrix_space.cell_shape
        assert self.block_matrix_space.is_single_cell(other.shape)
        return self.base.rsub(other.base, op)

    def iadd(self, other: OptimizedMatrix, op: Monoid):
        assert isinstance(other, BlockMatrix)
        assert self.block_matrix_space.cell_shape == other.block_matrix_space.cell_shape
        self.base.iadd(MatrixToOptimizedAdapter(self.block_matrix_space.reduce_hyper_vector_or_cell(other.to_unoptimized(), op)), op)

    def __sizeof__(self):
        return self.base.__sizeof__()


class VectorBlockMatrix(BlockMatrix):
    def __init__(
        self,
        base: OptimizedMatrix,
        block_matrix_space: BlockMatrixSpace,
        discard_base_on_reformat: bool = True,
    ):
        assert block_matrix_space.is_hyper_vector(base.shape)
        super().__init__(base, block_matrix_space)
        self.matrices = {block_matrix_space.get_block_matrix_orientation(base.shape): base}
        self.discard_base_on_reformat = discard_base_on_reformat

    def _force_init_orientation(self, desired_orientation: BlockMatrixOrientation) -> "OptimizedMatrix":
        if desired_orientation not in self.matrices:
            rotated_matrix = MatrixToOptimizedAdapter(self.block_matrix_space.hyper_rotate(self.base.to_unoptimized(), desired_orientation))
            self.matrices[desired_orientation] = self.base.optimize_similarly(rotated_matrix)
            if self.discard_base_on_reformat:
                base_shape = self.block_matrix_space.get_block_matrix_orientation(self.base.shape)
                del self.matrices[base_shape]
                self._base = self.matrices[desired_orientation]
        self.discard_base_on_reformat = False
        return self.matrices[desired_orientation]

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        assert isinstance(other, BlockMatrix)
        assert (
            self.block_matrix_space.cell_shape[1] == other.block_matrix_space.cell_shape[0]
            if not swap_operands
            else self.block_matrix_space.cell_shape[0] == other.block_matrix_space.cell_shape[1]
        )
        if other.block_matrix_space.is_single_cell(other.shape):
            return self._force_init_orientation(BlockMatrixOrientation.HORIZONTAL if swap_operands else BlockMatrixOrientation.VERTICAL).mxm(
                other, op, swap_operands=swap_operands
            )
        return self._force_init_orientation(BlockMatrixOrientation.VERTICAL if swap_operands else BlockMatrixOrientation.HORIZONTAL).mxm(
            MatrixToOptimizedAdapter(other.block_matrix_space.to_block_diag_matrix(other.to_unoptimized())), op=op, swap_operands=swap_operands
        )

    def rsub(self, other: OptimizedMatrix, op: Callable[["OptimizedMatrix", "OptimizedMatrix"], "OptimizedMatrix"]) -> OptimizedMatrix:
        assert isinstance(other, BlockMatrix)
        assert self.block_matrix_space.cell_shape == other.block_matrix_space.cell_shape
        if self.block_matrix_space.get_block_matrix_orientation(other.shape) not in self.matrices:
            my_shape = next(self.matrices.keys().__iter__())
            other = MatrixToOptimizedAdapter(self.block_matrix_space.hyper_rotate(other.to_unoptimized(), my_shape))
        other_shape = self.block_matrix_space.get_block_matrix_orientation(other.shape)
        return self.matrices[other_shape].rsub(other, op)

    def iadd(self, other: OptimizedMatrix, op: Monoid):
        assert isinstance(other, BlockMatrix)
        assert self.block_matrix_space.cell_shape == other.block_matrix_space.cell_shape
        if self.block_matrix_space.is_single_cell(other.shape):
            other = MatrixToOptimizedAdapter(self.block_matrix_space.repeat_into_hyper_column(other.to_unoptimized()))
        for orientation, m in self.matrices.items():
            m.iadd(MatrixToOptimizedAdapter(self.block_matrix_space.hyper_rotate(other.to_unoptimized(), orientation)), op=op)

    def __sizeof__(self):
        return sum(m.__sizeof__() for m in self.matrices.values())
