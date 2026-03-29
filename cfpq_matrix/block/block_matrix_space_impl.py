from typing import Tuple, List

import graphblas.ss
from graphblas.core.matrix import Matrix
import graphblas.monoid
from graphblas.core.operator import Monoid

from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.block.block_matrix import BlockMatrix, CellBlockMatrix, VectorBlockMatrix
from cfpq_matrix.block.block_matrix_space import BlockMatrixSpace, BlockMatrixOrientation


class BlockMatrixSpaceImpl(BlockMatrixSpace):
    def __init__(self, cell_shape: tuple[int, int], block_count: int):
        assert cell_shape[0] >= 0 and cell_shape[1] >= 0
        assert block_count >= 1
        self._cell_shape: tuple[int, int] = cell_shape
        self._block_count: int = block_count

    @property
    def block_count(self) -> int:
        return self._block_count

    @property
    def cell_shape(self) -> Tuple[int, int]:
        return self._cell_shape

    def is_single_cell(self, matrix_shape: Tuple[int, int]) -> bool:
        return matrix_shape == self.cell_shape

    def is_hyper_vector(self, matrix_shape: Tuple[int, int]) -> bool:
        return matrix_shape in [(self.cell_shape[0] * self.block_count, self.cell_shape[1]), (self.cell_shape[0], self.cell_shape[1] * self.block_count)]

    def get_block_matrix_orientation(self, matrix_shape: Tuple[int, int]) -> BlockMatrixOrientation:
        return {
            (self.cell_shape[0] * self.block_count, self.cell_shape[1]): BlockMatrixOrientation.VERTICAL,
            (self.cell_shape[0], self.cell_shape[1] * self.block_count): BlockMatrixOrientation.HORIZONTAL,
        }[matrix_shape]

    def reduce_hyper_vector_or_cell(self, hyper_vector_or_cell: Matrix, op: Monoid) -> Matrix:
        if self.is_single_cell(hyper_vector_or_cell.shape):
            return hyper_vector_or_cell
        input_orientation = self.get_block_matrix_orientation(hyper_vector_or_cell.shape)
        (rows, columns, values) = hyper_vector_or_cell.to_coo()
        if input_orientation == BlockMatrixOrientation.VERTICAL:
            rows = rows % self.cell_shape[0]
        elif input_orientation == BlockMatrixOrientation.HORIZONTAL:
            columns = columns % self.cell_shape[1]
        else:
            assert False
        return Matrix.from_coo(rows, columns, values, nrows=self.cell_shape[0], ncols=self.cell_shape[1], dup_op=op)

    def hyper_rotate(self, hyper_vector: Matrix, orientation: BlockMatrixOrientation) -> Matrix:
        input_orientation = self.get_block_matrix_orientation(hyper_vector.shape)
        if input_orientation == orientation:
            return hyper_vector
        (rows, columns, values) = hyper_vector.to_coo()
        if orientation == BlockMatrixOrientation.VERTICAL:
            rows = rows + (columns // self.cell_shape[1] * self.cell_shape[0])
            columns = columns % self.cell_shape[1]
        elif orientation == BlockMatrixOrientation.HORIZONTAL:
            columns = columns + (rows // self.cell_shape[0] * self.cell_shape[1])
            rows = rows % self.cell_shape[0]
        else:
            assert False

        return Matrix.from_coo(rows, columns, values, nrows=hyper_vector.ncols, ncols=hyper_vector.nrows)

    def to_block_diag_matrix(self, hyper_vector: Matrix) -> Matrix:
        input_orientation = self.get_block_matrix_orientation(hyper_vector.shape)
        (rows, columns, values) = hyper_vector.to_coo()
        if input_orientation == BlockMatrixOrientation.VERTICAL:
            columns = columns + (rows // self.cell_shape[0] * self.cell_shape[1])
        elif input_orientation == BlockMatrixOrientation.HORIZONTAL:
            rows = rows + (columns // self.cell_shape[1] * self.cell_shape[0])
        else:
            assert False

        return Matrix.from_coo(rows, columns, values, nrows=self.cell_shape[0] * self.block_count, ncols=self.cell_shape[1] * self.block_count)

    def create_hyper_vector(self, typ, orientation: BlockMatrixOrientation) -> Matrix:
        shape = {
            BlockMatrixOrientation.VERTICAL: (self.cell_shape[0] * self.block_count, self.cell_shape[1]),
            BlockMatrixOrientation.HORIZONTAL: (self.cell_shape[0], self.cell_shape[1] * self.block_count),
        }[orientation]
        return Matrix(dtype=typ, nrows=shape[0], ncols=shape[1])

    def create_cell(self, typ) -> Matrix:
        return Matrix(dtype=typ, nrows=self.cell_shape[0], ncols=self.cell_shape[1])

    def stack_into_hyper_column(self, matrices: List[Matrix]) -> Matrix:
        assert len(matrices) == self.block_count
        tiles = [[m] for m in matrices]
        return graphblas.ss.concat(tiles)

    def repeat_into_hyper_column(self, matrix: Matrix) -> Matrix:
        return self.stack_into_hyper_column([matrix] * self.block_count)

    def automize_block_operations(self, base: OptimizedMatrix) -> BlockMatrix:
        assert self.is_single_cell(base.shape) or self.is_hyper_vector(base.shape)
        return CellBlockMatrix(base, self) if self.is_single_cell(base.shape) else VectorBlockMatrix(base, self)

    def get_hyper_vector_blocks(self, hyper_vector: Matrix) -> List[Matrix]:
        return [cell for row in hyper_vector.ss.split((self.cell_shape[0], self.cell_shape[1])) for cell in row]
