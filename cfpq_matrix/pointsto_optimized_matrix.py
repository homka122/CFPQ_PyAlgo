from numba import njit, prange
import numba as nb
from cfpq_matrix.block.block_matrix_space_impl import BlockMatrixSpaceImpl
from numpy import block
from cfpq_model.cnf_grammar_template import Symbol
from networkx.drawing.nx_agraph import from_agraph
import test
from packaging.utils import _
from typing import Literal, Callable
from abc import ABC

import graphblas
import numpy as np
from graphblas.core.matrix import Matrix
from graphblas.core.operator import Monoid, Semiring

from cfpq_matrix.abstract_optimized_matrix_decorator import AbstractOptimizedMatrixDecorator
from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter
from cfpq_matrix.block.block_matrix_space import BlockMatrixSpace, BlockMatrixOrientation
from cfpq_matrix.block.block_matrix import BlockMatrix


class PointsToMatrix(AbstractOptimizedMatrixDecorator, ABC):
    def __init__(self, base: OptimizedMatrix, type: Literal["RSM", "Context", "State"], n: int, context_num: int, depth: int = -1):
        assert isinstance(base, BlockMatrix)
        self._base = base
        assert (type == "State" and depth != -1) or (type == "RSM") or (type == "Context")
        self.type: Literal["State", "RSM", "Context"] = type
        self.depth: int = depth
        self.n: int = n
        self.context_num: int = context_num
        self.block_space: BlockMatrixSpace = base.block_matrix_space
        self.reduced: BlockMatrix | None = None

    # written by Homka122
    def _is_flatted(self) -> bool:
        if self.type == "RSM" or self.type == "Context":
            return True
        if self._get_inner_shape()[0] == 1:
            return True

        return False

    def _is_grouped(self) -> bool:
        if self.type == "RSM" or self.type == "Context":
            return True
        if self._get_inner_shape()[0] == self.context_num:
            return True

        return False

    def _get_inner_shape(self) -> tuple[int, int]:
        return (self.block_space.cell_shape[0] // self.n, self.block_space.cell_shape[1] // self.n)

    @staticmethod
    def get_type_from_symbol(symbol: str) -> Literal["RSM", "Context", "State"]:
        if symbol.startswith(("(", ")")):
            return "Context"
        elif symbol.startswith(("alloc", "assign", "load", "store")):
            return "RSM"
        else:
            return "State"

    @staticmethod
    def get_depth_from_symbol(symbol: str) -> int:
        if symbol.startswith(("(", ")")):
            if "i" in symbol:
                return 1
            else:
                return 0
        elif not symbol.startswith("S"):
            return 0
        if "G" in symbol:
            # S_1_G0
            if symbol.endswith("_i"):
                return int(symbol.split("_")[-2].split("G")[-1])
            return int(symbol.split("G")[-1].split("_")[0])
        else:
            # S_1_(0, 0)
            return 0
            # return int(symbol.split("(")[-1].split(",")[0])

    def _transform_matrix(
        self,
        new_cell_shape,
        orientation: BlockMatrixOrientation,
        tranform: Callable[[np.ndarray, np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]],
        op: Semiring | None = None,
    ) -> BlockMatrix:
        assert isinstance(self.base, BlockMatrix)

        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not is_cell:
            # rotate vector
            orientation_cur = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation_cur != orientation:
                if orientation == BlockMatrixOrientation.VERTICAL:
                    rows = rows + (cols // cell_w * cell_h)
                    cols = cols % cell_w
                if orientation == BlockMatrixOrientation.HORIZONTAL:
                    cols = cols + (rows // cell_h * cell_w)
                    rows = rows % cell_h

        if is_cell:
            rows, cols, values, mask = tranform(rows, cols, values, cell_h, cell_w)
        elif orientation == BlockMatrixOrientation.VERTICAL:
            indecies = rows // cell_h
            rows = rows % cell_h
            rows, cols, values, mask = tranform(rows, cols, values, cell_h, cell_w)
            if mask is None:
                rows = rows + new_cell_shape[0] * indecies
            else:
                rows = rows + new_cell_shape[0] * indecies[mask]
        elif orientation == BlockMatrixOrientation.HORIZONTAL:
            indecies = cols // cell_w
            cols = cols % cell_w
            rows, cols, values, mask = tranform(rows, cols, values, cell_h, cell_w)
            if mask is None:
                cols = cols + new_cell_shape[1] * indecies
            else:
                cols = cols + new_cell_shape[1] * indecies[mask]

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            if orientation == BlockMatrixOrientation.VERTICAL:
                nrows *= self.block_space.block_count
            elif orientation == BlockMatrixOrientation.HORIZONTAL:
                ncols *= self.block_space.block_count

        base = Matrix.from_coo(rows, cols, values, nrows=nrows, ncols=ncols, dup_op=op)

        new_block_space = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = self.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_space)
        assert isinstance(base, BlockMatrix)

        return base

    # make from matrix with size 1 x nums^(depth) matrix with size nums x nums^(depth-1) if depth != 0
    def _flat_matrix(self) -> None:
        if self._is_flatted():
            return

        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()
        assert input_shape[0] == self.context_num
        output_shape = (1, input_shape[1] * self.context_num)

        new_cell_shape = (output_shape[0] * self.n, output_shape[1] * self.n)

        def transform(rows, cols, values, cell_h, cell_w):
            cols = cols % self.n + cols // self.n * cell_h + (rows // self.n * self.n)
            rows = rows % self.n

            return rows, cols, values, None

        base = self._transform_matrix(new_cell_shape, BlockMatrixOrientation.VERTICAL, transform)

        self._base = base
        self.block_space = base.block_matrix_space

    def _flat_matrix_rotate(self) -> OptimizedMatrix:
        assert self._is_flatted()

        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()

        if input_shape[0] == input_shape[1]:
            return self.base

        assert input_shape[0] == 1
        output_shape = (input_shape[1], input_shape[0])

        new_cell_shape = (output_shape[0] * self.n, output_shape[1] * self.n)

        def transform(rows, cols, values, cell_h, cell_w):
            rows = rows + (cols // cell_h * cell_h)
            cols = cols % cell_h

            return rows, cols, values, None

        base = self._transform_matrix(new_cell_shape, BlockMatrixOrientation.HORIZONTAL, transform)

        return base

    def _reduce_diag_matrix(self) -> OptimizedMatrix:
        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()
        assert input_shape[0] == input_shape[1]

        output_shape = (1, input_shape[1])

        new_cell_shape = (output_shape[0] * self.n, output_shape[1] * self.n)

        def transform(rows, cols, values, cell_h, cell_w):
            mask = (rows // self.n) == (cols // self.n)
            rows = rows[mask]
            cols = cols[mask]
            values = values[mask]

            rows = rows % self.n

            return rows, cols, values, mask

        base = self._transform_matrix(new_cell_shape, BlockMatrixOrientation.VERTICAL, transform)

        return base

    def _reduce_diag_context_matrix(self, op: Semiring) -> OptimizedMatrix:
        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()
        assert input_shape[0] == self.context_num

        output_shape = (self.context_num, input_shape[1] // self.context_num)

        new_cell_shape = (output_shape[0] * self.n, output_shape[1] * self.n)

        def transform(rows, cols, values, cell_h, cell_w):
            mask = (rows // self.n) == (cols % (self.n * self.context_num) // self.n)
            rows = rows[mask]
            cols = cols[mask]
            values = values[mask]

            rows = rows % self.n
            cols = cols % self.n + cols // (self.n * self.context_num) * self.n

            return rows, cols, values, mask

        base = self._transform_matrix(new_cell_shape, BlockMatrixOrientation.VERTICAL, transform, op=op)

        return base

    def _flat_matrix_rotate_reverse(self) -> OptimizedMatrix:

        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()
        # assert input_shape[0] == 1
        output_shape = (1, input_shape[1] * self.context_num)

        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        new_cell_shape = (cell_w, cell_h)
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not is_cell:
            orientation = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation == BlockMatrixOrientation.VERTICAL:
                cols = cols + (rows // cell_h * cell_w)
                rows = rows % cell_h

        cols = cols % cell_w + (cols // cell_w * cell_w) + (rows // cell_w * cell_h)
        rows = rows % cell_w

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            ncols *= self.block_space.block_count

        base = Matrix.from_coo(
            rows,
            cols,
            values,
            nrows=nrows,
            ncols=ncols,
        )

        new_block_space = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = self.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_space)
        return base

    def _get_hyper_vector_shape(self, matrix: Matrix) -> list[Matrix]:
        return [cell for row in matrix.ss.split((self.n, self.n)) for cell in row]

    def _group_matrix(self) -> None:
        if self._is_grouped():
            return

        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()
        assert input_shape[0] == 1
        output_shape = (self.context_num, input_shape[1] // self.context_num)

        new_cell_shape = (output_shape[0] * self.n, output_shape[1] * self.n)

        def transform(rows, cols, values, cell_h, cell_w):
            rows = rows + (cols // cell_h % self.context_num * cell_h)
            cols = cols % cell_h + cols // (cell_h * self.context_num) * cell_h

            return rows, cols, values, None

        base = self._transform_matrix(new_cell_shape, BlockMatrixOrientation.HORIZONTAL, transform)

        self._base = base
        self.block_space = base.block_matrix_space

    @staticmethod
    def get_block_diag_matrix(matrix: OptimizedMatrix, graph_size: int) -> OptimizedMatrix:
        assert isinstance(matrix, PointsToMatrix)
        assert isinstance(matrix.base, BlockMatrix)

        cell_h = matrix.block_space.cell_shape[0]
        cell_w = matrix.block_space.cell_shape[1]
        new_cell_shape = (cell_w, cell_w)
        is_cell = matrix.block_space.is_single_cell(matrix.shape)

        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        if not is_cell:
            orientation = matrix.block_space.get_block_matrix_orientation(matrix.shape)
            if orientation == BlockMatrixOrientation.HORIZONTAL:
                rows = rows + (cols // cell_w * cell_h)
                cols = cols % cell_w

        rows = rows % cell_h + rows // cell_h * cell_w + cols // cell_h * cell_h

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            nrows *= matrix.block_space.block_count

        base = Matrix.from_coo((rows), (cols), (values), nrows=nrows, ncols=ncols)

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, matrix.block_space.block_count)
        base = matrix.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        return base

    @staticmethod
    def get_hyper_column(matrix: OptimizedMatrix, count: int) -> OptimizedMatrix:
        assert isinstance(matrix, PointsToMatrix)
        assert isinstance(matrix.base, BlockMatrix)

        cell_h = matrix.block_space.cell_shape[0]
        new_cell_shape = (matrix.block_space.cell_shape[0] * count, matrix.block_space.cell_shape[1])
        is_cell = matrix.block_space.is_single_cell(matrix.shape)

        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        if not is_cell:
            orientation = matrix.block_space.get_block_matrix_orientation(matrix.shape)
            if orientation == BlockMatrixOrientation.VERTICAL:
                cols = cols + (rows // cell_h * cell_h)
                rows = rows % cell_h

        all_rows = []
        for i in range(count):
            all_rows.append(rows + cell_h * i)
        cols = [cols] * count
        values = [values] * count

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            ncols *= matrix.block_space.block_count

        base = Matrix.from_coo(
            np.concatenate(all_rows),
            np.concatenate(cols),
            np.concatenate(values),
            nrows=nrows,
            ncols=ncols,
        )

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, matrix.base.block_matrix_space.block_count)
        base = matrix.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        return base

    @staticmethod
    def get_hyper_row(matrix: OptimizedMatrix, count: int, vertex_count: int, block_count: int) -> OptimizedMatrix:
        assert isinstance(matrix, PointsToMatrix)
        assert isinstance(matrix.base, BlockMatrix)

        cell_h = matrix.block_space.cell_shape[0]
        new_cell_shape = (matrix.block_space.cell_shape[0], matrix.block_space.cell_shape[1] * count)
        is_cell = matrix.block_space.is_single_cell(matrix.shape)

        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        if not matrix.block_space.is_single_cell(matrix.shape):
            orientation = matrix.block_space.get_block_matrix_orientation(matrix.shape)
            if orientation == BlockMatrixOrientation.HORIZONTAL:
                rows = rows + (cols // cell_h * cell_h)
                cols = cols % cell_h

        rows = [rows] * count
        all_cols = []
        for i in range(count):
            all_cols.append(cols + cell_h * i)
        values = [values] * count

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            nrows *= matrix.block_space.block_count

        base = Matrix.from_coo(
            np.concatenate(rows),
            np.concatenate(all_cols),
            np.concatenate(values),
            nrows=nrows,
            ncols=ncols,
        )

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, matrix.base.block_matrix_space.block_count)
        base = matrix.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        return base

    @staticmethod
    def reduce_column(matrix: OptimizedMatrix, op: Monoid, vertex_count: int, block_count: int) -> OptimizedMatrix:
        assert isinstance(matrix, BlockMatrix)

        cell_h = matrix.block_matrix_space.cell_shape[0]
        cell_w = matrix.block_matrix_space.cell_shape[1]
        new_cell_shape = (vertex_count, vertex_count)
        is_cell = matrix.block_matrix_space.is_single_cell(matrix.shape)

        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        if not is_cell:
            orientation = matrix.block_matrix_space.get_block_matrix_orientation(matrix.shape)
            if orientation == BlockMatrixOrientation.VERTICAL:
                cols = cols + (rows // cell_h * cell_w)
                rows = rows % cell_h

        rows = rows % vertex_count

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            ncols *= matrix.block_matrix_space.block_count

        base = Matrix.from_coo((rows), (cols), (values), nrows=nrows, ncols=ncols, dup_op=op)

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, block_count)
        base = matrix.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        return base

    @staticmethod
    def reduce_row(matrix: OptimizedMatrix, op: Monoid, vertex_count: int, block_count: int) -> OptimizedMatrix:
        assert isinstance(matrix, BlockMatrix)

        cell_h = matrix.block_matrix_space.cell_shape[0]
        cell_w = matrix.block_matrix_space.cell_shape[1]
        new_cell_shape = (vertex_count, vertex_count)
        is_cell = matrix.block_matrix_space.is_single_cell(matrix.shape)

        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        if not is_cell:
            orientation = matrix.block_matrix_space.get_block_matrix_orientation(matrix.shape)
            if orientation == BlockMatrixOrientation.HORIZONTAL:
                rows = rows + (cols // cell_w * cell_h)
                cols = cols % cell_w

        cols = cols % vertex_count

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            ncols *= matrix.block_matrix_space.block_count

        base = Matrix.from_coo((rows), (cols), (values), nrows=nrows, ncols=ncols, dup_op=op)

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, block_count)
        base = matrix.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        return base

    @property
    def base(self) -> OptimizedMatrix:
        return self._base

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        assert isinstance(other, PointsToMatrix)
        left, right = (self, other) if not swap_operands else (other, self)
        assert right.type == "State"
        if left.type == "RSM":
            # RSM [1x1] x State [1 x nums^depth] = State [1 x nums^depth]
            right._flat_matrix()
            base = right.base.optimize_similarly(self.base.mxm(other.base, op, swap_operands=swap_operands))
            return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
        elif left.type == "Context":
            left_shape = left._get_inner_shape()
            if left_shape[0] == 1:
                # open context [(_0, ..., (_nums]
                right_shape = right._get_inner_shape()
                if right_shape[0] == 1 and right_shape[1] == 1:
                    # [(_0, ..., (_nums] x [S] => [(_0, ..., (_nums] x [S, ..., S]^T = [S]
                    if not swap_operands:
                        if self.reduced is None:
                            accum_contexts = self.reduce_row(self.base, op.monoid, self.n, self.block_space.block_count)
                            assert isinstance(accum_contexts, BlockMatrix)
                            self.reduced = accum_contexts
                        base = BlockMatrixSpaceImpl((self.n, self.n), self.block_space.block_count).automize_block_operations(
                            self.reduced.mxm(other.base, op, swap_operands=swap_operands)
                        )
                        return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                    else:
                        if other.reduced is None:
                            accum_contexts = self.reduce_row(other.base, op.monoid, self.n, self.block_space.block_count)
                            assert isinstance(accum_contexts, BlockMatrix)
                            other.reduced = accum_contexts
                        base = BlockMatrixSpaceImpl((self.n, self.n), self.block_space.block_count).automize_block_operations(
                            self.base.mxm(other.reduced, op, swap_operands=swap_operands)
                        )
                        return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                else:
                    if not swap_operands:
                        # [(_0, ..., (_nums] x State [nums x nums^(depth-1)] = State [1 x nums^(depth-1)]
                        other._group_matrix()
                        shape = other._get_inner_shape()
                        base = BlockMatrixSpaceImpl((self.n, self.n * shape[1]), self.block_space.block_count).automize_block_operations(
                            self.base.mxm(other.base, op, swap_operands=swap_operands)
                        )
                        return PointsToMatrix(base, "State", self.n, self.context_num, self.depth - 1)
                    if swap_operands:
                        rotated = other._flat_matrix_rotate()
                        new_block_space = BlockMatrixSpaceImpl(
                            (self.context_num, self.block_space.cell_shape[1] // self.context_num), self.block_space.block_count
                        )
                        base = new_block_space.automize_block_operations(self.base.mxm(rotated, op, swap_operands))
                        base = PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                        base = base._reduce_diag_context_matrix(op)
                        return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)

            else:
                # closed context [)_0, ..., )_nums]^T
                right_shape = right._get_inner_shape()
                if right_shape[0] == 1 and right_shape[1] == 1:
                    # [)_0, ..., )_nums]^T x [S] = [)_0*S, ..., )_nums*S]^T
                    base = left.base.optimize_similarly(self.base.mxm(other.base, op, swap_operands=swap_operands))
                    return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                else:
                    # [)_0, ..., )_nums]^T x State [1 x nums^(depth+1)] = State [nums x nums^(depth+1)]
                    right._flat_matrix()
                    shape = right._get_inner_shape()
                    base = BlockMatrixSpaceImpl((self.n * self.context_num, self.n * shape[1]), self.block_space.block_count).automize_block_operations(
                        self.base.mxm(other.base, op, swap_operands=swap_operands)
                    )
                    return PointsToMatrix(base, "State", self.n, self.context_num, right.depth)
                    # return self.base.mxm(other.base, op, swap_operands=swap_operands)
        elif left.type == "State":
            # State [1 x nums^depth] x State [1 x nums^depth] = State [1 x nums^depth] (wise multiplication)
            assert left.depth == right.depth

            left._flat_matrix()
            right._flat_matrix()

            if not swap_operands:
                assert isinstance(other.base, BlockMatrix)
                diag = other.get_block_diag_matrix(other, self.n)
                assert isinstance(diag, BlockMatrix)
                base = self.base.optimize_similarly(self.base.mxm(diag, op, swap_operands=swap_operands))
                return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
            if swap_operands:
                assert isinstance(other.base, BlockMatrix)
                diag = self.get_block_diag_matrix(self, self.n)
                assert isinstance(diag, BlockMatrix)
                base = self.base.optimize_similarly(diag.mxm(other.base, op, swap_operands=swap_operands))
                return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                rotated = other._flat_matrix_rotate()
                new_block_space = BlockMatrixSpaceImpl((self.block_space.cell_shape[1], self.block_space.cell_shape[1]), self.block_space.block_count)
                base = new_block_space.automize_block_operations(self.base.mxm(rotated, op, swap_operands=swap_operands))
                base = PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                assert isinstance(base, PointsToMatrix)
                base = base._reduce_diag_matrix()
                return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)

            return self.base.mxm(other.base, op, swap_operands=swap_operands)
        else:
            raise ValueError("Unknown matrix type")

    def iadd(self, other: OptimizedMatrix, op: Monoid) -> None:
        assert isinstance(other, PointsToMatrix)
        assert self.type == other.type

        if self._is_flatted():
            other._flat_matrix()
        else:
            other._group_matrix()

        self.base.iadd(other.base, op)

    def rsub(self, other: OptimizedMatrix, op: Callable[[OptimizedMatrix, OptimizedMatrix], OptimizedMatrix]) -> OptimizedMatrix:
        assert isinstance(other, PointsToMatrix)
        assert self.type == "State" and other.type == "State"

        if self._is_flatted():
            other._flat_matrix()
        else:
            other._group_matrix()

        # TODO: it must return MatrixToOptimizedAdapter
        return self.optimize_similarly(self.base.rsub(other.base, op))

    def to_unoptimized(self) -> Matrix:
        return self.base.to_unoptimized()

    def optimize_similarly(self, other: OptimizedMatrix) -> OptimizedMatrix:
        return PointsToMatrix(self.base.optimize_similarly(other), self.type, self.n, self.context_num, self.depth)


if __name__ == "__main__":
    testMatrix1 = Matrix.from_coo([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [True, False, True, True, True, False], nrows=1, ncols=9)
    testMatrix1 = PointsToMatrix(MatrixToOptimizedAdapter(testMatrix1), "State", 1, 3, 2)
    print(testMatrix1)
    grouped = testMatrix1._group_matrix()
    print(grouped)
    flatted = testMatrix1._flat_matrix()
    print(flatted)
