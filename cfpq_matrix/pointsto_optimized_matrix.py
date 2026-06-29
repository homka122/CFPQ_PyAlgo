from coverage.parser import Block
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
    def __init__(self, base: BlockMatrix, type: Literal["RSM", "Context", "State"], n: int, context_num: int, depth: int = -1):
        self._base: BlockMatrix = base
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
        op: Monoid | None = None,
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
        base = new_block_space.automize_block_operations(MatrixToOptimizedAdapter(base))

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

        base = self.base.optimize_similarly_with_block(base.base, base.block_matrix_space)
        assert(isinstance(base, BlockMatrix))

        self._base = base
        self.block_space = base.block_matrix_space

    def _flat_matrix_rotate(self) -> BlockMatrix:
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
        base = self.base.optimize_similarly_with_block(base.base, base.block_matrix_space)
        assert(isinstance(base, BlockMatrix))

        return base

    def _reduce_diag_matrix(self) -> BlockMatrix:
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

    def _reduce_diag_context_matrix(self, op: Monoid) -> BlockMatrix:
        assert isinstance(self.base, BlockMatrix)

        input_shape = self._get_inner_shape()
        assert input_shape[0] == self.context_num

        output_shape = (1, input_shape[1] // self.context_num)

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

    def get_block_diag_matrix(self) -> BlockMatrix:
        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        new_cell_shape = (cell_w, cell_w)
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not is_cell:
            orientation = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation == BlockMatrixOrientation.HORIZONTAL:
                rows = rows + (cols // cell_w * cell_h)
                cols = cols % cell_w

        rows = rows % cell_h + rows // cell_h * cell_w + cols // cell_h * cell_h

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            nrows *= self.block_space.block_count

        base = MatrixToOptimizedAdapter(Matrix.from_coo((rows), (cols), (values), nrows=nrows, ncols=ncols))

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = self.base.optimize_similarly_with_block(base, new_block_matrix)
        assert(isinstance(base, BlockMatrix))

        return base

    @staticmethod
    def get_hyper_column(matrix: OptimizedMatrix, count: int) -> OptimizedMatrix:
        assert isinstance(matrix, PointsToMatrix)
        assert isinstance(matrix.base, BlockMatrix)

        cell_h = matrix.block_space.cell_shape[0]
        cell_w = matrix.block_space.cell_shape[1]
        assert (cell_h == cell_w)
        new_cell_shape = (cell_h * count, cell_w)
        is_cell = matrix.block_space.is_single_cell(matrix.shape)

        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        if not is_cell:
            orientation = matrix.block_space.get_block_matrix_orientation(matrix.shape)
            if orientation == BlockMatrixOrientation.VERTICAL:
                cols = cols + (rows // cell_h * cell_w)
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

    def get_hyper_row(self, count: int) -> BlockMatrix:
        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        assert(cell_h == cell_w)
        new_cell_shape = (cell_h, cell_w * count)
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not self.block_space.is_single_cell(self.shape):
            orientation = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation == BlockMatrixOrientation.HORIZONTAL:
                rows = rows + (cols // cell_w * cell_h)
                cols = cols % cell_w

        rows = [rows] * count
        all_cols = []
        for i in range(count):
            all_cols.append(cols + cell_w * i)
        values = [values] * count

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            nrows *= self.block_space.block_count

        base = Matrix.from_coo(
            np.concatenate(rows),
            np.concatenate(all_cols),
            np.concatenate(values),
            nrows=nrows,
            ncols=ncols,
        )

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = new_block_matrix.automize_block_operations(MatrixToOptimizedAdapter(base))
        return base

    def reduce_column(self, op: Monoid) -> BlockMatrix:
        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        new_cell_shape = (self.n, self.n)
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not is_cell:
            orientation = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation == BlockMatrixOrientation.VERTICAL:
                cols = cols + (rows // cell_h * cell_w)
                rows = rows % cell_h

        rows = rows % self.n

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            ncols *= self.block_space.block_count

        base = Matrix.from_coo((rows), (cols), (values), nrows=nrows, ncols=ncols, dup_op=op)

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = new_block_matrix.automize_block_operations(MatrixToOptimizedAdapter(base))
        return base

    def reduce_row(self, op: Monoid) -> BlockMatrix:
        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        new_cell_shape = (self.n, self.n)
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not is_cell:
            orientation = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation == BlockMatrixOrientation.HORIZONTAL:
                rows = rows + (cols // cell_w * cell_h)
                cols = cols % cell_w

        cols = cols % self.n

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            nrows *= self.block_space.block_count

        base = Matrix.from_coo((rows), (cols), (values), nrows=nrows, ncols=ncols, dup_op=op)

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = self.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        assert isinstance(base, BlockMatrix)
        return base

    @property
    def base(self) -> BlockMatrix:
        return self._base

    # [1x1] matrix to [nums x nums] diag matrix
    def _to_context_diag_matrix(self) -> BlockMatrix:
        assert isinstance(self.base, BlockMatrix)

        cell_h = self.block_space.cell_shape[0]
        cell_w = self.block_space.cell_shape[1]
        assert (cell_h == self.n and cell_w == self.n)
        new_cell_shape = (self.context_num * self.n, self.context_num * self.n)
        is_cell = self.block_space.is_single_cell(self.shape)

        (rows, cols, values) = self.to_unoptimized().to_coo()
        if not is_cell:
            orientation = self.block_space.get_block_matrix_orientation(self.shape)
            if orientation == BlockMatrixOrientation.VERTICAL:
                cols = cols + (rows // cell_h * cell_w)
                rows = rows % cell_h

        all_rows = []
        all_cols = []
        for i in range(self.context_num):
            all_rows.append(rows + cell_h * i)
            all_cols.append(cols + cell_w * i)
        values = [values] * self.context_num

        nrows, ncols = new_cell_shape[0], new_cell_shape[1]
        if not is_cell:
            ncols *= self.block_space.block_count

        base = Matrix.from_coo(
            np.concatenate(all_rows),
            np.concatenate(all_cols),
            np.concatenate(values),
            nrows=nrows,
            ncols=ncols,
        )

        new_block_matrix = BlockMatrixSpaceImpl(new_cell_shape, self.block_space.block_count)
        base = self.base.optimize_similarly_with_block(MatrixToOptimizedAdapter(base), new_block_matrix)
        assert isinstance(base, BlockMatrix)
        return base

    def _mxm_rsm(self, other: "PointsToMatrix", op: Semiring, swap_operands: bool = False) -> BlockMatrix:
        # RSM [1x1] x State [1 x nums^depth] = State [1 x nums^depth]
        left, right = (self, other) if not swap_operands else (other, self)
        left_block, right_block = (left.base, right.base)

        if left.nvals <= right.nvals:
            if right._is_grouped():
                # diag RSM [nums x nums] x State [nums x nums^depth-1] = State [nums x nums^depth-1]
                left_block = left._to_context_diag_matrix()
        else:
            right._flat_matrix()
            right_block = right.base

        if swap_operands:
            base = right_block.mxm(left_block, op, swap_operands)
        else:
            base = left_block.mxm(right_block, op, swap_operands)

        return BlockMatrixSpaceImpl(right_block.block_matrix_space.cell_shape, right_block.block_matrix_space.block_count).automize_block_operations(base)

    def _mxm_state(self, other: "PointsToMatrix", op: Semiring, swap_operands: bool = False) -> BlockMatrix:
        # State [1 x nums^depth] x State [1 x nums^depth] = State [1 x nums^depth] (wise multiplication)
        left, right = (self, other) if not swap_operands else (other, self)
        left._flat_matrix()
        right._flat_matrix()

        left_block, right_block = (left.base, right.base)

        # TODO calculate diag nvals so we can decide what cost less
        if left.nvals <= right.nvals:
            left_block = left._flat_matrix_rotate()
            if swap_operands:
                base = right_block.mxm(left_block, op, swap_operands)
            else:
                base = left_block.mxm(right_block, op, swap_operands)
            new_block_space = BlockMatrixSpaceImpl((right.block_space.cell_shape[1], right.block_space.cell_shape[1]), right.block_space.block_count)
            base = new_block_space.automize_block_operations(base)
            base = PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
            base = base._reduce_diag_matrix().base
        else:
            right_block = right.get_block_diag_matrix()
            if swap_operands:
                base = right_block.mxm(left_block, op, swap_operands)
            else:
                base = left_block.mxm(right_block, op, swap_operands)

        return BlockMatrixSpaceImpl(right.block_space.cell_shape, right.block_space.block_count).automize_block_operations(base)

    def _mxm_open_context_single(self, other: "PointsToMatrix", op: Semiring, swap_operands: bool = False) -> BlockMatrix:
        # [(_0, ..., (_nums] x [S] => [(_0, ..., (_nums] x [S, ..., S]^T = [S]
        assert isinstance(other, PointsToMatrix)
        left, right = (self, other) if not swap_operands else (other, self)
        left_block, right_block = (left.reduced, right.base)

        if left_block is None:
            left.reduced = left.reduce_row(op.monoid)
            left_block = left.reduced

        left_shape = left_block.shape[0] // self.n, left_block.shape[1] // self.n
        right_shape = right._get_inner_shape()
        assert(left_shape == (1, 1) and right_shape == (1, 1))

        if swap_operands:
            base = right_block.mxm(left_block, op, swap_operands)
        else:
            base = left_block.mxm(right_block, op, swap_operands)

        base = BlockMatrixSpaceImpl((right.n, right.n), right.block_space.block_count).automize_block_operations(
            base
        )

        return base

    def _mxm_open_context_multiple(self, other: "PointsToMatrix", op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        # [(_0, ..., (_nums] x State [nums x nums^(depth-1)] = State [1 x nums^(depth-1)]
        left, right = (self, other) if not swap_operands else (other, self)

        if left.nvals <= right.nvals:
            if right._is_grouped():
                shape = right._get_inner_shape()
                base = BlockMatrixSpaceImpl((self.n, self.n * shape[1]), self.block_space.block_count).automize_block_operations(
                    left.base.mxm(right.base, op)
                )
            else:
                rotated = left._flat_matrix_rotate()
                new_block_space = BlockMatrixSpaceImpl((left.block_space.cell_shape[1], right.block_space.cell_shape[1]), right.block_space.block_count)
                base = new_block_space.automize_block_operations(rotated.mxm(right.base, op))
                base = PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                base = base._reduce_diag_context_matrix(op.monoid)
        else:
            right._group_matrix()
            shape = right._get_inner_shape()
            base = BlockMatrixSpaceImpl((self.n, self.n * shape[1]), self.block_space.block_count).automize_block_operations(
                left.base.mxm(right.base, op)
            )

        return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)

    def _mxm_closed_context(self, other: "PointsToMatrix", op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        # closed context [)_0, ..., )_nums]^T
        left, right = (self, other) if not swap_operands else (other, self)

        right_shape = right._get_inner_shape()
        if right_shape[0] == 1 and right_shape[1] == 1:
            # [)_0, ..., )_nums]^T x [S] = [)_0*S, ..., )_nums*S]^T
            base = left.base.optimize_similarly(left.base.mxm(right.base, op))
            assert(isinstance(base, BlockMatrix))
        else:
            # [)_0, ..., )_nums]^T x State [1 x nums^(depth+1)] = State [nums x nums^(depth+1)]
            right._flat_matrix()
            shape = right._get_inner_shape()
            base = BlockMatrixSpaceImpl((self.n * self.context_num, self.n * shape[1]), self.block_space.block_count).automize_block_operations(
                left.base.mxm(right.base, op)
            )
            # return self.base.mxm(other.base, op, swap_operands=swap_operands)

        return PointsToMatrix(base, "State", self.n, self.context_num, right.depth)

    def mxm(self, other: OptimizedMatrix, op: Semiring, swap_operands: bool = False) -> OptimizedMatrix:
        assert isinstance(other, PointsToMatrix)
        left, right = (self, other) if not swap_operands else (other, self)
        assert right.type == "State"
        if left.type == "RSM":
            base = self._mxm_rsm(other, op, swap_operands)
            return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
        elif left.type == "Context":
            left_shape = left._get_inner_shape()
            if left_shape[0] == 1:
                # open context [(_0, ..., (_nums]
                right_shape = right._get_inner_shape()
                if right_shape[0] == 1 and right_shape[1] == 1:
                    # [(_0, ..., (_nums] x [S] => [(_0, ..., (_nums] x [S, ..., S]^T = [S]
                    base = self._mxm_open_context_single(other, op, swap_operands)
                    return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
                else:
                    # [(_0, ..., (_nums] x State [nums x nums^(depth-1)] = State [1 x nums^(depth-1)]
                    return self._mxm_open_context_multiple(other, op, swap_operands)
            else:
                return self._mxm_closed_context(other, op, swap_operands=swap_operands)
        elif left.type == "State":
            # State [1 x nums^depth] x State [1 x nums^depth] = State [1 x nums^depth] (wise multiplication)
            base = self._mxm_state(other, op, swap_operands)
            return PointsToMatrix(base, "State", self.n, self.context_num, self.depth)
        else:
            raise ValueError("Unknown matrix type")

    def iadd(self, other: OptimizedMatrix, op: Monoid) -> None:
        assert isinstance(other, PointsToMatrix)
        assert isinstance(other.base, BlockMatrix)
        assert self.type == other.type

        self._flat_matrix()

        self_shape = self._get_inner_shape()
        other_shape = other._get_inner_shape()

        if self_shape != (1, 1) and other_shape == (1, 1):
            base = other.get_hyper_row(self.context_num**self.depth)
        elif self_shape == (1, 1) and other_shape != (1, 1):
            base = other.reduce_column(op)
        elif other._is_grouped() and other_shape[1] * self.context_num == self_shape[1]:
            other._flat_matrix()
            base = other.base
        else:
            base = other.base

        self.base.iadd(base, op)

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
        base = self.base.optimize_similarly(other)
        assert(isinstance(base, BlockMatrix))
        return PointsToMatrix(base, self.type, self.n, self.context_num, self.depth)
