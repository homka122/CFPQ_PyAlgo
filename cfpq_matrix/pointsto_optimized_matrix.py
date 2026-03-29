from cfpq_matrix.block.block_matrix_space_impl import BlockMatrixSpaceImpl
from numpy import block
from cfpq_model.cnf_grammar_template import Symbol
from networkx.drawing.nx_agraph import from_agraph
import test
from packaging.utils import _
from typing import Literal, Callable
from abc import ABC

import graphblas
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

    # written by Homka122
    def _is_flatted(self) -> bool:
        if self.type == "RSM" or self.type == "Context":
            return True
        if self._get_inner_shape()[0] == 1:
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
            return 1
        elif not symbol.startswith("S"):
            return 0
        return int(symbol.split("G")[-1])

    # make from matrix with size 1 x nums^(depth) matrix with size nums x nums^(depth-1) if depth != 0
    def _flat_matrix(self) -> None:
        if self._is_flatted():
            return

        input_shape = self._get_inner_shape()
        assert input_shape[0] == self.context_num
        output_shape = (1, input_shape[1] * self.context_num)

        matrices = self.block_space.get_hyper_vector_blocks(self.base.to_unoptimized())
        new_matrices: list[Matrix] = []
        for matrix in matrices:
            (rows, cols, values) = matrix.to_coo()
            cols = cols % self.n + (cols // self.n * self.context_num * self.n) + (rows // self.n * self.n)
            rows = rows % self.n
            new_matrices.append(Matrix.from_coo(rows, cols, values, nrows=output_shape[0] * self.n, ncols=output_shape[1] * self.n))

        base = MatrixToOptimizedAdapter(self.block_space.stack_into_hyper_column(new_matrices))
        assert isinstance(self.base, BlockMatrix)
        new_block_space = BlockMatrixSpaceImpl((output_shape[0] * self.n, output_shape[1] * self.n), self.block_space.block_count)
        self._base = self.base.optimize_similarly_with_block(
            base,
            new_block_space,
        )
        self.block_space = new_block_space

    def _group_matrix(self) -> None:
        if not self._is_flatted():
            return

        input_shape = self._get_inner_shape()
        assert input_shape[0] == 1
        output_shape = (self.context_num, input_shape[1] // self.context_num)

        matrices = self.block_space.get_hyper_vector_blocks(self.base.to_unoptimized())
        new_matrices: list[Matrix] = []
        for matrix in matrices:
            (rows, cols, values) = matrix.to_coo()
            rows = rows + (cols // self.n % self.context_num * self.n)
            cols = cols % self.n + cols // (self.n * self.context_num) * self.n
            new_matrices.append(Matrix.from_coo(rows, cols, values, nrows=output_shape[0] * self.n, ncols=output_shape[1] * self.n))

        base = MatrixToOptimizedAdapter(self.block_space.stack_into_hyper_column(new_matrices))
        assert isinstance(self.base, BlockMatrix)
        new_block_space = BlockMatrixSpaceImpl((output_shape[0] * self.n, output_shape[1] * self.n), self.block_space.block_count)
        self._base = self.base.optimize_similarly_with_block(
            base, new_block_space
        )
        self.block_space = new_block_space

    @staticmethod
    def get_block_diag_matrix(matrix: OptimizedMatrix, graph_size: int) -> OptimizedMatrix:
        assert matrix.shape[0] == graph_size
        (rows, cols, values) = matrix.to_unoptimized().to_coo()
        rows = rows + (cols // graph_size * graph_size)

        base = Matrix.from_coo(rows, cols, values, nrows=matrix.shape[1], ncols=matrix.shape[1])
        return MatrixToOptimizedAdapter(base)

    @staticmethod
    def get_hyper_column(matrix: OptimizedMatrix, count: int) -> OptimizedMatrix:
        assert isinstance(matrix, PointsToMatrix)
        assert isinstance(matrix.base, BlockMatrix)
        base = matrix.to_unoptimized()
        matrices = [[base] for _ in range(count)]

        base_adapter = MatrixToOptimizedAdapter(graphblas.ss.concat(matrices))
        base = matrix.base.optimize_similarly_with_block(base_adapter, BlockMatrixSpaceImpl(base_adapter.shape, matrix.base.block_matrix_space.block_count))
        return base

    @staticmethod
    def get_hyper_row(matrix: OptimizedMatrix, count: int, vertex_count: int) -> OptimizedMatrix:
        base = matrix.to_unoptimized()
        matrices = [[base for _ in range(count)]]
        return BlockMatrixSpaceImpl((vertex_count, vertex_count * count), 1).automize_block_operations(MatrixToOptimizedAdapter(graphblas.ss.concat(matrices)))

    @staticmethod
    def reduce_column(matrix: OptimizedMatrix, op: Monoid, vertex_count: int) -> OptimizedMatrix:
        base = matrix.to_unoptimized()
        (rows, columns, values) = base.to_coo()

        rows = rows % matrix.shape[1]

        return BlockMatrixSpaceImpl((vertex_count, vertex_count), 1).automize_block_operations(
            MatrixToOptimizedAdapter(Matrix.from_coo(rows, columns, values, nrows=matrix.shape[1], ncols=matrix.shape[1], dup_op=op))
        )

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
            return self.base.mxm(other.base, op, swap_operands=swap_operands)
        elif left.type == "Context":
            left_shape = left._get_inner_shape()
            if left_shape[0] == 1:
                # open context [(_0, ..., (_nums]
                right_shape = right._get_inner_shape()
                if right_shape[0] == 1 and right_shape[1] == 1:
                    # [(_0, ..., (_nums] x [S] => [(_0, ..., (_nums] x [S, ..., S]^T = [S]
                    if not swap_operands:
                        column = self.get_hyper_column(other, self.context_num)
                        assert isinstance(column, BlockMatrix)
                        return self.base.mxm(column, op, swap_operands=swap_operands)
                    else:
                        column = self.get_hyper_column(self, self.context_num)
                        assert isinstance(column, BlockMatrix)
                        return column.mxm(other.base, op, swap_operands=swap_operands)
                else:
                    # [(_0, ..., (_nums] x State [nums x nums^(depth-1)] = State [1 x nums^(depth-1)]
                    right._group_matrix()
                    return self.base.mxm(other.base, op, swap_operands=swap_operands)
            else:
                # closed context [)_0, ..., )_nums]^T
                right_shape = right._get_inner_shape()
                if right_shape[0] == 1 and right_shape[1] == 1:
                    # [)_0, ..., )_nums]^T x [S] = [)_0*S, ..., )_nums*S]^T
                    return self.base.mxm(other.base, op, swap_operands=swap_operands)
                else:
                    # [)_0, ..., )_nums]^T x State [1 x nums^(depth+1)] = State [1 x nums^(depth+1)]
                    right._flat_matrix()
                    return self.base.mxm(other.base, op, swap_operands=swap_operands)
        elif left.type == "State":
            # State [1 x nums^depth] x State [1 x nums^depth] = State [1 x nums^depth] (wise multiplication)
            assert left.depth == right.depth

            left._flat_matrix()
            right._flat_matrix()

            if not swap_operands:
                assert isinstance(other.base, BlockMatrix)
                diag_adapter = self.get_block_diag_matrix(other, self.n)
                diag = other.base.optimize_similarly_with_block(
                    diag_adapter,
                    BlockMatrixSpaceImpl(diag_adapter.shape, self.block_space.block_count),
                )
                assert isinstance(diag, BlockMatrix)
                return self.base.mxm(diag, op, swap_operands=swap_operands)
            if swap_operands:
                assert isinstance(other.base, BlockMatrix)
                diag_adapter = self.get_block_diag_matrix(self, self.n)
                diag = other.base.optimize_similarly_with_block(
                    diag_adapter,
                    BlockMatrixSpaceImpl(diag_adapter.shape, self.block_space.block_count),
                )
                assert isinstance(diag, BlockMatrix)
                return diag.mxm(self.base, op, swap_operands=swap_operands)

            return self.base.mxm(other.base, op, swap_operands=swap_operands)
        else:
            raise ValueError("Unknown matrix type")

    def iadd(self, other: OptimizedMatrix, op: Monoid) -> None:
        assert isinstance(other, PointsToMatrix)
        assert self.type == other.type

        self._flat_matrix()
        other._flat_matrix()

        self.base.iadd(other.base, op)

    def rsub(self, other: OptimizedMatrix, op: Callable[[OptimizedMatrix, OptimizedMatrix], OptimizedMatrix]) -> OptimizedMatrix:
        assert isinstance(other, PointsToMatrix)
        assert self.type == "State" and other.type == "State"

        self._flat_matrix()
        other._flat_matrix()

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
