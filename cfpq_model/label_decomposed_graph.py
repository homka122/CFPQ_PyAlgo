from cfpq_matrix.block.block_matrix_space import BlockMatrixSpace
from pandas.core.internals.blocks import new_block
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, Callable

import graphblas
import graphblas.core.matrix
import numpy as np
import pandas as pd
import graphblas
from graphblas.core.dtypes import DataType, BOOL
from graphblas.core.matrix import Matrix
from graphblas.core.operator import Monoid, Semiring
from graphblas.exceptions import IndexOutOfBound

from cfpq_matrix.block.block_matrix_space_impl import BlockMatrixSpaceImpl
from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter
from cfpq_matrix.pointsto_optimized_matrix import PointsToMatrix
from cfpq_matrix.block.block_matrix import BlockMatrix


# from cfpq_matrix.subtractable_semiring import SubOp
from cfpq_model.cnf_grammar_template import CnfGrammarTemplate, Symbol


class LabelDecomposedGraph:
    """
    Representation of an edge labeled graph where labels are of type `(Symbol, Optional[int])`,
    i.e. each label is represented by a combination of "label symbol" and an optional "label index".

    For each label string an adjacency matrix is stored.

    If "label symbol" is used without indices, then its adjacency
    matrix is a square matrix of shape `(vertex_count, vertex_count)`.

    If "label symbol" is used with indices, then its adjacency matrix is a
    block-matrix of shape `(block_matrix_space.block_count * vertex_count, vertex_count)`
    or `(vertex_count, block_matrix_space.block_count * vertex_count)`, where
    `block_matrix_space.block_count` is the largest "label index" in the entire graph.
    """

    def __init__(self, vertex_count: int, block_matrix_space: BlockMatrixSpace, dtype: DataType, matrices: Dict[Symbol, Matrix], contexts_num: int, depth: int):
        self.vertex_count: int = vertex_count
        self.block_matrix_space: BlockMatrixSpace = block_matrix_space
        self.dtype: DataType = dtype
        self.matrices: dict[Symbol, Matrix] = matrices
        self.contexts_num: int = contexts_num
        self.depth: int = depth

    @property
    def nvals(self) -> int:
        return sum(m.nvals for m in self.matrices.values())

    def group_contexts(self) -> None:
        tiles_open: list[list[Matrix]] = [[]]
        tiles_close: list[list[Matrix]] = []
        for i in range(self.contexts_num):
            for key, matrix in self.matrices.items():
                if key.label == f"({i}":
                    tiles_open[0].append(matrix)
                elif key.label == f"){i}":
                    tiles_close.append([matrix])
            if len(tiles_open[0]) != i + 1:
                tiles_open[0].append(Matrix(self.dtype, self.vertex_count, self.vertex_count))
            if len(tiles_close) != i + 1:
                tiles_close.append([Matrix(self.dtype, self.vertex_count, self.vertex_count)])

        self.matrices[Symbol("(i")] = graphblas.ss.concat(tiles_open)
        self.matrices[Symbol(")i")] = graphblas.ss.concat(tiles_close)
        for i in range(self.contexts_num):
            if Symbol(f"({i}") in self.matrices:
                del self.matrices[Symbol(f"({i}")]
            if Symbol(f"){i}") in self.matrices:
                del self.matrices[Symbol(f"){i}")]

    @staticmethod
    def read_from_pocr_graph_file(path: Union[Path, str], contexts_num: int, depth: int) -> "LabelDecomposedGraph":
        try:
            dfs = pd.read_csv(
                path,
                sep="\s+",
                header=None,
                names=["EDGE_SOURCE", "EDGE_DESTINATION", "EDGE_LABEL", "LABEL_INDEX"],
                dtype={"EDGE_SOURCE": np.int64, "EDGE_DESTINATION": np.int64, "EDGE_LABEL": str, "LABEL_INDEX": pd.Int64Dtype()},
                chunksize=1_000_000,
            )

            # edge_label -> (edge_source_chunks, edge_destination_chunks, label_index_chunks)
            data_chunks = defaultdict(lambda: ([], [], []))

            for df in dfs:
                df.fillna({"LABEL_INDEX": 0}, inplace=True)

                for label, group in df.groupby("EDGE_LABEL"):
                    symbol = Symbol(label)
                    (edge_sources, edge_destinations, label_indices) = data_chunks[symbol]
                    edge_sources.append(group["EDGE_SOURCE"].to_numpy(dtype=np.int64))
                    edge_destinations.append(group["EDGE_DESTINATION"].to_numpy(dtype=np.int64))
                    label_indices.append(group["LABEL_INDEX"].to_numpy(dtype=np.int64))

            vertex_count = 1
            block_count = 1

            # edge_label -> (edge_sources, edge_destinations, label_indices)
            data = {}

            for symbol, (edge_sources_chunks, edge_destinations_chunks, label_indices_chunks) in data_chunks.items():
                edge_sources = np.concatenate(edge_sources_chunks)
                edge_destinations = np.concatenate(edge_destinations_chunks)
                label_indices = np.concatenate(label_indices_chunks)

                data[symbol] = (edge_sources, edge_destinations, label_indices)

                vertex_count = max(vertex_count, int(edge_sources.max()) + 1, int(edge_destinations.max()) + 1)
                block_count = max(block_count, int(label_indices.max()) + 1)

            matrices: Dict[Symbol, Matrix] = {}
            for symbol, (edge_sources, edge_destinations, label_indices) in data.items():
                edge_sources += label_indices * vertex_count

                try:
                    matrices[symbol] = Matrix.from_coo(
                        rows=edge_sources,
                        columns=edge_destinations,
                        values=True,
                        nrows=block_count * vertex_count if symbol.is_indexed else vertex_count,
                        ncols=vertex_count,
                    )
                except IndexOutOfBound as e:
                    raise ValueError(
                        f"Failed to create adjacency matrix for label `{symbol.label}`.\n"
                        f"This issue is usually caused by using indexes for label without `_i` suffix.\n"
                        f"Consider adding `_i` suffix to label '{symbol.label}'."
                    ) from e

            return LabelDecomposedGraph(
                vertex_count=vertex_count,
                block_matrix_space=BlockMatrixSpaceImpl(cell_shape=(vertex_count, vertex_count), block_count=block_count),
                dtype=BOOL,
                matrices=matrices,
                contexts_num=contexts_num,
                depth=depth,
            )
        except Exception as e:
            raise ValueError(
                f"Invalid graph file '{path}'. All lines of graph file should have form:\n"
                "```\n"
                "<EDGE_SOURCE>\t<EDGE_DESTINATION>\t<EDGE_LABEL>\t[LABEL_INDEX]\n"
                "```\n"
                "Whitespace characters should be used to separate values on one line.\n"
                "[LABEL_INDEX] is optional.\n"
                "Indexed labels names must end with `_i`."
            ) from e

    def write_to_pocr_graph_file(self, path: Union[Path, str]):
        with open(path, "w", encoding="utf-8") as output_file:
            for symbol, matrix in self.matrices.items():
                edge_label = symbol.label
                (rows, columns, _) = matrix.to_coo()
                if matrix.shape[0] == self.vertex_count:
                    edges_df = pd.DataFrame({"source": rows, "destination": columns, "label": edge_label})
                else:
                    edges_df = pd.DataFrame(
                        {"source": rows % self.vertex_count, "destination": columns, "label": edge_label, "label_index": rows // self.vertex_count}
                    )
                csv_string = edges_df.to_csv(sep="\t", index=False, header=False)
                output_file.write(csv_string)

    def __sizeof__(self) -> int:
        return sum(m.__sizeof__() for m in self.matrices.values())

    def __getitem__(self, symbol: Symbol):
        return self.matrices[symbol] if symbol in self.matrices else self.block_matrix_space.create_space_element(self.dtype, is_vector=symbol.is_indexed)


class OptimizedLabelDecomposedGraph:
    """
    Representation of an edge labeled graph similar to `LabelDecomposedGraph`,
    but with `OptimizedMatrix` instead of regular `Matrix`.
    """

    def __init__(
        self,
        vertex_count: int,
        block_matrix_space: BlockMatrixSpace,
        dtype: DataType,
        matrix_optimizer: Callable[[Matrix], OptimizedMatrix],
        contexts_num: int,
        depth: int,
    ):
        self.vertex_count: int = vertex_count
        self.block_matrix_space: BlockMatrixSpace = block_matrix_space
        self.dtype: DataType = dtype
        self.matrix_optimizer: Callable[[Matrix], OptimizedMatrix] = matrix_optimizer
        self.matrices: Dict[Symbol, OptimizedMatrix] = {}
        self.contexts_num: int = contexts_num
        self.depth: int = depth

    @staticmethod
    def from_unoptimized(unoptimized_graph: LabelDecomposedGraph, matrix_optimizer: Callable[[Matrix], OptimizedMatrix]) -> "OptimizedLabelDecomposedGraph":
        optimized_graph = OptimizedLabelDecomposedGraph(
            vertex_count=unoptimized_graph.vertex_count,
            block_matrix_space=unoptimized_graph.block_matrix_space,
            dtype=unoptimized_graph.dtype,
            matrix_optimizer=matrix_optimizer,
            contexts_num=unoptimized_graph.contexts_num,
            depth=unoptimized_graph.depth,
        )

        for symbol, matrix in unoptimized_graph.matrices.items():
            # optimized_graph.matrices[symbol] = PointsToMatrix(
            #     BlockMatrixSpaceImpl(unoptimized_graph.vertex_count, 1).automize_block_operations(MatrixToOptimizedAdapter(matrix)),
            #     PointsToMatrix.get_type_from_symbol(symbol.label),
            #     unoptimized_graph.vertex_count,
            #     unoptimized_graph.contexts_num,
            #     depth=PointsToMatrix.get_depth_from_symbol(symbol.label),
            # )
            type = PointsToMatrix.get_type_from_symbol(symbol.label)
            depth_local = PointsToMatrix.get_depth_from_symbol(symbol.label)
            # if type == "State":
            #     depth_local = PointsToMatrix.get_depth_from_symbol(symbol.label)
            # else:
            #     depth_local = 1
            n = unoptimized_graph.vertex_count
            new_block_space_size = (n, n * (unoptimized_graph.contexts_num**depth_local))
            if symbol == Symbol(")i"):
                new_block_space_size = (n * (unoptimized_graph.contexts_num**depth_local), n)
            new_block_space = BlockMatrixSpaceImpl(new_block_space_size, unoptimized_graph.block_matrix_space.block_count)
            optimized_graph.matrices[symbol] = PointsToMatrix(
                base=new_block_space.automize_block_operations(MatrixToOptimizedAdapter(matrix)),
                type=type,
                n=n,
                context_num=unoptimized_graph.contexts_num,
                depth=PointsToMatrix.get_depth_from_symbol(symbol.label),
            )
            # optimized_graph.iadd_by_symbol(symbol, MatrixToOptimizedAdapter(matrix), op=graphblas.monoid.any)
        # optimized_graph.iadd(unoptimized_graph, op=graphblas.monoid.any)

        return optimized_graph

    def empty_copy(self) -> "OptimizedLabelDecomposedGraph":
        return OptimizedLabelDecomposedGraph(
            vertex_count=self.vertex_count,
            block_matrix_space=self.block_matrix_space,
            matrix_optimizer=self.matrix_optimizer,
            dtype=self.dtype,
            contexts_num=self.contexts_num,
            depth=self.depth,
        )

    def to_unoptimized(self) -> LabelDecomposedGraph:
        return LabelDecomposedGraph(
            vertex_count=self.vertex_count,
            block_matrix_space=self.block_matrix_space,
            matrices={symbol: matrix.to_unoptimized() for symbol, matrix in self.matrices.items()},
            dtype=self.dtype,
            contexts_num=self.contexts_num,
            depth=self.depth,
        )

    @property
    def nvals(self) -> int:
        return sum(matrix.nvals for matrix in self.matrices.values())

    def iadd_by_symbol(self, symbol: Symbol, matrix: OptimizedMatrix, op: Monoid) -> None:
        # if symbol not in self:
        #     type = PointsToMatrix.get_type_from_symbol(symbol.label)
        #     depth_local = PointsToMatrix.get_depth_from_symbol(symbol.label)

        #     nrows = self.vertex_count
        #     if depth_local == self.depth + 1:
        #         ncols = self.vertex_count
        #     else:
        #         ncols = self.vertex_count * (self.contexts_num**depth_local)
        #     base = Matrix(self.dtype, nrows, ncols, name=symbol.label)
        #     self.matrices[symbol] = PointsToMatrix(
        #         MatrixToOptimizedAdapter(base),
        #         type,
        #         self.vertex_count,
        #         self.contexts_num,
        #         depth_local,
        #     )
        self.matrices[symbol] = self[symbol]
        self[symbol].iadd((matrix), op)

    def iadd(self, other: "OptimizedLabelDecomposedGraph", op: Monoid) -> "OptimizedLabelDecomposedGraph":
        for symbol, matrix in other.matrices.items():
            self.iadd_by_symbol(symbol, matrix, op)
        return self

    def rsub(
        self, other: "OptimizedLabelDecomposedGraph", op: Callable[[OptimizedMatrix, OptimizedMatrix], OptimizedMatrix]
    ) -> "OptimizedLabelDecomposedGraph":
        result = OptimizedLabelDecomposedGraph(
            vertex_count=self.vertex_count,
            block_matrix_space=self.block_matrix_space,
            matrix_optimizer=self.matrix_optimizer,
            dtype=self.dtype,
            contexts_num=self.contexts_num,
            depth=self.depth,
        )

        result.matrices = {symbol: (self.matrices[symbol].rsub(matrix, op) if symbol in self else matrix) for symbol, matrix in other.matrices.items()}

        return result

    def mxm(
        self,
        other: "OptimizedLabelDecomposedGraph",
        grammar: CnfGrammarTemplate,
        op: Semiring,
        accum: Optional["OptimizedLabelDecomposedGraph"] = None,
        swap_operands: bool = False,
    ) -> "OptimizedLabelDecomposedGraph":
        if accum is None:
            accum = self.empty_copy()
        for lhs, rhs1, rhs2 in grammar.complex_rules:
            leftrhs = rhs1
            rightrhs = rhs2
            if swap_operands:
                rhs1, rhs2 = rhs2, rhs1
            if rhs1 in self.matrices and rhs2 in other.matrices:
                left = self.matrices[leftrhs]
                right = other.matrices[rightrhs]

                mxm = self.matrices[rhs1].mxm(
                    other.matrices[rhs2],
                    swap_operands=swap_operands,
                    op=op,
                )

                if (
                    leftrhs.label.startswith(")")
                    and (PointsToMatrix.get_depth_from_symbol(lhs.label) == 0 or PointsToMatrix.get_depth_from_symbol(lhs.label) == self.depth + 1)
                ):
                    assert isinstance(left, PointsToMatrix) and isinstance(right, PointsToMatrix)
                    if rightrhs.label == "S_10_G0_i":
                        pass
                    new_cell_shape = (left.block_space.cell_shape[0], right.block_space.cell_shape[1])
                    base = PointsToMatrix.reduce_column(
                        BlockMatrixSpaceImpl(new_cell_shape, self.block_matrix_space.block_count).automize_block_operations(mxm),
                        op.monoid,
                        self.vertex_count,
                        self.block_matrix_space.block_count,
                    )
                    mxm = PointsToMatrix(
                        base,
                        "State",
                        self.vertex_count,
                        self.contexts_num,
                        PointsToMatrix.get_depth_from_symbol(lhs.label),
                    )
                elif leftrhs.label.startswith("(") and PointsToMatrix.get_depth_from_symbol(lhs.label) == self.depth:
                    mxm = PointsToMatrix(
                        (
                            PointsToMatrix.get_hyper_row(
                                mxm, self.contexts_num**self.depth, vertex_count=self.vertex_count, block_count=self.block_matrix_space.block_count
                            )
                        ),
                        "State",
                        self.vertex_count,
                        self.contexts_num,
                        PointsToMatrix.get_depth_from_symbol(lhs.label),
                    )
                else:
                    assert isinstance(left, PointsToMatrix) and isinstance(right, PointsToMatrix)
                    new_cell_shape = (left.block_space.cell_shape[0], right.block_space.cell_shape[1])
                    mxm = PointsToMatrix(
                        BlockMatrixSpaceImpl(new_cell_shape, self.block_matrix_space.block_count).automize_block_operations(mxm),
                        "State",
                        self.vertex_count,
                        self.contexts_num,
                        PointsToMatrix.get_depth_from_symbol(lhs.label),
                    )

                accum.iadd_by_symbol(lhs, mxm, op.monoid)
        return accum

    def rmxm(
        self, other: "OptimizedLabelDecomposedGraph", grammar: CnfGrammarTemplate, op: Semiring, accum: Optional["OptimizedLabelDecomposedGraph"] = None
    ) -> "OptimizedLabelDecomposedGraph":
        return self.mxm(other, grammar, op, accum, swap_operands=True)

    def __getitem__(self, symbol: Symbol) -> OptimizedMatrix:
        if symbol in self:
            return self.matrices[symbol]

        type = PointsToMatrix.get_type_from_symbol(symbol.label)
        depth_local = PointsToMatrix.get_depth_from_symbol(symbol.label)

        if type == "State":
            nrows = self.vertex_count
            if depth_local == self.depth + 1:
                ncols = self.vertex_count
            else:
                ncols = self.vertex_count * (self.contexts_num**depth_local)
        elif type == "RSM":
            nrows = self.vertex_count
            ncols = self.vertex_count
        else:
            if symbol.label == "(i":
                nrows = self.vertex_count
                ncols = self.vertex_count * (self.contexts_num)
            else:
                nrows = self.vertex_count * (self.contexts_num)
                ncols = self.vertex_count

        nrows_base = nrows
        if symbol.is_indexed:
            nrows_base *= self.block_matrix_space.block_count

        base = MatrixToOptimizedAdapter(Matrix(self.dtype, nrows_base, ncols, name=symbol.label))
        block_space = BlockMatrixSpaceImpl((nrows, ncols), self.block_matrix_space.block_count)
        base = block_space.automize_block_operations(base)

        return PointsToMatrix(base, type, self.vertex_count, self.contexts_num, depth_local)

    def _create_matrix_for_symbol(self, symbol) -> Matrix:
        return self.block_matrix_space.create_space_element(self.dtype, is_vector=symbol.is_indexed)

    def __contains__(self, symbol: Symbol) -> bool:
        return symbol in self.matrices

    def __sizeof__(self) -> int:
        return sum(m.__sizeof__() for m in self.matrices.values())
