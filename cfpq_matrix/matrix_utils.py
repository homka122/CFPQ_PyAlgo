from typing import Any

import graphblas
from graphblas.core.dtypes import DataType
from graphblas.core.matrix import Matrix
from graphblas.core.vector import Vector

from cfpq_matrix.optimized_matrix import OptimizedMatrix
from cfpq_matrix.matrix_to_optimized_adapter import MatrixToOptimizedAdapter


def complimentary_mask(matrix: OptimizedMatrix, mask: OptimizedMatrix) -> OptimizedMatrix:
    matrix_base = matrix.to_unoptimized()
    mask_base = mask.to_unoptimized()
    larger_matrix = matrix_base if matrix.nvals > mask.nvals else mask_base
    zero = Matrix(matrix.dtype, nrows=matrix_base.nrows, ncols=matrix_base.ncols)
    zero.ss.config["format"] = larger_matrix.ss.config["format"]
    res = Matrix(matrix.dtype, nrows=matrix_base.nrows, ncols=matrix_base.ncols)
    res.ss.config["format"] = larger_matrix.ss.config["format"]
    res(~mask_base.S) << zero.ewise_add(matrix_base, op=graphblas.monoid.any)
    return MatrixToOptimizedAdapter(res)


def identity_matrix(one: Any, dtype: DataType, size: int) -> Matrix:
    return Vector.from_scalar(value=one, size=size, dtype=dtype).diag()
