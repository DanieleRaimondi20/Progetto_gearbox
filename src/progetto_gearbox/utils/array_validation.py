from numpy.typing import NDArray


def check_column_vector(vector: NDArray) -> bool:
    return len(vector.shape) == 2 and vector.shape[1] == 1


def check_row_vector(vector: NDArray) -> bool:
    return len(vector.shape) == 2 and vector.shape[0] == 1


def check_matrix(matrix: NDArray) -> bool:
    return len(matrix.shape) == 2
