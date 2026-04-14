from abc import ABC
from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np
from progetto_gearbox.utils.array_validation import (
    check_column_vector,
    check_row_vector,
    check_matrix,
)


class Model(ABC):

    state_names: list[str]
    state_uoms: list[str]
    input_names: list[str]
    input_uoms: list[str]
    output_names: list[str]
    output_uoms: list[str]
    ns: int
    no: int
    ni: int
    input_funcs: dict[str, Callable]
    state_transition_matrix: NDArray
    input_matrix: NDArray
    output_matrix: NDArray
    feedthrough_matrix: NDArray
    non_linear_process: Callable | None
    non_linear_output: Callable | None
    initial_conditions: NDArray

    def set_initial_conditions(self, init_conditions_dict: dict[float]):
        self.initial_conditions = np.array(
            [
                (
                    init_conditions_dict[state_name]
                    if (state_name in init_conditions_dict.keys())
                    else 0
                )
                for state_name in self.state_names
            ]
        )

    def init_linear_model(self) -> None:
        self.non_linear_process = None
        self.non_linear_output = None

    def get_matrices_dimensions(self) -> None:
        self.ns = len(self.state_names)
        self.ni = len(self.input_names)
        self.no = len(self.output_names)

    def check_matrices_dimensions(self) -> None:
        assert self.state_transition_matrix.shape == (self.ns, self.ns)
        assert self.input_matrix.shape == (self.ns, self.ni)
        assert self.output_matrix.shape == (self.no, self.ns)
        assert self.feedthrough_matrix.shape == (self.no, self.ni)

    def get_inputs(self, time: float | NDArray) -> NDArray:
        inputs = np.array(
            [self.input_funcs[input_name](time) for input_name in self.input_names]
        )
        return inputs

    def process_equation(self, time: float, state_vector: NDArray) -> NDArray:
        state_vector = state_vector[:, np.newaxis]
        assert check_column_vector(state_vector)
        input_vector = self.get_inputs(time)[:, np.newaxis]
        assert check_column_vector(input_vector)
        state_vector_derivative = (
            self.state_transition_matrix @ state_vector
            + self.input_matrix @ input_vector
        )
        if self.non_linear_process:
            state_vector_derivative += self.non_linear_process(
                time, state_vector, input_vector
            )

        return state_vector_derivative.flatten()

    def get_outputs(self, time: NDArray, state_vector: NDArray) -> NDArray:
        assert check_matrix(state_vector)
        inputs = self.get_inputs(time)
        assert check_row_vector(inputs)
        outputs = self.output_matrix @ state_vector + self.feedthrough_matrix @ inputs
        if self.non_linear_output:
            outputs += self.non_linear_output(time, state_vector, inputs)

        return outputs
