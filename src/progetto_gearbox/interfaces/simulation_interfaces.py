from logging import getLogger
from abc import ABC, abstractmethod
from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np
from progetto_gearbox.utils.array_validation import (
    check_column_vector,
    check_matrix,
)
from bokeh.plotting import figure
from bokeh.models import Renderer, ColumnDataSource, GlyphRenderer

logger = getLogger(__name__)

class Model(ABC):
    name: str
    state_names: list[str]
    state_uoms: list[str]
    input_names: list[str]
    input_uoms: list[str]
    output_names: list[str]
    output_uoms: list[str]
    ns: int
    no: int
    ni: int
    input_funcs: list[Callable | None]
    state_transition_matrix: NDArray
    input_matrix: NDArray
    output_matrix: NDArray
    feedthrough_matrix: NDArray
    non_linear_process: Callable | None
    non_linear_output: Callable | None
    initial_conditions: NDArray

    def __init__(self, name: str):
        logger.info("Initializing the model...")
        self.name = name
        self.state_names = None
        self.state_uoms = None
        self.input_names = None
        self.input_uoms = None
        self.output_names = None
        self.output_uoms = None
        self.ns = None
        self.no = None
        self.ni = None
        self.input_funcs = None
        self.state_transition_matrix = None
        self.input_matrix = None
        self.output_matrix = None
        self.feedthrough_matrix = None
        self.non_linear_process = None
        self.non_linear_output = None
        self.initial_conditions = None

    @abstractmethod
    def get_state_space():
        """Sets:
        - self.state_names
        - self.state_uoms
        - self.input_names
        - self.input_uoms
        - self.output_names
        - self.output_uoms
        - self.state_transition_matrix
        - self.input_matrix
        - self.output_matrix
        - self.feedthrough_matrix
        Eventually:
        - non_linear_process
        - non_linear_output
        
        Gets with self.get_matrices_dimensions()
            - self.ns
            - self.ni
            - self.no

        Checks with self.check_matrices_dimensions()
        """
        raise NotImplementedError

    @abstractmethod
    def plot(fig: figure, state_vector: NDArray) -> tuple[figure, ColumnDataSource, GlyphRenderer]:
        raise NotImplementedError

    @abstractmethod
    def update_plot(sources: ColumnDataSource, state_vector: NDArray):
        raise NotImplementedError

    def set_initial_conditions(self, init_conditions_dict: dict[str, float] = {}):
        logger.debug("Setting initial conditions for the model...")
        for state_idx, state_name in enumerate(self.state_names):
            if state_name in init_conditions_dict.keys():
                self.initial_conditions[state_idx] = init_conditions_dict[state_name]
            else:
                logger.warning("Initial condition for dof '%s' not set. Initialising to 0.",state_name)
                self.initial_conditions[state_idx] = 0

    def set_input_functions(self, input_func_dict: dict[str, Callable] = {}):
        logger.debug("Setting input functions for the model...")
        inputs_to_be_removed = []
        for input_idx, input_name in enumerate(self.input_names):
            if input_name in input_func_dict.keys():
                self.input_funcs[input_idx] = input_func_dict[input_name]
            else:
                logger.warning("Initial condition for dof '%s' not set. Removing the input.",input_name)
                inputs_to_be_removed.append(input_name)
        self.remove_unset_inputs(inputs_to_be_removed=inputs_to_be_removed)

    def remove_unset_inputs(self,inputs_to_be_removed: list[str]) -> None:
        for input_name in inputs_to_be_removed:
            input_idx = self.input_names.index(input_name)
            self.input_names.remove(input_name)
            self.input_funcs.pop(input_idx)
            self.input_matrix = np.delete(self.input_matrix, input_idx, axis=1)
            self.feedthrough_matrix = np.delete(self.feedthrough_matrix, input_idx, axis=1)

    def get_matrices_dimensions(self) -> None:
        self.ns = len(self.state_names)
        self.ni = len(self.input_names)
        self.no = len(self.output_names)

    def check_matrices_dimensions(self) -> None:
        logger.debug("Checking state space matrices dimensions...")
        assert self.state_transition_matrix.shape == (self.ns, self.ns)
        assert self.input_matrix.shape == (self.ns, self.ni)
        assert self.output_matrix.shape == (self.no, self.ns)
        assert self.feedthrough_matrix.shape == (self.no, self.ni)

    def get_inputs(self, time: float | NDArray) -> NDArray:
        inputs = np.array(
            [input_func(time) for input_func in self.input_funcs]
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
        assert check_matrix(inputs)
        outputs = self.output_matrix @ state_vector + self.feedthrough_matrix @ inputs
        if self.non_linear_output:
            outputs += self.non_linear_output(time, state_vector, inputs)

        return outputs
