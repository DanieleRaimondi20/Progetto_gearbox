from logging import getLogger
from abc import ABC, abstractmethod
from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np
from progetto_gearbox.utils.array_validation import (
    check_column_vector,
    check_matrix,
    check_row_vector,
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
        logger.info("Initializing model '%s'...", name)
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
        self._initial_conditions_set = False
        logger.info("Model '%s' initialized.", name)

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
    def _update_plot(sources: ColumnDataSource, state_vector: NDArray):
        raise NotImplementedError

    def set_initial_conditions(self, init_conditions_dict: dict[str, float] = {}):
        logger.debug("Setting initial conditions for model '%s'...", self.name)
        logger.debug("Checking initial conditions for model '%s'...", self.name)
        for init_state in init_conditions_dict.keys():
            assert init_state in self.state_names, f"Initial condition for state '{init_state}' was provided, but state '{init_state}' was not found in model '{self.name}'."
        logger.debug("Initial conditions for model '%s' checked.", self.name)

        for state_idx, state_name in enumerate(self.state_names):
            logger.debug("Setting initial condition for dof '%s'...",state_name)
            if state_name in init_conditions_dict.keys():
                logger.debug("Initial condition for dof '%s' set...",state_name)
                self.initial_conditions[state_idx] = init_conditions_dict[state_name]
            else:
                logger.warning("Initial condition for dof '%s' was not assigned in model '%s'. Keeping a 'None'.", state_name, self.name)
                self.initial_conditions[state_idx] = None
        self._initial_conditions_set = True
        logger.debug("Initial conditions for model '%s' set.", self.name)

    def set_input_functions(self, input_func_dict: dict[str, Callable] = {}):
        logger.debug("Setting input functions for model '%s'...", self.name)
        logger.debug("Checking input functions for model '%s'...", self.name)
        for input in input_func_dict.keys():
            assert input in self.input_names, f"Input function for state '{input}' was provided, but input '{input}' was not found in model '{self.name}'."
        logger.debug("Input functions for model '%s' checked.", self.name)

        inputs_to_be_removed = []
        for input_idx, input_name in enumerate(self.input_names):
            logger.debug("Setting input '%s'...", input_name)
            if input_name in input_func_dict.keys():
                logger.debug("Input '%s' set.", input_name)
                self.input_funcs[input_idx] = input_func_dict[input_name]
            else:
                logger.warning("Input function for dof '%s' was not assigned. Input will be removed from model '%s'.", input_name, self.name)
                inputs_to_be_removed.append(input_name)
                
        self.remove_unset_inputs(inputs_to_be_removed=inputs_to_be_removed)
        logger.debug("Input functions for model '%s' set.", self.name)

    def remove_unset_inputs(self,inputs_to_be_removed: list[str]) -> None:
        logger.debug("Removing unset inputs from model '%s'...", self.name)
        for input_name in inputs_to_be_removed:
            self.remove_input(input_name=input_name)
        logger.debug("Unset inputs from model '%s' removed.", self.name)

    def get_matrices_dimensions(self) -> None:
        logger.debug("Getting dimensions of state space matrices for model '%s'...", self.name)
        self.ns = len(self.state_names)
        self.ni = len(self.input_names)
        self.no = len(self.output_names)
        logger.debug("Dimensions of state space matrices for model '%s' obtained.", self.name)

    def check_matrices_dimensions(self) -> None:
        logger.debug("Checking state space matrices dimensions...")
        assert self.state_transition_matrix.shape == (self.ns, self.ns)
        assert self.input_matrix.shape == (self.ns, self.ni)
        assert self.output_matrix.shape == (self.no, self.ns)
        assert self.feedthrough_matrix.shape == (self.no, self.ni)
        logger.debug("State space matrices dimensions checked.")

    def get_inputs(self, time: float | NDArray) -> NDArray:
        inputs = np.array(
            [input_func(time) for input_func in self.input_funcs]
        )
        return inputs

    def process_equation(self, time: float, state_vector: NDArray) -> NDArray:
        print(f"{time:.02f} s")
        state_vector = state_vector[:, np.newaxis]
        # assert check_column_vector(state_vector)
        input_vector = self.get_inputs(time)[:, np.newaxis]
        # assert check_column_vector(input_vector)
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
        # assert check_matrix(state_vector)
        inputs = self.get_inputs(time)
        # assert check_matrix(inputs)
        outputs = self.output_matrix @ state_vector + self.feedthrough_matrix @ inputs
        if self.non_linear_output:
            outputs += self.non_linear_output(time, state_vector, inputs)

        return outputs
    
    def get_state_idx(self, state_name: str) -> int:
        logger.debug("Getting index of state '%s'...", state_name)
        assert state_name in self.state_names, f"State '{state_name}' not found in model {self.name}."
        logger.debug("Index of state '%s' obtained.", state_name)
        return self.state_names.index(state_name)

    def remove_state(self, state_name: str) -> None:
        logger.debug("Removing state '%s' from the model...", state_name)
        state_idx = self.get_state_idx(state_name=state_name)
        self.state_names.remove(state_name)
        self.state_uoms.pop(state_idx)
        self.state_transition_matrix = np.delete(self.state_transition_matrix, state_idx, axis=0)
        self.state_transition_matrix = np.delete(self.state_transition_matrix, state_idx, axis=1)
        self.input_matrix = np.delete(self.input_matrix, state_idx, axis=0)
        self.output_matrix = np.delete(self.output_matrix, state_idx, axis=1)
        self.initial_conditions = np.delete(self.initial_conditions, state_idx, axis=0)
        self.ns -= 1
        logger.debug("State '%s' removed.", state_name)
    
    def get_input_idx(self, input_name: str) -> int:
        logger.debug("Getting index of input '%s'...", input_name)
        assert input_name in self.input_names, f"Input '{input_name}' not found in model {self.name}."
        logger.debug("Index of input '%s' obtained.", input_name)
        return self.input_names.index(input_name)
    
    def remove_input(self, input_name: str) -> None:
        logger.debug("Removing input '%s' from the model...", input_name)
        input_idx = self.get_input_idx(input_name=input_name)
        self.input_names.remove(input_name)
        self.input_uoms.pop(input_idx)
        self.input_funcs.pop(input_idx)
        self.input_matrix = np.delete(self.input_matrix, input_idx, axis=1)
        self.feedthrough_matrix = np.delete(self.feedthrough_matrix, input_idx, axis=1)
        self.ni -= 1
        logger.debug("Input '%s' removed.", input_name)

    def get_output_idx(self, output_name: str) -> int:
        logger.debug("Getting index of output '%s'...", output_name)
        assert output_name in self.output_names, f"Output '{output_name}' not found in model {self.name}."
        logger.debug("Index of output '%s' obtained.", output_name)
        return self.output_names.index(output_name)
    
    def remove_output(self, output_name: str) -> None:
        logger.debug("Removing output '%s' from the model...", output_name)
        output_idx = self.get_output_idx(output_name=output_name)
        self.output_names.remove(output_name)
        self.output_uoms.pop(output_idx)
        self.output_matrix = np.delete(self.output_matrix, output_idx, axis=0)
        self.feedthrough_matrix = np.delete(self.feedthrough_matrix, output_idx, axis=0)
        self.no -= 1
        logger.debug("Output '%s' removed.", output_name)

    def remove_dof(self, dof_name: str) -> None:
        logger.debug("Removing dof '%s' from the model...", dof_name)
        assert dof_name in ["x", "y", "t"], f"DOF '{dof_name}' not recognized. It should be either 'x', 'y' or 't'."
        dof_pos = f"{dof_name}_pos"
        dof_vel = f"{dof_name}_vel"
        dof_inp = f"{dof_name}_force" if dof_name in ["x", "y"] else f"{dof_name}_torque"
        # dof can be either a state, an input or an output
        
        self.remove_state(state_name=dof_pos)
        self.remove_output(output_name=dof_pos)

        self.remove_state(state_name=dof_vel)
        self.remove_output(output_name=dof_vel)

        self.remove_input(input_name=dof_inp)
        logger.debug("DOF '%s' removed.", dof_name)
        
    def add_output(self, output_name: str, output_uom: str, output_matrix_row: NDArray, feedthrough_matrix_row: NDArray | None = None) -> None:
        logger.debug("Adding output '%s' to '%s'.", output_name, self.name)
        assert check_row_vector(output_matrix_row), f"Output matrix row for new output '{output_name}' is not a row vector."
        assert output_matrix_row.shape[1] == self.ns, f"Output matrix row for new output '{output_name}' has {output_matrix_row.shape[1]} elements, but it should have {self.ns} elements to match the number of states."
        if feedthrough_matrix_row is None:
            feedthrough_matrix_row = np.zeros((1, self.ni))
        assert check_row_vector(feedthrough_matrix_row), f"Feedthrough matrix row for new output '{output_name}' is not a row vector."
        assert feedthrough_matrix_row.shape[1] == self.ni, f"Feedthrough matrix row for new output '{output_name}' has {feedthrough_matrix_row.shape[1]} elements, but it should have {self.ni} elements to match the number of inputs."
        self.output_names.append(output_name)
        self.output_uoms.append(output_uom)
        self.output_matrix = np.vstack((self.output_matrix, output_matrix_row))
        self.feedthrough_matrix = np.vstack((self.feedthrough_matrix, feedthrough_matrix_row))
        self.no += 1
        logger.debug("Output '%s' added to '%s'.", output_name, self.name)

    def add_input(self, input_name: str, input_uom: str, input_function: Callable, input_matrix_column: NDArray, feedthrough_matrix_column: NDArray | None = None) -> None:
        logger.debug("Adding input '%s' to '%s'.", input_name, self.name)
        assert check_column_vector(input_matrix_column), f"Input matrix column for new input '{input_name}' is not a column vector."
        assert input_matrix_column.shape[0] == self.ns, f"Input matrix column for new input '{input_name}' has {input_matrix_column.shape[0]} elements, but it should have {self.ns} elements to match the number of states."
        if feedthrough_matrix_column is None:
            feedthrough_matrix_column = np.zeros((self.no, 1))
        assert check_column_vector(feedthrough_matrix_column), f"Feedthrough matrix column for new input '{input_name}' is not a column vector."
        assert feedthrough_matrix_column.shape[0] == self.no, f"Feedthrough matrix column for new input '{input_name}' has {feedthrough_matrix_column.shape[0]} elements, but it should have {self.no} elements to match the number of outputs."
        self.input_names.append(input_name)
        self.input_uoms.append(input_uom)
        self.input_funcs.append(input_function)
        self.input_matrix = np.hstack((self.input_matrix, input_matrix_column))
        self.feedthrough_matrix = np.hstack((self.feedthrough_matrix, feedthrough_matrix_column))
        self.ni += 1
        logger.debug("Input '%s' added to '%s'.", input_name, self.name)