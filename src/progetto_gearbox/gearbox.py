from logging import getLogger

from progetto_gearbox import gear
from progetto_gearbox.interfaces.simulation_interfaces import Model
from progetto_gearbox.gear import SpurGear
from progetto_gearbox.utils.input_functions import constant
from progetto_gearbox.utils.angles import wrapToPi
from copy import deepcopy
from scipy.linalg import block_diag
from scipy.integrate import trapezoid
import numpy as np
from numpy import pi
from numpy.typing import NDArray
from collections.abc import Callable
from bokeh.plotting import figure
from bokeh.models import GlyphRenderer, ColumnDataSource

logger = getLogger(__name__)

class GearBox(Model):
    
    def __init__(self, name: str = "gearbox") -> None:
        super().__init__(name)
        self.gears: list[SpurGear] = []
        self.gear_names: list[str] = []
        self.gear_damages: list[NDArray] = []
        self.meshed_gear_names: list[tuple[str, str]] = []  # List of tuples (driving_gear_name, driven_gear_name)
        self.meshed_gears: list[tuple[int,'SpurGear',int,'SpurGear']] = []
        self.meshed_dofs_idx: list[tuple[int,int,int,int]] = []
        self.meshed_gamma: list[float] = []
        self._gears_added = False
        self._initial_conditions_set = False
        self._input_functions_set = False
        self._state_space_set = False

        



        # self.lumped_masses: list[tuple[str, float]] = []  # List
        
    def add_gears(self, gears: list[SpurGear], damages: list[NDArray]) -> None:
        logger.info("Adding gears to the gearbox...")
        
        for gear, gear_damages in zip(gears, damages):
            logger.debug("Adding gear '%s' to the gearbox...",gear.name)
            self.gears.append(deepcopy(gear))
            self.gear_names.append(gear.name)
            assert np.logical_and(np.all(gear_damages >= 0), np.all(gear_damages <= 1)), "'damages' must be a list of NDArrays, with the same length of 'gears', and with all elements within [0; 1]"
        
        self.gear_damages.extend(deepcopy(damages))
        self._gears_added = True
            
    def get_state_space(self) -> None:
        logger.info("Setting up the state space model...")
        self.state_names = []
        self.state_uoms = []
        self.input_names = []
        self.input_uoms = []
        self.input_funcs = []
        self.output_names = []
        self.output_uoms = []
        
        initial_conditions = []
        for gear in self.gears:
            logger.debug("Inheriting state space from gear '%s'...", gear.name)
            # gear.get_state_space() --> questo conviene farlo fare ai singoli gears... setuppo i singoli e poi aggiungo le connesioni
            self.state_names.extend([f"{gear.name}_{state_name}" for state_name in gear.state_names])
            self.state_uoms.extend(gear.state_uoms)
            self.input_names.extend([f"{gear.name}_{input_name}" for input_name in gear.input_names])
            self.input_uoms.extend(gear.input_uoms)
            self.input_funcs.extend(gear.input_funcs)
            self.output_names.extend([f"{gear.name}_{output_name}" for output_name in gear.output_names])
            self.output_uoms.extend(gear.output_uoms)
            initial_conditions.extend(gear.initial_conditions)
            logger.debug("State space from gear '%s' inherited.", gear.name)

        self.get_matrices_dimensions()
        self.state_transition_matrix = block_diag(*[gear.state_transition_matrix for gear in self.gears])
        self.input_matrix = block_diag(*[gear.input_matrix for gear in self.gears])
        self.output_matrix = block_diag(*[gear.output_matrix for gear in self.gears])
        self.feedthrough_matrix = block_diag(*[gear.feedthrough_matrix for gear in self.gears])
        self.check_matrices_dimensions()
        self.initial_conditions = np.array(initial_conditions)
        logger.info("State space model for gearbox '%s' set up.", self.name)
        self._state_space_set = True

    def set_initial_conditions(self, init_conditions_dict = dict[str, dict[str, float]]) -> None:
        logger.info("Setting initial conditions for gearbox '%s'...", self.name)
        assert self._state_space_set, "You should set state space before setting initial conditions."
        for gear_name in init_conditions_dict.keys():
            logger.debug("Setting initial conditions for gear '%s'...", gear_name)
            assert gear_name in self.gear_names, f"Initial condition for gear '{gear_name}' was provided, but gear '{gear_name}' was not found in gearbox '{self.name}'."
            gear_init_conditions_dict = init_conditions_dict[gear_name]
            for init_state in gear_init_conditions_dict.keys():
                logger.debug("Setting initial condition for dof '%s'...", init_state)
                init_state_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name=init_state)
                self.initial_conditions[init_state_idx] = gear_init_conditions_dict[init_state]
                logger.debug("Initial condition for dof '%s' set.", init_state)
            logger.debug("Initial conditions for gear '%s' set.", gear_name)
        
        logger.debug("Deriving initial conditions for meshed gears in gearbox '%s'...", self.name)
        for (driving_gear_name, driven_gear_name), gamma in zip(self.meshed_gear_names, self.meshed_gamma):
            logger.debug("Setting gear '%s' initial conditions...", driving_gear_name)
            self._get_driven_gear_initial_conditions_from(driven_gear_name=driven_gear_name,driving_gear_name=driving_gear_name, gamma=gamma)
            logger.debug("Initial conditions for gear '%s' set.", driving_gear_name)
        logger.debug("Initial conditions for meshed gears in gearbox '%s' derived.", self.name)
        self._initial_conditions_set = True
        logger.info("Initial conditions for gearbox '%s' set.", self.name)

    def add_proportional_derivative_feedback(self, gear_name: str, dof: str, kp: float, kd: float):
        logger.debug("Adding proportional derivative feedback in gearbox '%s' for gear '%s' on dof '%s'...", self.name, gear_name, dof)
        assert not self._input_functions_set, "You should add feedback before setting input functions."
        _, gear = self._get_gear(gear_name=gear_name)
        input_suffix = "_torque" if dof == "t" else "_force"
        input_name = gear_name + "_" + dof + input_suffix
        self.remove_input(input_name=input_name)

        output_name = input_name + "_feedback"
        output_uom = "Nm" if dof == "t" else "_torque"
        output_matrix_row = np.zeros((1,self.ns))
        feedthrough_matrix_row = np.zeros((1,self.ni))
        gear_vel_state_name = dof + "_vel"
        vel_state_idx, vel_state_name = self._get_state_idx_and_name(gear_name=gear_name,gear_state_name=gear_vel_state_name)

        if kp:
            logger.debug("Adding proportional feedback...")
            gear_pos_state_name = dof + "_pos"
            pos_state_idx, pos_state_name = self._get_state_idx_and_name(gear_name=gear_name,state_name=gear_pos_state_name)
            pos_input_uom = self.state_uoms[pos_state_idx]
            pos_input_name = pos_state_name + "_ref"
            pos_input_matrix_column = np.zeros((self.ns,1))
            pos_input_matrix_column[vel_state_idx,0] = kp/gear.inertia[dof]
            self.add_input(
                input_name=pos_input_name,
                input_uom=pos_input_uom,
                input_function=None,
                input_matrix_column=pos_input_matrix_column,
                feedthrough_matrix_column=np.zeros((self.no,1))
                )
            self.state_transition_matrix[vel_state_idx,pos_state_idx] -= kp/gear.inertia[dof]
            output_matrix_row[0,pos_state_idx] = - kp
            feedthrough_matrix_row = np.concatenate([feedthrough_matrix_row, np.array([[kp]])],axis=1)
            logger.debug("Proportional feedback added...")
        
        if kd:
            logger.debug("Adding derivative feedback...")
            vel_input_uom = self.state_uoms[vel_state_idx]
            vel_input_name = vel_state_name + "_ref"
            vel_input_matrix_column = np.zeros((self.ns,1))
            vel_input_matrix_column[vel_state_idx,0] = kd/gear.inertia[dof]
            self.add_input(
                input_name=vel_input_name,
                input_uom=vel_input_uom,
                input_function=None,
                input_matrix_column=vel_input_matrix_column,
                feedthrough_matrix_column=np.zeros((self.no,1))
                )
            self.state_transition_matrix[vel_state_idx,vel_state_idx] -= kd/gear.inertia[dof]
            output_matrix_row[0,vel_state_idx] = - kd
            feedthrough_matrix_row = np.concatenate([feedthrough_matrix_row, np.array([[kd]])],axis=1)
            logger.debug("Derivative feedback added...")
        
        self.add_output(output_name=output_name,output_uom=output_uom,output_matrix_row=output_matrix_row, feedthrough_matrix_row=feedthrough_matrix_row)
        logger.debug("Proportional derivative feedback in gearbox '%s' for gear '%s' on dof '%s' added.", self.name, gear_name, dof)
        
    def set_input_functions(self, input_func_dict = dict[str, dict[str, Callable]]) -> None:
        logger.info("Setting input functions for gearbox '%s'...", self.name)
        assert self._state_space_set, "You should get the state space model before setting inputs."
        logger.debug("Checking input functions for gearbox '%s'...", self.name)
        for gear_name in input_func_dict.keys():
            assert gear_name in self.gear_names, f"Input functions for gear '{gear_name}' were provided, but gear '{gear_name}' was not found in gearbox '{self.name}'."
        logger.debug("Input functions for gearbox '%s' checked.", self.name)
        
        inputs_to_be_removed = []
        for gear_name in self.gear_names:
            logger.debug("Setting input functions for gear '%s'...",gear_name)
            _, gear = self._get_gear(gear_name)
            gear_input_names = [input_name for input_name in self.input_names if input_name.startswith(gear_name)]
            if gear_name in input_func_dict.keys():
                gear_input_func_dict = input_func_dict[gear_name]
                for gear_input_name in gear_input_names:
                    logger.debug("Setting input '%s'...", gear_input_name)
                    gear_input_idx = self.get_input_idx(input_name=gear_input_name)
                    local_gear_input_name = gear_input_name.replace(gear_name + "_","")
                    if local_gear_input_name in gear_input_func_dict.keys():
                        self.input_funcs[gear_input_idx] = gear_input_func_dict[local_gear_input_name]
                        logger.debug("Input '%s' set.", gear_input_name)
                    else:
                        inputs_to_be_removed.append(gear_input_name)
                        logger.warning("Input function for input '%s' was not assigned. Input will be removed from gearbox '%s'.", gear_input_name, self.name)
                    logger.debug("Input '%s' set.", gear_input_name)
            else:
                for gear_input_name in gear_input_names:
                    logger.debug("Setting input '%s'...", gear_input_name)
                    inputs_to_be_removed.append(gear_input_name)
                    logger.warning("Input function for input '%s' was not assigned. Input will be removed from gearbox '%s'.", gear_input_name, self.name)
                    logger.debug("Input '%s' set.", gear_input_name)
            
            for gear_input_name in gear_input_names:
                logger.debug("Setting input '%s'...", gear_input_name)
                gear_input_idx = self.get_input_idx(input_name=gear_input_name)
            logger.debug("Input functions for gear '%s' set.",gear_name)
        
        self.remove_unset_inputs(inputs_to_be_removed=inputs_to_be_removed)
        self._input_functions_set = True
        logger.info("Input functions for gearbox '%s' set.", self.name)
    
    def add_spring_damper(self, to_gear_name: str, from_gear_name: str | None = None, stiffness: dict[str, float] = {}, damping: dict[str, float] = {}, origin: dict[str, float] = {}):
        # connecting_from -- C,K -- connecting_to
        # ground -- C,K -- connecting_to
        logger.debug("Adding spring damper connection from '%s' to '%s'...", from_gear_name if from_gear_name is not None else "ground", to_gear_name)
        assert self._initial_conditions_set, "You should set initial conditions before adding spring and dampers."
        logger.debug("Checking spring damper connection dofs...")
        setting_dofs = set(stiffness.keys()).union(set(damping.keys())).union(set(origin.keys()))
        for dof_name in setting_dofs:
            assert dof_name in ["x", "y", "t"], f"Degree of freedom '{dof_name}' is not valid. Valid dofs are 'x', 'y', 't'."
        logger.debug("Spring damper connection dofs checked.")
        
        logger.debug("Getting involved gears...")
        _, to_gear = self._get_gear(to_gear_name)
        if from_gear_name is not None:
            _, from_gear = self._get_gear(from_gear_name)
        logger.debug("Involved gears obtained.")
                    
        logger.debug("Adding spring damper connections...")
        for dof_name in setting_dofs:
            logger.debug("Adding spring damper connection for dof '%s'...", dof_name)
            dof_pos_name = dof_name + "_pos"
            dof_vel_name = dof_name + "_vel"
            to_dof_pos_idx, _ = self._get_state_idx_and_name(gear_name=to_gear_name, gear_state_name=dof_pos_name)
            to_dof_vel_idx, _ = self._get_state_idx_and_name(gear_name=to_gear_name, gear_state_name=dof_vel_name)
            sign = 1 if dof_name in ["x", "y", "t"] else -1
            if from_gear_name is not None:
                from_dof_pos_idx, _ = self._get_state_idx_and_name(gear_name=from_gear_name, gear_state_name=dof_pos_name)
                from_dof_vel_idx, _ = self._get_state_idx_and_name(gear_name=from_gear_name, gear_state_name=dof_vel_name)
            else:
                if (dof_name in origin.keys() or self.initial_conditions[to_dof_pos_idx]) and dof_name in stiffness.keys():
                    logger.debug("Adding ground spring damper input for dof '%s' in gearbox '%s'...", dof_name, self.name)
                    new_input_name = f"Offset_force_spring_damper_{to_gear_name}_{dof_name}"
                    new_input_uom = "m" if dof_name in ["x", "y"] else "rad"
                    new_input_value = origin[dof_name] if dof_name in origin.keys() else self.initial_conditions[to_dof_pos_idx]
                    new_input_func = constant(value=new_input_value)
                    new_input_matrix_col = np.zeros((self.ns, 1))
                    new_input_matrix_col[to_dof_vel_idx, 0] = sign*stiffness[dof_name]/to_gear.inertia[dof_name]
                    new_feedthrough_matrix_col = np.zeros((self.no, 1))
                    self.add_input(input_name=new_input_name, input_uom=new_input_uom, input_function=new_input_func, input_matrix_column=new_input_matrix_col, feedthrough_matrix_column=new_feedthrough_matrix_col)
                    logger.debug("Ground spring damper input for dof '%s' in gearbox '%s' added.", dof_name, self.name)

                new_output_name = f"Ground_force_spring_damper_{to_gear_name}_{dof_name}"
                new_output_uom = "N" if dof_name in ["x", "y"] else "Nm"
                new_output_matrix_row = np.zeros((1, self.ns))
                new_feedthrough_matrix_row = np.zeros((1, self.ni))

            if dof_name in stiffness.keys():
                logger.debug("Adding spring connection for dof '%s'...", dof_name)
                logger.debug("Adding spring connection for gear '%s'...", to_gear_name)
                self.state_transition_matrix[to_dof_vel_idx, to_dof_pos_idx] -= sign*stiffness[dof_name]/to_gear.inertia[dof_name]
                if from_gear_name is not None:
                    logger.debug("Adding spring connection for gear '%s'...", from_gear_name)
                    self.state_transition_matrix[to_dof_vel_idx, from_dof_pos_idx] += sign*stiffness[dof_name]/to_gear.inertia[dof_name]
                    self.state_transition_matrix[from_dof_vel_idx, from_dof_pos_idx] -= sign*stiffness[dof_name]/from_gear.inertia[dof_name]
                    self.state_transition_matrix[from_dof_vel_idx, to_dof_pos_idx] += sign*stiffness[dof_name]/from_gear.inertia[dof_name]
                else:
                    logger.debug("Adding ground spring connection...")
                    new_output_matrix_row[0, to_dof_pos_idx] = sign*stiffness[dof_name]
                    if dof_name in origin.keys():
                        new_feedthrough_matrix_row[0, -1] = -sign*stiffness[dof_name]
                logger.debug("Spring connection for dof '%s' added.", dof_name)

            if dof_name in damping.keys():
                logger.debug("Adding damper connection for dof '%s'...", dof_name)
                logger.debug("Adding damper connection for gear '%s'...", to_gear_name)
                self.state_transition_matrix[to_dof_vel_idx, to_dof_vel_idx] -= sign*damping[dof_name]/to_gear.inertia[dof_name]
                if from_gear_name is not None:
                    logger.debug("Adding damper connection for gear '%s'...", from_gear_name)
                    self.state_transition_matrix[to_dof_vel_idx, from_dof_vel_idx] += sign*damping[dof_name]/to_gear.inertia[dof_name]
                    self.state_transition_matrix[from_dof_vel_idx, from_dof_vel_idx] -= sign*damping[dof_name]/from_gear.inertia[dof_name]
                    self.state_transition_matrix[from_dof_vel_idx, to_dof_vel_idx] += sign*damping[dof_name]/from_gear.inertia[dof_name]
                else:
                    logger.debug("Adding ground damper connection...")
                    new_output_matrix_row[0, to_dof_vel_idx] = sign*damping[dof_name]
                logger.debug("Damper connection for dof '%s' added.", dof_name)

            if from_gear_name is None:
                logger.debug("Adding ground spring damper output for dof '%s' in gearbox '%s'...", dof_name, self.name)
                self.add_output(output_name=new_output_name, output_uom=new_output_uom, output_matrix_row=new_output_matrix_row)
            
            logger.debug("Spring damper connection for dof '%s' added.", dof_name)
        logger.debug("Spring damper connections added.")
        logger.debug("Spring damper connection from '%s' to '%s' added.", from_gear_name if from_gear_name is not None else "ground", to_gear_name)

    def add_meshing_constraint(self, driving_gear_name: str, driven_gear_name: str, gamma: float=0.0):
        logger.info("Adding meshing constraint between driving gear '%s' and driven gear '%s'...", driving_gear_name, driven_gear_name)
        assert self._state_space_set, "You should set state space before adding meshing constraints."
        self.meshed_gear_names.append((driving_gear_name, driven_gear_name))
        driving_gear_idx, driving_gear = self._get_gear(driving_gear_name)
        driven_gear_idx, driven_gear = self._get_gear(driven_gear_name)
        self.meshed_gears.append((driving_gear_idx,driving_gear,driven_gear_idx,driven_gear))
        driving_dof_x_pos_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name, gear_state_name="x_pos")
        driving_dof_y_pos_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name, gear_state_name="y_pos")
        driving_dof_pos_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name, gear_state_name="t_pos")
        driving_dof_vel_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name, gear_state_name="t_vel")
        driven_dof_x_pos_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name, gear_state_name="x_pos")
        driven_dof_y_pos_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name, gear_state_name="y_pos")
        driven_dof_pos_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name, gear_state_name="t_pos")
        driven_dof_vel_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name, gear_state_name="t_vel")
        self.meshed_dofs_idx.append((driving_dof_x_pos_idx,driving_dof_y_pos_idx,driving_dof_pos_idx,driving_dof_vel_idx,driven_dof_x_pos_idx,driven_dof_y_pos_idx,driven_dof_pos_idx,driven_dof_vel_idx))
        self.meshed_gamma.append(gamma)
        if self.non_linear_process is None:
            self.non_linear_process = self._compute_meshing_constraints
        logger.info("Meshing constraint between driving gear '%s' and driven gear '%s' added.", driving_gear_name, driven_gear_name)

    def _compute_meshing_constraints(self, time, state_vector, input_vector):
        delta_state_matrix = np.zeros((self.ns,self.ns))
        delta_state_vector = np.zeros((self.ns,1))
        for meshed_idx, (meshed_gears, meshed_dofs_idx) in enumerate(zip(self.meshed_gears,self.meshed_dofs_idx)):
            driving_gear_idx, driving_gear, driven_gear_idx, driven_gear = meshed_gears
            driving_dof_x_pos_idx, driving_dof_y_pos_idx, driving_dof_pos_idx, driving_dof_vel_idx, driven_dof_x_pos_idx, driven_dof_y_pos_idx, driven_dof_pos_idx, driven_dof_vel_idx = meshed_dofs_idx
 
            driving_gear_teeth_angles = driving_gear._get_teeth_centre_angle(state_vector[driving_dof_pos_idx])
            driven_gear_teeth_angles = driven_gear._get_teeth_centre_angle(state_vector[driven_dof_pos_idx])
            gamma = np.atan2(state_vector[driven_dof_y_pos_idx]-state_vector[driving_dof_y_pos_idx], state_vector[driven_dof_x_pos_idx]-state_vector[driving_dof_x_pos_idx])
            centre_distance = driving_gear.radiuses["pitch"] + driven_gear.radiuses["pitch"]
            
            _, _, driving_gear_engagement = self._compute_teeth_engagement(driving_gear, driven_gear, driving_gear_teeth_angles, "driving", state_vector[driving_dof_vel_idx], centre_distance, gamma)
            _, _, driven_gear_engagement = self._compute_teeth_engagement(driven_gear, driving_gear, driven_gear_teeth_angles, "driven", state_vector[driven_dof_vel_idx], centre_distance, gamma)
                                    
            driving_gear_teeth_engagement_angle = self._compute_teeth_engagement_angle(gear=driving_gear,gear_teeth_pos=driving_gear_teeth_angles,gear_engagement=driving_gear_engagement,gear_mode="driving",gear_vel=state_vector[driving_dof_vel_idx],gamma=gamma)
            driven_gear_teeth_engagement_angle = self._compute_teeth_engagement_angle(gear=driven_gear,gear_teeth_pos=driven_gear_teeth_angles,gear_engagement=driven_gear_engagement,gear_mode="driven",gear_vel=state_vector[driven_dof_vel_idx],gamma=gamma)

            driving_mesh_stiffness = self._compute_mesh_stiffness(gear=driving_gear, gear_teeth_engagement_angle=driving_gear_teeth_engagement_angle)
            driven_mesh_stiffness = self._compute_mesh_stiffness(gear=driven_gear, gear_teeth_engagement_angle=driven_gear_teeth_engagement_angle)
            mesh_stiffness = 1/(1/driving_mesh_stiffness + 1/driven_mesh_stiffness)
            raise NotImplementedError
            mesh_damping = self._example_mesh_damping(time, state_vector)
            
            delta_state_matrix[driving_dof_vel_idx, driving_dof_pos_idx] -= mesh_stiffness*driving_gear.radiuses["base"]**2/driving_gear.inertia["t"]
            delta_state_matrix[driving_dof_vel_idx, driving_dof_vel_idx] -= mesh_damping*driving_gear.radiuses["base"]**2/driving_gear.inertia["t"]
            delta_state_matrix[driving_dof_vel_idx, driven_dof_pos_idx] -= mesh_stiffness*driving_gear.radiuses["base"]*driven_gear.radiuses["base"]/driving_gear.inertia["t"]
            delta_state_matrix[driving_dof_vel_idx, driven_dof_vel_idx] -= mesh_damping*driving_gear.radiuses["base"]*driven_gear.radiuses["base"]/driving_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driving_dof_pos_idx] -= mesh_stiffness*driven_gear.radiuses["base"]*driving_gear.radiuses["base"]/driven_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driving_dof_vel_idx] -= mesh_damping*driven_gear.radiuses["base"]*driving_gear.radiuses["base"]/driven_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driven_dof_pos_idx] -= mesh_stiffness*driven_gear.radiuses["base"]**2/driven_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driven_dof_vel_idx] -= mesh_damping*driven_gear.radiuses["base"]**2/driven_gear.inertia["t"]
            delta_state_vector[driving_dof_pos_idx] -= self.initial_conditions[driving_dof_pos_idx]
            delta_state_vector[driven_dof_pos_idx] -= self.initial_conditions[driven_dof_pos_idx]
        
        return delta_state_matrix @ (state_vector + delta_state_vector)


    def add_lumped_mass(self, name: str = "lumped_mass", mass: float = 0.0):
        raise NotImplementedError

    def add_ground_constraint(self, gear_name: str, dofs: list[str]):
        logger.debug("Adding ground constraint to gear '%s' for dofs '%s'...", gear_name, dofs)
        assert self._state_space_set == False, "You should apply ground constraints BEFORE getting state space."
        # es: gear_name = "gear", dofs = ["x", "y"]
        gear_idx, gear = self._get_gear(gear_name)
        for dof in dofs:
            gear.remove_dof(dof_name=dof)

    def _get_driven_gear_initial_conditions_from(self,
            driven_gear_name: str, 
            driving_gear_name: str,
            gamma: float, 
            driving_gear_tooth_idx: int = 0, 
            driven_gear_tooth_idx: int = 0) -> dict[str, float]:
        
        
        logger.debug("Computing initial conditions for driven gear '%s' from driving gear '%s'.", driven_gear_name, driving_gear_name)
        
        logger.debug("Getting driving and driven gears...")
        _, driving_gear = self._get_gear(driving_gear_name)
        _, driven_gear = self._get_gear(driven_gear_name)
        logger.debug("Driving and driven gears obtained.")

        logger.debug("Setting initial condition for dof 'x_pos'...")
        driving_x_pos_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name,gear_state_name="x_pos")
        driven_x_pos_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name,gear_state_name="x_pos")
        centres_distance = driving_gear.radiuses["pitch"] + driven_gear.radiuses["pitch"]
        self.initial_conditions[driven_x_pos_idx] = self.initial_conditions[driving_x_pos_idx] + centres_distance*np.cos(gamma)
        logger.debug("Initial condition for dof 'x_pos' set.")

        logger.debug("Setting initial condition for dof 'x_vel'...")
        driving_x_vel_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name,gear_state_name="x_vel")
        driven_x_vel_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name,gear_state_name="x_vel")
        self.initial_conditions[driven_x_vel_idx] = self.initial_conditions[driving_x_vel_idx]
        logger.debug("Initial condition for dof 'x_vel' set.")


        logger.debug("Setting initial condition for dof 'y_pos'...")
        driving_y_pos_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name,gear_state_name="y_pos")
        driven_y_pos_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name,gear_state_name="y_pos")
        self.initial_conditions[driven_y_pos_idx] = self.initial_conditions[driving_y_pos_idx] + centres_distance*np.sin(gamma)
        logger.debug("Initial condition for dof 'y_pos' set.")


        logger.debug("Setting initial condition for dof 'y_vel'...")
        driving_y_vel_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name,gear_state_name="y_vel")
        driven_y_vel_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name,gear_state_name="y_vel")
        self.initial_conditions[driven_y_vel_idx] = self.initial_conditions[driving_y_vel_idx]
        logger.debug("Initial condition for dof 'y_vel' set.")


        logger.debug("Setting initial condition for dof 't_pos'...")
        driving_t_pos_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name,gear_state_name="t_pos")
        driven_t_pos_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name,gear_state_name="t_pos")
        self.initial_conditions[driven_t_pos_idx] = pi - driving_gear.teeth_number / driven_gear.teeth_number * self.initial_conditions[driving_t_pos_idx] + (driving_gear.teeth_number / driven_gear.teeth_number + 1) * gamma - pi / 2 / driven_gear.teeth_number * (4 * driving_gear_tooth_idx + 4 * driven_gear_tooth_idx - 6)
        logger.debug("Initial condition for dof 't_pos' set.")


        logger.debug("Setting initial condition for dof 't_vel'...")
        driving_t_vel_idx, _ = self._get_state_idx_and_name(gear_name=driving_gear_name,gear_state_name="t_vel")
        driven_t_vel_idx, _ = self._get_state_idx_and_name(gear_name=driven_gear_name,gear_state_name="t_vel")
        self.initial_conditions[driven_t_vel_idx] = - self.initial_conditions[driving_t_vel_idx] * driving_gear.teeth_number / driven_gear.teeth_number
        logger.debug("Initial condition for dof 't_vel' set.")

        logger.debug("Initial conditions for driven gear '%s' from driving gear '%s' computed.", driven_gear_name, driving_gear_name)
        
    def _compute_teeth_engagement(self, gear: 'SpurGear', meshed_gear: 'SpurGear', gear_teeth_pos: NDArray, gear_mode: str, gear_vel: float, centre_distance: float, gamma: float): ### fix
        meshed_gear_csi_addendum = meshed_gear._compute_csi_angle(meshed_gear.diameters["addendum"])
        meshed_gear_radial_distance_addendum = np.sqrt(meshed_gear.radiuses["addendum"] ** 2 + centre_distance ** 2 - 2 * meshed_gear.radiuses["addendum"] * centre_distance * np.cos(meshed_gear_csi_addendum - gear.pressure_angle))
        normalised_gear_first_angle = gear._compute_phi_angle(2 * meshed_gear_radial_distance_addendum)-gear.alpha[2]
        normalised_gear_second_angle = gear._compute_phi_angle(gear.diameters["addendum"])-gear.alpha[2]
        visual_normalised_gear_first_angle = gear._compute_csi_angle(2 * meshed_gear_radial_distance_addendum)
        visual_normalised_gear_second_angle = gear._compute_csi_angle(gear.diameters["addendum"])
        
        match (gear_mode, gear_vel):
            case ("driving", gv) if gv >= 0:
                gear_entry_angle = wrapToPi(gamma - gear.pressure_angle + normalised_gear_first_angle)
                gear_exit_angle = wrapToPi(gamma - gear.pressure_angle + normalised_gear_second_angle)
                visual_gear_entry_angle = wrapToPi(gamma - gear.pressure_angle + visual_normalised_gear_first_angle)
                visual_gear_exit_angle = wrapToPi(gamma - gear.pressure_angle + visual_normalised_gear_second_angle)
            case ("driven", gv) if gv <= 0:
                gear_entry_angle = wrapToPi(pi - gear.pressure_angle + gamma + normalised_gear_first_angle)
                gear_exit_angle = wrapToPi(pi - gear.pressure_angle + gamma + normalised_gear_second_angle)
                visual_gear_entry_angle = wrapToPi(pi - gear.pressure_angle + gamma + visual_normalised_gear_first_angle)
                visual_gear_exit_angle = wrapToPi(pi - gear.pressure_angle + gamma + visual_normalised_gear_second_angle)
            case ("driving", gv) if gv < 0:
                gear_exit_angle = wrapToPi(gamma + gear.pressure_angle - normalised_gear_first_angle)
                gear_entry_angle = wrapToPi(gamma + gear.pressure_angle - normalised_gear_second_angle)
                visual_gear_exit_angle = wrapToPi(gamma + gear.pressure_angle - visual_normalised_gear_first_angle)
                visual_gear_entry_angle = wrapToPi(gamma + gear.pressure_angle - visual_normalised_gear_second_angle)
            case ("driven", gv) if gv > 0:
                gear_exit_angle = wrapToPi(pi + gear.pressure_angle - gamma - normalised_gear_first_angle)
                gear_entry_angle = wrapToPi(pi + gear.pressure_angle - gamma - normalised_gear_second_angle)
                visual_gear_exit_angle = wrapToPi(pi + gear.pressure_angle - gamma - visual_normalised_gear_first_angle)
                visual_gear_entry_angle = wrapToPi(pi + gear.pressure_angle - gamma - visual_normalised_gear_second_angle)
        
        if gear_entry_angle <= gear_exit_angle:
            gear_engagement = np.logical_and(gear_teeth_pos >= gear_entry_angle, gear_teeth_pos <= gear_exit_angle)
        else:
            gear_engagement = np.logical_or(
                np.logical_and(gear_teeth_pos >= gear_entry_angle, gear_teeth_pos <= pi),
                np.logical_and(gear_teeth_pos >= -pi, gear_teeth_pos <= gear_exit_angle))
        return visual_gear_entry_angle, visual_gear_exit_angle, gear_engagement

    def _compute_teeth_engagement_angle(self, gear: "SpurGear", gear_teeth_pos: NDArray, gear_engagement: NDArray, gear_mode: str, gear_vel: float, gamma: float):
        gear_engaged_teeth_position = gear_teeth_pos[gear_engagement]
        match (gear_mode, gear_vel):
            case ("driving", gv) if gv >= 0:
                gear_teeth_engagement_angle = gear_engaged_teeth_position - gamma + gear.pressure_angle
            case ("driven", gv) if gv <= 0:
                gear_teeth_engagement_angle = gear_engaged_teeth_position - pi - gamma + gear.pressure_angle
            case ("driving", gv) if gv < 0:
                gear_teeth_engagement_angle = gamma + gear.pressure_angle - gear_engaged_teeth_position
            case ("driven", gv) if gv > 0:
                gear_teeth_engagement_angle = pi + gamma + gear.pressure_angle - gear_engaged_teeth_position
        return wrapToPi(gear_teeth_engagement_angle)

    def _compute_mesh_stiffness(self, gear: "SpurGear", gear_teeth_engagement_angle: NDArray, n_integration_points: int = 50):
        E = gear.young
        L = gear.thickness
        nu = gear.poisson

        if gear.root_greater_than_base:
            alpha = np.linspace(-gear_teeth_engagement_angle, gear.alpha[5], n_integration_points)
        else:
            alpha = np.linspace(-gear_teeth_engagement_angle, gear.alpha[2], n_integration_points)
            bending_constant = ((1 - (gear.teeth_number - 2.5) * np.cos(gear_teeth_engagement_angle) * np.cos(gear.alpha[3]) / (gear.teeth_number * np.cos(gear.pressure_angle))) ** 3 - (1 - np.cos(gear_teeth_engagement_angle) * np.cos(gear.alpha[2])) ** 3) / (2 * E * L * (np.cos(gear_teeth_engagement_angle) * np.sin(alpha[2]) ** 3))
            shear_constant = 1.2 * (1 + nu) * np.cos(gear_teeth_engagement_angle) ** 2 * (np.cos(alpha[2]) - (gear.teeth_number - 2.5) * np.cos(alpha[3]) / (gear.teeth_number * np.cos(gear.pressure_angle))) / (E * L * np.sin(alpha[2]))
            axial_constant = np.sin(gear_teeth_engagement_angle) ** 2 * (np.cos(alpha[2]) - (gear.teeth_number - 2.5) * np.cos(alpha[3]) / (gear.teeth_number * np.cos(gear.pressure_angle))) / (2 * E * L * (np.sin(alpha[2])))
        
        bending_integrand = 3 * (1 + np.cos(gear_teeth_engagement_angle) * ((gear.alpha[2] - alpha) * np.sin(alpha) - np.cos(alpha))) ** 2 * (gear.alpha[2] - alpha) * np.cos(alpha) / (2 * E * L * (np.sin(alpha) + (gear.alpha[2] - alpha) * np.cos(alpha)) ** 3)
        shear_integrand =  1.2 * (1 + nu) * (gear.alpha[2] - alpha) * np.cos(alpha) * np.cos(gear_teeth_engagement_angle) ** 2 / (E * L * (np.sin(alpha) + (gear.alpha[2] - alpha) * np.cos(alpha)))
        axial_integrand = (gear.alpha[2] - alpha) * np.cos(alpha) * np.sin(gear_teeth_engagement_angle) ** 2 / (2 * E * L * (np.sin(alpha) + (gear.alpha[2] - alpha) * np.cos(alpha)))
        
        hertz_inverse = 4 * (1 - nu ** 2) / (pi * E * L)
        bending_inverse = trapezoid(bending_integrand, x=alpha, axis=0)
        shear_inverse = trapezoid(shear_integrand, x=alpha, axis=0)
        axial_inverse = trapezoid(axial_integrand, x=alpha, axis=0)
        
        if gear.root_greater_than_base:
            bending_inverse += bending_constant
            shear_inverse += shear_constant
            axial_inverse += axial_constant

        gear_mesh_stiffness_inverse = bending_inverse + shear_inverse + axial_inverse + hertz_inverse
        gear_mesh_stiffness = np.sum(1 / gear_mesh_stiffness_inverse)
        return gear_mesh_stiffness
    
   
    def _compute_mesh_stiffness_damaged(self, gear: "SpurGear", gear_teeth_engagement_angle: NDArray, n_integration_points: int = 50):
        
        # Parameters
        E = gear.young
        L = gear.thickness
        nu = gear.poisson
        v = pi/4
        hq1 = damage * hr
        ha = hr/2 - hq1
        q1 = hq1 / np.sin(v)

        if gear.root_greater_than_base:
            pass
        else:
            """ casi su ha, hc, alpha1 --< ogni dente ha un suo alpha1
                
                se damaga <= 50% --> q1 --> ha
                se damage < 50% --> q2 --> hc """
            alphao = gear.alpha[4] - gear._compute_phi_angle(diameter=gear.diameters["addendum"])
            ho = gear.radiuses["base"] * (np.sin(alphao) + (gear.alpha[2]-alphao) * np.cos(alphao))
            ha = gear.radiuses["base"] * np.sin(alpha[2]) - q1 * np.sin(v)
            pass
        
        # Define alfa range
 
        #phig_r - self.angle_at_base/2
        #np.linspace(0, 1, num_points)[None, :] * (stop - start)[:, None] + start[:, None]
        
        # ha =
        # hc =
        # h0 =
        
        if self.root_greater_than_base: #case 1 from paper
                # Compute Ib, Is, Ia
            alpha = np.linspace(-alpha1, self.alpha[5], num_points)
            match ("mode", ha, hc, h0, alpha1, alpha4, alphaa, alphac): #resta da includere engagement
                
                case ("condition 1",ha, hc, h0, alpha1, alpha4, alphaa, alphac) if ha > h0 and alpha1 > alpha4:
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-(q1*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(np.sin(alpha3)+np.sin(alpha)-(q1*np.sin(v))/(root_radius))**3)
                    Ib1 = (4*(1-((self.teethNumber-2.5)*np.cos(alpha1)*np.cos(alpha3))/(self.teethNumber*np.cos(self.pressureAngle)))**3-4(1-np.cos(alpha1)*np.cos(alpha2)**3))/(E*L*np.cos(alpha1)*(2*np.sin(alpha2)-(q1*np.sin(v))/(base_radius))**3)
                    Ib2 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha2)-(q1*np.sin(v))/(root_radius)+np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))**3
                    Ib3 = 3*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha)/(2*E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha))**3)
                    Ib = Ib0 + Ib1 + Ib2 + Ib3
            
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha3)-(q1*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*np.cos(alpha1)**2*(np.cos(alpha2)-(self.teethNumber-2.5)/(self.teethNumber*np.cos(self.pressureAngle))*np.cos(alpha3)))/(E*L*(2*np.sin(alpha2)-(q1*np.sin(v))/(base_radius)))
                    Is2 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha2)-(q1*np.sin(v))/(base_radius)+np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))
                    Is3 = (1.2*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))
                    Is = Is0 + Is1 + Is2 + Is3
                    
                case ("condition 2",ha, hc, h0, alpha1, alpha4, alphaa, alphac) if ha < h0 or (ha > h0 and alpha1 > alpha4):
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-(q1*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(np.sin(alpha3)+np.sin(alpha)-(q1*np.sin(v))/(root_radius))**3)
                    Ib1 = (4*(1-((self.teethNumber-2.5)*np.cos(alpha1)*np.cos(alpha3))/(self.teethNumber*np.cos(self.pressureAngle)))**3-4(1-np.cos(alpha1)*np.cos(alpha2)**3))/(E*L*np.cos(alpha1)*(2*np.sin(alpha2)-(q1*np.sin(v))/(base_radius))**3)
                    Ib2 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha2)-(q1*np.sin(v))/(root_radius)+np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))**3
                    Ib = Ib0 + Ib1 + Ib2
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha3)-(q1*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*np.cos(alpha1)**2*(np.cos(alpha2)-(self.teethNumber-2.5)/(self.teethNumber*np.cos(self.pressureAngle))*np.cos(alpha3)))/(E*L*(2*np.sin(alpha2)-(q1*np.sin(v))/(base_radius)))
                    Is2 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha2)-(q1*np.sin(v))/(base_radius)+np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))
                    Is = Is0 + Is1 + Is2
                    
                case ("condition 3",ha, hc, h0, alpha1, alpha4, alphaa, alphac) if hc < h0 or (hc > h0 and alpha1 > alphac):
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-(((np.sin(alpha3)/(np.sin(v)))-(q2/root_radius))*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(np.sin(alpha)-(q2*np.sin(v))/(root_radius))**3)
                    Ib1 = (4*(1-((self.teethNumber-2.5)*np.cos(alpha1)*np.cos(alpha3))/(self.teethNumber*np.cos(self.pressureAngle)))**3-4(1-np.cos(alpha1)*np.cos(alpha2)**3))/(E*L*np.cos(alpha1)*(np.sin(alpha2)-(q2*np.sin(v))/(base_radius))**3)
                    Ib2 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)+np.sin(alpha)-(q2*np.sin(v))/(base_radius)))**3
                    Ib = Ib0 + Ib1 + Ib2
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha)-(q2*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*np.cos(alpha1)**2*(np.cos(alpha2)-(self.teethNumber-2.5)/(self.teethNumber*np.cos(self.pressureAngle))*np.cos(alpha3)))/(E*L*(np.sin(alpha2)-(q2*np.sin(v))/(base_radius)))
                    Is2 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)-(q2*np.sin(v))/(base_radius)))
                    Is = Is0 + Is1 + Is2
                    
                case ("condition 4",ha, hc, h0, alpha1, alpha4, alphaa, alphac) if hc >= h0 and alpha1 > alphac:
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-(((np.sin(alpha3)/(np.sin(v)))-(q2/root_radius))*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(np.sin(alpha)-(q2*np.sin(v))/(root_radius))**3)
                    Ib1 = (4*(1-((self.teethNumber-2.5)*np.cos(alpha1)*np.cos(alpha3))/(self.teethNumber*np.cos(self.pressureAngle)))**3-4(1-np.cos(alpha1)*np.cos(alpha2)**3))/(E*L*np.cos(alpha1)*(np.sin(alpha2)-(q2*np.sin(v))/(base_radius))**3)
                    Ib2 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)+np.sin(alpha)-(q2*np.sin(v))/(base_radius)))**3
                    Ib = Ib0 + Ib1 + Ib2
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha)-(q2*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*np.cos(alpha1)**2*(np.cos(alpha2)-(self.teethNumber-2.5)/(self.teethNumber*np.cos(self.pressureAngle))*np.cos(alpha3)))/(E*L*(np.sin(alpha2)-(q2*np.sin(v))/(base_radius)))
                    Is2 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)-(q2*np.sin(v))/(base_radius)))
                    Is = Is0 + Is1 + Is2
            
        else: #case 2 from paper
            alpha = np.linspace(-alpha1, self.alpha[2], num_points)
            
            match ("mode", ha, hc, h0, alpha1, alpha4, alphaa, alphac):
                
                case ("condition 1", ha, hc, h0, alpha1, alpha4, alphaa, alphac) if ha >= h0 and alpha1 > alphaa:
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-((q1/root_radius)*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(np.sin(alpha4)-(q1*np.sin(v))/(root_radius)+np.sin(alpha))**3)
                    Ib1 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)+np.sin(alpha)-(q1*np.sin(v))/(base_radius)+((self.teethNumber-2.5)/(self.teethNumber*np.cos(alpha0)))*np.sin(alpha4))**3)
                    Ib2 = 3*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha)/(2*E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha))**3)
                    Ib = Ib0 + Ib1 + Ib2
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha3)-(q1*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (1.2*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))
                    Is2 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(((self.teethNumber-2.5)/(self.teethNumber*np.cos(self.pressureAngle)))*np.sin(alpha4)-(q1*np.sin(v))/(base_radius)+np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))
                    Is = Is0 + Is1
                    
                case ("condition 2", ha, hc, h0, alpha1, alpha4, alphaa, alphac) if ha < h0 or (ha >= h0 and alpha1 <= alphaa):
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-((q1/root_radius)*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(np.sin(alpha4)-(q1*np.sin(v))/(root_radius)+np.sin(alpha))**3)
                    Ib1 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)+np.sin(alpha)-(q1*np.sin(v))/(base_radius)+((self.teethNumber-2.5)/(self.teethNumber*np.cos(alpha0)))*np.sin(alpha4))**3)
                    Ib = Ib0 + Ib1
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha3)-(q1*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(((self.teethNumber-2.5)/(self.teethNumber*np.cos(self.pressureAngle)))*np.sin(alpha4)-(q1*np.sin(v))/(base_radius)+np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)))
                    Is = Is0 + Is1
                    
                case ("condition 3", ha, hc, h0, alpha1, alpha4, alphaa, alphac) if hc < h0 or (hc >= h0 and alpha1 <= alphac):
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-(((np.sin(alpha4)/np.sin(v))-(q2/root_radius))*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(-(q2*np.sin(v))/(root_radius)+np.sin(alpha))**3)
                    Ib1 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)+np.sin(alpha)-(q2*np.sin(v))/(base_radius))**3)
                    Ib = Ib0 + Ib1
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha)-(q2*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha)-(q2*np.sin(v))/(base_radius)+(alpha2-alpha)*np.cos(alpha)))
                    Is = Is0 + Is1
                    
                case ("condition 4", ha, hc, h0, alpha1, alpha4, alphaa, alphac) if hc >= h0 and alpha1 > alphac:
                    Ib0 = (12*np.sin(alpha)*((self.teethNumber*np.cos(self.pressureAngle))/(self.teethNumber-2.5)-(np.cos(alpha)+np.cos(alpha3)-np.cos(alphar)-(((np.sin(alpha4)/np.sin(v))-(q2/root_radius))*np.cos(v))/(root_radius))*np.cos(alpha1))**2)/(E*L(-(q2*np.sin(v))/(root_radius)+np.sin(alpha))**3)
                    Ib1 = (12*(1+np.cos(alpha1)*((alpha2-alpha)*np.sin(alpha)-np.cos(alpha)))**2*(alpha2-alpha)*np.cos(alpha))/(E*L*(np.sin(alpha)+(alpha2-alpha)*np.cos(alpha)+np.sin(alpha)-(q2*np.sin(v))/(base_radius))**3)
                    Ib = Ib0 + Ib1
                    
                    Is0 = (2.4*(1+v)*np.cos(alpha1)**2(np.sin(alpha)))/(E*L(np.sin(alpha)-(q2*np.sin(v))/(root_radius)*np.sin(v))**3)
                    Is1 = (2.4*(1+v)*(alpha2-alpha)*np.cos(alpha)*np.cos(alpha1)**2)/(E*L*(np.sin(alpha)-(q2*np.sin(v))/(base_radius)+(alpha2-alpha)*np.cos(alpha)))
                    Is = Is0 + Is1
    
        # Integrate over alfa
        Kb_inv = np.sum(sc.integrate.trapezoid(Ib, x=alpha, axis=0))
        Ks_inv = np.sum(sc.integrate.trapezoid(Is, x=alpha, axis=0))
        Ka_inv = np.sum(sc.integrate.trapezoid(Ia, x=alpha, axis=0))
        
        if self.root_greater_than_base:
            kb_inv += Ib0
            ks_inv += Is0
        
        # Total stiffness
        Kt = 1 / (Kb_inv + Ks_inv + Ka_inv)
        if all(engagement == 0):
            print("No engagement detected.")
        
        raise NotImplementedError
        return Kt




    def _example_mesh_damping(self, t, x): ### fix
        return 2.5*1e3

    def plot(self,
             fig: figure,
             state_vector: NDArray,
             additional_outputs: dict[str, list[NDArray]],
             n_points_involutes: int = 10,
             n_points_tips: int = 5,
             ) -> tuple[figure, ColumnDataSource, GlyphRenderer]:
        
        x_pos = {}
        y_pos = {}
        t_pos = {}
        sources = {}
        renderers = {}

        try:
            engaged_teeth = additional_outputs["engaged_teeth"]
        except:
            engaged_teeth = None
        
        try:
            engagement_angles = additional_outputs["engagement_angles"]
        except:
            engagement_angles = None

        for gear_name, gear in zip(self.gear_names, self.gears):
            gear_x_pos_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name="x_pos")
            gear_y_pos_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name="y_pos")
            gear_t_pos_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name="t_pos")

            x_pos[gear_name] = state_vector[gear_x_pos_idx]
            y_pos[gear_name] = state_vector[gear_y_pos_idx]
            t_pos[gear_name] = state_vector[gear_t_pos_idx]

        for gear_idx, (gear_name, gear) in enumerate(zip(self.gear_names, self.gears)):
            
            fig, teeth_source, teeth_renderer = gear._teeth_plot(
                fig=fig,
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
                t_pos=t_pos[gear_name],
                engaged_teeth=engaged_teeth[gear_idx] if engaged_teeth is not None else None,
                n_points_involutes=n_points_involutes,
                n_points_tips=n_points_tips,
            )
            sources[f"{gear_name}_teeth"] = teeth_source
            renderers[f"{gear_name}_teeth"] = teeth_renderer

        for gear_name, gear in zip(self.gear_names, self.gears):
            fig, root_source, root_renderer = gear._teeth_root_plot(
                fig=fig,
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )
            sources[f"{gear_name}_root"] = root_source
            renderers[f"{gear_name}_root"] = root_renderer

        for gear_name, gear in zip(self.gear_names, self.gears):
            fig, ref_source, ref_renderer = gear._reference_circles_plot(
                fig=fig,
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )
            sources[f"{gear_name}_reference_circles"] = ref_source
            renderers[f"{gear_name}_reference_circles"] = ref_renderer

        for meshed_idx, (driving_gear_name, driven_gear_name) in enumerate(self.meshed_gear_names):
            fig, ref_source, ref_renderer = self._engagement_angles_plot(
                fig=fig,
                driving_gear_name=driving_gear_name,
                driven_gear_name=driven_gear_name,
                driving_x_pos=x_pos[driving_gear_name],
                driving_y_pos=y_pos[driving_gear_name],
                driven_x_pos=x_pos[driven_gear_name],
                driven_y_pos=y_pos[driven_gear_name],
                driving_entry_angle=engagement_angles[meshed_idx][0] if engagement_angles is not None else 0,
                driving_exit_angle=engagement_angles[meshed_idx][1] if engagement_angles is not None else 0,
                driven_entry_angle=engagement_angles[meshed_idx][2] if engagement_angles is not None else 0,
                driven_exit_angle=engagement_angles[meshed_idx][3] if engagement_angles is not None else 0
            )
            sources[f"mesh_{meshed_idx}_engagements"] = ref_source
            renderers[f"mesh_{meshed_idx}_engagements"] = ref_renderer

        return fig, sources, renderers

    def _update_plot(self,
            sources: dict,
            state_vector: NDArray,
            additional_outputs: dict[str, list[NDArray]],
            n_points_involutes: int = 10,
            n_points_tips: int = 5,
        ) -> None:

        engaged_teeth = additional_outputs["engaged_teeth"]
        engagement_angles = additional_outputs["engagement_angles"]
        x_pos = {}
        y_pos = {}
        t_pos = {}  
        for gear_idx, (gear_name, gear) in enumerate(zip(self.gear_names, self.gears)):
            gear_x_pos_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name="x_pos")
            gear_y_pos_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name="y_pos")
            gear_t_pos_idx, _ = self._get_state_idx_and_name(gear_name=gear_name, gear_state_name="t_pos")

            x_pos[gear_name] = state_vector[gear_x_pos_idx]
            y_pos[gear_name] = state_vector[gear_y_pos_idx]
            t_pos[gear_name] = state_vector[gear_t_pos_idx]

            gear._update_teeth_plot(
                source=sources[f"{gear_name}_teeth"],
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
                t_pos=t_pos[gear_name],
                engaged_teeth=engaged_teeth[gear_idx],
                n_points_involutes=n_points_involutes,
                n_points_tips=n_points_tips,
            )

            gear._update_teeth_root_plot(
                source=sources[f"{gear_name}_root"],
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )

            gear._update_reference_circles_plot(
                source=sources[f"{gear_name}_reference_circles"],
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )
        
        for meshed_idx, (driving_gear_name, driven_gear_name) in enumerate(self.meshed_gear_names):
            self._update_engagement_angles_plot(
                driving_gear_name=driving_gear_name,
                driven_gear_name=driven_gear_name,
                source=sources[f"mesh_{meshed_idx}_engagements"],
                driving_x_pos=x_pos[driving_gear_name],
                driving_y_pos=y_pos[driving_gear_name],
                driven_x_pos=x_pos[driven_gear_name],
                driven_y_pos=y_pos[driven_gear_name],
                driving_entry_angle=engagement_angles[meshed_idx][0],
                driving_exit_angle=engagement_angles[meshed_idx][1],
                driven_entry_angle=engagement_angles[meshed_idx][2],
                driven_exit_angle=engagement_angles[meshed_idx][3]
            )
    
    def _get_gear(self, gear_name: str) -> tuple[int, SpurGear]:
        assert gear_name in self.gear_names, f"Gear '{gear_name}' not found in gearbox'{self.name}'."
        gear_idx = self.gear_names.index(gear_name)
        return gear_idx, self.gears[gear_idx]
    
    def _get_state_idx_and_name(self, gear_name: str, gear_state_name: str) -> tuple[int, str]:
        logger.debug("Getting index of state '%s' for gear '%s'...", gear_state_name, gear_name)
        state_name = f"{gear_name}_{gear_state_name}"
        assert state_name in self.state_names, f"State '{gear_state_name}' not found in gear '{gear_name}'."
        state_idx = self.state_names.index(state_name)
        logger.debug("Index of state '%s' for gear '%s' obtained.", gear_state_name, gear_name)
        return state_idx, state_name
    
    def _get_additional_outputs(self, time: NDArray, state_matrix: NDArray):
        engaged_teeth = []
        for gear in self.gears:
            engaged_teeth.append(np.zeros((gear.teeth_number, time.shape[0]),dtype=np.bool))

        engagement_angles = []
        for meshed_idx, (meshed_gears, meshed_dofs_idx) in enumerate(zip(self.meshed_gears,self.meshed_dofs_idx)):
            engagement_angles.append(np.zeros((4, time.shape[0])))

        for idx, state_vector in enumerate(state_matrix.T):
            for meshed_idx, (meshed_gears, meshed_dofs_idx) in enumerate(zip(self.meshed_gears,self.meshed_dofs_idx)):
                driving_gear_idx, driving_gear, driven_gear_idx, driven_gear = meshed_gears
                driving_dof_x_pos_idx, driving_dof_y_pos_idx, driving_dof_pos_idx, driving_dof_vel_idx, driven_dof_x_pos_idx, driven_dof_y_pos_idx, driven_dof_pos_idx, driven_dof_vel_idx = meshed_dofs_idx
    
                driving_gear_teeth_angles = driving_gear._get_teeth_centre_angle(state_vector[driving_dof_pos_idx])
                driven_gear_teeth_angles = driven_gear._get_teeth_centre_angle(state_vector[driven_dof_pos_idx])
                gamma = np.atan2(state_vector[driven_dof_y_pos_idx]-state_vector[driving_dof_y_pos_idx], state_vector[driven_dof_x_pos_idx]-state_vector[driving_dof_x_pos_idx])
                centre_distance = driving_gear.radiuses["pitch"] + driven_gear.radiuses["pitch"]
                
                driving_gear_entry_angle, driving_gear_exit_angle, driving_gear_engagement = self._compute_teeth_engagement(driving_gear, driven_gear, driving_gear_teeth_angles, "driving", state_vector[driving_dof_vel_idx], centre_distance, gamma)
                driven_gear_entry_angle, driven_gear_exit_angle, driven_gear_engagement = self._compute_teeth_engagement(driven_gear, driving_gear, driven_gear_teeth_angles, "driven", state_vector[driven_dof_vel_idx], centre_distance, gamma)
                engagement_angles[meshed_idx][:,idx] = (driving_gear_entry_angle, driving_gear_exit_angle, driven_gear_entry_angle, driven_gear_exit_angle)
                engaged_teeth[driving_gear_idx][:,idx] = np.logical_or(engaged_teeth[driving_gear_idx][:,idx],driving_gear_engagement)
                engaged_teeth[driven_gear_idx][:,idx] = np.logical_or(engaged_teeth[driven_gear_idx][:,idx],driven_gear_engagement)

        return {"engagement_angles": engagement_angles,"engaged_teeth": engaged_teeth}
    
    def _engagement_angles_plot(
            self,
            fig: figure,
            driving_gear_name,
            driven_gear_name,
            driving_x_pos = 0,
            driving_y_pos = 0,
            driven_x_pos = 0,
            driven_y_pos = 0,
            driving_entry_angle = -pi/2,
            driving_exit_angle = pi/2,
            driven_entry_angle = -pi/2,
            driven_exit_angle = pi/2
            ):
        
        _, driving_gear = self._get_gear(driving_gear_name)
        _, driven_gear = self._get_gear(driven_gear_name)
        driving_entry_xs = [driving_x_pos + driving_gear.radiuses["root"] * np.cos(driving_entry_angle),
                            driving_x_pos + driving_gear.radiuses["addendum"] * np.cos(driving_entry_angle)]
        driving_entry_ys = [driving_y_pos + driving_gear.radiuses["root"] * np.sin(driving_entry_angle),
                            driving_y_pos + driving_gear.radiuses["addendum"] * np.sin(driving_entry_angle)]
        driving_entry_color = ["blue"]
        driving_exit_xs = [driving_x_pos + driving_gear.radiuses["root"] * np.cos(driving_exit_angle),
                            driving_x_pos + driving_gear.radiuses["addendum"] * np.cos(driving_exit_angle)]
        driving_exit_ys = [driving_y_pos + driving_gear.radiuses["root"] * np.sin(driving_exit_angle),
                            driving_y_pos + driving_gear.radiuses["addendum"] * np.sin(driving_exit_angle)]
        driving_exit_color = ["red"]
        driven_entry_xs = [driven_x_pos + driven_gear.radiuses["root"] * np.cos(driven_entry_angle),
                            driven_x_pos + driven_gear.radiuses["addendum"] * np.cos(driven_entry_angle)]
        driven_entry_ys = [driven_y_pos + driven_gear.radiuses["root"] * np.sin(driven_entry_angle),
                            driven_y_pos + driven_gear.radiuses["addendum"] * np.sin(driven_entry_angle)]
        driven_entry_color = ["blue"]
        driven_exit_xs = [driven_x_pos + driven_gear.radiuses["root"] * np.cos(driven_exit_angle),
                            driven_x_pos + driven_gear.radiuses["addendum"] * np.cos(driven_exit_angle)]
        driven_exit_ys = [driven_y_pos + driven_gear.radiuses["root"] * np.sin(driven_exit_angle),
                            driven_y_pos + driven_gear.radiuses["addendum"] * np.sin(driven_exit_angle)]
        driven_exit_color = ["red"]

        source = ColumnDataSource(data=dict(
            xs=[driving_entry_xs,driving_exit_xs,driven_entry_xs,driven_exit_xs],
            ys=[driving_entry_ys,driving_exit_ys,driven_entry_ys,driven_exit_ys],
            colors=driving_entry_color+driving_exit_color+driven_entry_color+driven_exit_color
        ))

        renderer = fig.multi_line(
            xs="xs",
            ys="ys",
            source=source,
            line_color="colors",
            line_width=1,
            line_dash="dashed"
        )

        return fig, source, renderer
    
    def _update_engagement_angles_plot(
            self,
            driving_gear_name,
            driven_gear_name,
            source=ColumnDataSource,
            driving_x_pos = 0,
            driving_y_pos = 0,
            driven_x_pos = 0,
            driven_y_pos = 0,
            driving_entry_angle = -pi/2,
            driving_exit_angle = pi/2,
            driven_entry_angle = -pi/2,
            driven_exit_angle = pi/2
            ):
        
        _, driving_gear = self._get_gear(driving_gear_name)
        _, driven_gear = self._get_gear(driven_gear_name)
        driving_entry_xs = [driving_x_pos + driving_gear.radiuses["root"] * np.cos(driving_entry_angle),
                            driving_x_pos + driving_gear.radiuses["addendum"] * np.cos(driving_entry_angle)]
        driving_entry_ys = [driving_y_pos + driving_gear.radiuses["root"] * np.sin(driving_entry_angle),
                            driving_y_pos + driving_gear.radiuses["addendum"] * np.sin(driving_entry_angle)]
        driving_entry_color = ["blue"]
        driving_exit_xs = [driving_x_pos + driving_gear.radiuses["root"] * np.cos(driving_exit_angle),
                            driving_x_pos + driving_gear.radiuses["addendum"] * np.cos(driving_exit_angle)]
        driving_exit_ys = [driving_y_pos + driving_gear.radiuses["root"] * np.sin(driving_exit_angle),
                            driving_y_pos + driving_gear.radiuses["addendum"] * np.sin(driving_exit_angle)]
        driving_exit_color = ["red"]
        driven_entry_xs = [driven_x_pos + driven_gear.radiuses["root"] * np.cos(driven_entry_angle),
                            driven_x_pos + driven_gear.radiuses["addendum"] * np.cos(driven_entry_angle)]
        driven_entry_ys = [driven_y_pos + driven_gear.radiuses["root"] * np.sin(driven_entry_angle),
                            driven_y_pos + driven_gear.radiuses["addendum"] * np.sin(driven_entry_angle)]
        driven_entry_color = ["blue"]
        driven_exit_xs = [driven_x_pos + driven_gear.radiuses["root"] * np.cos(driven_exit_angle),
                            driven_x_pos + driven_gear.radiuses["addendum"] * np.cos(driven_exit_angle)]
        driven_exit_ys = [driven_y_pos + driven_gear.radiuses["root"] * np.sin(driven_exit_angle),
                            driven_y_pos + driven_gear.radiuses["addendum"] * np.sin(driven_exit_angle)]
        driven_exit_color = ["red"]

        source.data = dict(
            xs=[driving_entry_xs,driving_exit_xs,driven_entry_xs,driven_exit_xs],
            ys=[driving_entry_ys,driving_exit_ys,driven_entry_ys,driven_exit_ys],
            colors=driving_entry_color+driving_exit_color+driven_entry_color+driven_exit_color
        )
