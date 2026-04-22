from logging import getLogger

from progetto_gearbox import gear
from progetto_gearbox.interfaces.simulation_interfaces import Model
from progetto_gearbox.gear import SpurGear
from progetto_gearbox.utils.input_functions import constant
from copy import deepcopy
from scipy.linalg import block_diag
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
        self.meshed_gears: list[tuple[str, str]] = []  # List of tuples (driving_gear_name, driven_gear_name)
        # self.lumped_masses: list[tuple[str, float]] = []  # List
        
    def add_gears(self, gears: list[SpurGear]) -> None:
        logger.info("Adding gears to the gearbox...")
        
        for gear in gears:
            logger.debug("Adding gear '%s' to the gearbox...",gear.name)
            self.gears.append(deepcopy(gear))
            self.gear_names.append(gear.name)
            
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

    def set_initial_conditions(self, init_conditions_dict = dict[str, dict[str, float]]) -> None:
        logger.info("Setting initial conditions for gearbox '%s'...", self.name)
        logger.debug("Checking initial conditions for gearbox '%s'...", self.name)
        for gear_name in init_conditions_dict.keys():
            assert gear_name in self.gear_names, f"Initial condition for gear '{gear_name}' was provided, but gear '{gear_name}' was not found in gearbox '{self.name}'."
        logger.debug("Initial conditions for gearbox '%s' checked.", self.name)

        for gear_name in self.gear_names:
            logger.debug("Setting initial condition for gear '%s'...",gear_name)
            _, gear = self.get_gear(gear_name)
            if gear_name in init_conditions_dict.keys():
                gear.set_initial_conditions(init_conditions_dict=init_conditions_dict[gear_name])
                logger.debug("Initial conditions for gear '%s' set.", gear_name)
            else:
                logger.warning("Initial conditions for gear '%s' were not assigned in gearbox '%s'. Initialising them to 0.", gear_name, self.name)
                gear.set_initial_conditions(init_conditions_dict={})
        logger.info("Initial conditions for gearbox '%s' set.", self.name)

    def set_input_functions(self, input_func_dict = dict[str, dict[str, Callable]]) -> None:
        logger.info("Setting input functions for gearbox '%s'...", self.name)
        logger.debug("Checking input functions for gearbox '%s'...", self.name)
        for gear_name in input_func_dict.keys():
            assert gear_name in self.gear_names, f"Input functions for gear '{gear_name}' were provided, but gear '{gear_name}' was not found in gearbox '{self.name}'."
        logger.debug("Input functions for gearbox '%s' checked.", self.name)

        for gear_name in self.gear_names:
            logger.debug("Setting input functions for gear '%s'...",gear_name)
            _, gear = self.get_gear(gear_name)
            if gear_name in input_func_dict.keys():
                gear.set_input_functions(input_func_dict=input_func_dict[gear_name])
                logger.debug("Input functions for gear '%s' set.", gear_name)
            else:
                logger.warning("Input functions for gear '%s' were not assigned in gearbox '%s'. Removing them.", gear_name, self.name)
                gear.set_input_functions(input_func_dict={})
        logger.info("Input functions for gearbox '%s' set.", self.name)

    def add_lumped_mass(self, name: str = "lumped_mass", mass: float = 0.0):
        pass

    def add_ground_constraint(self, gear_name: str, dofs: list[str]):
        logger.debug("Adding ground constraint to gear '%s' for dofs '%s'...", gear_name, dofs)
        # es: gear_name = "gear", dofs = ["x", "y"]
        gear_idx, gear = self.get_gear(gear_name)
        for dof in dofs:
            gear.remove_dof(dof_name=dof)

    def add_spring_damper(self, to_gear_name: str, from_gear_name: str | None = None, stiffness: dict[str, float] = {}, damping: dict[str, float] = {}, origin: dict[str, float] = {}):
        # connecting_from -- C,K -- connecting_to
        # ground -- C,K -- connecting_to
        logger.debug("Adding spring damper connection from '%s' to '%s'...", from_gear_name if from_gear_name is not None else "ground", to_gear_name)
        logger.debug("Checking spring damper connection dofs...")
        setting_dofs = set(stiffness.keys()).union(set(damping.keys())).union(set(origin.keys()))
        for dof_name in setting_dofs:
            assert dof_name in ["x", "y", "t"], f"Degree of freedom '{dof_name}' is not valid. Valid dofs are 'x', 'y', 't'."
        logger.debug("Spring damper connection dofs checked.")
        
        logger.debug("Getting involved gears...")
        _, to_gear = self.get_gear(to_gear_name)
        if from_gear_name is not None:
            _, from_gear = self.get_gear(from_gear_name)
        logger.debug("Involved gears obtained.")
                    
        logger.debug("Adding spring damper connections...")
        for dof_name in setting_dofs:
            logger.debug("Adding spring damper connection for dof '%s'...", dof_name)
            dof_pos_name = dof_name + "_pos"
            dof_vel_name = dof_name + "_vel"
            to_dof_pos_idx, _ = self.get_state_idx_and_name(gear_name=to_gear_name, gear_state_name=dof_pos_name)
            to_dof_vel_idx, _ = self.get_state_idx_and_name(gear_name=to_gear_name, gear_state_name=dof_vel_name)
            sign = 1 if dof_name in ["x", "y", "t"] else -1
            if from_gear_name is not None:
                from_dof_pos_idx, _ = self.get_state_idx_and_name(gear_name=from_gear_name, gear_state_name=dof_pos_name)
                from_dof_vel_idx, _ = self.get_state_idx_and_name(gear_name=from_gear_name, gear_state_name=dof_vel_name)
            else:
                if dof_name in origin.keys() and dof_name in stiffness.keys():
                    logger.debug("Adding ground spring damper input for dof '%s' in gearbox '%s'...", dof_name, self.name)
                    new_input_name = f"Offset_force_spring_damper_{to_gear_name}_{dof_name}"
                    new_input_uom = "m" if dof_name in ["x", "y"] else "rad"
                    new_input_func = constant(value=origin[dof_name])
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

    def add_meshing_constraint(self, driving_gear_name: str, driven_gear_name: str):
        logger.info("Adding meshing constraint between driving gear '%s' and driven gear '%s'...", driving_gear_name, driven_gear_name)
        self.meshed_gears.append((driving_gear_name, driven_gear_name))
        if self.non_linear_process is None:
            self.non_linear_process = self._compute_meshing_constraints
        logger.info("Meshing constraint between driving gear '%s' and driven gear '%s' added.", driving_gear_name, driven_gear_name)
        pass

    def _compute_meshing_constraints(self, time, state_vector, input_vector):
        delta_state_matrix = np.zeros((self.ns,self.ns))
        for meshed_driving_gear_name, meshed_driven_gear_name in self.meshed_gears:
            meshed_driving_gear_idx, meshed_driving_gear = self.get_gear(meshed_driving_gear_name)
            meshed_driven_gear_idx, meshed_driven_gear = self.get_gear(meshed_driven_gear_name)
            # driving_stiffness = meshed_driving_gear.example_mesh_stiffness(time, state_vector)
            # driving_damping = meshed_driving_gear.example_mesh_damping(time, state_vector)
            # driven_stiffness = meshed_driven_gear.example_mesh_stiffness(time, state_vector)
            # driven_damping = meshed_driven_gear.example_mesh_damping(time, state_vector)
            # mesh_stiffness = 1/(1/driving_stiffness + 1/driven_stiffness)
            # mesh_damping = 1/(1/driving_damping + 1/driven_damping)
            mesh_stiffness = self.example_mesh_stiffness(time,state_vector)
            mesh_damping = self.example_mesh_damping(time,state_vector)
            driving_dof_pos_idx, _ = self.get_state_idx_and_name(gear_name=meshed_driving_gear_name, gear_state_name="t_pos")
            driving_dof_vel_idx, _ = self.get_state_idx_and_name(gear_name=meshed_driving_gear_name, gear_state_name="t_vel")
            driven_dof_pos_idx, _ = self.get_state_idx_and_name(gear_name=meshed_driven_gear_name, gear_state_name="t_pos")
            driven_dof_vel_idx, _ = self.get_state_idx_and_name(gear_name=meshed_driven_gear_name, gear_state_name="t_vel")
            delta_state_matrix[driving_dof_vel_idx, driving_dof_pos_idx] -= mesh_stiffness*meshed_driving_gear.radiuses["base"]**2/meshed_driving_gear.inertia["t"]
            delta_state_matrix[driving_dof_vel_idx, driving_dof_vel_idx] -= mesh_damping*meshed_driving_gear.radiuses["base"]**2/meshed_driving_gear.inertia["t"]
            delta_state_matrix[driving_dof_vel_idx, driven_dof_pos_idx] -= mesh_stiffness*meshed_driving_gear.radiuses["base"]*meshed_driven_gear.radiuses["base"]/meshed_driving_gear.inertia["t"]
            delta_state_matrix[driving_dof_vel_idx, driven_dof_vel_idx] -= mesh_damping*meshed_driving_gear.radiuses["base"]*meshed_driven_gear.radiuses["base"]/meshed_driving_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driving_dof_pos_idx] -= mesh_stiffness*meshed_driven_gear.radiuses["base"]*meshed_driving_gear.radiuses["base"]/meshed_driven_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driving_dof_vel_idx] -= mesh_damping*meshed_driven_gear.radiuses["base"]*meshed_driving_gear.radiuses["base"]/meshed_driven_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driven_dof_pos_idx] -= mesh_stiffness*meshed_driven_gear.radiuses["base"]**2/meshed_driven_gear.inertia["t"]
            delta_state_matrix[driven_dof_vel_idx, driven_dof_vel_idx] -= mesh_damping*meshed_driven_gear.radiuses["base"]**2/meshed_driven_gear.inertia["t"]
        return delta_state_matrix @ state_vector

    def get_driven_gear_initial_conditions_from(self,
            driven_gear_name: str, 
            driving_gear_name: str, 
            driving_gear_init_conditions_dict: dict[str, float], 
            gamma: float, 
            driving_gear_tooth_idx: int = 0, 
            driven_gear_tooth_idx: int = 0) -> dict[str, float]: 
        
        logger.debug("Computing initial conditions for driven gear '%s' from driving gear '%s'.", driven_gear_name, driving_gear_name)
        
        logger.debug("Getting driving and driven gears...")
        _, driving_gear = self.get_gear(driving_gear_name)
        _, driven_gear = self.get_gear(driven_gear_name)
        logger.debug("Driving and driven gears obtained.")

        logger.debug("Extracting driving gear initial conditions...")
        driving_gear_x_pos = driving_gear_init_conditions_dict["x_pos"] if "x_pos" in driving_gear_init_conditions_dict.keys() else 0.0
        driving_gear_x_vel = driving_gear_init_conditions_dict["x_vel"] if "x_vel" in driving_gear_init_conditions_dict.keys() else 0.0
        driving_gear_y_pos = driving_gear_init_conditions_dict["y_pos"] if "y_pos" in driving_gear_init_conditions_dict.keys() else 0.0
        driving_gear_y_vel = driving_gear_init_conditions_dict["y_vel"] if "y_vel" in driving_gear_init_conditions_dict.keys() else 0.0
        driving_gear_t_pos = driving_gear_init_conditions_dict["t_pos"] if "t_pos" in driving_gear_init_conditions_dict.keys() else 0.0
        driving_gear_t_vel = driving_gear_init_conditions_dict["t_vel"] if "t_vel" in driving_gear_init_conditions_dict.keys() else 0.0
        logger.debug("Driving gear initial conditions extracted.")

        logger.debug("Computing driven gear initial conditions...")
        logger.warning("Assuming that gamma_vel is 0...")
        driven_gear_x_pos, driven_gear_x_vel, driven_gear_y_pos, driven_gear_y_vel = self.get_driven_gear_position_and_velocity_from(
            driven_gear=driven_gear, 
            driving_gear=driving_gear, 
            driving_gear_x_pos=driving_gear_x_pos, 
            driving_gear_x_vel=driving_gear_x_vel, 
            driving_gear_y_pos=driving_gear_y_pos, 
            driving_gear_y_vel=driving_gear_y_vel, 
            gamma=gamma
        )
        driven_gear_t_pos, driven_gear_t_vel = self.get_driven_gear_angular_position_and_velocity_from(
            driven_gear=driven_gear,
            driving_gear=driving_gear,
            driving_gear_t_pos=driving_gear_t_pos,
            driving_gear_t_vel=driving_gear_t_vel,
            gamma=gamma,
            driving_gear_tooth_idx=driving_gear_tooth_idx,
            driven_gear_tooth_idx=driven_gear_tooth_idx
        )
        logger.debug("Driven gear initial conditions computed.")

        return {"x_pos": driven_gear_x_pos, "x_vel": driven_gear_x_vel, "y_pos": driven_gear_y_pos, "y_vel": driven_gear_y_vel, "t_pos": driven_gear_t_pos, "t_vel": driven_gear_t_vel}

    def get_driven_gear_position_and_velocity_from(self,
            driven_gear: "SpurGear", 
            driving_gear: "SpurGear", 
            driving_gear_x_pos: float,
            driving_gear_x_vel: float,
            driving_gear_y_pos: float,
            driving_gear_y_vel: float,
            gamma: float) -> tuple[float, float, float, float]:
        
        centres_distance = driving_gear.radiuses["pitch"] + driven_gear.radiuses["pitch"]
        driven_gear_x_pos = driving_gear_x_pos + centres_distance*np.cos(gamma)
        driven_gear_x_vel = driving_gear_x_vel
        driven_gear_y_pos = driving_gear_y_pos + centres_distance*np.sin(gamma)
        driven_gear_y_vel = driving_gear_y_vel
        logger.warning("Assuming that gamma_vel is 0...")
        return driven_gear_x_pos, driven_gear_x_vel, driven_gear_y_pos, driven_gear_y_vel

    def get_driven_gear_angular_position_and_velocity_from(self,
            driven_gear: "SpurGear", 
            driving_gear: "SpurGear", 
            driving_gear_t_pos: float,
            driving_gear_t_vel: float,
            gamma: float,
            driving_gear_tooth_idx: int = 0,
            driven_gear_tooth_idx: int = 0
            ) -> tuple[float, float]:
        
        driven_gear_t_pos = pi - driving_gear.teeth_number / driven_gear.teeth_number * driving_gear_t_pos + (driving_gear.teeth_number / driven_gear.teeth_number + 1) * gamma - pi / 2 / driven_gear.teeth_number * (4 * driving_gear_tooth_idx + 4 * driven_gear_tooth_idx - 6)
        driven_gear_t_vel = -driving_gear_t_vel * driving_gear.teeth_number / driven_gear.teeth_number
        return driven_gear_t_pos, driven_gear_t_vel

    def compute_teeth_engagement(self, teeth_position, mode, gamma, gear_rotation, other_gear): ### fix
        pass
        # csi_a2 = other_gear.compute_csi_angle(other_gear.diameter["addendum"])
        # a0 = self.radius["pitch"] + other_gear.radius["pitch"]
        # ro2 = np.sqrt(
        #     other_gear.radius["addendum"] ** 2
        #     + a0**2
        #     - 2
        #     * other_gear.radius["addendum"]
        #     * a0
        #     * np.cos(csi_a2 - self.pressureAngle)
        # )
        # phi_2 = self.compute_phi_angle(2 * ro2)
        # phi_a = self.compute_phi_angle(self.diameter["addendum"])

        # match (mode, gear_rotation):
        #     case ("driving", gr) if gr > 0:  #
        #         thetagi = wrapToPi(teethPosition - gamma)
        #         theta_start = wrapToPi(
        #             +(-self.pressureAngle + phi_2 - self.angle_at_base / 2)
        #         )
        #         theta_end = wrapToPi(
        #             +(-self.pressureAngle + phi_a - self.angle_at_base / 2)
        #         )
        #     case ("driving", gr) if gr < 0:
        #         thetagi = wrapToPi(teethPosition - gamma)
        #         theta_start = wrapToPi(
        #             -(-self.pressureAngle + phi_2 - self.angle_at_base / 2)
        #         )
        #         theta_end = wrapToPi(
        #             -(-self.pressureAngle + phi_a - self.angle_at_base / 2)
        #         )
        #     case ("driven", gr) if gr > 0:
        #         thetagi = wrapToPi(teethPosition - gamma - pi)
        #         theta_start = wrapToPi(
        #             -(self.pressureAngle - phi_a - self.angle_at_base / 2)
        #         )
        #         theta_end = wrapToPi(
        #             -(self.pressureAngle - phi_2 - self.angle_at_base / 2)
        #         )
        #     case ("driven", gr) if gr < 0:  #
        #         thetagi = wrapToPi(teethPosition - gamma - pi)
        #         theta_start = wrapToPi(
        #             +(self.pressureAngle - phi_a - self.angle_at_base / 2)
        #         )
        #         theta_end = wrapToPi(
        #             +(self.pressureAngle - phi_2 - self.angle_at_base / 2)
        #         )

        # # --- 6) Condizione di ingaggio corretta ---

        # engagement = np.logical_and(thetagi >= theta_start, thetagi < theta_end)

        # if not np.any(engagement):
        #     print("No engagement detected.")

        # # Angolo utile per cinematica
        # alfa1_i = -self.pressureAngle + thetagi

        # return engagement, alfa1_i

    def computeStiffnessAngles(self, t, x): ### fix
        """to be updated"""
        pass

    def meshStiffness(self, engagement, alpha1): ### fix
        # Parameters
        E = self.young
        L = self.thickness
        v = self.poisson

        num_points = 100

        # Define alfa range

        # phig_r - self.angle_at_base/2
        # np.linspace(0, 1, num_points)[None, :] * (stop - start)[:, None] + start[:, None]

        if self.root_greater_than_base:
            # Compute Ib, Is, Ia
            alpha = np.linspace(-alpha1, self.alpha[5], num_points)

            Ib = (
                (
                    3
                    * (
                        1
                        + np.cos(alpha1)
                        * ((self.alpha[2] - alpha) * np.sin(alpha) - np.cos(alpha))
                    )
                    ** 2
                    * (self.alpha[2] - alpha)
                    * np.cos(alpha)
                )
                / (
                    2
                    * E
                    * L
                    * (np.sin(alpha) + (self.alpha[2] - alpha) * np.cos(alpha)) ** 3
                )
                * engagement
            )
            Is = (
                (
                    1.2
                    * (1 + v)
                    * (self.alpha[2] - alpha)
                    * np.cos(alpha)
                    * np.cos(alpha1) ** 2
                )
                / (E * L * (np.sin(alpha) + (self.alpha[2] - alpha) * np.cos(alpha)))
                * engagement
            )
            Ia = (
                ((self.alpha[2] - alpha) * np.cos(alpha) * np.sin(alpha1) ** 2)
                / (
                    2
                    * E
                    * L
                    * (np.sin(alpha) + (self.alpha[2] - alpha) * np.cos(alpha))
                )
                * engagement
            )

        else:
            alpha = np.linspace(-alpha1, self.alpha[2], num_points)

            Ib0 = (
                ## primo termine
                (
                    (
                        1
                        - ((self.teethNumber - 2.5) * np.cos(alpha1) * np.cos(alpha[3]))
                        / (self.teethNumber * np.cos(self.pressureAngle))
                    )
                    ** 3
                    - (1 - np.cos(alpha1) * np.cos(alpha[2] ** 3))
                )
                / (2 * E * L * (np.cos(alpha1) * np.sin(alpha[2]) ** 3))
            )

            ## secondo termine

            Ib = (
                (
                    3
                    * (
                        1
                        + np.cos(alpha1)
                        * ((self.alpha[2] - alpha) * np.sin(alpha) - np.cos(alpha))
                    )
                    ** 2
                    * (self.alpha[2] - alpha)
                    * np.cos(alpha)
                )
                / (
                    2
                    * E
                    * L
                    * (np.sin(alpha) + (self.alpha[2] - alpha) * np.cos(alpha)) ** 3
                )
                * engagement
            )

            Is0 = (
                ## primo termine
                (
                    1.2
                    * (1 + v)
                    * np.cos(alpha1) ** 2
                    * (
                        np.cos(alpha[2])
                        - (
                            (self.teethNumber - 2.5)
                            / (self.teethNumber * np.cos(self.pressureAngle))
                        )
                        * np.cos(alpha[3])
                    )
                )
                / (E * L * (np.sin(alpha[2])))
            )

            ## secondo termine

            Is = (
                (
                    1.2
                    * (1 + v)
                    * (self.alpha[2] - alpha)
                    * np.cos(alpha)
                    * np.cos(alpha1) ** 2
                )
                / (E * L * (np.sin(alpha) + (self.alpha[2] - alpha) * np.cos(alpha)))
                * engagement
            )

            Ia0 = (
                ## primo termine
                (
                    np.sin(alpha1) ** 2
                    * (
                        np.cos(
                            alpha[2]
                            - (
                                (self.teethNumber - 2.5)
                                / (self.teethNumber * np.cos(self.pressureAngle))
                            )
                            * np.cos(alpha[3])
                        )
                    )
                )
                / (2 * E * L * (np.sin(alpha[2])))
            )

            ## secondo termine

            Ia = (
                ((self.alpha[2] - alpha) * np.cos(alpha) * np.sin(alpha1) ** 2)
                / (
                    2
                    * E
                    * L
                    * (np.sin(alpha) + (self.alpha[2] - alpha) * np.cos(alpha))
                )
                * engagement
            )

        # Integrate over alfa
        # Kb_inv = np.sum(sc.integrate.trapezoid(Ib, x=alpha, axis=0))
        # Ks_inv = np.sum(sc.integrate.trapezoid(Is, x=alpha, axis=0))
        # Ka_inv = np.sum(sc.integrate.trapezoid(Ia, x=alpha, axis=0))

        # if self.root_greater_than_base:
        #     kb_inv += Ib0
        #     ks_inv += Is0
        #     ka_inv += Ia0

        # # Total stiffness
        # Kt = 1 / (Kb_inv + Ks_inv + Ka_inv)
        # if all(engagement == 0):
        #     print("No engagement detected.")
        # return Kt

    def example_mesh_stiffness(self, t, x): ### fix
        return 100000 + 200 * np.sin(t)
    
    def example_mesh_damping(self, t, x): ### fix
        # return 0
        return 100 + 20 * np.sin(t)

    




    


    def plot(self,
             fig: figure,
             state_vector: NDArray,
             engaged_teeth = None,
             n_points_involutes: int = 10,
             n_points_tips: int = 5,
             ) -> tuple[figure, ColumnDataSource, GlyphRenderer]:
        
        x_pos = {}
        y_pos = {}
        t_pos = {}
        sources = {}
        renderers = {}
        for gear_name, gear in zip(self.gear_names, self.gears):
            gear_x_pos_idx, _ = self.get_state_idx_and_name(gear_name=gear_name, gear_state_name="x_pos")
            gear_y_pos_idx, _ = self.get_state_idx_and_name(gear_name=gear_name, gear_state_name="y_pos")
            gear_t_pos_idx, _ = self.get_state_idx_and_name(gear_name=gear_name, gear_state_name="t_pos")

            x_pos[gear_name] = state_vector[gear_x_pos_idx]
            y_pos[gear_name] = state_vector[gear_y_pos_idx]
            t_pos[gear_name] = state_vector[gear_t_pos_idx]

        for gear_name, gear in zip(self.gear_names, self.gears):
            fig, teeth_source, teeth_renderer = gear.teeth_plot(
                fig=fig,
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
                t_pos=t_pos[gear_name],
                engaged_teeth=engaged_teeth,
                n_points_involutes=n_points_involutes,
                n_points_tips=n_points_tips,
            )
            sources[f"{gear_name}_teeth"] = teeth_source
            renderers[f"{gear_name}_teeth"] = teeth_renderer

        for gear_name, gear in zip(self.gear_names, self.gears):
            fig, root_source, root_renderer = gear.teeth_root_plot(
                fig=fig,
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )
            sources[f"{gear_name}_root"] = root_source
            renderers[f"{gear_name}_root"] = root_renderer

        for gear_name, gear in zip(self.gear_names, self.gears):
            fig, ref_source, ref_renderer = gear.reference_circles_plot(
                fig=fig,
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )
            sources[f"{gear_name}_reference_circles"] = ref_source
            renderers[f"{gear_name}_reference_circles"] = ref_renderer

        return fig, sources, renderers

    def update_plot(self,
            sources: dict,
            state_vector: NDArray,
            engaged_teeth = None,
            n_points_involutes: int = 10,
            n_points_tips: int = 5,
        ) -> None:

        x_pos = {}
        y_pos = {}
        t_pos = {}
        for gear_name, gear in zip(self.gear_names, self.gears):
            gear_x_pos_idx, _ = self.get_state_idx_and_name(gear_name=gear_name, gear_state_name="x_pos")
            gear_y_pos_idx, _ = self.get_state_idx_and_name(gear_name=gear_name, gear_state_name="y_pos")
            gear_t_pos_idx, _ = self.get_state_idx_and_name(gear_name=gear_name, gear_state_name="t_pos")

            x_pos[gear_name] = state_vector[gear_x_pos_idx]
            y_pos[gear_name] = state_vector[gear_y_pos_idx]
            t_pos[gear_name] = state_vector[gear_t_pos_idx]

            gear.update_teeth_plot(
                source=sources[f"{gear_name}_teeth"],
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
                t_pos=t_pos[gear_name],
                engaged_teeth=engaged_teeth,
                n_points_involutes=n_points_involutes,
                n_points_tips=n_points_tips,
            )

            gear.update_teeth_root_plot(
                source=sources[f"{gear_name}_root"],
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )

            gear.update_reference_circles_plot(
                source=sources[f"{gear_name}_reference_circles"],
                x_pos=x_pos[gear_name],
                y_pos=y_pos[gear_name],
            )
    
    def get_gear(self, gear_name: str) -> tuple[int, SpurGear]:
        assert gear_name in self.gear_names, f"Gear '{gear_name}' not found in gearbox'{self.name}'."
        gear_idx = self.gear_names.index(gear_name)
        return gear_idx, self.gears[gear_idx]
    
    def get_state_idx_and_name(self, gear_name: str, gear_state_name: str) -> tuple[int, str]:
        logger.debug("Getting index of state '%s' for gear '%s'...", gear_state_name, gear_name)
        state_name = f"{gear_name}_{gear_state_name}"
        assert state_name in self.state_names, f"State '{gear_state_name}' not found in gear '{gear_name}'."
        state_idx = self.state_names.index(state_name)
        logger.debug("Index of state '%s' for gear '%s' obtained.", gear_state_name, gear_name)
        return state_idx, state_name