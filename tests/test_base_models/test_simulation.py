from pathlib import Path
from logging import getLogger
from datetime import datetime
from progetto_gearbox.logging.logger_configuration import setup_logger
from progetto_gearbox.interfaces.simulation_interfaces import Model
from progetto_gearbox.utils.input_functions import sinusoidal
from progetto_gearbox.simulation import Simulation
import numpy as np

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

class OneDofSecondOrderModel(Model):

    def from_dimensional_parameters(self, mass: float, damping: float, stiffness: float):
        self.get_state_space(mass=mass, damping=damping, stiffness=stiffness)

    def from_adimensional_parameters(self, mass: float, critical_damping: float, natural_frequency: float):
        natural_pulsation = 2 * np.pi * natural_frequency
        stiffness = natural_pulsation**2 * mass
        damping = critical_damping * (2 * mass * natural_pulsation)
        self.get_state_space(mass=mass, damping=damping, stiffness=stiffness)

    def get_state_space(self, mass: float, damping: float, stiffness: float):
        logger.info("Setting up the state space model...")
        assert(isinstance(mass, float))
        assert(isinstance(damping, float))
        assert(isinstance(stiffness, float))
        self.state_names = ["vel", "pos"]
        self.state_uoms = ["m/s", "m"]
        self.input_names = ["pos"]
        self.input_uoms = ["N"]
        self.input_funcs = [None]
        self.output_names = ["vel", "pos"]
        self.output_uoms = ["m/s", "m"]
        self.get_matrices_dimensions()

        self.state_transition_matrix = np.array(
            [[-damping / mass, -stiffness / mass], [1, 0]]
        )
        self.input_matrix = np.array([[1 / mass], [0]])
        self.output_matrix = np.eye(2)
        self.feedthrough_matrix = np.zeros((2, 1))
        self.check_matrices_dimensions()
        self.initial_conditions = np.full(self.ns,None)

    def plot():
        pass

    def animate():
        pass



def create_underdamped_system() -> OneDofSecondOrderModel:
    name = "underdamped_system"
    mass = 1.0  # kg
    critical_damping = 0.1  # -
    natural_frequency = 10.0  # Hz
    underdamped_model = OneDofSecondOrderModel(name=name)
    underdamped_model.from_adimensional_parameters(
        mass=mass,
        critical_damping=critical_damping,
        natural_frequency=natural_frequency,
    )
    return underdamped_model


def create_critically_damped_system() -> OneDofSecondOrderModel:
    name = "critically_damped_system"
    mass = 1.0  # kg
    critical_damping = 1.0  # -
    natural_frequency = 10.0  # Hz
    critically_damped_model = OneDofSecondOrderModel(name=name)
    critically_damped_model.from_adimensional_parameters(
        mass=mass,
        critical_damping=critical_damping,
        natural_frequency=natural_frequency,
    )
    return critically_damped_model


def create_overdamped_system() -> OneDofSecondOrderModel:
    name = "overdamped_system"
    mass = 1.0  # kg
    critical_damping = 3.0  # -
    natural_frequency = 10.0  # Hz
    overdamped_model = OneDofSecondOrderModel(name=name)
    overdamped_model.from_adimensional_parameters(
        mass=mass,
        critical_damping=critical_damping,
        natural_frequency=natural_frequency,
    )
    return overdamped_model


def simulate_underdamped_system() -> Simulation:
    initial_conditions = {"pos": 0.2, "vel": 1.0}
    input_funcs = {"pos": sinusoidal(amplitude=10, frequency=5)}

    underdamped_model = create_underdamped_system()
    underdamped_model.set_initial_conditions(init_conditions_dict=initial_conditions)
    underdamped_model.set_input_functions(input_func_dict=input_funcs)
    
    underdamped_model_simulation = Simulation(
        t_tot=5.0,
        deltat=0.0005,
        model=underdamped_model,
    )
    underdamped_model_simulation.solve()
    underdamped_model_simulation.plot()
    return underdamped_model_simulation


def simulate_critically_damped_system() -> Simulation:
    initial_conditions = {"pos": 0.2}
    input_funcs = {"pos": sinusoidal(amplitude=10, frequency=5)}

    critically_damped_model = create_critically_damped_system()
    critically_damped_model.set_initial_conditions(init_conditions_dict=initial_conditions)
    critically_damped_model.set_input_functions(input_func_dict=input_funcs)
    
    critically_damped_model_simulation = Simulation(
        t_tot=5.0,
        deltat=0.0005,
        model=critically_damped_model,
    )
    critically_damped_model_simulation.solve()
    critically_damped_model_simulation.plot()
    return critically_damped_model_simulation


def simulate_overdamped_system() -> Simulation:
    initial_conditions = {"pos": 0.2, "vel": 1.0}
    input_funcs = {"pos": sinusoidal(amplitude=10, frequency=5)}

    overdamped_model = create_overdamped_system()
    overdamped_model.set_initial_conditions(init_conditions_dict=initial_conditions)
    overdamped_model.set_input_functions(input_func_dict=input_funcs)
    
    overdamped_model_simulation = Simulation(
        t_tot=5.0,
        deltat=0.0005,
        model=overdamped_model,
    )
    overdamped_model_simulation.solve()
    overdamped_model_simulation.plot()
    return overdamped_model_simulation


def main():
    logger.info("Simulation of the underdamped system...")
    underdamped_model_simulation = simulate_underdamped_system()

    logger.info("Simulation of the critically damped system...")
    critically_damped_model_simulation = simulate_critically_damped_system()

    logger.info("Simulation of the overdamped system...")
    overdamped_model_simulation = simulate_overdamped_system()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
