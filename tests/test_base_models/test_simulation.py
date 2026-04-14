from progetto_gearbox.interfaces.simulation_interfaces import Model
from progetto_gearbox.simulation import Simulation
import numpy as np


class OneDofSecondOrderModel(Model):
    def __init__(self, mass: float, damping: float, stiffness: float):
        self.init_linear_model()
        self.state_names = ["vel", "pos"]
        self.state_uoms = ["m/s", "m"]
        self.input_names = ["pos"]
        self.input_uoms = ["N"]
        self.output_names = ["vel", "pos"]
        self.output_uoms = ["m/s", "m"]
        self.get_matrices_dimensions()

        self.input_funcs = {"pos": lambda t: 10 * np.sin(t)}
        self.state_transition_matrix = np.array(
            [[-damping / mass, -stiffness / mass], [1, 0]]
        )
        self.input_matrix = np.array([[1 / mass], [0]])
        self.output_matrix = np.eye(2)
        self.feedthrough_matrix = np.zeros((2, 1))
        self.check_matrices_dimensions()
        self.initial_conditions = np.zeros(self.ns)

    @classmethod
    def from_adimensional_parameters(
        cls, mass: float, critical_damping: float, natural_frequency: float
    ):
        natural_pulsation = 2 * np.pi * natural_frequency
        stiffness = natural_pulsation**2 * mass
        damping = critical_damping * (2 * mass * natural_pulsation)
        return cls(mass, damping, stiffness)


def create_underdamped_system() -> OneDofSecondOrderModel:
    mass = 1  # kg
    critical_damping = 0.1  # -
    natural_frequency = 10  # Hz
    model = OneDofSecondOrderModel.from_adimensional_parameters(
        mass=mass,
        critical_damping=critical_damping,
        natural_frequency=natural_frequency,
    )
    return model


def create_critically_damped_system() -> OneDofSecondOrderModel:
    mass = 1  # kg
    critical_damping = 1  # -
    natural_frequency = 10  # Hz
    model = OneDofSecondOrderModel.from_adimensional_parameters(
        mass=mass,
        critical_damping=critical_damping,
        natural_frequency=natural_frequency,
    )
    return model


def create_overdamped_system() -> OneDofSecondOrderModel:
    mass = 1  # kg
    critical_damping = 1.5  # -
    natural_frequency = 10  # Hz
    model = OneDofSecondOrderModel.from_adimensional_parameters(
        mass=mass,
        critical_damping=critical_damping,
        natural_frequency=natural_frequency,
    )
    return model


def simulate_underdamped_system() -> Simulation:
    initial_conditions = {"pos": 0.2, "vel": 1.0}
    underdamped_model = create_underdamped_system()
    underdamped_model.set_initial_conditions(init_conditions_dict=initial_conditions)
    underdamped_model_simulation = Simulation(
        name="underdamped_model_simulation",
        t_tot=5.0,
        deltat=0.0005,
        model=underdamped_model,
    )
    underdamped_model_simulation.solve()
    underdamped_model_simulation.plot()
    return underdamped_model_simulation


def simulate_critically_damped_system() -> Simulation:
    initial_conditions = {"pos": 0.2}
    critically_damped_model = create_critically_damped_system()
    critically_damped_model.set_initial_conditions(
        init_conditions_dict=initial_conditions
    )
    critically_damped_model_simulation = Simulation(
        name="critically_damped_model_simulation",
        t_tot=5.0,
        deltat=0.0005,
        model=critically_damped_model,
    )
    critically_damped_model_simulation.solve()
    critically_damped_model_simulation.plot()
    return critically_damped_model_simulation


def simulate_overdamped_system() -> Simulation:
    initial_conditions = {"pos": 0.2}
    overdamped_model = create_critically_damped_system()
    overdamped_model.set_initial_conditions(init_conditions_dict=initial_conditions)
    overdamped_model_simulation = Simulation(
        name="overdamped_model_simulation",
        t_tot=5.0,
        deltat=0.0005,
        model=overdamped_model,
    )
    overdamped_model_simulation.solve()
    overdamped_model_simulation.plot()
    return overdamped_model_simulation


def main():
    underdamped_model_simulation = simulate_underdamped_system()
    critically_damped_model_simulation = simulate_critically_damped_system()
    overdamped_model_simulation = simulate_overdamped_system()


if __name__ == "__main__":
    main()
