from datetime import datetime
from pathlib import Path
from logging import getLogger
from progetto_gearbox.logging.logger_configuration import setup_logger
from progetto_gearbox.utils.input_functions import sinusoidal, step, ramp
from progetto_gearbox.simulation import Simulation
from progetto_gearbox.gear import SpurGear
from numpy import pi, deg2rad


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

def create_gear() -> SpurGear:
    logger.info("Creating a gear...")
    gear = SpurGear(name="gear")
    gear.set_material(young=2.068 * 1e11, poisson=0.3)
    gear.set_geometry(module=3.2,teeth_number=31,thickness=0.0381)
    gear.set_inertias(inertia_x=1,inertia_y=1,inertia_t=0.5)
    gear.get_state_space()
    return gear

def simulate_gear() -> Simulation:
    logger.info("Simulating a gear...")
    initial_conditions = {"x_pos": 0, "x_vel": 0}
    input_funcs = {
        "x_force": sinusoidal(amplitude=5,frequency=5, phase=pi/2),
        "t_torque": step(step_value=10)}
    
    gear = create_gear()
    gear.set_initial_conditions(init_conditions_dict=initial_conditions)
    gear.set_input_functions(input_func_dict=input_funcs)
    
    gear_simulation = Simulation(
        t_tot=5.0,
        deltat=0.0005,
        model=gear,
    )
    gear_simulation.solve()
    gear_simulation.plot()
    return gear_simulation


def main(doc=None, server=None):
    gear_simulation = simulate_gear()
    gear_simulation.plot_initial_conditions()
    gear_simulation.animate(doc=doc, server=server)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e