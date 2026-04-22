from datetime import datetime
from pathlib import Path
from logging import getLogger
from progetto_gearbox.logging.logger_configuration import setup_logger
from progetto_gearbox.gear import SpurGear
from progetto_gearbox.gearbox import GearBox
from progetto_gearbox.simulation import Simulation
from progetto_gearbox.utils.input_functions import sinusoidal, step, ramp, constant

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

def create_gear():
    logger.info("Creating the driving gear...")
    gear = SpurGear(name="driving_gear")
    gear.set_geometry(module=3.2 * 1e-3,teeth_number=31,thickness=0.0381)
    gear.set_material(young=2.068 * 1e5, poisson=0.3, density = 7900)
    # gear.set_inertias(inertia_x=1,inertia_y=1,inertia_t=0.5)
    gear.get_state_space()
    return gear

def create_pinion():
    logger.info("Creating the pinion...")
    gear = SpurGear(name="driven_pinion")
    gear.set_geometry(module=3.2 * 1e-3,teeth_number=19,thickness=0.0381)
    gear.set_material(young=2.068 * 1e5, poisson=0.3, density = 7900)
    # gear.set_inertias(inertia_x=0.5,inertia_y=0.5,inertia_t=2)
    gear.get_state_space()
    return gear

def create_gearbox():
    logger.info("Creating the gearbox...")
    gearbox = GearBox(name="gearbox")
    gear = create_gear()
    pinion = create_pinion()
    gearbox.add_gears([gear, pinion])
    # gearbox.add_ground_constraint(gear_name="driven_pinion", dofs=["x","y"])
    # gearbox.add_ground_constraint(gear_name="driving_gear", dofs=["x","y"])
    return gearbox


def main(doc=None, server=None):
    gearbox = create_gearbox()
    driving_gear_initial_conditions = {
        "x_pos": 0,
        "x_vel": 0,
        "y_pos": 0,
        "y_vel": 0,
        "t_pos": 0,
        "t_vel": 0
    }
    # driven_pinion_initial_conditions = gearbox.get_driven_gear_initial_conditions_from(
    #     driven_gear_name="driven_pinion", 
    #     driving_gear_name="driving_gear", 
    #     driving_gear_init_conditions_dict=driving_gear_initial_conditions, 
    #     gamma=0.0)
    
    driven_pinion_initial_conditions = {
        "x_pos": 80 * 1e-3,
        "x_vel": 0,
        "y_pos": 0,
        "y_vel": 0,
        "t_pos": 0,
        "t_vel": 0
    }

    initial_conditions = {
        "driving_gear": driving_gear_initial_conditions,
        "driven_pinion": driven_pinion_initial_conditions,
    }

    gearbox.set_initial_conditions(init_conditions_dict=initial_conditions)

    
    # input_functions = {"driving_gear": {"t_torque": PD(kp=100, pos_set = 10, pos_idx = gearbox.get_state_idx("driving_gear_t_vel"))}}
    gearbox.add_proportional_derivative_feedback(gear_name="driving_gear", dof = "t", kp = 0, kd = 100)
    input_functions = {
        "driving_gear": {
            # "t_torque": constant(value = 0.1)
            "t_vel_ref": step(step_value = 1, t_start = 0.1)
            },
        "driven_pinion": {
            "t_torque": constant(value=1)
        }
    }
    gearbox.set_input_functions(input_func_dict=input_functions)
    gearbox.get_state_space()
    
    gearbox.add_spring_damper(to_gear_name="driving_gear", stiffness={"x": 10000, "y":10000}, damping={"x": 10, "y": 10})
    gearbox.add_spring_damper(to_gear_name="driven_pinion", stiffness={"x": 10000, "y":10000}, damping={"x": 10, "y": 10}, origin={"x": 80*1e-3})
    gearbox.add_meshing_constraint(driving_gear_name="driving_gear", driven_gear_name="driven_pinion")
    
    gearbox_simulation = Simulation(
        t_tot=1.0,
        deltat=0.00005,
        model=gearbox,
    )
    
    gearbox_simulation.solve()
    gearbox_simulation.plot_initial_conditions()
    gearbox_simulation.plot()
    gearbox_simulation.animate(doc=doc, server=server)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e