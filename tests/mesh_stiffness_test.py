from datetime import datetime
from pathlib import Path
from logging import getLogger
from progetto_gearbox.logging.logger_configuration import setup_logger
from progetto_gearbox.gear import SpurGear
from progetto_gearbox.gearbox import GearBox
from bokeh.plotting import figure, output_file, show
import numpy as np

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

def create_gear():
    logger.info("Creating the driving gear...")
    gear = SpurGear(name="driving_gear")
    gear.set_geometry(module=3.2 * 1e-3,teeth_number=31,thickness=0.0381)
    gear.set_material(young=2.068 * 1e11, poisson=0.3, density = 7900)
    gear.get_state_space()
    return gear

def create_pinion():
    logger.info("Creating the pinion...")
    gear = SpurGear(name="driven_pinion")
    gear.set_geometry(module=3.2 * 1e-3,teeth_number=19,thickness=0.0381)
    gear.set_material(young=2.068 * 1e11, poisson=0.3, density = 7900)
    gear.get_state_space()
    return gear

def create_gearbox():
    logger.info("Creating the gearbox...")
    gearbox = GearBox(name="gearbox")
    gear = create_gear()
    pinion = create_pinion()
    gearbox.add_gears([gear, pinion], [np.zeros(gear.teeth_number),np.zeros(pinion.teeth_number)])
    return gearbox


def main(doc=None, server=None):
    gearbox = create_gearbox()
    gearbox.get_state_space()

    driving_gear_initial_conditions = {
        "x_pos": 0,
        "x_vel": 0,
        "y_pos": 0,
        "y_vel": 0,
        "t_pos": 0,
        "t_vel": 0
    }
    initial_conditions = {
        "driving_gear": driving_gear_initial_conditions,
    }
    gearbox.add_meshing_constraint(driving_gear_name="driving_gear", driven_gear_name="driven_pinion", gamma=0)
    gearbox.set_initial_conditions(init_conditions_dict=initial_conditions)

    output_file = "plots/mesh_stiffness_test_plot.html"
    fig = figure(
        x_axis_label="Drving Gear Angle [°]",
        y_axis_label="Mesh Stiffness [N/m]"
    )

    gamma = 0
    driving_gear = gearbox.gears[0]
    driven_gear = gearbox.gears[1]
    centre_distance = driving_gear.radiuses["pitch"] + driven_gear.radiuses["pitch"]
    
    driving_angles = np.arange(0,2*np.pi,np.pi/1800)
    driven_angle_idx, _ = gearbox._get_state_idx_and_name("driven_pinion","t_pos")
    driven_angles = gearbox.initial_conditions[driven_angle_idx] - np.arange(0,2*np.pi,np.pi/1800) * driving_gear.teeth_number/driven_gear.teeth_number

    driving_mesh_stiffness = []
    driven_mesh_stiffness = []
    for driving_angle, driven_angle in zip(driving_angles,driven_angles):
        driving_gear_teeth_angles = driving_gear._get_teeth_centre_angle(driving_angle)
        driven_gear_teeth_angles = driven_gear._get_teeth_centre_angle(driven_angle)
        
        _, _, driving_gear_engagement = gearbox._compute_teeth_engagement(driving_gear, driven_gear, driving_gear_teeth_angles, "driving", +1, centre_distance, gamma)
        _, _, driven_gear_engagement = gearbox._compute_teeth_engagement(driven_gear, driving_gear, driven_gear_teeth_angles, "driven", -1, centre_distance, gamma)
                                
        driving_gear_teeth_engagement_angle = gearbox._compute_teeth_engagement_angle(gear=driving_gear,gear_teeth_pos=driving_gear_teeth_angles,gear_engagement=driving_gear_engagement,gear_mode="driving",gear_vel=+1,gamma=gamma)
        driven_gear_teeth_engagement_angle = gearbox._compute_teeth_engagement_angle(gear=driven_gear,gear_teeth_pos=driven_gear_teeth_angles,gear_engagement=driven_gear_engagement,gear_mode="driven",gear_vel=-1,gamma=gamma)

        driving_mesh_stiffness.append(gearbox._compute_mesh_stiffness(gear=driving_gear, gear_teeth_engagement_angle=driving_gear_teeth_engagement_angle))
        driven_mesh_stiffness.append(gearbox._compute_mesh_stiffness(gear=driven_gear, gear_teeth_engagement_angle=driven_gear_teeth_engagement_angle))
    mesh_stiffness = 1/(1/np.array(driving_mesh_stiffness) + 1/np.array(driven_mesh_stiffness))

    fig.line(x=np.rad2deg(driving_angles), y=driving_mesh_stiffness, line_color="black", legend_label="driving")
    fig.line(x=np.rad2deg(driving_angles), y=driven_mesh_stiffness, line_color="blue", legend_label="driven")
    fig.line(x=np.rad2deg(driving_angles), y=mesh_stiffness, line_color="red", legend_label="total")
    show(fig)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e