from progetto_gearbox.interfaces.simulation_interfaces import Model
import numpy as np
from datetime import datetime
from pathlib import Path
from progetto_gearbox.logging.logger_configuration import setup_logger
from logging import getLogger
from progetto_gearbox.utils.input_functions import step
from progetto_gearbox.simulation import Simulation

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

class SimpleGearboxModel(Model):

    def from_parameters(
            self, 
            m1, kx1, cx1,
            m2, kx2, cx2,
            ky1, cy1, kt, rb1, rb2,
            ct,
            ky2, cy2,
            J1, kp, cp, 
            J2, kg, cg,
            Jm, Jb):
        
        self.m1 = m1
        self.kx1 = kx1
        self.cx1 = cx1
        self.m2 = m2
        self.kx2 = kx2
        self.cx2 = cx2
        self.ky1 = ky1
        self.cy1 = cy1
        self.kt = kt
        self.rb1 = rb1
        self.rb2 = rb2
        self.ct = ct
        self.ky2 = ky2
        self.cy2 = cy2
        self.J1 = J1
        self.kp = kp
        self.cp = cp
        self.J2 = J2
        self.kg = kg
        self.cg = cg
        self.Jm = Jm
        self.Jb = Jb
        self.get_state_space()


    def get_state_space(self):
        logger.info("Setting up the state space model...")
        self.state_names = ["x1_vel", "x1_pos", "x2_vel", "x2_pos",
                            "y1_vel", "y1_pos", "y2_vel", "y2_pos",
                            "t1_vel", "t1_pos", "t2_vel", "t2_pos",
                            "tm_vel", "tm_pos", "tb_vel", "tb_pos"]
        self.state_uoms = ["m/s", "m","m/s", "m",
                           "m/s", "m", "m/s", "m",
                           "rad/s", "rad", "rad/s", "rad",
                           "rad/s", "rad", "rad/s", "rad"]
        self.input_names = ["M1", "M2"]
        self.input_uoms = ["Nm", "Nm"]
        self.input_funcs = [None, None]
        self.output_names = ["x1_vel", "x1_pos", "x2_vel", "x2_pos",
                            "y1_vel", "y1_pos", "y2_vel", "y2_pos",
                            "t1_vel", "t1_pos", "t2_vel", "t2_pos",
                            "tm_vel", "tm_pos", "tb_vel", "tb_pos"]
        self.output_uoms = ["m/s", "m","m/s", "m",
                           "m/s", "m", "m/s", "m",
                           "rad/s", "rad", "rad/s", "rad",
                           "rad/s", "rad", "rad/s", "rad"]
        self.get_matrices_dimensions()

        self.state_transition_matrix = np.zeros((16,16))
        self.state_transition_matrix[0,0] = - self.cx1 / self.m1
        self.state_transition_matrix[0,1] = - self.kx1 / self.m1
        self.state_transition_matrix[1,0] = 1
        self.state_transition_matrix[2,2] = - self.cx1 / self.m1
        self.state_transition_matrix[2,3] = - self.kx1 / self.m1
        self.state_transition_matrix[3,2] = 1
        self.state_transition_matrix[4,4] = ( - self.cy1 - self.ct) / self.m1
        self.state_transition_matrix[4,5] = ( - self.ky1 - self.kt) / self.m1
        self.state_transition_matrix[4,8] = self.ct * self.rb1 / self.m1
        self.state_transition_matrix[4,9] = self.kt * self.rb1 / self.m1
        self.state_transition_matrix[4,10] = - self.ct * self.rb2 / self.m1
        self.state_transition_matrix[4,11] = - self.kt * self.rb2 / self.m1
        self.state_transition_matrix[5,4] = 1
        self.state_transition_matrix[6,4] = ( - self.cy2 - self.ct) / self.m2
        self.state_transition_matrix[6,5] = ( - self.ky2 - self.kt) / self.m2
        self.state_transition_matrix[6,8] = self.ct * self.rb1 / self.m2
        self.state_transition_matrix[6,9] = self.kt * self.rb1 / self.m2
        self.state_transition_matrix[6,10] = - self.ct * self.rb2 / self.m2
        self.state_transition_matrix[6,11] = - self.kt * self.rb2 / self.m2
        self.state_transition_matrix[7,6] = 1
        self.state_transition_matrix[8,4] = + self.rb1 * self.ct / self.J1
        self.state_transition_matrix[8,5] = + self.rb1 * self.kt / self.J1
        self.state_transition_matrix[8,6] = - self.rb1 * self.ct / self.J1
        self.state_transition_matrix[8,7] = - self.rb1 * self.kt / self.J1
        self.state_transition_matrix[8,8] = ( - self.cp - self.rb1 **2 * self.ct ) / self.J1 
        self.state_transition_matrix[8,9] = ( - self.kp - self.rb1 **2 * self.kt ) / self.J1
        self.state_transition_matrix[8,10] = + self.rb1 * self.rb2 * self.ct / self.J1 
        self.state_transition_matrix[8,11] = + self.rb1 * self.rb2 * self.kt / self.J1
        self.state_transition_matrix[8,12] =  self.cp / self.J1 
        self.state_transition_matrix[8,13] =  self.kp / self.J1
        self.state_transition_matrix[9,8] = 1
        self.state_transition_matrix[10,4] = - self.rb2 * self.ct / self.J2
        self.state_transition_matrix[10,5] = - self.rb2 * self.kt / self.J2
        self.state_transition_matrix[10,6] = + self.rb2 * self.ct / self.J2
        self.state_transition_matrix[10,7] = + self.rb2 * self.kt / self.J2
        self.state_transition_matrix[10,8] = self.rb2 * self.rb1 * self.ct / self.J2
        self.state_transition_matrix[10,9] = self.rb2 * self.rb1 * self.kt / self.J2
        self.state_transition_matrix[10,10] = ( - self.cg - self.rb2 **2 * self.ct ) / self.J2
        self.state_transition_matrix[10,11] = ( - self.kg - self.rb2 **2 * self.kt ) / self.J2
        self.state_transition_matrix[10,14] =  self.cg / self.J2
        self.state_transition_matrix[10,15] =  self.kg / self.J2
        self.state_transition_matrix[11,10] = 1
        self.state_transition_matrix[12,8] = self.cp / self.Jm
        self.state_transition_matrix[12,9] = self.kp / self.Jm
        self.state_transition_matrix[12,12] = - self.cp / self.Jm
        self.state_transition_matrix[12,13] = - self.kp / self.Jm
        self.state_transition_matrix[13,12] = 1
        self.state_transition_matrix[14,10] = self.cg / self.Jb
        self.state_transition_matrix[14,11] = self.kg / self.Jb
        self.state_transition_matrix[14,14] = - self.cg / self.Jb
        self.state_transition_matrix[14,15] = - self.kg / self.Jb
        self.state_transition_matrix[15,14] = 1

        self.input_matrix = np.zeros((16,2))
        self.input_matrix[12,0] = 1 / self.Jm
        self.input_matrix[14,0] = - 1 / self.Jb

        self.output_matrix = np.eye(16)
        self.feedthrough_matrix = np.zeros((16, 2))
        self.check_matrices_dimensions()
        self.initial_conditions = np.full(self.ns,None)

    def plot():
        pass
    
    def update_plot():
        pass

    def animate():
        pass

def create_simple_gearbox() -> SimpleGearboxModel:
    name = "simple_gearbox"
    simple_gearbox = SimpleGearboxModel(name=name)
    simple_gearbox.from_parameters(
            m1 = 10,
            kx1 = 100000,
            cx1 = 0.01,
            m2 = 20,
            kx2 = 100000,
            cx2 = 0.01,
            ky1 = 100000,
            cy1 = 0.01,
            kt = 100000,
            rb1 = 0.2,
            rb2 = 0.4,
            ct = 0.01,
            ky2 = 100000,
            cy2 = 0.01,
            J1 = 10*0.2**2, 
            kp = 10000,
            cp = 0.01, 
            J2 = 20*0.4**2,
            kg = 20000,
            cg = 0.02,
            Jm = 15*0.1**2,
            Jb = 30*0.2**2 
        )
    return simple_gearbox


def main():
    simple_gearbox = create_simple_gearbox()

    initial_conditions = {}
    input_funcs = {"M1": step(step_value=10, t_start=2, t_end=5),
                   "M2": step(step_value=10, t_start=2, t_end=5)}

    simple_gearbox.set_initial_conditions(init_conditions_dict=initial_conditions)
    simple_gearbox.set_input_functions(input_func_dict=input_funcs)

    simple_gearbox_simulation = Simulation(
        t_tot=10.0,
        deltat=0.0005,
        model=simple_gearbox,
    )
    simple_gearbox_simulation.solve()
    simple_gearbox_simulation.plot()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e