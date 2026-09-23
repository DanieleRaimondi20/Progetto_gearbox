from bokeh.plotting import figure, show
from matplotlib.patches import Circle
import numpy as np
from progetto_gearbox.gearbox import _find_alpha_bisection
from progetto_gearbox.demos.gearbox_demo import create_gear
from progetto_gearbox.demos.gearbox_demo import create_pinion
from datetime import datetime
from pathlib import Path
from logging import getLogger
from progetto_gearbox.logging.logger_configuration import setup_logger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.linalg import block_diag
from scipy.integrate import trapezoid


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

driving_gear= create_gear()
#test con create pinion

hr = 2 * driving_gear.radiuses["root"] * np.sin(driving_gear.alpha[3])
hb = 2 * driving_gear.radiuses["base"] * np.sin(driving_gear.alpha[2])
alpha0 = driving_gear.alpha[2] - driving_gear._compute_phi_angle(driving_gear.diameters["addendum"])
h0 = driving_gear.radiuses["base"] * (np.sin(alpha0) + (driving_gear.alpha[2] - alpha0) * np.cos(alpha0))

#h0 = 0.0011844071314226702
#tooth_damage = 0.1
tooth_damage = 0.5 - h0/hr
hq = tooth_damage * hr
v = np.pi/6

ha = hr/2 - hq

# alpha_k, rho, phi, iterations = _find_alpha_bisection(
#     min_radius = driving_gear.radiuses["base"],
#     max_radius = driving_gear.radiuses["addendum"],
#     base_radius = driving_gear.radiuses["base"],
#     alpha2 = driving_gear.alpha[2],
#     h_target = h0 + 0.00001,
#     )

alpha_I = alpha0
# alpha_J = - alpha_k
driving_gear.alpha[1] = alpha0

if tooth_damage <= 0.5:
    
    q1 = hq/np.sin(v) 
    ha = hr/2 - hq
    
    if ha >= h0:
        alpha_k, rho, phi, iterations = _find_alpha_bisection(
            min_radius = driving_gear.radiuses["base"],
            max_radius = driving_gear.radiuses["addendum"],
            base_radius = driving_gear.radiuses["base"],
            alpha2 = driving_gear.alpha[2],
            h_target = ha,
        )
        print("alpha_a =", alpha_k, "rho =", rho, "phi =", phi)
        alphaa = alpha_k
        rho0 = 2.0
        r_root = np.linspace(0, driving_gear.radiuses["root"], 20)
        r_base = np.linspace(0, driving_gear.radiuses["base"], 20)
        r_evolv = np.linspace(driving_gear.radiuses["base"], driving_gear.radiuses["addendum"], 100)

        psi_evolv = driving_gear._compute_psi_angle(2*r_evolv)
        x_evolv = r_evolv * np.cos(driving_gear.alpha[2] - psi_evolv)
        y_evolv = r_evolv * np.sin(driving_gear.alpha[2] - psi_evolv)

        theta = driving_gear.alpha[2] - 0.1 #driving_gear.psi
        x_root = r_root * np.cos(driving_gear.alpha[3])
        y_root = r_root * np.sin(driving_gear.alpha[3])
        x_base = r_base * np.cos(driving_gear.alpha[2])
        y_base = r_base * np.sin(driving_gear.alpha[2])
        x_rho = rho * np.cos(alpha_k)
        y_rho = rho * np.sin(alpha_k)
    
        # Cerchio base e cerchio root
        fig, ax = plt.subplots(figsize=(7,7))
    
        circle_base = Circle((0,0), driving_gear.radiuses["base"], fill=False, linestyle='--', color='k')
        ax.add_patch(circle_base)

        # Cerchio root
        circle_root = Circle((0,0), driving_gear.radiuses["root"], fill=False, linestyle='--', color='k')
        ax.add_patch(circle_root)

        # Raggi
        ax.plot(x_root, y_root, 'r--', linewidth=1, label='r_root')
        ax.plot(x_base, y_base, 'b--', linewidth=1, label='r_base')

        # Linea orizzontale (se serve)
        #ax.axhline(0, color='gray', linewidth=1)
        ax.hlines(y=ha, xmin=0, xmax=0.05, color='green', linewidth=1, linestyle='--')

        mx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3])
        my = driving_gear.radiuses["root"] * np.sin(driving_gear.alpha[3])

        qx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3]) - q1 * np.sin(np.pi/2 - v)
        qy = driving_gear.radiuses["root"] * np.sin(driving_gear.alpha[3]) - q1 * np.cos(np.pi/2 - v)

        nx = driving_gear.radiuses["base"] * np.cos(driving_gear.alpha[2])
        ny = driving_gear.radiuses["base"] * np.sin(driving_gear.alpha[2])

        ax.plot([mx, qx], [my, qy], linestyle='--', color='purple', linewidth=1)
        ax.plot([mx, nx], [my, ny], linestyle='--', color='red', linewidth=1)
        ax.plot(x_evolv, y_evolv, linestyle='-.', color='blue', linewidth=1)
    
        ox = 0
        oy = 0
    
        alpha1 = - 0.3
    
        aax = driving_gear.radiuses["base"] * np.cos(- alpha1)
        aay = driving_gear.radiuses["base"] * np.sin(- alpha1)
    
        h1 = driving_gear.radiuses["base"] * (np.sin(alpha1) + (driving_gear.alpha[2] - alpha1) * np.cos(alpha1))

        #fx = driving_gear.radiuses["base"]*np.cos(alpha1) - driving_gear.radiuses["base"] * (driving_gear.alpha[2] - alpha1)* np.sin(alpha1)  - driving_gear.radiuses["root"] * np.cos(alpha1)  
        #fy = h1
    
        f1x = driving_gear.radiuses["base"]*np.cos(alpha1)
        f1y = driving_gear.radiuses["base"]*np.sin(alpha1)
    
        fx = - h1*np.tan(alpha1) + f1y*np.tan(alpha1) + f1x
        fy = h1
    
        deltah = 0.3
    
        f2x = - (h1 + deltah)*np.tan(alpha1) + f1y*np.tan(alpha1) + f1x

        f2y = h1 + deltah
    
        #ax.plot([f1x, f2x], [f1y, f2y], linestyle='--', color='orange', linewidth=1)
        ax.plot([ox, f1x], [oy, f1y], linestyle='--', color='orange', linewidth=1)
        ax.plot([f1x, fx], [f1y, fy], linestyle='--', color='orange', linewidth=1)
        ax.plot([fx, f2x], [fy, f2y], linestyle='--', color='blue', linewidth=1)
    
        ax.plot([ox, aax], [oy, aay], linestyle='--', color='red', linewidth=1)
    
        # for rho_it, alpha_it, phi_it in iterations:
        #     x_iter = rho_it * np.cos(alpha_it)
        #     y_iter = rho_it * np.sin(alpha_it)
        #     ax.scatter(x_iter, y_iter, color='orange', s=20, label='Iterazione' if 'Iterazione' not in ax.get_legend_handles_labels()[1] else None)
        ax.plot([0, x_rho], [0, y_rho], 'g--', linewidth=1, label='rho')


        # Aspetto grafico
        ax.set_aspect('equal', 'box')
        #ax.set_xlim(left=0)
        #ax.set_ylim(0.002, 0.004)
        ax.grid(True)
        ax.legend()
        ax.set_title("Geometria del modello")

        plt.show()
     
        # if driving_gear.alpha[1] >= driving_gear.alpha[2]:
        #     print("condition 1")
        
        # elif driving_gear.alpha[1] < alphaa: #"condition 2"
        #     print("condition 2")
        
        # Cerchio base e cerchio root
        fig, ax = plt.subplots(figsize=(7,7))
    
        circle_base = Circle((0,0), driving_gear.radiuses["base"], fill=False, linestyle='--', color='k')
        ax.add_patch(circle_base)

        # Cerchio root
        circle_root = Circle((0,0), driving_gear.radiuses["root"], fill=False, linestyle='--', color='k')
        ax.add_patch(circle_root)

        # Raggi
        ax.plot(x_root, y_root, 'r--', linewidth=1, label='r_root')
        ax.plot(x_base, y_base, 'b--', linewidth=1, label='r_base')

        # Punto P
        #ax.plot(Px, Py, 'bo', label='P = (rho0 cos a, rho0 sin a)')

        # Linea orizzontale (se serve)
        #ax.axhline(0, color='gray', linewidth=1)
        ax.hlines(y=ha, xmin=0, xmax=0.06, color='green', linewidth=1, linestyle='--')

        mx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3])
        my = driving_gear.radiuses["root"] * np.sin(driving_gear.alpha[3])

        qx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3]) - q1 * np.sin(np.pi/2 - v)
        qy = driving_gear.radiuses["root"] * np.sin(driving_gear.alpha[3]) - q1 * np.cos(np.pi/2 - v)

        nx = driving_gear.radiuses["base"] * np.cos(driving_gear.alpha[2])
        ny = driving_gear.radiuses["base"] * np.sin(driving_gear.alpha[2])

        ax.plot([mx, qx], [my, qy], linestyle='--', color='purple', linewidth=1)
        ax.plot([mx, nx], [my, ny], linestyle='--', color='red', linewidth=1)
        ax.plot(x_evolv, y_evolv, linestyle='-.', color='blue', linewidth=1)

        # for rho_it, alpha_it, phi_it in iterations:
        #     x_iter = rho_it * np.cos(alpha_it)
        #     y_iter = rho_it * np.sin(alpha_it)
        #     ax.scatter(x_iter, y_iter, color='orange', s=20, label='Iterazione' if 'Iterazione' not in ax.get_legend_handles_labels()[1] else None)
        ax.plot([0, x_rho], [0, y_rho], 'g--', linewidth=1, label='rho')


        # Aspetto grafico
        ax.set_aspect('equal', 'box')
        ax.set_xlim(left=0)
        #ax.set_ylim(0.002, 0.004)
        ax.grid(True)
        ax.legend()
        ax.set_title("Geometria del modello")

        plt.show()
        
        if driving_gear.alpha[1] > alphaa:
            Ib0 = (12*np.sin(driving_gear.alpha)*((driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))/(driving_gear.teethNumber-2.5)-(np.cos(driving_gear.alpha)+np.cos(driving_gear.alpha[3])-np.cos(alphar)-(q1*np.cos(v))/(driving_gear.radiuses["root"]))*np.cos(driving_gear.alpha[1]))**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[3])+np.sin(driving_gear.alpha)-(q1*np.sin(v))/(driving_gear.radiuses["root"]))**3)
            Ib1 = (4*(1-((driving_gear.teethNumber-2.5)*np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[3]))/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle)))**3-4(1-np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[1])**3))/(driving_gear.young * driving_gear.thickness *np.cos(driving_gear.alpha[1])*(2*np.sin(driving_gear.alpha[1])-(q1*np.sin(v))/(driving_gear.radiuses["base"]))**3)
            Ib2 = (12*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[1] - driving_gear.alpha)*np.cos(driving_gear.alpha))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))**3
            Ib3 = 3*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)/(2*driving_gear.young * driving_gear.thickness * (np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha))**3)
            
            Ib = Ib0 + Ib1 + Ib2 + Ib3
            
            Is0 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2(np.sin(driving_gear.alpha)))/(driving_gear.young * driving_gear.thickness * (np.sin(driving_gear.alpha[3])-(q1*np.sin(v))/(driving_gear.radiuses["root"])*np.sin(v))**3)
            Is1 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.cos(driving_gear.alpha[2])-(driving_gear.teethNumber-2.5)/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))*np.cos(driving_gear.alpha[3])))/(driving_gear.young * driving_gear.thickness *(2*np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])))
            Is2 = (2.4*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness * (np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
            Is3 = (1.2*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
            
            Is = Is0 + Is1 + Is2 + Is3
                
        else:
    
            Ib0 = (12*np.sin(driving_gear.alpha)*((driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))/(driving_gear.teethNumber-2.5)-(np.cos(driving_gear.alpha)+np.cos(driving_gear.alpha[3])-np.cos(alphar)-(q1*np.cos(v))/(driving_gear.radiuses["root"]))*np.cos(driving_gear.alpha[1]))**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[3])+np.sin(driving_gear.alpha)-(q1*np.sin(v))/(driving_gear.radiuses["root"]))**3)
            Ib1 = (4*(1-((driving_gear.teethNumber-2.5)*np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[3]))/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle)))**3-4(1-np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[1])**3))/(driving_gear.young * driving_gear.thickness *np.cos(driving_gear.alpha[1])*(2*np.sin(driving_gear.alpha[1])-(q1*np.sin(v))/(driving_gear.radiuses["base"]))**3)
            Ib2 = (12*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[1] - driving_gear.alpha)*np.cos(driving_gear.alpha))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))**3
            
            Ib = Ib0 + Ib1 + Ib2 
                    
            Is0 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2(np.sin(driving_gear.alphaalpha)))/(driving_gear.young * driving_gear.thickness * (np.sin(driving_gear.alpha[3])-(q1*np.sin(v))/(driving_gear.radiuses["root"])*np.sin(v))**3)
            Is1 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.cos(driving_gear.alpha[2])-(driving_gear.teethNumber-2.5)/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))*np.cos(driving_gear.alpha[3])))/(driving_gear.young * driving_gear.thickness *(2*np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])))
            Is2 = (2.4*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
        
            Is = Is0 + Is1 + Is2 
        
    else:
        Ib0 = (12*np.sin(driving_gear.alpha)*((driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))/(driving_gear.teethNumber-2.5)-(np.cos(driving_gear.alpha)+np.cos(driving_gear.alpha[3])-np.cos(alphar)-(q1*np.cos(v))/(driving_gear.radiuses["root"]))*np.cos(driving_gear.alpha[1]))**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[3])+np.sin(driving_gear.alpha)-(q1*np.sin(v))/(driving_gear.radiuses["root"]))**3)
        Ib1 = (4*(1-((driving_gear.teethNumber-2.5)*np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[3]))/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle)))**3-4(1-np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[1])**3))/(driving_gear.young * driving_gear.thickness *np.cos(driving_gear.alpha[1])*(2*np.sin(driving_gear.alpha[1])-(q1*np.sin(v))/(driving_gear.radiuses["base"]))**3)
        Ib2 = (12*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[1] - driving_gear.alpha)*np.cos(driving_gear.alpha))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))**3
            
        Ib = Ib0 + Ib1 + Ib2 
                    
        Is0 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2(np.sin(driving_gear.alphaalpha)))/(driving_gear.young * driving_gear.thickness * (np.sin(driving_gear.alpha[3])-(q1*np.sin(v))/(driving_gear.radiuses["root"])*np.sin(v))**3)
        Is1 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.cos(driving_gear.alpha[2])-(driving_gear.teethNumber-2.5)/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))*np.cos(driving_gear.alpha[3])))/(driving_gear.young * driving_gear.thickness *(2*np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])))
        Is2 = (2.4*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q1*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
        
        Is = Is0 + Is1 + Is2 
            
else:
    
    hc = hq - hr/2
    q2 = hc/np.sin(v)
    
    
    if hc >= h0:
        alpha_k, rho, phi, iterations = _find_alpha_bisection(
            min_radius = driving_gear.radiuses["base"],
            max_radius = driving_gear.radiuses["addendum"],
            base_radius = driving_gear.radiuses["base"],
            alpha2 = driving_gear.alpha[2],
            h_target = hc,
        )
        print("alpha_c =", alpha_k, "rho =", rho, "phi =", phi)
        alphac = - alpha_k
    
        rho0 = 2.0
        r_root = np.linspace(0, driving_gear.radiuses["root"], 20)
        r_base = np.linspace(0, driving_gear.radiuses["base"], 20)
        r_evolv = np.linspace(driving_gear.radiuses["base"], driving_gear.radiuses["addendum"], 100)

        psi_evolv = driving_gear._compute_psi_angle(2*r_evolv)
        x_evolv = r_evolv * np.cos(driving_gear.alpha[2] - psi_evolv)
        y_evolv = r_evolv * np.sin(-driving_gear.alpha[2] + psi_evolv)

        theta = driving_gear.alpha[2] - 0.1 #driving_gear.psi
        x_root = r_root * np.cos(driving_gear.alpha[3])
        y_root = r_root * np.sin(driving_gear.alpha[3])
        x_base = r_base * np.cos(driving_gear.alpha[2])
        y_base = r_base * np.sin(driving_gear.alpha[2])
        x_rho = rho * np.cos(alpha_k)
        y_rho = rho * np.sin(alpha_k)
    
        rho0 = 2.0
        r_root = np.linspace(0, driving_gear.radiuses["root"], 20)
        r_base = np.linspace(0, driving_gear.radiuses["base"], 20)
        r_evolv = np.linspace(driving_gear.radiuses["base"], driving_gear.radiuses["addendum"], 100)

        theta = driving_gear.alpha[2] - 0.1 #driving_gear.psi
        x_root = r_root * np.cos(driving_gear.alpha[3])
        y_root = r_root * np.sin(-driving_gear.alpha[3])
        x_base = r_base * np.cos(driving_gear.alpha[2])
        y_base = r_base * np.sin(-driving_gear.alpha[2])
        x_rho = rho * np.cos(alpha_k)
        y_rho = rho * np.sin(- alpha_k)
    
        # Cerchio base e cerchio root
        fig, ax = plt.subplots(figsize=(7,7))
    
        circle_base = Circle((0,0), driving_gear.radiuses["base"], fill=False, linestyle='--', color='k')
        ax.add_patch(circle_base)

        circle_root = Circle((0,0), driving_gear.radiuses["root"], fill=False, linestyle='--', color='k')
        ax.add_patch(circle_root)

        # Raggi
        ax.plot(x_root, y_root, 'r--', linewidth=1, label='r_root')
        ax.plot(x_base, y_base, 'r--', linewidth=1, label='r_base')

        # Punto P
        #ax.plot(Px, Py, 'bo', label='P = (rho0 cos a, rho0 sin a)')

        # Linea orizzontale 
        #ax.axhline(0, color='gray', linewidth=1)
        ax.hlines(y=-hc, xmin=0, xmax=0.05, color='green', linewidth=1, linestyle='--')

        dx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3])
        dy = driving_gear.radiuses["root"] * np.sin(-driving_gear.alpha[3])

        cx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3])  - ((hr/2 - hc)/np.sin(v)) * np.cos(v) #- hr/2 * np.sin(np.pi/2 - v) + q2 * np.cos(v)
        cy = driving_gear.radiuses["root"] * np.sin(-driving_gear.alpha[3]) + ((hr/2 - hc)/np.sin(v)) * np.sin(v) #- hr/2 * np.cos(np.pi/2 - v) - q2 * np.sin(v)

        d1x = driving_gear.radiuses["base"] * np.cos(driving_gear.alpha[2])
        d1y = driving_gear.radiuses["base"] * np.sin(-driving_gear.alpha[2])
    
        bx = driving_gear.radiuses["root"] * np.cos(driving_gear.alpha[3]) - (hr/2 / np.sin(v)) * np.cos(v)
        by = 0

        ax.plot([bx, cx], [by, cy], linestyle='--', color='purple', linewidth=1)
        ax.plot([dx, d1x], [dy, d1y], linestyle='--', color='red', linewidth=1)
        ax.plot(x_evolv, y_evolv, linestyle='-.', color='blue', linewidth=1)

        for rho_it, alpha_it, phi_it in iterations:
            x_iter = rho_it * np.cos(alpha_it)
            y_iter = rho_it * np.sin(alpha_it)
            ax.scatter(x_iter, -y_iter, color='orange', s=20, label='Iterazione' if 'Iterazione' not in ax.get_legend_handles_labels()[1] else None)
        ax.plot([0, x_rho], [0, y_rho], 'g--', linewidth=1, label='rho')


        # Aspetto grafico
        ax.set_aspect('equal', 'box')
        ax.set_xlim(left=0)
        #ax.set_ylim(0.002, 0.004)
        ax.grid(True)
        ax.legend()
        ax.set_title("Geometria del modello")

        plt.show()
    
        if driving_gear.alpha[1] <= alphac:
            Ib1 = (4*(1-((driving_gear.teethNumber-2.5)*np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[3]))/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle)))**3-4(1-np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[1])**3))/(driving_gear.young * driving_gear.thickness *np.cos(driving_gear.alpha[1])*(2*np.sin(driving_gear.alpha[1])-(q2*np.sin(v))/(driving_gear.radiuses["base"]))**3)
            Ib2 = (12*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[1]-driving_gear.alpha)*np.cos(driving_gear.alpha))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))**3
            
            Ib = Ib1 + Ib2

            Is0 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.sin(driving_gear.alpha)))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha)-(q2*np.sin(v))/(driving_gear.radiuses["root"])*np.sin(v))**3)
            Is1 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.cos(driving_gear.alpha[2])-(driving_gear.teethNumber-2.5)/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))*np.cos(driving_gear.alpha[3])))/(driving_gear.young * driving_gear.thickness *(2*np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["base"])))
            Is2 = (2.4*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness(np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["base"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
    
            Is = Is0 + Is1 + Is2
            
        else:
            Ib0 = (12*np.sin(driving_gear.alpha)*((driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))/(driving_gear.teethNumber-2.5)-(np.cos(driving_gear.alpha)+np.cos(driving_gear.alpha[3])-np.cos(alphar)-(((np.sin(driving_gear.alpha[3])/(np.sin(v)))-(q2/driving_gear.radiuses["root"]))*np.cos(v))/(driving_gear.radiuses["root"]))*np.cos(driving_gear.alpha[1]))**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha)-(q2*np.sin(v))/(driving_gear.radiuses["root"]))**3)
            Ib1 = (4*(1-((driving_gear.teethNumber-2.5)*np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[3]))/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle)))**3-4(1-np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[1])**3))/(driving_gear.young * driving_gear.thickness *np.cos(driving_gear.alpha[1])*(2*np.sin(driving_gear.alpha[1])-(q2*np.sin(v))/(driving_gear.radiuses["base"]))**3)
            Ib2 = (12*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[1]-driving_gear.alpha)*np.cos(driving_gear.alpha))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))**3
            
            Ib = Ib0 + Ib1 + Ib2
                    
            Is0 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.sin(driving_gear.alpha)))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha)-(q2*np.sin(v))/(driving_gear.radiuses["root"])*np.sin(v))**3)
            Is1 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.cos(driving_gear.alpha[2])-(driving_gear.teethNumber-2.5)/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))*np.cos(driving_gear.alpha[3])))/(driving_gear.young * driving_gear.thickness *(2*np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["base"])))
            Is2 = (2.4*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["base"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
        
            Is = Is0 + Is1 + Is2
            
    else: 
        Ib1 = (4*(1-((driving_gear.teethNumber-2.5)*np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[3]))/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle)))**3-4(1-np.cos(driving_gear.alpha[1])*np.cos(driving_gear.alpha[1])**3))/(driving_gear.young * driving_gear.thickness *np.cos(driving_gear.alpha[1])*(2*np.sin(driving_gear.alpha[1])-(q2*np.sin(v))/(driving_gear.radiuses["base"]))**3)
        Ib2 = (12*(1+np.cos(driving_gear.alpha[1])*((driving_gear.alpha[2]-driving_gear.alpha)*np.sin(driving_gear.alpha)-np.cos(driving_gear.alpha)))**2*(driving_gear.alpha[1]-driving_gear.alpha)*np.cos(driving_gear.alpha))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["root"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))**3
            
        Ib = Ib1 + Ib2

        Is0 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.sin(driving_gear.alpha)))/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha)-(q2*np.sin(v))/(driving_gear.radiuses["root"])*np.sin(v))**3)
        Is1 = (2.4*(1+v)*np.cos(driving_gear.alpha[1])**2*(np.cos(driving_gear.alpha[2])-(driving_gear.teethNumber-2.5)/(driving_gear.teethNumber*np.cos(driving_gear.pressureAngle))*np.cos(driving_gear.alpha[3])))/(driving_gear.young * driving_gear.thickness *(2*np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["base"])))
        Is2 = (2.4*(1+v)*(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)*np.cos(driving_gear.alpha[1])**2)/(driving_gear.young * driving_gear.thickness *(np.sin(driving_gear.alpha[2])-(q2*np.sin(v))/(driving_gear.radiuses["base"])+np.sin(driving_gear.alpha)+(driving_gear.alpha[2]-driving_gear.alpha)*np.cos(driving_gear.alpha)))
    
        Is = Is0 + Is1 + Is2
            

Kb_inv = np.sum(trapezoid(Ib, x=driving_gear.alpha, axis=0))
Ks_inv = np.sum(trapezoid(Is, x=driving_gear.alpha, axis=0))

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETRI DEL MODELLO
# -------------------------
