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
import scipy as sc
import scipy.integrate


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

#tooth damage prima dello spigolo inferiore
tooth_damage = 0.01 #0.5 - h0/hr - 0.001

#tooth damage sullo spigolo superiore
#tooth_damage = 0.5 - h0/hr

#tooth damage fino a metà dente
#tooth_damage = 0.5

#tooth damage prima dello spigolo inferiore
#tooth_damage = 0.5 + h0/hr - 0.001

#tooth damage sullo spigolo inferiore
#tooth_damage = 0.5 + h0/hr

#tooth damage dopo lo spigolo inferiore
#tooth_damage = 0.5 + h0/hr + 0.001

#tooth damage limite
#tooth_damage = 1

hq = tooth_damage * hr
v = np.pi/6

hq = tooth_damage * hr


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

# alphar (alpha_r del paper, estremo superiore del primo integrale di Ib0/Is0)
# e' l'angolo polare del punto del cerchio di radice sulla verticale di A
# (condizioni 1-2) o di C (condizioni 3-4), Fig. 10; viene calcolato piu' sotto.

a1 = abs(driving_gear.alpha[1])
a2 = driving_gear.alpha[2]
a3 = driving_gear.alpha[3]
E  = driving_gear.young
L  = driving_gear.thickness
Rb = driving_gear.radiuses["base"]
Rr = driving_gear.radiuses["root"]
N  = driving_gear.teeth_number
pa = driving_gear.pressure_angle
nu = driving_gear.poisson   # coefficiente di Poisson (nu del paper), NON l'angolo v

if tooth_damage <= 0.5:

    q1 = hq/np.sin(v)
    ha = hr/2 - tooth_damage * hr
    alphar = np.arccos(np.cos(a3) - (q1/Rr)*np.cos(v))

    if ha >= h0:
        alpha_k, rho, phi, iterations = _find_alpha_bisection(
            min_radius = driving_gear.radiuses["base"],
            max_radius = driving_gear.radiuses["addendum"],
            base_radius = driving_gear.radiuses["base"],
            alpha2 = driving_gear.alpha[2],
            h_target = ha,
        )
        print("alpha_a =", alpha_k, "rho =", rho, "phi =", phi)
        # alpha_k (= alpha2 - psi) e' l'angolo polare, usato solo per i grafici;
        # l'alpha_a del paper e' phi - alpha2 (positivo)
        alphaa = phi - a2
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

        if a1 > alphaa:

            # ---- Condition 1: ha >= h0 & alpha1 > alphaa (eq. 5.5, 5.6) ----
            def Ib0(a):
                return (12 * np.sin(a) * (N * np.cos(pa) / (N - 2.5) - (np.cos(a) + np.cos(a3) - np.cos(alphar) - (q1 / Rr) * np.cos(v)) * np.cos(a1))**2
                        / (E * L * (np.sin(a3) + np.sin(a) - (q1 / Rr) * np.sin(v))**3))

            Ib1 = ((4 * (1 - (N - 2.5) * np.cos(a1) * np.cos(a3) / (N * np.cos(pa)))**3 - 4 * (1 - np.cos(a1) * np.cos(a2))**3)
                   / (E * L * np.cos(a1) * (2 * np.sin(a2) - (q1 / Rb) * np.sin(v))**3))

            def Ib2(a):
                return (12 * (1 + np.cos(a1) * ((a2 - a) * np.sin(a) - np.cos(a)))**2 * (a2 - a) * np.cos(a)
                        / (E * L * (np.sin(a2) - (q1 / Rb) * np.sin(v) + np.sin(a) + (a2 - a) * np.cos(a))**3))

            def Ib3(a):
                return (3 * (1 + np.cos(a1) * ((a2 - a) * np.sin(a) - np.cos(a)))**2 * (a2 - a) * np.cos(a)
                        / (2 * E * L * (np.sin(a) + (a2 - a) * np.cos(a))**3))

            Ib0_val, _ = sc.integrate.quad(Ib0, a3, alphar)
            Ib1_val = Ib1
            Ib2_val, _ = sc.integrate.quad(Ib2, -alphaa, a2)
            Ib3_val, _ = sc.integrate.quad(Ib3, -a1, -alphaa)

            Ib = Ib0_val + Ib1_val + Ib2_val + Ib3_val

            def Is0(a):
                return (2.4 * (1 + nu) * np.cos(a1)**2 * np.sin(a)
                        / (E * L * (np.sin(a3) - (q1 / Rr) * np.sin(v) + np.sin(a))**3))

            Is1 = (2.4 * (1 + nu) * np.cos(a1)**2 * (np.cos(a2) - (N - 2.5) / (N * np.cos(pa)) * np.cos(a3))
                   / (E * L * (2 * np.sin(a2) - (q1 / Rb) * np.sin(v))))

            def Is2(a):
                return (2.4 * (1 + nu) * (a2 - a) * np.cos(a) * np.cos(a1)**2
                        / (E * L * (np.sin(a2) - (q1 / Rb) * np.sin(v) + np.sin(a) + (a2 - a) * np.cos(a))))

            def Is3(a):
                return (1.2 * (1 + nu) * (a2 - a) * np.cos(a) * np.cos(a1)**2
                        / (E * L * (np.sin(a) + (a2 - a) * np.cos(a))))

            Is0_val, _ = sc.integrate.quad(Is0, a3, alphar)
            Is1_val = Is1
            Is2_val, _ = sc.integrate.quad(Is2, -alphaa, a2)
            Is3_val, _ = sc.integrate.quad(Is3, -a1, -alphaa)

            Is = Is0_val + Is1_val + Is2_val + Is3_val
            
            Kt = 1 / (Ib + Is)


        else:

            # ---- Condition 2 (caso ha >= h0 & alpha1 <= alphaa) (eq. 5.9, 5.10) ----
            def Ib0(a):
                return (12 * np.sin(a) * (N * np.cos(pa) / (N - 2.5) - (np.cos(a) + np.cos(a3) - np.cos(alphar) - (q1 / Rr) * np.cos(v)) * np.cos(a1))**2
                        / (E * L * (np.sin(a3) + np.sin(a) - (q1 / Rr) * np.sin(v))**3))

            Ib1 = ((4 * (1 - (N - 2.5) * np.cos(a1) * np.cos(a3) / (N * np.cos(pa)))**3 - 4 * (1 - np.cos(a1) * np.cos(a2))**3)
                   / (E * L * np.cos(a1) * (2 * np.sin(a2) - (q1 / Rb) * np.sin(v))**3))

            def Ib2(a):
                return (12 * (1 + np.cos(a1) * ((a2 - a) * np.sin(a) - np.cos(a)))**2 * (a2 - a) * np.cos(a)
                        / (E * L * (np.sin(a2) - (q1 / Rb) * np.sin(v) + np.sin(a) + (a2 - a) * np.cos(a))**3))

            Ib0_val, _ = sc.integrate.quad(Ib0, a3, alphar)
            Ib1_val = Ib1
            Ib2_val, _ = sc.integrate.quad(Ib2, -a1, a2)

            Ib = Ib0_val + Ib1_val + Ib2_val

            def Is0(a):
                return (2.4 * (1 + nu) * np.cos(a1)**2 * np.sin(a)
                        / (E * L * (np.sin(a3) - (q1 / Rr) * np.sin(v) + np.sin(a))**3))

            Is1 = (2.4 * (1 + nu) * np.cos(a1)**2 * (np.cos(a2) - (N - 2.5) / (N * np.cos(pa)) * np.cos(a3))
                   / (E * L * (2 * np.sin(a2) - (q1 / Rb) * np.sin(v))))

            def Is2(a):
                return (2.4 * (1 + nu) * (a2 - a) * np.cos(a) * np.cos(a1)**2
                        / (E * L * (np.sin(a2) - (q1 / Rb) * np.sin(v) + np.sin(a) + (a2 - a) * np.cos(a))))

            Is0_val, _ = sc.integrate.quad(Is0, a3, alphar)
            Is1_val = Is1
            Is2_val, _ = sc.integrate.quad(Is2, -a1, a2)

            Is = Is0_val + Is1_val + Is2_val
            
            Kt = 1 / (Ib + Is)


    else:

        # ---- Condition 2 (caso ha < h0) (eq. 5.9, 5.10) ----
        def Ib0(a):
            return (12 * np.sin(a) * (N * np.cos(pa) / (N - 2.5) - (np.cos(a) + np.cos(a3) - np.cos(alphar) - (q1 / Rr) * np.cos(v)) * np.cos(a1))**2
                    / (E * L * (np.sin(a3) + np.sin(a) - (q1 / Rr) * np.sin(v))**3))

        Ib1 = ((4 * (1 - (N - 2.5) * np.cos(a1) * np.cos(a3) / (N * np.cos(pa)))**3 - 4 * (1 - np.cos(a1) * np.cos(a2))**3)
               / (E * L * np.cos(a1) * (2 * np.sin(a2) - (q1 / Rb) * np.sin(v))**3))

        def Ib2(a):
            return (12 * (1 + np.cos(a1) * ((a2 - a) * np.sin(a) - np.cos(a)))**2 * (a2 - a) * np.cos(a)
                    / (E * L * (np.sin(a2) - (q1 / Rb) * np.sin(v) + np.sin(a) + (a2 - a) * np.cos(a))**3))

        Ib0_val, _ = sc.integrate.quad(Ib0, a3, alphar)
        Ib1_val = Ib1
        Ib2_val, _ = sc.integrate.quad(Ib2, -a1, a2)

        Ib = Ib0_val + Ib1_val + Ib2_val

        def Is0(a):
            return (2.4 * (1 + nu) * np.cos(a1)**2 * np.sin(a)
                    / (E * L * (np.sin(a3) - (q1 / Rr) * np.sin(v) + np.sin(a))**3))

        Is1 = (2.4 * (1 + nu) * np.cos(a1)**2 * (np.cos(a2) - (N - 2.5) / (N * np.cos(pa)) * np.cos(a3))
               / (E * L * (2 * np.sin(a2) - (q1 / Rb) * np.sin(v))))

        def Is2(a):
            return (2.4 * (1 + nu) * (a2 - a) * np.cos(a) * np.cos(a1)**2
                    / (E * L * (np.sin(a2) - (q1 / Rb) * np.sin(v) + np.sin(a) + (a2 - a) * np.cos(a))))

        Is0_val, _ = sc.integrate.quad(Is0, a3, alphar)
        Is1_val = Is1
        Is2_val, _ = sc.integrate.quad(Is2, -a1, a2)

        Is = Is0_val + Is1_val + Is2_val
        
        Kt = 1 / (Ib + Is)

else:

    hc = tooth_damage * hr - hr/2
    q2 = hc/np.sin(v)
    alphar = np.arccos(np.cos(a3) - (np.sin(a3)/np.sin(v) - q2/Rr)*np.cos(v))

    if hc >= h0:
        alpha_k, rho, phi, iterations = _find_alpha_bisection(
            min_radius = driving_gear.radiuses["base"],
            max_radius = driving_gear.radiuses["addendum"],
            base_radius = driving_gear.radiuses["base"],
            alpha2 = driving_gear.alpha[2],
            h_target = hc,
        )
        print("alpha_c =", alpha_k, "rho =", rho, "phi =", phi)
        # alpha_c del paper = phi - alpha2 (positivo); qui alphac = -alpha_c
        alphac = -(phi - a2)

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

        # nel codice alphac = -alpha_k, mentre nel paper alpha_c e' positivo:
        # la condizione alpha1 <= alpha_c diventa alpha1 <= -alphac
        if a1 <= -alphac:

            # ---- Condition 3 (caso hc >= h0 & alpha1 <= alpha_c) (eq. 5.13, 5.14) ----
            def Ib0(a):
                return (12 * np.sin(a) * (N * np.cos(pa) / (N - 2.5) - (np.cos(a) + np.cos(a3) - np.cos(alphar) - (np.sin(a3) / np.sin(v) - q2 / Rr) * np.cos(v)) * np.cos(a1))**2
                        / (E * L * (np.sin(a) - (q2 / Rr) * np.sin(v))**3))

            Ib1 = ((4 * (1 - (N - 2.5) * np.cos(a1) * np.cos(a3) / (N * np.cos(pa)))**3 - 4 * (1 - np.cos(a1) * np.cos(a2))**3)
                   / (E * L * np.cos(a1) * (np.sin(a2) - (q2 / Rb) * np.sin(v))**3))

            def Ib2(a):
                return (12 * (1 + np.cos(a1) * ((a2 - a) * np.sin(a) - np.cos(a)))**2 * (a2 - a) * np.cos(a)
                        / (E * L * (np.sin(a) + (a2 - a) * np.cos(a) - (q2 / Rb) * np.sin(v))**3))

            Ib0_val, _ = sc.integrate.quad(Ib0, a3, alphar)
            Ib1_val = Ib1
            Ib2_val, _ = sc.integrate.quad(Ib2, -a1, a2)

            Ib = Ib0_val + Ib1_val + Ib2_val

            def Is0(a):
                return (2.4 * (1 + nu) * np.cos(a1)**2 * np.sin(a)
                        / (E * L * (np.sin(a) - (q2 / Rr) * np.sin(v))**3))

            Is1 = (2.4 * (1 + nu) * np.cos(a1)**2 * (np.cos(a2) - (N - 2.5) / (N * np.cos(pa)) * np.cos(a3))
                   / (E * L * (np.sin(a2) - (q2 / Rb) * np.sin(v))))

            def Is2(a):
                return (2.4 * (1 + nu) * (a2 - a) * np.cos(a) * np.cos(a1)**2
                        / (E * L * (np.sin(a) + (a2 - a) * np.cos(a) - (q2 / Rb) * np.sin(v))))

            Is0_val, _ = sc.integrate.quad(Is0, a3, alphar)
            Is1_val = Is1
            Is2_val, _ = sc.integrate.quad(Is2, -a1, a2)

            Is = Is0_val + Is1_val + Is2_val
            
            Kt = 1 / (Ib + Is)

        else:

            # ---- Condition 4: hc >= h0 & alpha1 > alpha_c (eq. 5.15, 5.16) ----
            # Per alpha1 > alpha_c il tratto di evolvente usato nel paper arriva fino al
            # punto E' dove (hx - hc)^3 -> 0: Ib e Is divergono e la rigidezza tende a 0.
            # quad darebbe valori arbitrari/negativi, quindi si impone Kt = 0.
            Ib = np.inf
            Is = np.inf
            Kt = 0.0

    else:

        # ---- Condition 3 (caso hc < h0) (eq. 5.13, 5.14) ----
        def Ib0(a):
            return (12 * np.sin(a) * (N * np.cos(pa) / (N - 2.5) - (np.cos(a) + np.cos(a3) - np.cos(alphar) - (np.sin(a3) / np.sin(v) - q2 / Rr) * np.cos(v)) * np.cos(a1))**2
                    / (E * L * (np.sin(a) - (q2 / Rr) * np.sin(v))**3))

        Ib1 = ((4 * (1 - (N - 2.5) * np.cos(a1) * np.cos(a3) / (N * np.cos(pa)))**3 - 4 * (1 - np.cos(a1) * np.cos(a2))**3)
               / (E * L * np.cos(a1) * (np.sin(a2) - (q2 / Rb) * np.sin(v))**3))

        def Ib2(a):
            return (12 * (1 + np.cos(a1) * ((a2 - a) * np.sin(a) - np.cos(a)))**2 * (a2 - a) * np.cos(a)
                    / (E * L * (np.sin(a) + (a2 - a) * np.cos(a) - (q2 / Rb) * np.sin(v))**3))

        Ib0_val, _ = sc.integrate.quad(Ib0, a3, alphar)
        Ib1_val = Ib1
        Ib2_val, _ = sc.integrate.quad(Ib2, -a1, a2)

        Ib = Ib0_val + Ib1_val + Ib2_val

        def Is0(a):
            return (2.4 * (1 + nu) * np.cos(a1)**2 * np.sin(a)
                    / (E * L * (np.sin(a) - (q2 / Rr) * np.sin(v))**3))

        Is1 = (2.4 * (1 + nu) * np.cos(a1)**2 * (np.cos(a2) - (N - 2.5) / (N * np.cos(pa)) * np.cos(a3))
               / (E * L * (np.sin(a2) - (q2 / Rb) * np.sin(v))))

        def Is2(a):
            return (2.4 * (1 + nu) * (a2 - a) * np.cos(a) * np.cos(a1)**2
                    / (E * L * (np.sin(a) + (a2 - a) * np.cos(a) - (q2 / Rb) * np.sin(v))))

        Is0_val, _ = sc.integrate.quad(Is0, a3, alphar)
        Is1_val = Is1
        Is2_val, _ = sc.integrate.quad(Is2, -a1, a2)

        Is = Is0_val + Is1_val + Is2_val
        
        Kt = 1 / (Ib + Is)



import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETRI DEL MODELLO
# -------------------------