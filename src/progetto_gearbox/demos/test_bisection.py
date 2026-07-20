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


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

driving_gear= create_gear()
#test con create pinion

tooth_damage = 0.95
hr = 2 * driving_gear.radiuses["root"] * np.sin(driving_gear.alpha[3])
hb = 2 * driving_gear.radiuses["base"] * np.sin(driving_gear.alpha[2])
hq = tooth_damage * hr
ha = hr/2 - hq
v = np.pi/6
alpha0 = driving_gear.alpha[2] - driving_gear.compute_phi_angle(driving_gear.radiuses["addendum"])
h0 = driving_gear.radiuses["base"] * (np.sin(alpha0) + (driving_gear.alpha[2] - alpha0) * np.cos(alpha0))

q1 = hq/np.sin(v) 
hq1 = hq
ha = hr/2 - hq1

q1 = hq/np.sin(v) 
hq1 = hq
hc = hr/2 - hq1


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
    
    if driving_gear.alpha[1] >= driving_gear.alpha[2]:
        print("ciao")
        
    elif driving_gear.alpha[1] < alphaa: #"condition 2"
        print("ciao")
        
else: "condition 2"
               
        
if hc >= h0:
    alpha_k, rho, phi, iterations = _find_alpha_bisection(
        min_radius = driving_gear.radiuses["base"],
        max_radius = driving_gear.radiuses["addendum"],
        base_radius = driving_gear.radiuses["base"],
        alpha2 = driving_gear.alpha[2],
        h_target = hc,
    )
    print("alpha_c =", alpha_k, "rho =", rho, "phi =", phi)
    alphac = alpha_k
    
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
    
    if driving_gear.alpha[1] <= alphac:
        print("ciao")
    
    elif driving_gear.alpha[1] > alphac:
        print("ciao")
        
else: 
    print("ciao")






# if ha > h0 and driving_gear.alpha[1] > driving_gear.alpha[2]:
#     q1 = hq/np.sin(v) 
#     hq1 = hq
#     ha = hr/2 - hq1
#     alpha_k, rho, phi, iterations = _find_alpha_bisection(
#         min_radius=driving_gear.radiuses["base"],
#         max_radius=driving_gear.radiuses["addendum"],
#         base_radius=driving_gear.radiuses["base"],
#         alpha2=driving_gear.alpha[2],
#         h_target=ha,
#     )
#     print("alpha_k =", alpha_k, "rho =", rho, "phi =", phi)
    
#     rho0 = 2.0
#     r_root = np.linspace(0, driving_gear.radiuses["root"], 20)
#     r_base = np.linspace(0, driving_gear.radiuses["base"], 20)
#     r_evolv = np.linspace(driving_gear.radiuses["base"], driving_gear.radiuses["addendum"], 100)

#     psi_evolv = driving_gear._compute_psi_angle(2*r_evolv)
#     x_evolv = r_evolv * np.cos(driving_gear.alpha[2] - psi_evolv)
#     y_evolv = r_evolv * np.sin(driving_gear.alpha[2] - psi_evolv)

#     theta = driving_gear.alpha[2] - 0.1 #driving_gear.psi
#     x_root = r_root * np.cos(driving_gear.alpha[3])
#     y_root = r_root * np.sin(driving_gear.alpha[3])
#     x_base = r_base * np.cos(driving_gear.alpha[2])
#     y_base = r_base * np.sin(driving_gear.alpha[2])
#     x_rho = rho * np.cos(alpha_k)
#     y_rho = rho * np.sin(alpha_k)
    
# elif ha < h0 or (ha >= h0 and driving_gear.alpha[1] <= alphaa):
#     q1 = hq/np.sin(v) 
#     hq1 = hq
#     ha = hr/2 - hq1
#     alpha_k, rho, phi, iterations = _find_alpha_bisection(
#         min_radius=driving_gear.radiuses["base"],
#         max_radius=driving_gear.radiuses["addendum"],
#         base_radius=driving_gear.radiuses["base"],
#         alpha2=driving_gear.alpha[2],
#         h_target=ha,
#     )
#     print("alpha_k =", alpha_k, "rho =", rho, "phi =", phi)
    
#     rho0 = 2.0
#     r_root = np.linspace(0, driving_gear.radiuses["root"], 20)
#     r_base = np.linspace(0, driving_gear.radiuses["base"], 20)
#     r_evolv = np.linspace(driving_gear.radiuses["base"], driving_gear.radiuses["addendum"], 100)

#     psi_evolv = driving_gear._compute_psi_angle(2*r_evolv)
#     x_evolv = r_evolv * np.cos(driving_gear.alpha[2] - psi_evolv)
#     y_evolv = r_evolv * np.sin(driving_gear.alpha[2] - psi_evolv)

#     theta = driving_gear.alpha[2] - 0.1 #driving_gear.psi
#     x_root = r_root * np.cos(driving_gear.alpha[3])
#     y_root = r_root * np.sin(driving_gear.alpha[3])
#     x_base = r_base * np.cos(driving_gear.alpha[2])
#     y_base = r_base * np.sin(driving_gear.alpha[2])
#     x_rho = rho * np.cos(alpha_k)
#     y_rho = rho * np.sin(alpha_k)
    

# elif hc < h0 or (hc >= h0 and driving_gear.alpha[1] <= alphac): 
    
    
# elif hc >= h0 and driving_gear.alpha[1] > alphac: 
#     he = hq - hr/2 
#     q2 = he/np.sin(v)
#     alpha_e, rho, phi, iterations = _find_alpha_bisection(
#         min_radius=driving_gear.radiuses["base"],
#         max_radius=driving_gear.radiuses["addendum"],
#         base_radius=driving_gear.radiuses["base"],
#         alpha2=driving_gear.alpha[2],
#         h_target=he,
#     )
#     print("alpha_e =", alpha_e, "rho =", rho, "phi =", phi)
    
#     rho0 = 2.0
#     r_root = np.linspace(0, driving_gear.radiuses["root"], 20)
#     r_base = np.linspace(0, driving_gear.radiuses["base"], 20)
#     r_evolv = np.linspace(driving_gear.radiuses["base"], driving_gear.radiuses["addendum"], 100)

#     psi_evolv = driving_gear._compute_psi_angle(2*r_evolv)
#     x_evolv = r_evolv * np.cos(driving_gear.alpha[2] - psi_evolv)
#     y_evolv = r_evolv * np.sin(driving_gear.alpha[2] - psi_evolv)

#     theta = driving_gear.alpha[2] - 0.1 #driving_gear.psi
#     x_root = r_root * np.cos(driving_gear.alpha[3])
#     y_root = r_root * np.sin(driving_gear.alpha[3])
#     x_base = r_base * np.cos(driving_gear.alpha[2])
#     y_base = r_base * np.sin(driving_gear.alpha[2])
#     x_rho = rho * np.cos(alpha_e)
#     y_rho = rho * np.sin(alpha_e)    
    
# else:
#     raise ValueError("Tooth damage must be between 0 and 1.")








import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETRI DEL MODELLO
# -------------------------


# Cerchio base e cerchio root

fig, ax = plt.subplots(figsize=(7,7))

# Cerchio base
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

for rho_it, alpha_it, phi_it in iterations:
    x_iter = rho_it * np.cos(alpha_it)
    y_iter = rho_it * np.sin(alpha_it)
    ax.scatter(x_iter, y_iter, color='orange', s=20, label='Iterazione' if 'Iterazione' not in ax.get_legend_handles_labels()[1] else None)
ax.plot([0, x_rho], [0, y_rho], 'g--', linewidth=1, label='rho')


# Aspetto grafico
ax.set_aspect('equal', 'box')
ax.set_xlim(left=0)
#ax.set_ylim(0.002, 0.004)
ax.grid(True)
ax.legend()
ax.set_title("Geometria del modello")

plt.show()
