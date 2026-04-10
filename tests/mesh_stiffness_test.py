import progetto_gearbox.gears_module_v2 as gm
import numpy as np
import matplotlib.pyplot as plt
  # --- MODELLO 5: due gear accoppiati (con mesh connection) ---
    # qui vincolo su traslazioni X,Y lascia libera solo la rotazione T per entrambi
    # cm5 = gmc.ConstraintManager("cm_model5")
    # cm5.addGroundConstraint("gear", ["X", "Y"])  # blocco traslazioni, lascio rotazione
    # cm5.addGroundConstraint(
    #     "pinion", ["X", "Y"]
    # )  # blocco traslazioni, lascio rotazione
    # cm5.addMeshConnection("gear", "pinion")  # mesh
    # lm5 = gmc.LoadManager("lm_model5")

    # lm5.addInputFunction("gear", {"Tt": lambda t, theta: 10})  # forzante costante
    # lm5.addInputFunction("pinion", {"Tt": lambda t, theta: 0 * t})  # forzante rampa

    # model5 = gmc.GenericModel(name="Model_5_coupled_rot")
    # model5.addGear(gear)
    # model5.addGear(pinion)
    # model5.addConstraintManager(cm5)
    # model5.addLoadManager(lm5)
    # model5.assembleModel("cm_model5", "lm_model5")

    # x0_5 = np.zeros(model5.Ac.shape[0])
    # x0_5[0] = 1
    # x0_5[2] = 0

    # sim5 = simulation(t_tot=3.0, x0=x0_5, deltat=0.0005, model=model5)
    # print("Simulazione Model 5: due gear accoppiati (rotazioni legate)")
    # sim5.solve()
    # sim5.plot()
    # plt.show()
    
# --- crea gli oggetti gear (riporto la stessa procedura già usata) ---
gear = gm.spurGear(name="gear")
gear.assignGeometricalProperties(
    module=3.2,
    teethNumber=31,
    pressureAngle=20 * np.pi / 180,
    thickness=0.0381 * 1e3,
)
gear.assignMaterialProperties(young=2.068 * 1e5, poisson=0.3)
gear.assignDynamicProperties(massX=1, massY=1, inertiaT=0.5)

pinion = gm.spurGear(name="pinion")
pinion.assignGeometricalProperties(
    module=3.2,
    teethNumber=19,
    pressureAngle=20 * np.pi / 180,
    thickness=0.0381 * 1e3,
)
pinion.assignMaterialProperties(young=2.068 * 1e5, poisson=0.3)
pinion.assignDynamicProperties(massX=0.5, massY=0.5, inertiaT=0.2)
        
gear_angles = np.arange(0, 2 * np.pi, 0.001)
gear_mesh_stiffness = []
pinion_mesh_stiffness = []

for gear_angle in gear_angles:
    gear_teeth_pos = gear.computeTeethPosition(gear_angle)
    gear_engagement, gear_alpha1 = gear.computeTeethEngagement(gear_teeth_pos, "driving", 0, +100 , pinion)
    gear_mesh_stiffness.append(gear.meshStiffness(gear_engagement, gear_alpha1))
    pinion_angle = -gear_angle * gear.teethNumber / pinion.teethNumber
    pinion_teeth_pos = pinion.computeTeethPosition(pinion_angle)
    pinion_engagement, pinion_alpha1 = pinion.computeTeethEngagement(pinion_teeth_pos, "driven", 0, -100, gear)
    pinion_mesh_stiffness.append(pinion.meshStiffness(pinion_engagement, pinion_alpha1))
gear_mesh_stiffness = np.array(gear_mesh_stiffness)
pinion_mesh_stiffness = np.array(pinion_mesh_stiffness)
total_mesh_stiffness = 1 / (1/gear_mesh_stiffness + 1/pinion_mesh_stiffness)

fig, ax = plt.subplots()
ax.plot(gear_angles, gear_mesh_stiffness, label="Gear", color='blue')
ax.plot(gear_angles, pinion_mesh_stiffness, label="Pinion", color='red')
ax.set_xlabel("Gear Angle")
ax.set_ylabel("Mesh Stiffness")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(gear_angles, total_mesh_stiffness, label="Total", color='blue')
ax.set_xlabel("Gear Angle")
ax.set_ylabel("Mesh Stiffness")
ax.legend()
plt.show()
print("end")