# import numpy.typing as npt
# import matplotlib.pyplot as plt
# import scipy as sc
# import progetto_gearbox.gears_module_v2 as gm
# import progetto_gearbox.generic_model_class as gmc

# from scipy.integrate import solve_ivp


def main_debug():
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

    # --- MODELLO 1: gear che può muoversi solo in X ---
    cm1 = gmc.ConstraintManager("cm_model1")
    # vincolo rigido su Y e T => lascia solo X
    cm1.addGroundConstraint("gear", ["Y", "T"])
    cm1.addGroundElasticConnection("gear", dofs=["X"], stiffness=[246.5], damping=[5])
    lm1 = gmc.LoadManager("lm_model1")
    # lm1.addInputFunction("gear", {"Fx": lambda t,x: 10*np.sin(2*np.pi*5*t)}) #forzante armonica
    # lm1.addInputFunction("gear", {"Fx": lambda t,x: 10}) #forzante step
    lm1.addInputFunction("gear", {"Fx": lambda t, x: 1 * t})  # forzante rampa

    gear.add_initial_conditions(
        [0 for _ in range(6)]
    )  # inizializzo condizioni iniziali a zero (pos e vel)

    pinion.add_initial_conditions(
        [0 for _ in range(6)]
    )  # inizializzo condizioni iniziali a zero (pos e vel)

    model1 = gmc.GenericModel(name="Model_1_Xonly")
    model1.addGear(gear)
    model1.addConstraintManager(cm1)
    model1.addLoadManager(lm1)
    model1.assembleModel("cm_model1", "lm_model1")

    # azzero Bc per rimuovere forzanti esterni e simulare vibrazione libera
    # model1.Bc = np.zeros_like(model1.Bc)
    # condizione iniziale: piccolo spostamento sulle posizioni (ogni DOF ha [vel,pos])
    x0_1 = np.zeros(model1.Ac.shape[0])
    # setto piccolo spostamento su tutte le posizioni (indici 1,3,5,...)
    for i in range(model1.Ac.shape[0] // 2):
        x0_1[2 * i + 1] = 0.3

    sim1 = simulation(t_tot=5.0, x0=x0_1, deltat=0.0005, model=model1)
    print("Simulazione Model 1: gear solo X (vibrazione libera)")
    sim1.solve()
    sim1.plot()
    plt.show()

    # --- MODELLO 2: un gear che può muoversi solo in Y (X e rotazione vincolate) ---
    cm2 = gmc.ConstraintManager("cm_model2")
    # vincolo rigido su X e T => lascia solo Y
    cm2.addGroundConstraint("gear", ["X", "T"])
    cm2.addGroundElasticConnection("gear", dofs=["Y"], stiffness=[246.5], damping=[5])
    lm2 = gmc.LoadManager("lm_model2")
    # lm2.addInputFunction("gear", {"Fy": lambda t,y: 10*np.sin(2*np.pi*5*t)}) #forzante armonica
    # lm2.addInputFunction("gear", {"Fy": lambda t,y: 10}) #forzante step
    lm2.addInputFunction("gear", {"Fy": lambda t, y: 10 * t})  # forzante rampa
    model2 = gmc.GenericModel(name="Model_2_Yonly")
    model2.addGear(gear)
    model2.addConstraintManager(cm2)
    model2.addLoadManager(lm2)
    model2.assembleModel("cm_model2", "lm_model2")

    x0_2 = np.zeros(model2.Ac.shape[0])
    for i in range(model2.Ac.shape[0] // 2):
        x0_2[2 * i + 1] = 1

    sim2 = simulation(t_tot=5.0, x0=x0_2, deltat=0.0005, model=model2)
    print("Simulazione Model 2: pinion solo Y (vibrazione libera)")
    sim2.solve()
    sim2.plot()
    plt.show()

    # --- MODELLO 3: Model1 + Model2 assieme (gear libero solo X, pinion libero solo Y) ---

    cm3 = gmc.ConstraintManager("cm_model3")
    cm3.addGroundConstraint("gear", ["Y", "T"])  # gear: solo X
    cm3.addGroundElasticConnection("gear", dofs=["X"], stiffness=[246.5], damping=[5])
    cm3.addGroundConstraint("pinion", ["X", "T"])  # pinion: solo Y
    cm3.addGroundElasticConnection("pinion", dofs=["Y"], stiffness=[246.5], damping=[5])
    lm3 = gmc.LoadManager("lm_model3")
    # lm3.addInputFunction("gear", {"Fx": lambda t,x: 10*np.sin(2*np.pi*5*t)}) #forzante armonica
    # lm3.addInputFunction("gear", {"Fx": lambda t,x: 10}) #forzante step
    lm3.addInputFunction("gear", {"Fx": lambda t, x: 10 * t})  # forzante rampa
    # lm3.addInputFunction("pinion", {"Fy": lambda t,y: 10*np.sin(2*np.pi*5*t)}) #forzante armonica
    # lm3.addInputFunction("pinion", {"Fy": lambda t,y: 10}) #forzante step
    lm3.addInputFunction("pinion", {"Fy": lambda t, y: 10 * t})  # forzante rampa

    model3 = gmc.GenericModel(name="Model_3_X_and_Y")
    model3.addGear(gear)
    model3.addGear(pinion)
    model3.addConstraintManager(cm3)
    model3.addLoadManager(lm3)
    model3.assembleModel("cm_model3", "lm_model3")

    x0_3 = np.zeros(model3.Ac.shape[0])
    for i in range(model3.Ac.shape[0] // 2):
        x0_3[2 * i + 1] = 1

    sim3 = simulation(t_tot=5.0, x0=x0_3, deltat=0.0005, model=model3)
    print("Simulazione Model 3: gear (X only) + pinion (Y only) assieme")
    sim3.solve()
    sim3.plot()
    plt.show()

    # --- MODELLO 4: due gear disaccoppiati (nessuna mesh connection) ---
    # qui vincolo su traslazioni X,Y lascia libera solo la rotazione T per entrambi
    cm4 = gmc.ConstraintManager("cm_model4")
    cm4.addGroundConstraint("gear", ["X", "Y"])  # blocco traslazioni, lascio rotazione
    cm4.addGroundConstraint(
        "pinion", ["X", "Y"]
    )  # blocco traslazioni, lascio rotazione
    lm4 = gmc.LoadManager("lm_model4")
    cm4.addGroundElasticConnection("gear", dofs=["T"], stiffness=[246.5], damping=[5])
    cm4.addGroundElasticConnection("pinion", dofs=["T"], stiffness=[246.5], damping=[5])

    lm4.addInputFunction(
        "gear", {"Tt": lambda t, theta: 10 * np.sin(2 * np.pi * 5 * t)}
    )  # forzante costante
    lm4.addInputFunction("pinion", {"Tt": lambda t, theta: 10 * t})  # forzante costante

    model4 = gmc.GenericModel(name="Model_4_decoupled_rot")
    model4.addGear(gear)
    model4.addGear(pinion)
    model4.addConstraintManager(cm4)
    model4.addLoadManager(lm4)
    model4.assembleModel("cm_model4", "lm_model4")

    x0_4 = np.zeros(model4.Ac.shape[0])
    for i in range(model4.Ac.shape[0] // 2):
        x0_4[2 * i] = 1

    sim4 = simulation(t_tot=3.0, x0=x0_4, deltat=0.0005, model=model4)
    print("Simulazione Model 4: due gear disaccoppiati (rotazioni libere)")
    sim4.solve()
    sim4.plot()
    plt.show()

    # --- MODELLO 5: due gear accoppiati (con mesh connection) ---
    # qui vincolo su traslazioni X,Y lascia libera solo la rotazione T per entrambi
    cm5 = gmc.ConstraintManager("cm_model5")
    cm5.addGroundConstraint("gear", ["X", "Y"])  # blocco traslazioni, lascio rotazione
    cm5.addGroundConstraint(
        "pinion", ["X", "Y"]
    )  # blocco traslazioni, lascio rotazione
    cm5.addMeshConnection("gear", "pinion")  # mesh
    lm5 = gmc.LoadManager("lm_model5")

    lm5.addInputFunction("gear", {"Tt": lambda t, theta: 10})  # forzante costante
    lm5.addInputFunction("pinion", {"Tt": lambda t, theta: 0 * t})  # forzante rampa

    model5 = gmc.GenericModel(name="Model_5_coupled_rot")
    model5.addGear(gear)
    model5.addGear(pinion)
    model5.addConstraintManager(cm5)
    model5.addLoadManager(lm5)
    model5.assembleModel("cm_model5", "lm_model5")

    x0_5 = np.zeros(model5.Ac.shape[0])
    x0_5[0] = 1
    x0_5[2] = 0

    sim5 = simulation(t_tot=3.0, x0=x0_5, deltat=0.0005, model=model5)
    print("Simulazione Model 5: due gear accoppiati (rotazioni legate)")
    sim5.solve()
    sim5.plot()
    plt.show()


def main():

    gear = gm.spurGear(name="gear")
    gear.assignGeometricalProperties(
        module=3.2,
        teethNumber=31,
        pressureAngle=20 * np.pi / 180,
        thickness=0.0381 * 1e3,
    )
    gear.assignMaterialProperties(young=2.068 * 1e5, poisson=0.3)
    gear.assignDynamicProperties(massX=1, massY=1, inertiaT=0.5)
    print("object created")

    pinion = gm.spurGear(name="pinion")
    pinion.assignGeometricalProperties(
        module=3.2,
        teethNumber=19,
        pressureAngle=20 * np.pi / 180,
        thickness=0.0381 * 1e3,
    )
    pinion.assignMaterialProperties(young=2.068 * 1e5, poisson=0.3)
    pinion.assignDynamicProperties(massX=0.5, massY=0.5, inertiaT=0.2)
    print("object created")

    # gear = gm.spurGear(
    #     name="driving_gear",
    #     module=3.2,
    #     teeth_number=31,
    #     pressure_angle=20*np.pi/180,
    #     thickness=0.0381*1e3,
    #     young=2.068*1e5,
    #     poisson=0.3,
    #     rotation_direction="counterclockwise")
    # gear.assign_dynamic_properties(inertia=1)

    # pinion = gm.spurGear.driven_gear(
    #     name="driven_gear",
    #     teeth_number=19,
    #     thickness=0.0381*1e3,
    #     young=2.068*1e5,
    #     poisson=0.3,
    #     other=gear)
    # pinion.assign_dynamic_properties(inertia=0.5)

    modelCM = gmc.ConstraintManager("constraints_sim")
    modelCM.addGroundConstraint("gear", ["X", "Y"])
    modelCM.addGroundConstraint("pinion", ["X", "Y"])
    modelCM.addMeshConnection("gear", "pinion")

    generic_model = gmc.GenericModel(name="ModelloSimulazione")
    generic_model.addGear(gear)
    generic_model.addGear(pinion)
    generic_model.addConstraintManager(modelCM)
    generic_model.assembleModel("constraints_sim")

    simulation1 = simulation(
        t_tot=2, x0=[0.2, 0.5, 0.1, 0.3], deltat=0.0001, model=generic_model
    )
    simulation1.solve()
    simulation1.plot()
    plt.show()

    # model1 = gbm.SpurConnectionRotational(gear, pinion)
    # simulation1 = simulation(
    #     t_tot=20, x0=[0.2, 0.5, 0.1, 0.3], deltat=0.01, model=model1)
    # simulation1.solve()
    # simulation1.plot()
    # plt.show()


if __name__ == "__main__":
    main_debug()
