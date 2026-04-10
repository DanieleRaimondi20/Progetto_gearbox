import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy as sc
import progetto_gearbox.gears_module_v2 as gm

"""
Modulo generico per la costruzione di un modello di ingranaggi.

Questo file definisce due entità principali:
- ConstraintManager: contenitore modulare per vincoli e connessioni (rigide, elastiche, mesh)
- GenericModel: struttura che raccoglie ingranaggi e applica i vincoli definiti dal manager

Nota: il codice contiene parti ancora in sviluppo (metodi placeholder) — qui aggiungo docstring
e commenti esplicativi in italiano per facilitare la manutenzione.
"""

# -------------------------
# Gestione dei vincoli
# -------------------------


class ConstraintManager:
    """Manager modulare per vincoli e ingressi esterni.

    Questa classe conserva elenchi di vincoli di diverso tipo e funge da
    contenitore che poi il modello principale può interrogare per assemblare
    le equazioni del sistema.

    Attributi principali:
    - name: nome del manager
    - groundConstraints: lista di vincoli rigidi verso terra
    - groundElasticConnections: lista di collegamenti elastici/smorzati a terra
    - meshConnections: lista di collegamenti (mesh) tra ingranaggi
    """

    def __init__(self, name):
        # nome identificativo del manager
        self.name = name

        # vincoli rigidi applicati a singoli ingranaggi (es: blocco traslazioni)
        self.groundConstraints = []

        # collegamenti elastici tra un ingranaggio e la 'terra' (k, c per gdl)
        self.groundElasticConnections = []

        # collegamenti tra due ingranaggi (mesh)
        self.meshConnections = []

    def addGroundConstraint(self, gearName: str, dofs: list):
        """Aggiunge un vincolo rigido su `gearName` per i gradi di libertà `dofs`.

        Esempio: addGroundConstraint("pinion", ["X","Y"]) blocca le traslazioni X e Y.
        """
        self.groundConstraints.append(
            {"constrainingGear": gearName, "constrainingDofs": dofs}
        )

    def addGroundElasticConnection(
        self, gearName: str, dofs: list, stiffness: list, damping: list
    ):
        """Aggiunge una connessione elastica/smorzata tra `gearName` e la terra.

        - `dofs`: lista di gradi di libertà coinvolti (ordine coerente con stiffness/damping)
        - `stiffness`, `damping`: valori (o liste) per ciascun dof
        """
        self.groundElasticConnections.append(
            {
                "constrainingGear": gearName,
                "constrainingDofs": dofs,
                "stiffness": stiffness,
                "damping": damping,
            }
        )

    def addMeshConnection(self, gearName1, gearName2):
        """Aggiunge una connessione tra due ingranaggi (mesh).

        `stiffness` e `damping` possono essere valori scalari o callable(t) per dipendenza dal tempo.
        """
        self.meshConnections.append(
            {"constrainingGear1": gearName1, "constrainingGear2": gearName2}
        )

    # # Nota: i metodi assembleConstraints / assemble_inputs qui sotto sono indicativi.
    # # In alcune versioni del progetto il manager tiene riferimenti a `self.model` e a
    # # `self.constraints` — al momento questi attributi non sono definiti in questa
    # # semplice classe. L'implementazione effettiva dipende dall'architettura dell'app.

    # def assembleConstraints(self, A_c: np.ndarray):
    #     """(Placeholder) Aggiorna la matrice A_c con i vincoli noti.

    #     Implementazione attuale: scorre `self.constraints` e modifica A_c. Qui è
    #     lasciata come esempio; l'implementazione reale dovrebbe usare gli elenchi
    #     definiti sopra (groundConstraints, groundElasticConnections, meshConnections)
    #     e l'eventuale mappatura dei gdl del modello.
    #     """
    #     for c in getattr(self, "constraints", []):
    #         if c["type"] == "ground":
    #             for dof in c["dof"]:
    #                 i = self.model.gdl.index(dof)
    #                 A_c[i, :] = 0.0
    #                 A_c[:, i] = 0.0
    #                 A_c[i, i] = 1.0
    #         elif c["type"] == "elastic":
    #             for gdl_i, k, cval in zip(c["gdl"], c["k"], c["c"]):
    #                 i = self.model.gdl.index(gdl_i)
    #                 A_c[i, i] -= (k + cval)
    #     return A_c

    # # Metodo per assemblare ingressi esterni (es. forze o coppie) — placeholder
    # # def assemble_inputs(self, Y_c: np.ndarray, t: float):
    # #     for inp in self.inputs:
    # #         i = self.model.gdl.index(inp["gdl"])
    # #         Y_c[i] += inp["function"](t)
    # #     return Y_c


# -------------------------
# Modello generico che raccoglie ingranaggi e applica vincoli
# -------------------------


class LoadManager:
    """Gestisce funzioni di input (forze/coppie) e rimozioni temporanee."""

    def __init__(self, name: str):

        self.name = name
        # mapping name -> callable(t) -> float
        self.inputFunctions: list[dict[str, callable]] = []
        # nomi marcati per rimozione durante assembleLoads
        self.inputsToRemove: list[str] = []

    def removeInput(self, gearName: str, input: list):
        """Segnala rimozione dell'input (viene applicata da assembleLoads)."""
        self.inputsToRemove.append({"targetGear": gearName, "removingInputs": input})

    def addInputFunction(self, gearName: str, inputFunctionDict: dict[str, callable]):
        inputDict = {
            ":".join([gearName, key]): value for key, value in inputFunctionDict.items()
        }
        self.inputFunctions.append(inputDict)

    # def applyRemovals(self, input_names: list[str]) -> list[str]:
    #     """Ritorna la lista di input_names filtrata per rimozioni."""
    #     return [n for n in input_names if n not in self.inputsToRemove]

    # def evaluate(self, t: float, input_names: list[str]) -> np.ndarray:
    #     """Valuta tutte le inputFunctions richieste e ritorna vettore (m,)"""
    #     vals = []
    #     for name in input_names:
    #         func = self.inputFunctions.get(name, lambda tt: 0.0)
    #         vals.append(func(t))
    #     return np.asarray(vals).reshape(-1)


class GenericModel:
    """Classe contenitore per un insieme di ingranaggi e vincoli.

    Questa classe conserva matrici/state-space parziali per ogni ingranaggio
    ed è responsabile di 'assemblare' il modello completo a partire dai
    singoli componenti e dai ConstraintManager registrati.
    """

    def __init__(self, name):
        self.name = name

        # struttura che raccoglie informazioni per ogni gear aggiunto
        # - names: nomi degli ingranaggi
        # - A, B: matrici di stato locali (possono essere liste di array)
        # - dofs, inputs: metadati sui gradi di libertà e sugli ingressi
        self.gears = {
            "names": [],
            "gears": [],
            "A": [],
            "Aoff": {},
            "B": [],
            "dofs": [],
            "inputs": [],
            "initial_conditions": [],
        }

        # lista di manager (ConstraintManager) registrati
        self.constraintManagers = {"names": [], "managers": []}

        self.loadManagers = {"names": [], "managers": []}

        # Load manager e mappa per gli input
        # self.loadManager = LoadManager()
        # self.input_index_map: dict[str, int] = {}

        # Nota: se si desidera un singolo manager 'attivo', si può aggiungere
        # qui un attributo `self.manager = ...` oppure scegliere il manager
        # tramite `assembleModel(manager_name)` come implementato sotto.

    def addGear(self, gear: gm.spurGear):
        """Aggiunge uno spurGear al modello raccogliendone lo state-space.F

        Si assume che `gear.stateSpace()` ritorni (A, B, dofs, inputs).
        """
        A, B, dofs, inputs = gear.stateSpace()
        self.gears["names"].append(gear.name)
        self.gears["gears"].append(gear)
        self.gears["A"].append(A)
        self.gears["B"].append(B)
        self.gears["dofs"].append(dofs)
        self.gears["inputs"].append(inputs)
        self.gears["initial_conditions"].append(gear.initial_conditions)

    def addConstraintManager(self, constraintManager):
        """Registra un ConstraintManager nel modello (identificato dal .name)."""
        self.constraintManagers["names"].append(constraintManager.name)
        self.constraintManagers["managers"].append(constraintManager)

    def addLoadManager(self, loadManager):
        """Registra un LoadManager nel modello (identificato dal .name)."""
        self.loadManagers["names"].append(loadManager.name)
        self.loadManagers["managers"].append(loadManager)

    def assembleGroundConstraint(self, gear, dofs):
        """Applica un vincolo rigido su `gear` per i `dofs`.

        Trova gli indici locali di stato corrispondenti ai dof richiesti e:
        - annulla riga e colonna corrispondenti nella matrice A locale
        - imposta la diagonale a 1 per ciascun dof vincolato
        - annulla la riga corrispondente in B se le dimensioni lo permettono
        """
        print(f"assembling ground constraint on gear='{gear}' dofs={dofs}")

        if gear not in self.gears["names"]:
            raise ValueError(f"Gear '{gear}' not found in model")

        gidx = self.gears["names"].index(gear)

        # Supportiamo diverse strutture possibili di `dof_meta`.
        for dof in dofs:
            vel = dof + "d"
            if dof == "T":
                inp = "Tt"
            else:
                inp = "F" + dof.lower()

            if dof not in self.gears["dofs"][gidx]:
                raise ValueError(f"Dof '{dof}' not found in Gear '{gear}'")
            if vel not in self.gears["dofs"][gidx]:
                raise ValueError(f"Dof '{vel}' not found in Gear '{gear}'")
            if inp not in self.gears["inputs"][gidx]:
                raise ValueError(f"Input '{inp}' not found in Gear '{gear}'")

            idx_pos = self.gears["dofs"][gidx].index(dof)
            idx_vel = self.gears["dofs"][gidx].index(vel)
            idx_inp = self.gears["inputs"][gidx].index(inp)
            self.gears["A"][gidx] = np.delete(
                self.gears["A"][gidx], [idx_pos, idx_vel], 0
            )
            self.gears["A"][gidx] = np.delete(
                self.gears["A"][gidx], [idx_pos, idx_vel], 1
            )
            self.gears["B"][gidx] = np.delete(
                self.gears["B"][gidx], [idx_pos, idx_vel], 0
            )
            self.gears["B"][gidx] = np.delete(self.gears["B"][gidx], idx_inp, 1)
            del self.gears["dofs"][gidx][idx_pos]
            del self.gears["dofs"][gidx][idx_vel]
            del self.gears["inputs"][gidx][idx_inp]
            del self.gears["initial_conditions"][gidx][idx_pos]
            del self.gears["initial_conditions"][gidx][idx_vel]

    def assembleElasticGroundConnection(self, gear, dofs, stiffness, damping):
        """Applica un vincolo elastico su `gear` per i `dofs` al suolo.

        Trova gli indici locali di stato corrispondenti ai dof richiesti e:
        - aggiunge rigidezza e damping ai 'dofs'
        """
        print(f"assembling ground elastic connection on gear='{gear}' dofs={dofs}")

        if gear not in self.gears["names"]:
            raise ValueError(f"Gear '{gear}' not found in model")

        gidx = self.gears["names"].index(gear)

        for ndof, dof in enumerate(dofs):
            vel = dof + "d"
            if dof not in self.gears["dofs"][gidx]:
                raise ValueError(f"Dof '{dof}' not found in Gear '{gear}'")
            if vel not in self.gears["dofs"][gidx]:
                raise ValueError(f"Dof '{vel}' not found in Gear '{gear}'")

            idx_pos = self.gears["dofs"][gidx].index(dof)
            idx_vel = self.gears["dofs"][gidx].index(vel)

            self.gears["A"][gidx][idx_vel, idx_vel] -= (
                damping[ndof] / self.gears["gears"][gidx].inertia[dof]
            )
            self.gears["A"][gidx][idx_vel, idx_pos] -= (
                stiffness[ndof] / self.gears["gears"][gidx].inertia[dof]
            )

    def checkMeshConnection(self, g1name, g2name):
        if g1name not in self.gears["names"]:
            raise ValueError(f"Gear '{g1name}' not found in model")
        if g2name not in self.gears["names"]:
            raise ValueError(f"Gear '{g2name}' not found in model")

        dofPos = "T"  # posizione angolare
        dofVel = "Td"  # velocità angolare

        # Validazione e ricerca indici per gear1
        if f"{g1name}:{dofPos}" not in self.dofs:
            raise ValueError(f"Dof '{g1name}:{dofPos}' not found in dofs")
        if f"{g1name}:{dofPos}" not in self.dofs:
            raise ValueError(f"Dof '{g1name}:{dofVel}' not found in dofs")

        # Validazione e ricerca indici per gear2
        if f"{g2name}:{dofPos}" not in self.dofs:
            raise ValueError(f"Dof '{g2name}:{dofPos}' not found in dofs")
        if f"{g2name}:{dofPos}" not in self.dofs:
            raise ValueError(f"Dof '{g2name}:{dofVel}' not found in dofs")

    def assembleMeshConnection(self, g1name, g2name, t, x, deltaAc):

        print(f"Sim Time: {t}")
        # ============================================================
        # 1: Identificazione indici dei due ingranaggi
        # ============================================================

        #gear1 sempre guidante e gear2 guidato, passare stringa a engagement function per segno formule
        #usare match case al posto di if else con case driving o driven
        gidx1 = self.gears["names"].index(g1name)
        gidx2 = self.gears["names"].index(g2name)

        gear1 = self.gears["gears"][gidx1]
        gear2 = self.gears["gears"][gidx2]

        """Qui devo in qualche modo richiamare le funzioni dinamiche dei gear.
        Per esempio, se devo calcolare la posizione dei denti di gear 1:
        >> gear1angle = x[idxPos1, 0]  # estraggo l'angolo di gear1 dallo stato
        >> gear1teethPos = gear1.TeethPosition(gear1angle)
        """
        gear1_angle_idx = self.dofs.index(f"{g1name}:T")
        gear1_omega_idx = self.dofs.index(f"{g1name}:Td")
        gear1_x_idx = self.dofs.index(f"{g1name}:X") 
        gear1_y_idx = self.dofs.index(f"{g1name}:Y")
        gear1_angle = x[gear1_angle_idx, 0] 
        gear1_omega = x[gear1_omega_idx, 0] 
        gear1_x = x[gear1_x_idx, 0] 
        gear1_y = x[gear1_y_idx, 0] 
        
        gear2_angle_idx = self.dofs.index(f"{g2name}:T")
        gear2_omega_idx = self.dofs.index(f"{g2name}:Td")
        gear2_x_idx = self.dofs.index(f"{g2name}:X") 
        gear2_y_idx = self.dofs.index(f"{g2name}:Y")
        gear2_angle = x[gear2_angle_idx, 0]
        gear2_omega = x[gear2_omega_idx, 0]
        gear2_x = x[gear2_x_idx, 0] 
        gear2_y = x[gear2_y_idx, 0] 
        
        gamma = np.arctan((gear2_y - gear1_y)/(gear2_x - gear1_x)) # angolo di linea d'azione
        gear1_teeth_pos = gear1.computeTeethPosition(gear1_angle)
        gear2_teeth_pos = gear2.computeTeethPosition(gear2_angle)
        gear1_engagement, gear1_alpha1 = gear1.computeTeethEngagement(gear1_teeth_pos, "driving", gamma, gear1_omega, gear2)
        gear2_engagement, gear2_alpha2 = gear2.computeTeethEngagement(gear2_teeth_pos, "driven", gamma, gear2_omega, gear1)
        gear1_mesh_stiffness = gear1.meshStiffness(gear1_engagement, gear1_alpha1)
        gear2_mesh_stiffness = gear2.meshStiffness(gear2_engagement, gear2_alpha2)

        # Normalizza stiffness e damping (possono essere scalari o liste)
        I1 = gear1.inertia["T"]
        I2 = gear2.inertia["T"]
        Rb1 = gear1.radius["base"]
        Rb2 = gear2.radius["base"]
        #km1 = gear1.exampleMeshStiffness(alpha1_i)
        #km2 = gear2.exampleMeshStiffness(alpha2i)
        #km = 1 / (1 / km1 + 1 / km2)
        kt = gear1.meshStiffness(gear1_engagement, alpha1)
        #cm1 = gear1.exampleMeshDamping(t,x)
        #cm2 = gear2.exampleMeshDamping(t,x)
        #cm = 1 / (1 / cm1 + 1 / cm2)

        # ============================================================
        # 2: Identificazione posizione (indice) del gdl "T" per i due ingranaggi
        # ============================================================
        dofPos = "T"  # posizione angolare
        dofVel = "Td"  # velocità angolare

        idxPos1 = self.dofs.index(f"{g1name}:{dofPos}")
        idxVel1 = self.dofs.index(f"{g1name}:{dofVel}")
        
        gear1angle = x[idxPos1, 0]  # estraggo l'angolo di gear1 dallo stato
        gear1teethPos = gear1.TeethPosition(gear1angle)

        idxPos2 = self.dofs.index(f"{g2name}:{dofPos}")
        idxVel2 = self.dofs.index(f"{g2name}:{dofVel}")

        gear2angle = x[idxPos2, 0]  # estraggo l'angolo di gear2 dallo stato
        gear2teethPos = gear1.TeethPosition(gear2angle)
        
        # ============================================================
        # 3: Costruzione matrice A(gidx1, gidx2) con coupling mesh
        # ============================================================

        # Modifica matrice A di gear1 - coupling con gear2

        deltaAc[idxVel1, idxVel1] -= cm * Rb1**2 / I1  # -c/I1 on θ̇1 term
        deltaAc[idxVel1, idxPos1] -= kt * Rb1**2 / I1  # -k/I1 on θ1 term
        deltaAc[idxVel1, idxVel2] -= cm * Rb1 * Rb2 / I1  # -c/I1 on θ̇1 term
        deltaAc[idxVel1, idxPos2] -= kt * Rb1 * Rb2 / I1  # -k/I1 on θ1 term

        deltaAc[idxVel2, idxVel2] -= cm * Rb2**2 / I2  # -c/I1 on θ̇1 term
        deltaAc[idxVel2, idxPos2] -= kt * Rb2**2 / I2  # -k/I1 on θ1 term
        deltaAc[idxVel2, idxVel1] -= cm * Rb1 * Rb2 / I2  # -c/I1 on θ̇1 term
        deltaAc[idxVel2, idxPos1] -= kt * Rb1 * Rb2 / I2  # -k/I1 on θ1 term
        return deltaAc

    def assembleRemovedInput(self, gear, inputs):
        """Sincronizza `self.inputs` con `self.Bc` usando il LoadManager."""
        print(f"Removing inputs={inputs} on gear='{gear}'")

        if gear not in self.gears["names"]:
            raise ValueError(f"Gear '{gear}' not found in model")

        gidx = self.gears["names"].index(gear)

        # Supportiamo diverse strutture possibili di `dof_meta`.
        for inp in inputs:
            if inp == "Tt":
                dof = "T"
                vel = "Td"
            elif inp == "Fx":
                dof = "X"
                vel = "Xd"
            elif inp == "Fy":
                dof = "Y"
                vel = "Yd"
            else:
                raise ValueError(
                    f"Input='{inp}' is invalid. It must be either 'Fx', 'Fy' or 'Tt'"
                )

            if dof not in self.gears["dofs"][gidx]:
                raise ValueError(f"Dof '{dof}' not found in Gear '{gear}'")
            if vel not in self.gears["dofs"][gidx]:
                raise ValueError(f"Dof '{vel}' not found in Gear '{gear}'")
            if inp not in self.gears["inputs"][gidx]:
                raise ValueError(f"Input '{inp}' not found in Gear '{gear}'")

            idx_pos = self.gears["dofs"][gidx].index(dof)
            idx_vel = self.gears["dofs"][gidx].index(vel)
            idx_inp = self.gears["inputs"][gidx].index(inp)
            self.gears["B"][gidx] = np.delete(self.gears["B"][gidx], idx_inp, 1)
            del self.gears["inputs"][gidx][idx_inp]

    def assembleModel(self, constraintManagerName, loadManagerName):
        """Costruisce il modello completo a partire da un ConstraintManager.

        Viene recuperato il manager registrato con `constraintManagerName` e
        vengono iterate le sue liste per applicare i vincoli (rigidi, elastici,
        mesh). Al momento i metodi di applicazione sono placeholder.
        """

        # 1) Recupera il manager registrato
        if constraintManagerName not in self.constraintManagers["names"]:
            raise ValueError(f"ConstraintManager '{constraintManagerName}' not found")
        cmIdx = self.constraintManagers["names"].index(constraintManagerName)
        cmanager = self.constraintManagers["managers"][cmIdx]
        self.activeConstraintManager = cmanager

        # 1) Recupera il manager registrato
        if loadManagerName not in self.loadManagers["names"]:
            raise ValueError(f"LoadManager '{loadManagerName}' not found")
        lmIdx = self.loadManagers["names"].index(loadManagerName)
        lmanager = self.loadManagers["managers"][lmIdx]
        self.activeLoadManager = lmanager

        for groundConstraint in cmanager.groundConstraints:
            self.assembleGroundConstraint(
                groundConstraint["constrainingGear"],
                groundConstraint["constrainingDofs"],
            )

        for groundElasticConnection in cmanager.groundElasticConnections:
            self.assembleElasticGroundConnection(
                groundElasticConnection["constrainingGear"],
                groundElasticConnection["constrainingDofs"],
                groundElasticConnection["stiffness"],
                groundElasticConnection["damping"],
            )

        for inputToRemove in lmanager.inputsToRemove:
            self.assembleRemovedInput(
                inputToRemove["targetGear"], inputToRemove["removingInputs"]
            )

        self.dofs = []
        self.inputs = []
        self.inputFunctions = {}

        for gidx, gname in enumerate(self.gears["names"]):
            print(f"Assembling Gear N° {gidx}: '{gname}'")
            for dof in self.gears["dofs"][gidx]:
                self.dofs.append(f"{gname}:{dof}")
            for inp in self.gears["inputs"][gidx]:
                self.inputs.append(f"{gname}:{inp}")

        self.Ac = sc.linalg.block_diag(*self.gears["A"])
        self.Bc = sc.linalg.block_diag(*self.gears["B"])

        for meshConnection in cmanager.meshConnections:
            self.checkMeshConnection(
                meshConnection["constrainingGear1"], meshConnection["constrainingGear2"]
            )
            
            cgidx1 = self.gears["names"].index(meshConnection["constrainingGear1"])
            cgidx2 = self.gears["names"].index(meshConnection["constrainingGear2"])

            cgear1 = self.gears["gears"][cgidx1]
            cgear2 = self.gears["gears"][cgidx2]
            
            theta0_2, thetad0_2 = cgear2.derive_initial_conditions(cgear1, gamma)
            self.gears["initial_conditions"][cgidx2]["T"] = theta0_2
            self.gears["initial_conditions"][cgidx2]["Td"] = thetad0_2
            
            
        for inputFunction in lmanager.inputFunctions:
            for targetInput, targetFunction in inputFunction.items():
                if targetInput not in self.inputs:
                    raise ValueError(f"Input '{targetInput}' not found in inputs.")
                else:
                    self.inputFunctions[targetInput] = targetFunction

        print("Model succesfully assembled")

    def equations(self, t=None, x=None):
        """Costruisce il vettore risultato delle equazioni di stato: A*x + B*u.

        - Valida la shape di x
        - Aggrega deltaAc dalle mesh
        - Valuta gli ingressi tramite `self.loadManager` e restituisce f(t,x) = (Ac+deltaAc)@x + Bc@u
        """
        if x is None:
            raise ValueError("State vector x must be provided")
        x = np.asarray(x).reshape((-1, 1))
        # if x.size != self.Ac.shape[0]:
        # raise ValueError(f"State vector length {x.size} != {self.Ac.shape[0]} (Ac.shape[0])")

        deltaAc = np.zeros(self.Ac.shape)
        for meshConnection in self.activeConstraintManager.meshConnections:
            deltaAc = self.assembleMeshConnection(
                meshConnection["constrainingGear1"],
                meshConnection["constrainingGear2"],
                t,
                x,
                deltaAc,
            )

        # valuta ingressi tramite LoadManager (1D array)

        u = np.array([self.inputFunctions[inp](t, x) for inp in self.inputs]).reshape(
            [-1, 1]
        )
        # u = self.loadManager.evaluate(t, self.inputs) if len(self.inputs) > 0 else np.zeros(0)

        return (self.Ac + deltaAc) @ x + self.Bc @ u


def main():
    """Esempio minimale di costruzione oggetti e assemblaggio modello.

    Il main crea due ingranaggi di prova, un ConstraintManager e prova a
    registrare i componenti e assemblare il modello. Serve principalmente a
    esemplificare l'uso delle classi definite in questo modulo.
    """
    gear = gm.spurGear(name="gearProva")
    gear.assignGeometricalProperties(
        module=3.2,
        teethNumber=31,
        pressureAngle=20 * np.pi / 180,
        thickness=0.0381 * 1e3,
    )
    gear.assignMaterialProperties(young=2.068 * 1e5, poisson=0.3)
    gear.assignDynamicProperties(massX=1, massY=1, inertiaT=0.5)
    print("object created")

    pinion = gm.spurGear(name="pinionProva")
    pinion.assignGeometricalProperties(
        module=3.2,
        teethNumber=19,
        pressureAngle=20 * np.pi / 180,
        thickness=0.0381 * 1e3,
    )
    pinion.assignMaterialProperties(young=2.068 * 1e5, poisson=0.3)
    pinion.assignDynamicProperties(massX=0.5, massY=0.5, inertiaT=0.2)
    print("object created")

    model = GenericModel(name="ModelloProva")
    model.addGear(gear)
    model.addGear(pinion)

    modelCM = ConstraintManager("constraints1")
    modelCM.addGroundConstraint("gearProva", ["Y"])
    modelCM.addGroundConstraint("pinionProva", ["X", "Y"])
    # modelCM.addGroundElasticConnection(
    #     "pinionProva", ["X", "Y"], [1e4, 1e5], [0.5, 0.8])
    modelCM.addMeshConnection("gearProva", "pinionProva")

    modelLM = LoadManager("loads1")
    modelLM.removeInput("gearProva", ["Fx"])
    modelLM.addInputFunction("gearProva", {"Tt": lambda t, x: np.sin(t)})
    modelLM.addInputFunction("pinionProva", {"Tt": lambda t, x: 1.0})

    model.addConstraintManager(modelCM)
    model.addLoadManager(modelLM)

    model.assembleModel("constraints1", "loads1")

    x0 = np.zeros(model.Ac.shape[0])
    xd = model.equations(0.01, x0)
    print("gear aggiunto al modello")


if __name__ == "__main__":
    main()
