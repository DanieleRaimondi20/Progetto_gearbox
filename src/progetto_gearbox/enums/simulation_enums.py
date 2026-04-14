from enum import Enum


class SimulationMethod(str, Enum):
    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    RADAU = "Radau"
    BDF = "BDF"
    LSODA = "LSODA"
