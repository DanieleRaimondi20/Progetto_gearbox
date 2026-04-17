from numpy import pi, sin, full

def sinusoidal(amplitude: float=1, frequency: float=1, phase: float = 0):
    return lambda t: amplitude*sin(2*pi*frequency*t + phase)

def step(step_value=1):
    return lambda t: step_value if isinstance(t,float) else full(t.shape,step_value)

def ramp(angular_coefficient=1):
    return lambda t: angular_coefficient * t

