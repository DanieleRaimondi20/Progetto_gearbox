from numpy import pi, sin, full, zeros, logical_and, inf
from numpy.typing import NDArray

def sinusoidal(amplitude: float=1, frequency: float=1, phase: float = 0):
    return lambda t: amplitude*sin(2*pi*frequency*t + phase)

def step(step_value: float = 1, t_start: float = 0, t_end: float = inf):
    def _step_float(t, step_value: float = 1, t_start: float = 0, t_end: float = inf):
        return step_value if t>=t_start and t <= t_end else 0.0
    
    def _step_array(t: NDArray, step_value: float = 1, t_start: float = 0, t_end: float = inf):
        input = zeros(t.shape, dtype=float)
        input[logical_and(t >= t_start, t <= t_end)] = step_value
        return input

    return lambda t: _step_float(t, step_value=step_value, t_start=t_start, t_end=t_end) if isinstance(t, float) else _step_array(t, step_value=step_value, t_start=t_start, t_end=t_end)
        

def constant(value: float = 1):
    return step(step_value=value)

def ramp(angular_coefficient=1):
    return lambda t: angular_coefficient * t

def PD(kp = 1000, kd = 0, pos_set = 0, vel_set = 0, pos_idx = 0, vel_idx = 0):
    return lambda t, x: kp*(x[pos_idx] - pos_set)

