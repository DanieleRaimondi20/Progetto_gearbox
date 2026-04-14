from numpy.typing import NDArray
from numpy import pi

def wrapTo2Pi(angle: NDArray) -> NDArray:
    return angle % (2 * pi)


def wrapToPi(angles: NDArray) -> NDArray:
    return (angles + pi) % (2 * pi) - pi