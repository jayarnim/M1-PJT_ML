from .fdd import FusedDirectionalDistance
from .mse import MeanSquaredError


LOSS_FN_REGISTRY = {
    "fdd": FusedDirectionalDistance,
    "mse": MeanSquaredError,
}