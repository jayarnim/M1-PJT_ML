from .opt import opt_dataLoader
from .msr import msr_dataLoader


DATALOADER_REGISTRY = {
    "opt": opt_dataLoader,
    "msr": msr_dataLoader,
}