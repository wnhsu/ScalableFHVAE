from .fhvae import FHVAE
from .simple_fhvae import SimpleFHVAE

def load_model(name):
    if name == "fhvae":
        return FHVAE
    elif name == "simple_fhvae":
        return SimpleFHVAE
    else:
        raise ValueError
