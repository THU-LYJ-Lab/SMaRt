# python3.8
"""Contains the official constructor of EDM."""

from third_party.edm.networks import EDMPrecond

__all__ = ['build_edm_model']

def build_edm_model(backbone_kwargs):
    return EDMPrecond(**backbone_kwargs)
