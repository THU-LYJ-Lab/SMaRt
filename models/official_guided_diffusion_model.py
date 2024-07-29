# python3.8
"""Contains the official constructor of ADM.
"""

from third_party.guided_diffusion.script_util import create_model_from_config

__all__ = ['build_diffusion_model']

def build_diffusion_model(backbone_kwargs):
    return create_model_from_config(backbone_kwargs)
