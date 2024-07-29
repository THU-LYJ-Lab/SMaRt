# python3.7
"""Collects all models."""

from .stylegan2_generator import StyleGAN2Generator
from .text_generator import Text2ImageGenerator
from .official_guided_diffusion_model import build_diffusion_model
from .official_edm import build_edm_model

__all__ = ['build_model']

_MODELS = {
    'StyleGAN2Generator': StyleGAN2Generator,
    'Text2ImageGenerator': Text2ImageGenerator,
    'GuidedDiffusionModel': build_diffusion_model,
    'EDMModel': build_edm_model,
}


def build_model(model_type, **kwargs):
    """Builds a model based on its class type.

    Args:
        model_type: Class type to which the model belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the model.

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[model_type](**kwargs)
