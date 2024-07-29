# python3.8

import third_party.guided_diffusion as adm
import third_party.edm as edm


_MODELS = {
    'adm_imagenet128_cond': 'GuidedDiffusionModel',
    'adm_imagenet64_cond': 'GuidedDiffusionModel',
    'adm_lsun_bedroom256_uncond': 'GuidedDiffusionModel',
    'edm_imagenet64_cond': 'EDMModel',
    'edm_cifar10_cond': 'EDMModel',
}


def get_model_type(config_type):
    """Get the model type based on the config type.
    
    Args:
        config_type: Config type to which the model belongs, which is case
            sensitive.
            
    Raises:
        ValueError: If the `config_type` is not supported.
    """
    if config_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{config_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[config_type]


def get_backbone_config(config_type):
    """Get the backbone type based on the config type.
    
    Args:
        config_type: Config type to which the model belongs, which is case
            sensitive.
    """
    if 'adm' in config_type.lower():
        return adm.config_zoo.get_backbone_config(config_type)
    elif 'edm' in config_type.lower():
        return edm.get_backbone_config(config_type)
    else:
        raise ValueError(f'Wrong config type `{config_type}`.')


def get_diffusion_config(config_type):
    """Get the diffusion type based on the config type.
    
    Args:
        config_type: Config type to which the model belongs, which is case
            sensitive.
    """
    if 'adm' in config_type.lower():
        return adm.config_zoo.get_diffusion_config(config_type)
    elif 'edm' in config_type.lower():
        return edm.get_diffusion_config(config_type)
    else:
        raise ValueError(f'Wrong config type `{config_type}`.')


def create_gaussian_diffusion(config_type, diffusion_kwargs):
    if 'adm' in config_type.lower():
        return adm.script_util.create_gaussian_diffusion(**diffusion_kwargs)
    elif 'edm' in config_type.lower():
        return edm.GaussianDiffusion(**diffusion_kwargs)
    else:
        raise ValueError(f'Wrong config type `{config_type}`.')
