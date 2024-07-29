# python3.8

"""Contains all configurations for different model type."""

from .gaussian_diffusion import GaussianDiffusion

__all__ = ['get_backbone_config', 'get_diffusion_config', 'GaussianDiffusion']


_BACKBONE_CONFIG_ZOO = {
    'edm_imagenet64_cond': dict(
        img_resolution=64,
        img_channels=3,
        label_dim=1000,
        augment_dim=0,
        model_channels=192,
        channel_mult=(1, 2, 3, 4),
        dropout=0.1,
        use_fp16=True,
        model_type='DhariwalUNet',
    ),
    'edm_cifar10_cond': dict(
        img_resolution=32,
        img_channels=3,
        label_dim=10,
        resample_filter=[1, 1],
        embedding_type='positional',
        augment_dim=9,
        model_channels=128,
        channel_mult=(2, 2, 2),
        dropout=0.13,
        use_fp16=False,
        encoder_type='standard',
        channel_mult_noise=1,
        model_type='SongUNet',
    )
}


_DIFFUSION_CONFIG_ZOO = {
    'edm_imagenet64_cond': dict(
        num_timesteps=1000,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
    ),
    'edm_cifar10_cond': dict(
        num_timesteps=1000,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
    ),
}


def get_backbone_config(config_type):
    """Return the backbone configuration based on the resolution.
    
    Args:
        config_type: The type of configuration of the Diffusion.

    Raises:
        ValueError: If the `config_type` is not supported.
    """
    if config_type not in _BACKBONE_CONFIG_ZOO:
        raise ValueError(f'Invalid config type: `{config_type}`!\n'
                         f'Resolutions allowed: {list(_BACKBONE_CONFIG_ZOO)}.')
    return _BACKBONE_CONFIG_ZOO[config_type]


def get_diffusion_config(config_type):
    """Return the diffusion configuration based on the resolution.
    
    Args:
        config_type: The type of configuration of the Diffusion.

    Raises:
        ValueError: If the `config_type` is not supported.
    """
    if config_type not in _DIFFUSION_CONFIG_ZOO:
        raise ValueError(f'Invalid config type: `{config_type}`!\n'
                         f'Resolutions allowed: {list(_DIFFUSION_CONFIG_ZOO)}.')
    return _DIFFUSION_CONFIG_ZOO[config_type]
