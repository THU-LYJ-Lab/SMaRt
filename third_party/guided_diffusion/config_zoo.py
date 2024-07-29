# python3.8

"""Contains all configurations for different resolutions."""

__all__ = ['get_backbone_config', 'get_diffusion_config']


_BACKBONE_CONFIG_ZOO = {
    'adm_imagenet128_cond': dict(
        image_size=128,
        num_classes=1_000,
        num_channels=256,
        num_res_blocks=2,
        channel_mult='1,1,2,3,4',
        learn_sigma=True,
        class_cond=True,
        attention_resolutions=[32, 16, 8],
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_new_attention_order=False,
    ),
    'adm_imagenet64_cond': dict(
        image_size=64,
        num_classes=1_000,
        num_channels=192,
        num_res_blocks=3,
        channel_mult='1,2,3,4',
        learn_sigma=True,
        class_cond=True,
        attention_resolutions=(32, 16, 8),
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.1,
        resblock_updown=True,
        use_new_attention_order=True,
    ),
    'adm_lsun_bedroom256_uncond': dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        channel_mult='1,1,2,2,4,4',
        learn_sigma=True,
        class_cond=True,
        attention_resolutions=(32, 16, 8),
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.1,
        resblock_updown=True,
        use_new_attention_order=True,
    ),
}


_DIFFUSION_CONFIG_ZOO = {
    'adm_imagenet128_cond': dict(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing='',
    ),
    'adm_imagenet64_cond': dict(
        steps=1000,
        learn_sigma=True,
        noise_schedule='cosine',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing='',
    ),
    'adm_lsun_bedroom256_uncond': dict(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing='',
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
