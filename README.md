# SMaRt &mdash; Official PyTorch implementation

> **SMaRt: Improving GANs with Score Matching Regularity (ICML 2024)** <br>
> Mengfei Xia, Yujun Shen, Ceyuan Yang, Ran Yi, Wenping Wang, Yong-Jin Liu <br>

[[Paper](https://arxiv.org/pdf/2311.18208)]

Abstract: *Generative adversarial networks (GANs) usually struggle in learning from highly diverse data, whose underlying manifold is complex. In this work, we revisit the mathematical foundations of GANs, and theoretically reveal that the native adversarial loss for GAN training is insufficient to fix the problem of subsets with positive Lebesgue measure of the generated data manifold lying out of the real data manifold. Instead, we find that score matching serves as a promising solution to this issue thanks to its capability of persistently pushing the generated data points towards the real data manifold. We thereby propose to improve the optimization of GANs with score matching regularity (SMaRt). Regarding the empirical evidences, we first design a toy example to show that training GANs by the aid of a ground-truth score function can help reproduce the real data distribution more accurately, and then confirm that our approach can consistently boost the synthesis performance of various state-of-the-art GANs on real-world datasets with pre-trained diffusion models acting as the approximate score function. For instance, when training Aurora on the ImageNet 64 × 64 dataset, we manage to improve FID from 8.87 to 7.11, on par with the performance of one-step consistency model.*

## Installation

This repository is developed based on [Hammer](https://github.com/bytedance/Hammer), where you can find more detailed instructions on installation. Here, we summarize the necessary steps to facilitate reproduction.

1. Environment: CUDA version == 11.1.

2. Install package requirements with `conda`:

    ```shell
    conda create -n smart python=3.8  # create virtual environment with Python 3.8
    conda activate smart
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
    pip install -r requirements/minimal.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
    pip install protobuf==3.20
    pip install absl-py einops ftfy==6.1.1 
    ```

## Inference

First, please download the pre-trained model following the links below.

- [Aurora on ImageNet 64](https://drive.google.com/file/d/1vmJCW6gGV6Odzw6jqat8uQ02NcuQoIHP/view?usp=sharing)
- [StyleGAN2 on LSUN Bedroom 256](https://drive.google.com/file/d/1tBhDxa0ocjt0zmAAZoX1ISKTHB4msCV-/view?usp=sharing)

To synthesize images, you can use the following command

```bash
# Synthesize using Aurora on ImageNet64 with SMaRt.
python run_synthesize.py smart_aurora_imagenet64.pth --syn_num 50000

# Synthesize using StyleGAN2 on LSUN Bedroom with SMaRt.
python run_synthesize.py smart_stylegan2_lsun_bedroom256.pth --syn_num 50000
```

## TODO List

- [x] Release inference code, [Aurora](https://github.com/zhujiapeng/Aurora) generator on ImageNet 64x64, and [StyleGAN2](https://github.com/NVlabs/stylegan2) generator on LSUN Bedroom 256x256.
- [ ] Release training code.

## References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{xia2024smart,
  title={SMaRt: Improving GANs with Score Matching Regularity},
  author={Xia, Mengfei and Shen, Yujun and Yang, Ceyuan and Yi, Ran and Wang, Wenping and Liu, Yong-Jin},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.

## Acknowledgements

We highly appreciate [StyleGAN2](https://github.com/NVlabs/stylegan2), [Aurora](https://github.com/zhujiapeng/Aurora), [ADM](https://github.com/openai/guided-diffusion), [EDM](https://github.com/NVlabs/edm), and [Hammer](https://github.com/bytedance/Hammer) for their contributions to the community.
