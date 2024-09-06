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

## Training

Implementing SMaRt is based on the objective functions below:

$$\mathcal L_{score}=\mathbb E_{\mathbf z,\boldsymbol\epsilon,t}[\|\boldsymbol\epsilon_\theta(\alpha_tg_\phi(\mathbf z)+\sigma_t\boldsymbol\epsilon,t)-\boldsymbol\epsilon\|_2^2]\quad\text{unconditional GAN, Equation (10),}$$
$$\mathcal L_{score}=\mathbb E_{\mathbf z,\boldsymbol\epsilon,t,c}[\|\boldsymbol\epsilon_\theta(\alpha_tg_\phi(\mathbf z,c)+\sigma_t\boldsymbol\epsilon,c,t)-\boldsymbol\epsilon\|_2^2]\quad\text{conditional GAN, Equation (15).}$$

Therefore, it is necessary to implement the score matching objective using pre-trained DPMs. We provide the pseudo-code below for conditional generation:

```python
def forward_step(cur_iter, freq, G, DPM, z, c, t, lambda_score=0.1):
    """Define the forward process of one training step.

    Args:
        cur_iter: Current iteration, determining whether to use SMaRt.
        freq: Frequency to involve SMaRt.
        G: The generator module to learn.
        DPM: The pre-trained DPM, fixed while training.
        z: Random noise inputted to G.
        c: Condition inputted to G.
        t: Preset timestep for score matching.
        lambda_score: Loss weight.
    """
    # Directly skip.
    if cur_iter % freq != 0:
        return None

    # Generate fake images with G.
    image = G(z, c)

    # Forward diffusing process.
    noise = torch.randn_like(image)
    x_t = alpha_t * image + sigma_t * noise

    # Calculate the score matching regularity.
    noise_pred = DPM(x_t, c, t)
    loss = (noise - noise_pred).square().mean()

    return loss * lambda_score
```

According to Table 6 in Appendix, we provide the empirical value of hyper-parameters used in our experiments.

|           Dataset |     CIFAR10 | ImageNet 64 | ImageNet 128 |  LSUN Bedroom |
| :---------------: | :---------: | :---------: | :----------: | :-----------: |
|           Setting | Conditional | Conditional |  Conditional | Unconditional |
|     Dataset Scale |  50K Images | 1.3M Images |  1.3M Images |     3M Images |
| $\lambda_{score}$ |      $0.01$ |       $0.1$ |        $0.1$ |         $0.1$ |
|               $t$ |   $[40,60]$ |   $[25,35]$ |    $[25,35]$ |     $[25,35]$ |
|         Frequency |         $8$ |         $8$ |          $8$ |           $8$ |

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
