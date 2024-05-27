# SMaRt &mdash; Official PyTorch implementation

> **SMaRt: Improving GANs with Score Matching Regularity (ICML 2024)** <br>
> Mengfei Xia, Yujun Shen, Ceyuan Yang, Ran Yi, Wenping Wang, Yong-Jin Liu <br>

[[Paper](https://arxiv.org/pdf/2311.18208)]

Abstract: *Generative adversarial networks (GANs) usually struggle in learning from highly diverse data, whose underlying manifold is complex. In this work, we revisit the mathematical foundations of GANs, and theoretically reveal that the native adversarial loss for GAN training is insufficient to fix the problem of subsets with positive Lebesgue measure of the generated data manifold lying out of the real data manifold. Instead, we find that score matching serves as a promising solution to this issue thanks to its capability of persistently pushing the generated data points towards the real data manifold. We thereby propose to improve the optimization of GANs with score matching regularity (SMaRt). Regarding the empirical evidences, we first design a toy example to show that training GANs by the aid of a ground-truth score function can help reproduce the real data distribution more accurately, and then confirm that our approach can consistently boost the synthesis performance of various state-of-the-art GANs on real-world datasets with pre-trained diffusion models acting as the approximate score function. For instance, when training Aurora on the ImageNet 64 × 64 dataset, we manage to improve FID from 8.87 to 7.11, on par with the performance of one-step consistency model.*

## TODO List

- [ ] Release inference code and [Aurora](https://github.com/zhujiapeng/Aurora) generator on ImageNet 64x64.
- [ ] Release training code.

## References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{xia2024smart,
  title={SMaRt: Improving GANs with Score Matching Regularity},
  author={Xia, Mengfei and Shen, Yujun and Yang, Ceyuan and Yi, Ran and Wang, Wenping and Liu, Yong-Jin},
  booktitle={ICML},
  year={2024},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.
