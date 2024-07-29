# python3.8
"""Contains the code to synthesize images from a pre-trained models."""
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from models import build_model


def run_mapping(G, z, label):
    """Run mapping network of the generator."""
    with torch.no_grad():
        # For Aurora.
        try:
            mapping_results = G.mapping(z,
                                        label=label,
                                        context=None)
        # For StyleGAN2.
        except TypeError:
            mapping_results = G.mapping(z,
                                        label=label)
    return mapping_results['wp']


def run_synthesis(G, wp):
    """Run synthesis network of the generator."""
    with torch.no_grad():
        # For Aurora.
        try:
            res = G.synthesis(wp, context=None)
        # For StyleGAN2.
        except TypeError:
            res = G.synthesis(wp)
    return res


def read_text(text_path):
    """Prepare snapshot text that will be used for evaluation."""
    print(f'Loading text from {text_path}')
    with open(text_path) as f:
        text = [line.strip() for line in f.readlines()]
    return text


def parse_float(arg):
    """Parse float number in string."""
    if not arg:
        return None
    arg = arg.split(',')
    arg = [float(i) for i in arg]
    return arg


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', type=str, default='',
                        help='Path to the pre-trained models.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--syn_num', type=int, default=100,
                        help='Number of synthesized images.')
    parser.add_argument('--result_dir', type=str, default='work_dirs/synthesis',
                        help='Results directory.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--save_png', action='store_true',
                        help='Save png or npz.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    state = torch.load(args.weight_path, map_location='cpu')
    G = build_model(**state['model_kwargs_init']['generator_smooth'])
    G.load_state_dict(state['models']['generator_smooth'])
    G.eval().cuda()

    # Deterministic sampling.
    g1 = torch.Generator()
    g1.manual_seed(args.seed)
    g2 = torch.Generator()
    g2.manual_seed(args.seed)

    # Sampling.
    syn_num = args.syn_num
    batch_size = args.batch_size
    num_batches = (syn_num + batch_size - 1) // batch_size
    all_images = list()
    for idx in tqdm(range(num_batches)):
        with torch.no_grad():
            batch_codes = torch.randn((batch_size, G.z_dim),
                                      generator=g1).cuda()
            if G.label_dim == 0:
                batch_labels = torch.zeros(batch_size, 0).cuda()
            else:
                rnd_labels = torch.randint(
                    low=0, high=G.label_dim, size=(batch_size,),
                    generator=g2).cuda()
                batch_labels = F.one_hot(
                    rnd_labels, num_classes=G.label_dim)

            batch_wps = run_mapping(G, batch_codes, batch_labels)
            batch_images = run_synthesis(G, batch_wps)['image']
            batch_images = ((batch_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            batch_images = batch_images.permute(0, 2, 3, 1)
            batch_images = batch_images.contiguous().cpu().numpy()
            all_images.append(batch_images)

    all_images = np.concatenate(all_images, axis=0)[:syn_num]
    if args.save_png:
        for idx, image in enumerate(all_images):
            save_path = os.path.join(args.result_dir, f'{idx}.png')
            Image.fromarray(image).save(save_path)
    else:
        shape_str = "x".join([str(x) for x in all_images.shape])
        save_path = os.path.join(args.result_dir, f"{shape_str}-samples.npz")
        np.savez(save_path, all_images)


if __name__ == '__main__':
    main()
