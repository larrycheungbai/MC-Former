"""MC-Former inference using the original Restormer implementation.

This script requires the Restormer source code (not its pretrained weights).
See https://github.com/swz30/Restormer for installation instructions.

Example from the MC-Former repository root:
    python test.py --restormer_root ../Restormer --input_dir ./sample_images

The inference preprocessing, model architecture, checkpoint key, and image
postprocessing follow the original MC-Former test script.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from skimage import img_as_ubyte
from tqdm import tqdm


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_DIR / 'Deblurring_V3_1200k_with_overlap_downstream_last_clean_img.yml'
DEFAULT_WEIGHTS = REPO_DIR / 'Model' / 'net_g_1028000.pth'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Correct motion-corrupted brain MRI PNG/JPG images using MC-Former.'
    )
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='Directory of input PNG/JPG images (or one image file).')
    parser.add_argument('--result_dir', type=Path, default=REPO_DIR / 'results',
                        help='Output directory for restored PNG images.')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS,
                        help='MC-Former checkpoint; defaults to Model/net_g_1028000.pth.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='YAML configuration file containing network_g.')
    parser.add_argument('--restormer_root', type=Path, default=None,
                        help='Path to cloned Restormer root containing basicsr/ and Motion_Deblurring/.')
    return parser.parse_args()


def import_restormer_dependencies(restormer_root):
    """Use the external Restormer code without copying it into this repository."""
    if restormer_root is not None:
        root = restormer_root.expanduser().resolve()
        arch_path = root / 'basicsr' / 'models' / 'archs' / 'restormer_arch.py'
        utils_path = root / 'Motion_Deblurring' / 'utils.py'
        if not arch_path.is_file() or not utils_path.is_file():
            raise FileNotFoundError(
                f'Cannot find Restormer source files under {root}. Expected:\n'
                f'  {arch_path}\n  {utils_path}\n'
                'Pass --restormer_root pointing to the Restormer repository root.'
            )
        sys.path.insert(0, str(root / 'Motion_Deblurring'))
        sys.path.insert(0, str(root))
    try:
        import utils
        from basicsr.models.archs.restormer_arch import Restormer
    except ImportError as exc:
        raise ImportError(
            'Restormer modules could not be imported. Install the original Restormer '
            'project and specify --restormer_root /path/to/Restormer.'
        ) from exc
    return utils, Restormer


def find_images(input_path):
    input_path = input_path.expanduser().resolve()
    valid = {'.png', '.jpg', '.jpeg'}
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in valid else []
    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {input_path}')
    return natsorted([p for p in input_path.iterdir()
                      if p.is_file() and p.suffix.lower() in valid],
                     key=lambda p: str(p))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The current MC-Former inference script requires an NVIDIA CUDA GPU.')

    for path, label in ((args.config, 'Configuration'), (args.weights, 'Checkpoint')):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f'{label} file not found: {path}')

    images = find_images(args.input_dir)
    if not images:
        raise RuntimeError(f'No PNG/JPG images found at: {args.input_dir}')

    utils, Restormer = import_restormer_dependencies(args.restormer_root)

    with args.config.expanduser().open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or not isinstance(config.get('network_g'), dict):
        raise ValueError('The YAML file must contain a network_g configuration mapping.')
    network_params = dict(config['network_g'])
    if network_params.pop('type', None) != 'Restormer':
        raise ValueError('Expected network_g.type: Restormer')

    model = Restormer(**network_params)
    # Keep the same checkpoint convention as the original working test.py.
    checkpoint = torch.load(str(args.weights.expanduser()), map_location='cpu')
    if 'params' not in checkpoint:
        raise KeyError('Checkpoint is missing the expected "params" key.')
    model.load_state_dict(checkpoint['params'], strict=True)
    model = model.cuda().eval()

    output_dir = args.result_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'MC-Former weights: {args.weights}')
    print(f'Model config: {args.config}')
    print(f'Input images: {len(images)}')
    print(f'Output folder: {output_dir}')

    factor = 8
    with torch.no_grad():
        for image_path in tqdm(images, desc='MC-Former inference'):
            # Original MC-Former preprocessing: RGB, float32, divide by 255.
            image = np.float32(utils.load_img(str(image_path))) / 255.0
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'Expected a 3-channel image: {image_path}')
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
            height, width = tensor.shape[-2:]
            pad_h = (factor - height % factor) % factor
            pad_w = (factor - width % factor) % factor
            # Reflection padding matches the original script.
            if (pad_h and pad_h >= height) or (pad_w and pad_w >= width):
                raise ValueError(f'Input image is too small for reflect padding: {image_path}')
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

            restored = model(tensor)
            restored = restored[:, :, :height, :width]
            restored = (torch.clamp(restored, 0, 1).cpu()
                        .permute(0, 2, 3, 1).squeeze(0).numpy())
            output_path = output_dir / f'{image_path.stem}.png'
            utils.save_img(str(output_path), img_as_ubyte(restored))

    print(f'Completed: {len(images)} restored image(s) saved in {output_dir}')


if __name__ == '__main__':
    main()
