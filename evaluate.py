#!/usr/bin/env python3
"""Compute test-split VRMSE for a released checkpoint.

    export WELL_DATASETS_PATH=/path/to/well/datasets
    python evaluate.py --ckpt ckpts/shear_flow.pt

Checkpoints live at https://huggingface.co/yashpincha/TheWellBaselines, one
<dataset>.pt per task. Both the dataset and the architecture come from the
config the checkpoint recorded under 'args' when it was trained.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuraloperator"))

from neuralop.models import FNO
from dataloaders.dataset import get_the_well_dataset
from dataloaders.metadata import DATASET_DICT


LIFTING_RATIO = PROJECTION_RATIO = 4
CONFIG_KEYS = ('n_layers', 'modes', 'width', 'rank', 'factorization', 'domain_padding', 'normalize', 'batch_size')


def vrmse(pred, true, eps=1e-7):
    """sqrt(MSE / variance) per sample and channel, averaged. Matches The Well."""
    spatial = tuple(range(2, true.ndim)) # inputs are [B, C, *spatial]
    mse = ((pred - true) ** 2).mean(dim=spatial)
    var = true.std(dim=spatial, unbiased=True) ** 2
    return torch.sqrt(mse / (var + eps)).mean()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def training_args(ckpt):
    args = ckpt.get('args') if isinstance(ckpt, dict) else None
    if not args:
        sys.exit("checkpoint records no 'args', so its architecture is unknown")
    missing = [k for k in CONFIG_KEYS if k not in args]
    if missing:
        sys.exit(f"checkpoint's 'args' is missing {missing}")
    return {k: args[k] for k in CONFIG_KEYS}


def build_model(cfg, n_channels, is_3d):
    n_modes = (cfg['modes'],) * (3 if is_3d else 2)
    return FNO(
        n_modes=n_modes,
        hidden_channels=cfg['width'],
        in_channels=n_channels,
        out_channels=n_channels,
        n_layers=cfg['n_layers'],
        lifting_channel_ratio=LIFTING_RATIO,
        projection_channel_ratio=PROJECTION_RATIO,
        domain_padding=cfg['domain_padding'],
        factorization=cfg['factorization'],
        rank=cfg['rank'],
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', help='path to a <dataset>.pt checkpoint')
    p.add_argument('--dataset', help='dataset name, if not the checkpoint filename')
    p.add_argument('--split', default='test', choices=['test', 'train'])
    p.add_argument('--batch-size', type=int, help='batch size for evaluation')
    p.add_argument('--max-batches', type=int, default=0, help='stop early (0 runs the full split)')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p, p.parse_args()


def resolve_dataset(parser, cli_name, ckpt, ckpt_path):
    recorded = (ckpt.get('args') or {}).get('dataset') if isinstance(ckpt, dict) else None
    name = cli_name or recorded or os.path.splitext(os.path.basename(ckpt_path))[0]
    return name


def main():
    parser, args = parse_args()

    if not args.ckpt:
        parser.error('--ckpt is required')
    if not os.path.isfile(args.ckpt):
        parser.error(f"no such checkpoint: {args.ckpt}\n"
                     "get one from https://huggingface.co/yashpincha/TheWellBaselines")

    ckpt = load_checkpoint(args.ckpt)
    dataset = resolve_dataset(parser, args.dataset, ckpt, args.ckpt)
    cfg = training_args(ckpt)

    meta = DATASET_DICT[dataset]
    n_channels = len(meta['variable_ids'])
    is_3d = len(meta['in_size']) == 3
    print(f"{dataset}: {n_channels} channels on a {'x'.join(map(str, meta['in_size']))} grid")
    print(f"config: {cfg['n_layers']} layers, {cfg['modes']} modes, width {cfg['width']}, "
          f"{cfg['factorization']} rank {cfg['rank']}, domain_padding {cfg['domain_padding']}")

    model = build_model(cfg, n_channels, is_3d)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        sys.exit("these weights do not fit that architecture:\n"
                 + "\n".join(str(exc).splitlines()[:4]))
    if missing or unexpected:
        sys.exit(f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected "
                 f"keys, e.g. {(list(missing) + list(unexpected))[:3]}")

    device = torch.device(args.device)
    model = model.to(device).eval()
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    loader_args = argparse.Namespace(dataset=dataset, normalize=cfg['normalize'], t_in=1, t_out=1)
    data = get_the_well_dataset(loader_args, args.split)
    batch_size = args.batch_size or cfg['batch_size']
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=(device.type == 'cuda'))
    total = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    print(f"{args.split} split: {len(data)} samples in {total} batches of {batch_size}")

    scores = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = batch['x'].to(device), batch['y'].to(device)
            pred = data.denormalize_data(model(x))
            scores.append(vrmse(pred, data.denormalize_data(y)).item())
            if (i + 1) % 10 == 0:
                print(f"batch {i + 1}/{total}, VRMSE so far {np.mean(scores):.4f}")

    print(f"\nsingle-step VRMSE: {np.mean(scores):.6f} over {len(scores)} batches")

    reported = ckpt.get('avg_test_single_vrmse') if isinstance(ckpt, dict) else None
    if reported is not None and np.isfinite(reported):
        print(f"recorded in the checkpoint: {reported:.6f}")

if __name__ == '__main__':
    main()
