import sys
import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.utils.data  # torch.utils.data.DataLoader is used fully-qualified below

import wandb
from accelerate import Accelerator

from dataloaders.dataset import get_the_well_dataset
from dataloaders.metadata import DATASET_DICT

sys.path.append(os.path.join(os.path.dirname(__file__), "neuraloperator"))

from neuralop.models import FNO
from neuralop.utils import count_model_params, get_wandb_api_key

################################################################
# configs
################################################################
parser = argparse.ArgumentParser(description='Train an FNO on one dataset from The Well')

parser.add_argument('--model', type=str, default='FNO', choices=['FNO'])
parser.add_argument('--rollout_steps', type=int, default=12,
                    help='number of autoregressive steps to roll out during evaluation')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--modes', type=int, default=16)
parser.add_argument('--t_in', type=int, default=1)
parser.add_argument('--t_out', type=int, default=1)

### FNO params
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--width', type=int, default=16)
parser.add_argument('--rollout_eval_start', type=int, default=1)
parser.add_argument('--rollout_eval_end', type=int, default=2)
parser.add_argument('--test_every_n', type=int, default=1,
                    help='run the eval loop every N epochs; the final epoch is always evaluated')
parser.add_argument('--eval_at_end_only', type=int, default=0,
                    help='1: skip periodic eval and evaluate once, at the final epoch or the --max_train_hours deadline')
parser.add_argument('--max_train_hours', type=float, default=0.0,
                    help='>0: stop after this many wall-clock hours, running a final eval first')
parser.add_argument('--factorization', type=str, default='Tucker')
parser.add_argument('--rank', type=float, default=0.1)

### training params
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_dir', type=str, default='./runs')
parser.add_argument('--domain_padding', type=float, default=0)

parser.add_argument('--dataset', type=str, default='rayleigh_benard')
parser.add_argument('--wandb_project', type=str, default='the-well-baselines')
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--debug_mode', type=int, default=0)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def vrmse_per_batch(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Root Variance-Scaled RMSE (VRMSE) matching The Well's implementation:
      VRMSE = sqrt( MSE_spatial(y_pred, y_true) / ( Var_spatial(y_true) + eps ) )

    - Spatial reduction is over all dims after the channel dim.
    - Var_spatial uses sample variance (std with unbiased=True) to mirror the repo.

    Args:
        y_pred: [B, C, *spatial] predictions (denormalized/physical units)
        y_true: [B, C, *spatial] ground truth (denormalized/physical units)
        eps:    small constant to avoid divide-by-zero

    Returns:
        Scalar VRMSE averaged over batch and channels.
    """
    assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape"
    if y_pred.isnan().any() or y_true.isnan().any():
        raise ValueError("y_pred or y_true contains NaN values")
    spatial_dims = tuple(range(2, y_true.ndim))

    # MSE over space per (B, C)
    mse_spatial = ((y_pred - y_true) ** 2).mean(dim=spatial_dims)  # [B, C]

    # Variance of truth over space per (B, C)
    var_spatial = y_true.std(dim=spatial_dims, unbiased=True) ** 2  # [B, C]

    # VRMSE per (B, C), then average over channels and batch
    vrmse_bc = torch.sqrt(mse_spatial / (var_spatial + eps))        # [B, C]

    return vrmse_bc.mean()

def train(args):
    accelerator = Accelerator(split_batches=False)
    device = accelerator.device

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        torch.cuda.set_device(local_rank)
    except Exception:
        pass
    if torch.cuda.is_available():
        print(f"[Rank {accelerator.process_index}] using device: cuda:{local_rank}")
    else:
        print(f"[Rank {accelerator.process_index}] using CPU (no CUDA available)")

    set_seed(args.seed)

    train_dataset = get_the_well_dataset(args, 'train')
    test_dataset = get_the_well_dataset(args, 'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False
    )

    print(f"[Rank {accelerator.process_index}] train size: {len(train_loader)}, test size: {len(test_loader)}")

    V_id = DATASET_DICT[args.dataset]['variable_ids']
    dataset_in_size = DATASET_DICT[args.dataset]['in_size']
    is_3d_dataset = len(dataset_in_size) == 3

    if args.model == 'FNO':
        model = FNO(
            n_modes = (args.modes, args.modes),
            hidden_channels = args.width,
            in_channels=len(V_id),
            out_channels=len(V_id),
            projection_channel_ratio = 4,
            lifting_channel_ratio=4,
            n_layers = args.n_layers,
            domain_padding = args.domain_padding,
            factorization = args.factorization,
            rank = args.rank,
        )
        if is_3d_dataset:
            model = FNO(
            n_modes = (args.modes, args.modes, args.modes),
            hidden_channels = args.width,
            in_channels=len(V_id),
            out_channels=len(V_id),
            projection_channel_ratio = 4,
            lifting_channel_ratio=4,
            n_layers = args.n_layers,
            domain_padding = args.domain_padding,
            factorization = args.factorization,
            rank = args.rank,
        )
    else:
        raise ValueError(f"model {args.model} not supported")

    print(f"parameters: {count_model_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6)
    loss_fn = torch.nn.MSELoss()

    best_metric = float('inf')

    # Run identifier, used verbatim as the W&B run name, the checkpoint dir name,
    # and the checkpoint filename prefix: <config>__<dd-mm-yy>_<HH-MM-SS>
    config_name = (
        f"{args.dataset}_{args.model}"
        f"_L{args.n_layers}_M{args.modes}_W{args.width}"
        f"_tout{args.t_out}_norm{args.normalize}"
        f"_bs{args.batch_size}_lr{args.lr}_S{args.seed}"
    )
    run_date = time.strftime('%d-%m-%y')
    run_name = f"{config_name}__{run_date}_{time.strftime('%H-%M-%S')}"

    wandb_run_id = None

    if accelerator.is_main_process:
        base_log_dir = os.path.expanduser(os.path.expandvars(args.log_dir))
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"run: {run_name}")
        print(f"saving checkpoints to {log_dir}")

        if not args.debug_mode:
            wandb_dir = os.path.join(log_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            _key_file = os.environ.get("WANDB_KEY_FILE", "")
            if os.environ.get("WANDB_API_KEY"):
                wandb.login(key=os.environ["WANDB_API_KEY"])
            elif _key_file and os.path.isfile(_key_file):
                wandb.login(key=get_wandb_api_key(_key_file))
            else:
                wandb.login()
            wandb.init(
                project=args.wandb_project,
                config={**vars(args), 'run_name': run_name, 'run_date': run_date},
                name=run_name,
                dir=wandb_dir,
            )
            wandb_run_id = wandb.run.id if wandb.run is not None else None
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")
            wandb.define_metric("batch_step")
            wandb.define_metric("batch/*", step_metric="batch_step")
        else:
            print("debug mode: skipping wandb init")

        with open(os.path.join(log_dir, 'run_info.json'), 'w') as _f:
            json.dump({
                'run_name': run_name,
                'run_date': run_date,
                'wandb_run_id': wandb_run_id,
                'wandb_project': args.wandb_project,
                'log_dir': os.path.abspath(log_dir),
                'config': vars(args),
            }, _f, indent=2)
        print(f"wrote {os.path.join(log_dir, 'run_info.json')}")

    accelerator.wait_for_everyone()

    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    global_batch_step = 0
    train_start = time.time()  # wall-clock start, for the --max_train_hours deadline
    print(f"[Rank {accelerator.process_index}] starting training loop on device {device}")

    avg_test_loss = float('nan')
    avg_test_single = float('nan')
    avg_test_rollout = float('nan')

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = []
        start_time = time.time()

        for i, data in enumerate(train_loader):
            x, y = data['x'], data['y']  # accelerator.prepare dataloader will handle device placement
            optimizer.zero_grad()
            output_single = model(x)
            y_next = y if args.t_out == 1 else y[:, 0]
            loss = loss_fn(output_single, y_next)
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process and i % 10 == 0 and not args.debug_mode:
                wandb.log({
                    "batch/train_loss": loss.item(),
                    "batch_step": global_batch_step
                })

            global_batch_step += 1
            total_train_loss.append(loss.item())

        avg_train_loss = float(np.mean(total_train_loss)) if total_train_loss else float('nan')
        scheduler.step(avg_train_loss)
        epoch_time = time.time() - start_time

        if accelerator.is_main_process:
            print(f"epoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f}s")
            print(f"train loss: {avg_train_loss:.6f}")
            print(f"lr: {scheduler.get_last_lr()[0]:.2e}")

            if not args.debug_mode:
                wandb.log({
                    "train/loss": avg_train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch_time": epoch_time,
                    "epoch": epoch
                })

        elapsed_h = (time.time() - train_start) / 3600.0
        time_up = args.max_train_hours > 0 and elapsed_h >= args.max_train_hours
        if args.eval_at_end_only:
            do_eval = (epoch == args.epochs - 1) or time_up
        else:
            do_eval = ((epoch + 1) % args.test_every_n == 0) or (epoch == args.epochs - 1) or time_up
        if do_eval:
            model.eval()
            total_test_loss = []
            total_test_single_vrmse = []
            total_test_rollout_vrmse = []

            if accelerator.is_main_process:
                print(f"epoch {epoch}: evaluating on {len(test_loader)} test batches")

            with torch.no_grad():
                for i, test_data in enumerate(test_loader):
                    if accelerator.is_main_process and i % 5 == 0:
                        print(f"processing test batch {i+1}/{len(test_loader)}")

                    test_x, test_y = test_data['x'], test_data['y']
                    test_output_single = model(test_x)
                    test_y_next = test_y if args.t_out == 1 else test_y[:, 0]

                    if train_dataset.normalize:
                        test_output = train_dataset.denormalize_data(test_output_single)
                        test_y_next_denorm = train_dataset.denormalize_data(test_y_next)
                    else:
                        test_output = test_output_single
                        test_y_next_denorm = test_y_next

                    single_vrmse = vrmse_per_batch(test_output, test_y_next_denorm)
                    total_test_single_vrmse.append(single_vrmse.item())

                    mse_loss_val = loss_fn(test_output_single, test_y_next).item()
                    total_test_loss.append(mse_loss_val)

                    rollout_steps = args.rollout_steps
                    preds = []
                    x_t = test_x

                    for step in range(rollout_steps):
                        out_t = model(x_t)
                        preds.append(out_t.unsqueeze(1))  # [B,1,C,H,W]
                        if torch.isnan(out_t).any() or torch.isinf(out_t).any():
                            if accelerator.is_main_process:
                                print(f"WARNING: NaN detected at rollout step {step} in batch {i}")
                            break
                        x_t = out_t.detach()

                    preds = torch.cat(preds, dim=1) # [B, rollout_steps, C, H, W]

                    if args.t_out == 1:
                        preds_window = preds[:, :1] # [B, 1, C, *spatial]
                        B, T = preds_window.shape[:2]
                        preds_window = preds_window.reshape(B * T, *preds_window.shape[2:])
                        y_window = test_y.reshape(B, 1, *test_y.shape[1:]).reshape(B * T, *test_y.shape[1:])
                    else:
                        preds_window = preds[:, args.rollout_eval_start:args.rollout_eval_end] # [B, T, C, *spatial]
                        B, T = preds_window.shape[:2]
                        # reshapes to (B*T, C, *spatial)
                        preds_window = preds_window.reshape(B * T, *preds_window.shape[2:])
                        y_window = test_y[:, args.rollout_eval_start:args.rollout_eval_end].reshape(B * T, *test_y.shape[2:])

                    if train_dataset.normalize:
                        preds_window = train_dataset.denormalize_data(preds_window)
                        y_window = train_dataset.denormalize_data(y_window)

                    rollout_vrmse = vrmse_per_batch(preds_window, y_window)
                    total_test_rollout_vrmse.append(rollout_vrmse.item())

            avg_test_loss = float(np.mean(total_test_loss)) if total_test_loss else float('nan')
            avg_test_single = float(np.mean(total_test_single_vrmse)) if total_test_single_vrmse else float('nan')
            avg_test_rollout = float(np.mean(total_test_rollout_vrmse)) if total_test_rollout_vrmse else float('nan')

            if accelerator.is_main_process and not args.debug_mode:
                wandb.log({
                    'test/loss': avg_test_loss,
                    'test/single_step_vrmse': avg_test_single,
                    'test/rollout_vrmse': avg_test_rollout,
                    'test/rollout_window_start': args.rollout_eval_start,
                    'test/rollout_window_end': args.rollout_eval_end,
                    'epoch': epoch
                })

        if accelerator.is_main_process:
            # last.pt every epoch; best.pt only on eval epochs 
            ckpt = {
                'epoch': epoch,
                'model': accelerator.unwrap_model(model).state_dict(),
                'best_metric': best_metric,
                'avg_test_single_vrmse': avg_test_single,
                'avg_test_rollout_vrmse': avg_test_rollout,
                'avg_test_loss': avg_test_loss,
                'run_name': run_name,
                'run_date': run_date,
                'wandb_run_id': wandb_run_id,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(log_dir, f'{run_name}__last.pt'))
            if do_eval and avg_test_single < best_metric:
                best_metric = avg_test_single
                ckpt['best_metric'] = best_metric
                torch.save(ckpt, os.path.join(log_dir, f'{run_name}__best.pt'))
                print(f"new best single-step vRMSE: {best_metric:.6f} (saved {run_name}__best.pt)")

        if time_up:
            if accelerator.is_main_process:
                print(f"hit max_train_hours ({args.max_train_hours}h) at epoch {epoch+1} "
                      f"after {elapsed_h:.2f}h, stopping")
            break

    if accelerator.is_main_process:
        print("\ntraining complete")
        if not args.debug_mode:
            wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"args: {args}")

    set_seed(args.seed)
    train(args)