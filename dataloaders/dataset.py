import os
from functools import partial

import h5py
import torch
import yaml
from torch.utils.data import Dataset

from dataloaders.metadata import DATASET_DICT, CHANNELS


class WellHDF5Dataset(Dataset):
    def __init__(self, hdf5_dir, channels, n_spatial_dims,
                 t_in=1, t_out=1, trajs_len=None, normalize=False):
        self.channels = channels
        self.n_spatial_dims = n_spatial_dims
        self.trajs_len = trajs_len
        assert self.trajs_len is not None, "trajs_len must be provided"

        self.normalize = normalize
        self.t_in = t_in
        self.t_out = t_out

        self.hdf5_files = sorted(
            os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.hdf5')
        )

        # The trajectory count comes from the first time-varying channel: its
        # leading axis is the trajectory axis. Static fields are unusable here
        # ('global' ones have no trajectory axis at all).
        probe = next(c for c in self.channels if c.static is None)

        self.actual_trajs_per_file = []
        self.file_sample_counts = []
        self.cumulative_samples = [0]

        print(f"scanning {len(self.hdf5_files)} HDF5 files to determine trajectory counts...")
        for i, file_path in enumerate(self.hdf5_files):
            with h5py.File(file_path, 'r') as f:
                actual_trajs = f[probe.group][probe.field].shape[0]
            self.actual_trajs_per_file.append(actual_trajs)
            samples_in_file = actual_trajs * (self.trajs_len - (t_in + t_out) + 1)
            self.file_sample_counts.append(samples_in_file)
            self.cumulative_samples.append(self.cumulative_samples[-1] + samples_in_file)
            print(f"file {i+1}/{len(self.hdf5_files)}: {os.path.basename(file_path)} - "
                  f"{actual_trajs} trajectories, {samples_in_file} samples")

        self.total_samples_length = sum(self.file_sample_counts)
        self.extract_data_funcs = [partial(self.extract_data, hdf5_path=f) for f in self.hdf5_files]
        print(f"Total samples across all files: {self.total_samples_length}")
        # hdf5_dir is <root>/<dataset>/data/<split>; stats.yaml sits at <root>/<dataset>.
        self.stats = yaml.load(open(os.path.join(hdf5_dir.split('/data/')[0], 'stats.yaml')),
                               Loader=yaml.FullLoader)
        # Guards the normalization divisor. Unrelated to the eps inside VRMSE.
        self.norm_eps = 1e-6

    # -- reading ------------------------------------------------------------
    def extract_data(self, hdf5_path, trajectory_idx, start_idx, end_idx):

        time_steps = end_idx - start_idx
        reads = {}  # (group, field) -> array, so a vector field is read once and sliced
        out = []
        with h5py.File(hdf5_path, 'r') as f:
            for c in self.channels:
                key = (c.group, c.field)
                if key not in reads:
                    dset = f[c.group][c.field]
                    if c.static is None:
                        reads[key] = dset[trajectory_idx, start_idx:end_idx]
                    elif c.static == 'per_traj':
                        reads[key] = dset[trajectory_idx]
                    else:
                        reads[key] = dset[:]
                arr = reads[key]
                if c.component is not None:
                    comp = c.component if isinstance(c.component, tuple) else (c.component,)
                    arr = arr[(Ellipsis,) + comp]
                t = torch.from_numpy(arr)
                if c.static is not None:
                    # no time axis in the file, so repeat the field across the window
                    t = t.unsqueeze(0).repeat(time_steps, *([1] * t.ndim))
                out.append(t.unsqueeze(1))
        return torch.cat(out, dim=1)

    def _stat(self, kind, channel):
        """stats['mean'|'std'] for a channel, following its component index."""
        value = self.stats[kind][channel.field]
        if channel.component is not None:
            comp = channel.component if isinstance(channel.component, tuple) else (channel.component,)
            for i in comp:
                value = value[i]
        return value

    def _channel_axis(self, x):
        """0 for an unbatched [C, *spatial] tensor, 1 for a batched [B, C, *spatial] one."""
        if x.ndim == self.n_spatial_dims + 1:
            return 0
        if x.ndim == self.n_spatial_dims + 2:
            return 1
        raise ValueError(
            f"expected [C, *spatial] or [B, C, *spatial] with {self.n_spatial_dims} spatial dims, got shape {tuple(x.shape)}")

    def normalize_data(self, x):
        if not self.normalize:
            return x
        axis = self._channel_axis(x)
        out = x.clone()
        for i, c in enumerate(self.channels):
            sel = (slice(None),) * axis + (i,)
            out[sel] = (x[sel] - self._stat('mean', c)) / (self._stat('std', c) + self.norm_eps)
        return out

    def denormalize_data(self, x):
        if not self.normalize:
            return x
        axis = self._channel_axis(x)
        out = x.clone()
        for i, c in enumerate(self.channels):
            sel = (slice(None),) * axis + (i,)
            out[sel] = x[sel] * (self._stat('std', c) + self.norm_eps) + self._stat('mean', c)
        return out

    def get_corresponding_std(self):
        return torch.tensor([self._stat('std', c) for c in self.channels])
    def __len__(self):
        return self.total_samples_length

    def __getitem__(self, idx):
        # idx is a global index over every window in every file. Walk the
        # cumulative counts to find the file, then split the offset within it
        # into a trajectory and a start time.
        hdf5_file_idx = 0
        for i, cumulative_count in enumerate(self.cumulative_samples[1:]):
            if idx < cumulative_count:
                hdf5_file_idx = i
                break

        local_idx = idx - self.cumulative_samples[hdf5_file_idx]
        actual_trajs_in_file = self.actual_trajs_per_file[hdf5_file_idx]
        samples_per_traj = self.trajs_len - (self.t_in + self.t_out) + 1

        trajectory_idx = local_idx // samples_per_traj
        time_step_idx = local_idx % samples_per_traj
        assert trajectory_idx < actual_trajs_in_file, (
            f"Trajectory index {trajectory_idx} exceeds available trajectories {actual_trajs_in_file} in file {hdf5_file_idx}")

        extract = self.extract_data_funcs[hdf5_file_idx]
        x = extract(trajectory_idx=trajectory_idx,
                    start_idx=time_step_idx, end_idx=time_step_idx + self.t_in)
        y = extract(trajectory_idx=trajectory_idx,
                    start_idx=time_step_idx + self.t_in,
                    end_idx=time_step_idx + self.t_in + self.t_out)

        x, y = x.squeeze(0), y.squeeze(0) # drop the time axis: [C, *spatial]

        if self.normalize:
            x = self.normalize_data(x)
            y = self.normalize_data(y)

        return {'x': x, 'y': y}


def get_the_well_dataset(args, set_type):
    assert set_type in ['train', 'test'], "set_type must be either train or test"
    meta = DATASET_DICT[args.dataset]
    return WellHDF5Dataset(
        meta[f'{set_type}_dir'],
        channels=CHANNELS[args.dataset],
        n_spatial_dims=len(meta['in_size']),
        t_in=args.t_in,
        t_out=args.t_out,
        trajs_len=meta['t_total'],
        normalize=args.normalize,
    )
