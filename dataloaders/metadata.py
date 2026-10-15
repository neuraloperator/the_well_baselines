"""Per-dataset metadata for The Well benchmark.

DATASET_DICT maps a dataset name to its split directories, grid size, trajectory
length and channel (variable) names.
"""
import os

# Customizable path for the well datasets.
# Set WELL_DATASETS_PATH to the directory that contains the per-dataset folders
# (e.g. .../well/datasets, which holds acoustic_scattering_maze/data/train, ...).
BASE_DATASET_PATH = os.environ.get('WELL_DATASETS_PATH')
if BASE_DATASET_PATH is None:
    # Use relative path from project root (two levels up from dataloaders/metadata.py)
    project_root = os.path.dirname(os.path.dirname(__file__))
    BASE_DATASET_PATH = os.path.join(project_root, 'datasets')
BASE_DATASET_PATH = os.path.expanduser(os.path.expandvars(BASE_DATASET_PATH))

DATASET_DICT = {}

def get_dataset_paths(dataset_name):
    """Helper function to generate dataset paths"""
    base_path = os.path.join(BASE_DATASET_PATH, dataset_name, 'data')
    return {
        'train_dir': os.path.join(base_path, 'train'),
        'valid_dir': os.path.join(base_path, 'valid'),
        'test_dir': os.path.join(base_path, 'test')
    }

### The Well Benchmark
### Acoustic Scattering Maze DataSet
name = 'acoustic_scattering_maze'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 202
DATASET_DICT[name]['in_size'] = (256, 256)
DATASET_DICT[name]['variable_ids'] = ['density', 'pressure', 'speed_of_sound', 'u_x', 'u_y']

### Active Matter Dataset
name = 'active_matter'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 81
DATASET_DICT[name]['in_size'] = (256, 256)
DATASET_DICT[name]['variable_ids'] = ['concentration', 'u_x', 'u_y', 'D_xx', 'D_xy', 'D_yx', 'D_yy', 'E_xx', 'E_xy', 'E_yx', 'E_yy']

### Gray Scott Reaction Diffusion Dataset
name = 'gray_scott_reaction_diffusion'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 1001
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['variable_ids'] = ['A', 'B']

### Rayleigh Benard Dataset
name = 'rayleigh_benard'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 200
DATASET_DICT[name]['in_size'] = (512, 128)
DATASET_DICT[name]['variable_ids'] = ['b', 'p', 'u_x', 'u_y']

### Turbulent Radiative Layer 2D Dataset
name = 'turbulent_radiative_layer_2D'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 101
DATASET_DICT[name]['in_size'] = (128, 384)
DATASET_DICT[name]['variable_ids'] = ['density', 'pressure', 'velocity_x', 'velocity_y']

### Viscoelastic Instability Dataset
name = 'viscoelastic_instability'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 20
DATASET_DICT[name]['in_size'] = (512, 512)
DATASET_DICT[name]['variable_ids'] = ['c_zz', 'p', 'u_x', 'u_y', 'C_xx', 'C_xy', 'C_yy']

### Convective Envelope RSG Dataset
name = 'convective_envelope_rsg'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 100
DATASET_DICT[name]['in_size'] = (256, 128, 256)  # 3D dataset: [r, theta, phi]
DATASET_DICT[name]['variable_ids'] = ['density', 'energy', 'pressure', 'velocity_r', 'velocity_theta', 'velocity_phi']

### Euler Multi Quadrants Dataset
for name in ('euler_multi_quadrants_periodicBC', 'euler_multi_quadrants_openBC'):
    DATASET_DICT[name] = get_dataset_paths(name)

    DATASET_DICT[name]['t_total'] = 101
    DATASET_DICT[name]['in_size'] = (512, 512)
    DATASET_DICT[name]['variable_ids'] = ['density', 'energy', 'pressure', 'momentum_x', 'momentum_y']

DATASET_DICT['euler_multi_quadrants'] = DATASET_DICT['euler_multi_quadrants_periodicBC']

### Helmholtz Staircase Dataset
name = 'helmholtz_staircase'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 50
DATASET_DICT[name]['in_size'] = (1024, 256)
DATASET_DICT[name]['variable_ids'] = ['mask', 'pressure_re', 'pressure_im']

### MHD 64 Dataset
name = 'MHD_64'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 100
DATASET_DICT[name]['in_size'] = (64, 64, 64)
DATASET_DICT[name]['variable_ids'] = ['density', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z', 'velocity_x', 'velocity_y', 'velocity_z']

### Planet SWE Dataset
name = 'planetswe'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 1008
DATASET_DICT[name]['in_size'] = (256, 512)
DATASET_DICT[name]['variable_ids'] = ['height', 'velocity_theta', 'velocity_phi']

### Post Neutron Star Merger Dataset
name = 'post_neutron_star_merger'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 181
DATASET_DICT[name]['in_size'] = (192, 128, 66)
DATASET_DICT[name]['variable_ids'] = ['density', 'electron_fraction', 'entropy', 'internal_energy', 'pressure', 'temperature', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z', 'velocity_x', 'velocity_y', 'velocity_z']

### Rayleigh Taylor Instability Dataset
name = 'rayleigh_taylor_instability'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 119
DATASET_DICT[name]['in_size'] = (128, 128, 128)
DATASET_DICT[name]['variable_ids'] = ['density', 'velocity_x', 'velocity_y', 'velocity_z']

### Shear Flow Dataset
name = 'shear_flow'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 200
DATASET_DICT[name]['in_size'] = (256, 512)
DATASET_DICT[name]['variable_ids'] = ['pressure', 'tracer', 'velocity_x', 'velocity_y']

### Supernova Explosion 64 Dataset
name = 'supernova_explosion_64'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 59
DATASET_DICT[name]['in_size'] = (64, 64, 64)
DATASET_DICT[name]['variable_ids'] = ['density', 'pressure', 'temperature', 'velocity_x', 'velocity_y', 'velocity_z']

### Turbulence Gravity Cooling Dataset
name = 'turbulence_gravity_cooling'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 50
DATASET_DICT[name]['in_size'] = (64, 64, 64)
DATASET_DICT[name]['variable_ids'] = ['density', 'pressure', 'temperature', 'velocity_x', 'velocity_y', 'velocity_z']

### Turbulent Radiative Layer 3D Dataset
name = 'turbulent_radiative_layer_3D'
DATASET_DICT[name] = get_dataset_paths(name)

DATASET_DICT[name]['t_total'] = 101
DATASET_DICT[name]['in_size'] = (128, 128, 256)
DATASET_DICT[name]['variable_ids'] = ['density', 'pressure', 'velocity_x', 'velocity_y', 'velocity_z']

from collections import namedtuple

# CHANNELS[dataset] is the channel axis of every sample tensor, in order, so it
# has to line up with variable_ids. Each entry locates one channel:
#
#     ch(T0, 'pressure')       ->  f['t0_fields']['pressure']
#     ch(T1, 'velocity', 0)    ->  f['t1_fields']['velocity'][..., 0]
#     ch(T2, 'D', (0, 1))      ->  f['t2_fields']['D'][..., 0, 1]
#
# component is also the index into stats.yaml: the velocity channel above
# normalizes with stats['mean']['velocity'][0].
#
# Fields with no time axis are repeated across the window and say so with
# static='per_traj' (one per trajectory) or static='global' (one per file).
Channel = namedtuple('Channel', ['group', 'field', 'component', 'static'])


def ch(group, field, component=None, static=None):
    return Channel(group, field, component, static)


T0, T1, T2 = 't0_fields', 't1_fields', 't2_fields'

CHANNELS = {
    'acoustic_scattering_maze': [
        ch(T0, 'density', static='per_traj'),
        ch(T0, 'pressure'),
        ch(T0, 'speed_of_sound', static='per_traj'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
    ],
    'active_matter': [
        ch(T0, 'concentration'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
        ch(T2, 'D', (0, 0)), ch(T2, 'D', (0, 1)), ch(T2, 'D', (1, 0)), ch(T2, 'D', (1, 1)),
        ch(T2, 'E', (0, 0)), ch(T2, 'E', (0, 1)), ch(T2, 'E', (1, 0)), ch(T2, 'E', (1, 1)),
    ],
    'convective_envelope_rsg': [
        ch(T0, 'density'), ch(T0, 'energy'), ch(T0, 'pressure'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'euler_multi_quadrants_periodicBC': [
        ch(T0, 'density'), ch(T0, 'energy'), ch(T0, 'pressure'),
        ch(T1, 'momentum', 0), ch(T1, 'momentum', 1),
    ],
    'gray_scott_reaction_diffusion': [
        ch(T0, 'A'), ch(T0, 'B'),
    ],
    'helmholtz_staircase': [
        ch(T0, 'mask', static='global'),
        ch(T0, 'pressure_re'), ch(T0, 'pressure_im'),
    ],
    'MHD_64': [
        ch(T0, 'density'),
        ch(T1, 'magnetic_field', 0), ch(T1, 'magnetic_field', 1), ch(T1, 'magnetic_field', 2),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'planetswe': [
        ch(T0, 'height'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
    ],
    'post_neutron_star_merger': [
        ch(T0, 'density'), ch(T0, 'electron_fraction'), ch(T0, 'entropy'),
        ch(T0, 'internal_energy'), ch(T0, 'pressure'), ch(T0, 'temperature'),
        ch(T1, 'magnetic_field', 0), ch(T1, 'magnetic_field', 1), ch(T1, 'magnetic_field', 2),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'rayleigh_benard': [
        ch(T0, 'buoyancy'), ch(T0, 'pressure'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
    ],
    'rayleigh_taylor_instability': [
        ch(T0, 'density'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'shear_flow': [
        ch(T0, 'pressure'), ch(T0, 'tracer'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
    ],
    'supernova_explosion_64': [
        ch(T0, 'density'), ch(T0, 'pressure'), ch(T0, 'temperature'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'turbulence_gravity_cooling': [
        ch(T0, 'density'), ch(T0, 'pressure'), ch(T0, 'temperature'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'turbulent_radiative_layer_2D': [
        ch(T0, 'density'), ch(T0, 'pressure'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
    ],
    'turbulent_radiative_layer_3D': [
        ch(T0, 'density'), ch(T0, 'pressure'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1), ch(T1, 'velocity', 2),
    ],
    'viscoelastic_instability': [
        ch(T0, 'c_zz'), ch(T0, 'pressure'),
        ch(T1, 'velocity', 0), ch(T1, 'velocity', 1),
        ch(T2, 'C', (0, 0)), ch(T2, 'C', (0, 1)), ch(T2, 'C', (1, 1)),
    ],
}

CHANNELS['euler_multi_quadrants_openBC'] = CHANNELS['euler_multi_quadrants_periodicBC']
CHANNELS['euler_multi_quadrants'] = CHANNELS['euler_multi_quadrants_periodicBC']
