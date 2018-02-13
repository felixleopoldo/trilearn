"""
Generate a Markov chain from Particle Gibbs for different
parameter settings.
"""

import os.path
from os.path import basename

import numpy as np
import pandas as pd

import trilearn.smc as smc


def main(data_filename, pseudo_observations, trajectory_length, n_particles,
         alphas, betas, radii, seed, parallel,
         output_directory, **args):
    if seed is not None:
        np.random.seed(seed)

    filename_base = os.path.splitext(basename(data_filename))[0]

    df = pd.read_csv(data_filename, sep=',', header=[0, 1])
    X = df.values.astype(int)
    n_levels = [int(a[1]) for a in list(df.columns)]
    levels = np.array([range(l) for l in n_levels])

    if parallel is True:
        smc.gen_pgibbs_loglin_trajectories_parallel(
            X, levels, trajectory_length, n_particles,
            pseudo_observations=pseudo_observations,
            alphas=alphas, betas=betas, radii=radii, cache={},
            filename_prefix=output_directory + "/" + filename_base)
    else:
        smc.gen_pgibbs_loglin_trajectories(
            X, levels, trajectory_length, n_particles,
            pseudo_observations=pseudo_observations,
            alphas=alphas, betas=betas, radii=radii, cache={},
            filename_prefix=output_directory + "/" + filename_base)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Generate particle Gibbs trajectories och decomposable graphs.")

    parser.add_argument(
        '-M', '--trajectory_length',
        type=int, required=True, nargs='+',
        help="Number of Gibbs samples"
    )
    parser.add_argument(
        '-f', '--data_filename',
        required=True,
        help="Filename of dataset stored as row vectors och floats. "
    )
    parser.add_argument(
        '-N', '--n_particles',
        type=int, required=True, nargs='+',
        help="Number of SMC particles"
    )
    parser.add_argument(
        '-a', '--alphas',
        type=float, required=False, default=[0.5], nargs='+',
        help="Parameter for the Christmas tree algorithm"
    )
    parser.add_argument(
        '-b', '--betas',
        type=float, required=False, default=[0.5], nargs='+',
        help="Parameter for the Christmas tree algorithm"
    )
    parser.add_argument(
        '-r', '--radii',
        type=int, required=False, default=[None], nargs='+',
        help="The search neighborhood radius for the Gibbs sampler"
    )
    parser.add_argument(
        '-s', '--seed',
        type=int, required=False, default=None
    )
    parser.add_argument(
        '--parallel',
        required=False, action="store_true"
    )
    parser.add_argument(
        '-o', '--output_directory',
        required=False, default="./",
        help="Output directory"
    )
    parser.add_argument(
        '--pseudo_observations',
        type=float, required=False, default=[1.0], nargs='+',
        help="Total number of pseudo observations"
    )
    # parser.add_argument('--n_levels',
    #                     type=int, required=False, nargs='+',
    #                     help="Number of levels for each variable"
    # )

    args = parser.parse_args()
    main(**args.__dict__)