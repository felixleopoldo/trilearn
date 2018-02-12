import os.path

import numpy as np

import trilearn.smc as smc


def main(trajectory_length, n_particles, alphas, betas, radii, seed, parallel, data_filename, output_directory, **args):
    if seed is not None:
        np.random.seed(seed)

    X = np.matrix(np.loadtxt(data_filename, delimiter=','))
    filename_base = os.path.splitext(os.path.basename(data_filename))[0]
    if parallel is True:
        smc.gen_pgibbs_ggm_trajectories_parallel(X, trajectory_length, n_particles,
                                                 alphas=alphas, betas=betas, radii=radii, cache={},
                                                 filename_prefix=output_directory + "/" + filename_base)
    else:
        smc.gen_pgibbs_ggm_trajectories(X, trajectory_length, n_particles,
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

    args = parser.parse_args()
    main(**args.__dict__)