import os

import numpy as np


def main(precmat_filename, n_samples, output_directory, seed, **args):
    if seed > 0:
        np.random.seed(seed)

    n = n_samples
    filename = os.path.basename(precmat_filename)
    invwish = os.path.splitext(filename)[0]

    precmat = np.matrix(np.loadtxt(precmat_filename, delimiter=','))
    sigma = precmat.I
    p = len(sigma)

    X = np.matrix(np.random.multivariate_normal(np.zeros(p), sigma, n))

    tmp = output_directory+"/"+invwish+"_n_"+str(n)+".csv"
    np.savetxt(tmp, X, delimiter=',', fmt="%f")

    print "wrote"
    print tmp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sample normal data with mean zero "
                                                 "and specified precision matrix.")
    parser.add_argument(
        '-m', '--precmat_filename',
        required=True,
        help="Precision matrix file.",
    )
    parser.add_argument(
        '-n', '--n_samples',
        type=int, required=True,
        help="Number of samples"
    )
    parser.add_argument(
        '-d', '--output_directory',
        required=False, default="."
    )
    parser.add_argument(
        '-s', '--seed',
        type=int, required=False
    )

    args = parser.parse_args()
    main(**args.__dict__)
