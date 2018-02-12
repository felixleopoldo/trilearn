import argparse
from multiprocessing import Process

import numpy as np

import chordal_learning.distributions.SequentialJTDistributions as seqdist
from chordal_learning.smc import particle_gibbs_to_file

parser = argparse.ArgumentParser()
parser.add_argument('-T', '--trajectory_length', type=int, required=True,
                    nargs='+', help="Number of Gibbs samples")
parser.add_argument('-N', '--particles', type=int, required=True,
                    nargs='+', help="Number of SMC particles")
parser.add_argument('-a', '--alpha', type=float, required=True,
                    nargs='+', help="Parameter for the junction tree expander")
parser.add_argument('-b', '--beta', type=float, required=True,
                    nargs='+', help="Parameter for the junction tree expander")
parser.add_argument('-r', '--radius', type=int, required=True,
                    nargs='+', help="The search neighborhood" +
                    "radius for the Gibbs sampler")
parser.add_argument('-p', '--order', type=int, required=True,
                    help="The order of the underlying decompoasble graph")
parser.add_argument('-s', '--seed', type=int, required=True,
                    help="Random seed")
parser.add_argument('-o', '--output_directory', required=True,
                    help="Output directory")


args = parser.parse_args()

np.random.seed(args.seed)
p = args.order
filename_prefix = "uniform_jt_samples_p_"+str(p)


for N in args.particles:
    for T in args.trajectory_length:
        for radius in args.radius:
            for alpha in args.alpha:
                for beta in args.beta:
                    sd = seqdist.UniformJTDistribution(p)
                    print "Starting: "+str((N, alpha, beta, radius,
                                            T, sd, args.output_directory+"/" +
                                            filename_prefix,))

                    proc = Process(target=particle_gibbs_to_file,
                                   args=(N, alpha, beta, radius,
                                         T, sd, args.output_directory + "/" +
                                         filename_prefix,))
                    proc.start()


for N in args.particles:
    for T in args.trajectory_length:
        for radius in args.radius:
            for alpha in args.alpha:
                for beta in args.beta:
                    proc.join()
                    print "Completed: "+str((N, alpha, beta,
                                             radius,
                                             T,
                                             filename_prefix,))
