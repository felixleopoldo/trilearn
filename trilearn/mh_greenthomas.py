import numpy as np
import networkx as nx
from tqdm import tqdm

import trilearn.distributions.sequential_junction_tree_distributions as seqdist
import trilearn.graph.trajectory as mcmctraj
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.decomposable as dlib
import trilearn.graph.greenthomas as aglib


def mh(n_mh_samples, randomize, sd):
    graph = nx.Graph()
    graph.add_nodes_from(range(sd.p))
    jt = dlib.junction_tree(graph)
    assert (jtlib.is_junction_tree(jt))
    jt_traj = [None] * n_mh_samples
    graphs = [None] * n_mh_samples
    jt_traj[0] = jt
    graphs[0] = jtlib.graph(jt)
    log_prob_traj = [None] * n_mh_samples

    gtraj = mcmctraj.Trajectory()
    gtraj.set_sequential_distribution(sd)

    log_prob_traj[0] = 0.0
    log_prob_traj[0] = sd.log_likelihood(jtlib.graph(jt_traj[0]))
    log_prob_traj[0] += -jtlib.log_n_junction_trees(jt_traj[0], jtlib.separators(jt_traj[0]))

    accept_traj = [0] * n_mh_samples

    MAP_graph = (graphs[0], log_prob_traj[0])

    for i in tqdm(range(1, n_mh_samples), desc="Metropolis-Hastings samples"):
        if log_prob_traj[i-1] > MAP_graph[1]:
            MAP_graph = (graphs[i-1], log_prob_traj[i-1])

        if i % randomize == 0:
            print i
            print "SHUFFLE"
            jtlib.randomize(jt)
            graphs[i] = jtlib.graph(jt)  # TODO: Improve.
            log_prob_traj[i] = sd.log_likelihood(graphs[i]) - jtlib.log_n_junction_trees(jt, jtlib.separators(jt))

        r = np.random.randint(2)  # Connect / disconnect move
        #assert(jtlib.is_junction_tree(jt))
        num_seps = jt.size()
        log_p1 = log_prob_traj[i - 1]
        if r == 0:
            # Connect move
            num_cliques = jt.order()
            conn = aglib.connect_move(jt)  # need to move to calculate posterior

            #assert(jtlib.is_junction_tree(jt))
            seps_prop = jtlib.separators(jt)
            log_p2 = sd.log_likelihood(jtlib.graph(jt)) - jtlib.log_n_junction_trees(jt, seps_prop)

            C_disconn = conn[2] | conn[3] | conn[4]
            if conn[0] == "a":
                (case, log_q12, X, Y, S, CX_disconn, CY_disconn, XSneig, YSneig) = conn
                (NX_disconn, NY_disconn, N_disconn) = aglib.disconnect_get_neighbors(jt, C_disconn, X, Y)  # TODO: could this be done faster?
                log_q21 = aglib.disconnect_logprob_a(num_cliques - 1, X, Y, S, N_disconn)
                #print log_p2, log_q21, log_p1, log_q12
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                #print alpha
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    # print "Accept"
                    accept_traj[i] = 1
                    log_prob_traj[i] = log_p2
                    # jt_traj[i] = jt.copy()  # TODO: Improve.
                    graphs[i] = jtlib.graph(jt)  # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_a(jt, C_disconn, X, Y, CX_disconn, CY_disconn, XSneig, YSneig)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    # jt_traj[i] = jt_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif conn[0] == "b":
                (case, log_q12, X, Y, S, CX_disconn, CY_disconn) = conn
                log_q21 = aglib.disconnect_logprob_bcd(num_cliques, X, Y, S)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    #print "Accept"
                    accept_traj[i] = 1
                    log_prob_traj[i] = log_p2
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                    #jt_traj[i] = jt.copy()  # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_b(jt, C_disconn, X, Y, CX_disconn, CY_disconn)
                    #assert(jtlib.is_junction_tree(jt))
                    log_prob_traj[i] = log_prob_traj[i-1]
                    #jt_traj[i] = jt_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif conn[0] == "c":
                (case, log_q12, X, Y, S, CX_disconn, CY_disconn) = conn
                log_q21 = aglib.disconnect_logprob_bcd(num_cliques, X, Y, S)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2
                    #jt_traj[i] = jt.copy()  # TODO: Improve.
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_c(jt, C_disconn, X, Y, CX_disconn, CY_disconn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    #jt_traj[i] = jt_traj[i-1]
                    continue

            elif conn[0] == "d":
                (case, log_q12, X, Y, S,  CX_disconn, CY_disconn) = conn
                log_q21 = aglib.disconnect_logprob_bcd(num_cliques + 1, X, Y, S)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2
                    #jt_traj[i] = jt.copy()  # TODO: Improve.
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_d(jt, C_disconn, X, Y, CX_disconn, CY_disconn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    #jt_traj[i] = jt_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

        elif r == 1:
            # Disconnect move
            disconnect = aglib.disconnect_move(jt)  # need to move to calculate posterior
            seps_prop = jtlib.separators(jt)
            log_p2 = sd.log_likelihood(jtlib.graph(jt)) - jtlib.log_n_junction_trees(jt, seps_prop)

            #assert(jtlib.is_junction_tree(jt))
            #print "disconnect"
            if disconnect is not False:
                if disconnect[0] == "a":
                    (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                    log_q21 = aglib.connect_logprob(num_seps + 1, X, Y, CX_conn, CY_conn)
                    alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                    samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                    if samp == 1:
                        accept_traj[i] = 1
                        #print "Accept"
                        log_prob_traj[i] = log_p2
                        #jt_traj[i] = jt.copy()  # TODO: Improve.
                        graphs[i] = jtlib.graph(jt) # TODO: Improve.
                    else:
                        #print "Reject"
                        aglib.connect_a(jt, S, X, Y, CX_conn, CY_conn)
                        #assert(jtlib.is_junction_tree(jt))
                        log_prob_traj[i] = log_prob_traj[i-1]
                        #jt_traj[i] = jt_traj[i-1]
                        graphs[i] = graphs[i-1]
                        continue

                elif disconnect[0] == "b":
                    (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                    log_q21 = aglib.connect_logprob(num_seps, X, Y, CX_conn, CY_conn)
                    alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                    samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                    if samp == 1:
                        accept_traj[i] = 1
                        #print "Accept"
                        log_prob_traj[i] = log_p2
                        #jt_traj[i] = jt.copy()  # TODO: Improve.
                        graphs[i] = jtlib.graph(jt) # TODO: Improve.
                    else:
                        #print "Reject"
                        aglib.connect_b(jt, S, X, Y, CX_conn, CY_conn)
                        #assert(jtlib.is_junction_tree(jt))
                        log_prob_traj[i] = log_prob_traj[i-1]
                        #jt_traj[i] = jt_traj[i-1]
                        graphs[i] = graphs[i-1]
                        continue

                elif disconnect[0] == "c":
                    (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                    log_q21 = aglib.connect_logprob(num_seps, X, Y, CX_conn, CY_conn)
                    alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                    samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                    if samp == 1:
                        accept_traj[i] = 1
                        #print "Accept"
                        log_prob_traj[i] = log_p2
                        #jt_traj[i] = jt.copy()  # TODO: Improve.
                        graphs[i] = jtlib.graph(jt) # TODO: Improve.
                    else:
                        #print "Reject"
                        aglib.connect_c(jt, S, X, Y, CX_conn, CY_conn)
                        #assert(jtlib.is_junction_tree(jt))
                        log_prob_traj[i] = log_prob_traj[i-1]
                        #jt_traj[i] = jt_traj[i-1]
                        graphs[i] = graphs[i-1]
                        continue

                elif disconnect[0] == "d":
                    (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                    log_q21 = aglib.connect_logprob(num_seps - 1, X, Y, CX_conn, CY_conn)
                    alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                    samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                    if samp == 1:
                        #print "Accept"
                        accept_traj[i] = 1
                        log_prob_traj[i] = log_p2
                        #jt_traj[i] = jt.copy()  # TODO: Improve.
                        graphs[i] = jtlib.graph(jt) # TODO: Improve.
                    else:
                        #print "Reject"
                        aglib.connect_d(jt, S, X, Y, CX_conn, CY_conn)
                        #assert(jtlib.is_junction_tree(jt))
                        log_prob_traj[i] = log_prob_traj[i-1]
                        #jt_traj[i] = jt_traj[i-1]
                        graphs[i] = graphs[i-1]
                        continue
            else:
                log_prob_traj[i] = log_prob_traj[i-1]
                #jt_traj[i] = jt_traj[i-1]
                graphs[i] = graphs[i-1]
                continue

    gtraj.set_trajectory(graphs)
    return gtraj


def gen_ggm_trajectory(dataframe, n_samples, randomize=1000, D=None, delta=1.0, cache={}, alpha=0.5, beta=0.5, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    return mh(n_samples, randomize, sd)

