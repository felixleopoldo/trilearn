import numpy as np
import scipy.special as sp

import trilearn.auxiliary_functions as aux


def disconnect_select_subsets(tree, c):
    # 2. choose sets
    M = np.random.randint(2, high=len(c)+1)
    N = np.random.randint(1, high=M)
    X = frozenset(np.random.choice(list(c), size=N, replace=False))
    Y = frozenset(np.random.choice(list(c-X), size=M-N, replace=False))

    #print "X: " + str(X)
    #print "Y: " + str(Y)
    return (X, Y)


def disconnect_get_neighbors(tree, C, X, Y):
    # 3. define neighbors
    NX = set()
    NY = set()
    N = set()
    for n in tree.neighbors(C):
        # n interects both X and Y
        if len(n & X) > 0 and len(n & Y) > 0:  # TODO
            #print str(n) + " interects both " + str(X) + " and "+str(Y)
            return False
        # neighbors intersecting X
        if len(n & X) > 0 and len(n & Y) == 0:
            NX.add(n)
        # neighbors intersectin Y
        elif len(n & X) == 0 and len(n & Y) > 0:
            NY.add(n)
        # neighbors not intersecting eithor of X or Y
        else:
            N.add(n)
    return (NX, NY, N)


def disconnect_get_CXCY(C, X, Y, NX, NY):
    S = C - (X | Y)
    CX = None
    CY = None
    XS = X | S
    YS = Y | S

    if NX is not None:
        for n in NX:
            if XS <= n:
                CX = n
                break

    if NY is not None:
        for n in NY:
            if YS <= n:
                CY = n
                break
    return (CX, CY)


def connect_a(tree, S, X, Y, CX, CY):
    XYS = X | Y | S
    # print "a)"
    # print "added node: " + str(XYS)
    # print "removed node: " + str(CX)
    # print "removed node: " + str(CY)
    # print "removed edge: " + str((CX, CY))
    tree.remove_edge(CX, CY)
    XSneigh = set(tree.neighbors(CX))
    YSneigh = set(tree.neighbors(CY))

    tree.add_node(XYS)
    tree.add_edges_from([(XYS, n) for n in tree.neighbors(CX)])
    tree.add_edges_from([(XYS, n) for n in tree.neighbors(CY)])
    # print "added edges: " + str([(XYS, n) for n in tree.neighbors(CX)])
    # print "added edges: " + str([(XYS, n) for n in tree.neighbors(CY)])
    tree.remove_node(CY)
    tree.remove_node(CX)

    CX_disconn = None
    CY_disconn = None
    return (CX_disconn, CY_disconn, XSneigh, YSneigh)


def disconnect_a(tree, c, X, Y, CX, CY, XSneig, YSneig):
    S = c - (X | Y)
    XS = X | S
    YS = Y | S
    tree.add_node(XS)
    tree.add_node(YS)
    tree.add_edge(XS, YS)
    tree.add_edges_from([(XS, n) for n in XSneig])
    tree.add_edges_from([(YS, n) for n in YSneig])
    # print "a)"
    # print "removed node: " + str(c)
    # print "added edges: " + str([(XS, n) for n in XSneig])
    # print "added edges: " + str([(YS, n) for n in YSneig])
    # print "added edge: (" + str((XS, YS)) + ")"
    # print "removed edges: " + str([(c, n) for n in tree.neighbors(c)])
    tree.remove_node(c)

    CX_conn = XS
    CY_conn = YS
    return (CX_conn, CY_conn)


def disconnect_b(tree, c, X, Y, CX, CY):
    S = c - (X | Y)
    YS = Y | S
    # print "b) disconnect"
    # print "added node: " + str(YS)
    # print "removed node: " + str(c)
    # replace c=XYS by YS
    tree.add_node(YS)
    tree.add_edges_from([(YS, n) for n in tree.neighbors(c)])
    # print "added edges: " + str([(YS, n) for n in tree.neighbors(c)])
    # print "removed edges: " + str([(c, n) for n in tree.neighbors(c)])
    tree.remove_node(c)
    CX_conn = CX
    CY_conn = YS
    return (CX_conn, CY_conn)


def disconnect_c(tree, c, X, Y, CX, CY):
    S = c - (X | Y)
    XS = X | S
    # print "c)"
    # print "added node: " + str(XS)
    # print "removed node: " + str(c)
    # print "added edges: " + str([(XS, n) for n in tree.neighbors(c)])
    # print "removed edges: " + str([(c, n) for n in tree.neighbors(c)])
    # replace c=XYS by XS
    tree.add_node(XS)
    tree.add_edges_from([(XS, n) for n in tree.neighbors(c)])
    tree.remove_node(c)
    CX_conn = XS
    CY_conn = CY
    return (CX_conn, CY_conn)


def disconnect_d(tree, c, X, Y, CX, CY):
    # print "d)"
    # print "added edge: " + str((CX, CY))
    # print "removed node: " + str(c)
    # print "removed edges: " + str([(c, n) for n in tree.neighbors(c)])
    tree.add_edge(CX, CY)
    tree.remove_node(c)
    CX_conn = CX
    CY_conn = CY
    return (CX_conn, CY_conn)


def disconnect_move(tree):
    C = np.random.choice(tree.nodes())
    if len(C) < 2:
        #print "|C| < 2"
        return False
    (X, Y) = disconnect_select_subsets(tree, C)
    neigs = disconnect_get_neighbors(tree, C, X, Y)

    if neigs is False:
        #print "Some neighbor intersects both X and Y"
        return False
    else:
        (NX, NY, N) = neigs

    (CX, CY) = disconnect_get_CXCY(C, X, Y, NX, NY)
    S = C - (X | Y)
    # print "Disconnnect nodes: " + str(list(X)) + " and " + str(list(Y)) + " in " + str(C)
    # print "CX: " + str(CX)
    # print "CY: " + str(CY)

    num_cliques = tree.order()
    # case a
    if CX is None and CY is None:
        if N is not None:  # this should be taken out
            NXS = aux.random_subset(N)  # part of neigs N that will be assigned to XS
            NYS = N - NXS  # part of neigs N that will be assigned to YS
        # print "NXS: " + str(NXS)
        # print "NYS: " + str(NYS)
        XSneig = NX | NXS  # CX_disconn_neig
        YSneig = NY | NYS  # CY_disconn_neig
        (CX_conn, CY_conn) = disconnect_a(tree, C, X, Y, CX, CY, XSneig, YSneig)
        logprob = disconnect_logprob_a(num_cliques, X, Y, S, N)
        return ("a", logprob, X, Y, S, CX_conn, CY_conn)

    # case b
    elif CX is not None and CY is None:
        if len(NX) == 1:  # contains only CX
            (CX_conn, CY_conn) = disconnect_b(tree, C, X, Y, CX, CY)
            logprob = disconnect_logprob_bcd(num_cliques, X, Y, S)
            return ("b", logprob, X, Y, S, CX_conn, CY_conn)
        else:
            #print "NX contains more that one clique so nothing is done"
            return False

    # case c
    elif CX is None and CY is not None:
        if len(NY) == 1:  # contains only CX
            (CX_conn, CY_conn) = disconnect_c(tree, C, X, Y, CX, CY)
            logprob = disconnect_logprob_bcd(num_cliques, X, Y, S)
            return ("c", logprob, X, Y, S, CX_conn, CY_conn)
        else:
            #print "NY contains more that one clique so nothing is done"
            return False

    # case d
    elif CX is not None and CY is not None:
        if len(N) == 0 and len(NY) == 1 and len(NX) == 1:
            (CX_conn, CY_conn) = disconnect_d(tree, C, X, Y, CX, CY)
            logprob = disconnect_logprob_bcd(num_cliques, X, Y, S)
            return ("d", logprob, X, Y, S, CX_conn, CY_conn)
        else:
            #print "len(N) == 0 and len(NY) == 1 and len(NX) == 1 id False so nothing is changes"
            return False


def disconnect_logprob_a(num_cliques, X, Y, S, N):
    return disconnect_logprob_bcd(num_cliques, X, Y, S) - len(N) * np.log(2)


def disconnect_logprob_bcd(num_cliques, X, Y, S):
    C = X | Y | S
    M = len(X | Y)
    N = len(X)
    m = len(C)
    logprob = 0.0
    logprob += -np.log(num_cliques)
    logprob += np.log(2) - np.log((m - 1) * (M - 1))
    logprob += sp.gammaln(N+1) + sp.gammaln(M-N+1) + sp.gammaln(m - M+1) - sp.gammaln(m+1)
    return logprob


def connect_move(tree):
    num_seps = tree.size()
    if num_seps == 0:
        return False

    (X, Y, CX, CY) = connect_select_subsets(tree)
    logprob = connect_logprob(num_seps, X, Y, CX, CY)

    S = CX & CY
    XS = X | S
    YS = Y | S
    CX_disconn, CY_disconn = None, None
    # print "Connnect: " + str(list(X)) + "  and " + str(list(Y)) + " in " + str((CX, CY))
    # print "XS: " + str(XS)
    # print "CX: " + str(CX)
    # print "YS: " + str(YS)
    # print "CY: " + str(CY) 

    # a)
    if XS == CX and YS == CY:
        (CX_disconn, CY_disconn, XSneig, YSneig) = connect_a(tree, S, X, Y, CX, CY)
        return ("a", logprob, X, Y, S, CX_disconn, CY_disconn, XSneig, YSneig)
    # b)
    elif XS < CX and YS == CY:
        (CX_disconn, CY_disconn) = connect_b(tree, S, X, Y, CX, CY)
        return ("b", logprob, X, Y, S, CX_disconn, CY_disconn)
    # c)
    elif XS == CX and YS < CY:
        (CX_disconn, CY_disconn) = connect_c(tree, S, X, Y, CX, CY)
        return ("c", logprob, X, Y, S, CX_disconn, CY_disconn)
    # d)
    elif XS < CX and YS < CY:
        (CX_disconn, CY_disconn) = connect_d(tree, S, X, Y, CX, CY)
        return ("d", logprob, X, Y, S, CX_disconn, CY_disconn)


def connect_select_subsets(tree):
    # 1. choose separator
    SJ = tree.size()
    edgeind = np.random.randint(tree.size())
    edge = list(tree.edges())[edgeind]
    S = edge[0] & edge[1]
    CX = edge[0]
    CY = edge[1]
    NumX = np.random.randint(len(CX - S)) + 1
    NumY = np.random.randint(len(CY - S)) + 1
    X = frozenset(np.random.choice(list(CX - S), NumX, replace=False))
    Y = frozenset(np.random.choice(list(CY - S), NumY, replace=False))
    return (X, Y, CX, CY)


def connect_b(tree, S, X, Y, CX, CY):
    XYS = X | Y | S
    # print "b) connect"
    # print "added node: " + str(XYS)
    # print "removed node: " + str(CY)
    # print "removed edges: " + str([(CY, n) for n in tree.neighbors(CY)])
    # print "added edges: " + str([(XYS, n) for n in tree.neighbors(CY)])
    tree.add_node(XYS)
    tree.add_edges_from([(XYS, n) for n in tree.neighbors(CY)])
    tree.remove_node(CY)
    CX_disconn = CX
    CY_disconn = None
    return (CX_disconn, CY_disconn)


def connect_c(tree, S, X, Y, CX, CY):
    XYS = X | Y | S
    # print "c)"
    # print "added node: " + str(XYS)
    # print "removed node: " + str(CX)
    # print "removed edges: " + str([(CX, n) for n in tree.neighbors(CX)])
    # print "added edges: " + str([(XYS, n) for n in tree.neighbors(CX)])
    tree.add_node(XYS)
    tree.add_edges_from([(XYS, n) for n in tree.neighbors(CX)])
    tree.remove_node(CX)
    CX_disconn = None
    CY_disconn = CY
    return (CX_disconn, CY_disconn)


def connect_d(tree, S, X, Y, CX, CY):
    XYS = X | Y | S
    # print "d)"
    # print "added node: " + str(XYS)
    # print "added edge: " + str((XYS, CX))
    # print "added edge: " + str((XYS, CY))
    # print "removed edge: " + str((CX, CY))
    tree.add_node(XYS)
    tree.add_edge(XYS, CX)
    tree.add_edge(XYS, CY)
    tree.remove_edge(CX, CY)
    CX_disconn = CX
    CY_disconn = CY
    return (CX_disconn, CY_disconn)


def connect_logprob(num_seps, X, Y, CX, CY):
    S = CX & CY
    logprob = 0.0
    s = len(S)
    SJ = num_seps
    NX = len(X)
    NY = len(Y)
    mX = len(CX)
    mY = len(CY)
    logprob += -np.log(SJ)
    logprob += -np.log(mX - s)
    logprob += sp.gammaln(NX + 1) + sp.gammaln(mX - s - NX + 1) - sp.gammaln(mX - s + 1)
    logprob += -np.log(mY - s)
    logprob += sp.gammaln(NY + 1) + sp.gammaln(mY - s - NY + 1) - sp.gammaln(mY - s + 1)
    return logprob
