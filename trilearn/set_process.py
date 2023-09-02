import copy

import numpy as np

import trilearn.auxiliary_functions as aux

def backward_perm_traj_sample(p, radius):
    """ Samples a permutation tajectory with maximum p indices.
    """
    maxradius = radius >= p
    ind_perms = [None for i in range(p)]
    ind_perms[p-1] = list(range(p))
    n = p-2
    ind_perms[p-1]
    while n >= 0:
        ind_perms[n] = gen_backward_order_neigh(ind_perms[n+1], radius, maxradius)
        n -= 1
    return ind_perms


def order_neigh_set(current_order, radius, total_set):
    """ The set of neighbors of current_order with
        maximal distance radius (of total_set \ current_order).
    """

    if radius >= len(total_set) - 1:
        return set(total_set) - set(current_order)
    if current_order == []:
        return total_set
    available = set(total_set) - set(current_order)

    # we first clean away some of the unreachable
    # remove all those in available where >= max(current_order)  + radius

    lower_bound = max(0, min(current_order) - radius)
    upper_bound = min(len(total_set)-1, max(current_order) + radius)
    remove_from_below = set(range(0, lower_bound))
    remove_from_above = set(range(upper_bound+1, len(total_set)))

    available = available - remove_from_below - remove_from_above
    a = [{neig for neig in available if np.abs(neig - s) <= radius} for s in current_order]

    b = set.union(*a)

    return b


def order_neigh_log_prob(from_order, to_order, radius, total_set):
    """ Since the perm trajectory is Markovian, the ratio
    becomes only the probability.
    """
    new = list(set(to_order) - set(from_order))[0]
    neigs = order_neigh_set(from_order, radius, total_set)
    if new in neigs:
        return -np.log(len(neigs))
    else:
        return -np.inf


def gen_order_neigh(from_order, radius, total_set):
    """ Returns a list with one more element than from_order
        such that the new element is within the radius and belongs
        to total_set.

    Args:
        from_order (list): list of elements
        radius (int): specifies the radius within which the new element can be taken
        total_set (list): the full set of elements

    Returns:
        numpy array
    """
    
    
    neigs = order_neigh_set(from_order, radius, total_set)
    new = aux.random_element_from_coll(neigs)
    return from_order + [new]


def gen_backward_order_neigh(from_order, radius, maxradius):
    """ Returns: A permutation wit one less element than fromm_order
    """
    removable_neigs = backward_order_neigh_set(from_order, radius, maxradius)
    to_remove = aux.random_element_from_coll(removable_neigs)
    #to_order = from_order[:]
    to_order = list(from_order[:])
    to_order.remove(to_remove)
    return to_order


def backward_order_neigh_set(from_order, radius, maxradius):
    """ Returns the list of nodes that can be removed from from_order
    (the greater order).
    """
    if maxradius is True:
        return from_order

    n = len(from_order)
    from_order.sort()
    removable = [from_order[0], from_order[n-1]]
    removable2 = [val for i, val in enumerate(from_order[1:n-1])
                  if(np.abs(from_order[i-1] - from_order[i+1]) <= radius)]
    return removable + removable2


def backward_order_neigh_log_prob(from_order, to_order, radius, maxradius):
    """ Probability of generating order from_order from the larger order to_order
    under the restriction that no hole greater than radius is created.
    """
    removed = list(set(to_order) - set(from_order))[0]

    neigs = backward_order_neigh_set(to_order, radius, maxradius)  # TODO: bug? from_order?
    if removed in neigs:
        return -np.log(len(neigs))
    else:
        return -np.inf
