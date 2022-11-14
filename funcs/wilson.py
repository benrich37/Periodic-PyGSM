import numpy as np

# See wilson's Molecular Vibrations
# pg 55
# A nonredundant internal coordinate S_t = sum_N <s_t,N | p_N>,
# where |s_t,N> is projection

# S_t = sum_{i->3N} B_{t,i}f_i,
# where f_i is one of three cartesian coordinates for an atom,
# and B_{t,i} is its coefficient in expressing S_t

def unit_vec(posns, idx1, idx2):
    """
    :param posns: List of all positions (shape = (n,3))
    :param idx1: index for atom 1
    :param idx2: index for atom 2
    :return e12: unit vector for atom 1 to atom 2
    :return r12: length of distance from atom 1 to atom 2
    """
    vec = posns[idx2] - posns[idx1]
    r12 = np.linalg.norm(vec)
    e12 = vec/r12
    return e12, r12


def wilson_bond_vec(posns, idcs):
    # See Wilson pg 56, eqn 3
    st1 = unit_vec(posns, idcs[0], idcs[1])[0]
    st2 = -st1
    return [st1, st2]


def wilson_angle_vec(posns, idcs):
    # See Wilson pg 57, eqn 5-8
    e31, r31 = unit_vec(posns, idcs[2], idcs[0])
    e32, r32 = unit_vec(posns, idcs[2], idcs[1])
    theta = np.arccos(np.dot(e31, e32))
    st1 = ((np.cos(theta) * e31) - e32) / (r31 * np.sin(theta))
    st2 = ((np.cos(theta) * e32) - e31) / (r32 * np.sin(theta))
    st3 = (((r31 - (r32 * np.cos(theta))) * e31) + ((r32 - (r31 * np.cos(theta))) * e32)) / (r31 * r32 * np.sin(theta))
    return [st1, st2, st3]


def wilson_dihedral_vec(posns, idcs):
    # See Wilson pg 60, eqn 19, and pg 59, eqn 18
    e41, r41 = unit_vec(posns, idcs[3], idcs[0])
    e42, r42 = unit_vec(posns, idcs[3], idcs[1])
    e43, r43 = unit_vec(posns, idcs[3], idcs[2])
    phi1 = np.arccos(np.dot(e43, e42))
    theta = np.arcsin(np.dot((np.cross(e42, e43) / np.sin(phi1)), e41))
    st1 = (1 / r41) * (
            (np.cross(e42, e43) / (np.cos(theta) * np.sin(phi1))) -
            (np.tan(theta) * e41)
    )
    st2 = (1 / r42) * (
            (np.cross(e43, e41) / (np.cos(theta) * np.sin(phi1))) -
            (np.tan(theta) / ((np.sin(phi1)) ** 2)) * (e42 - (np.cos(phi1) * e43))
    )
    st3 = (1 / r43) * (
            (np.cross(e41, e42) / (np.cos(theta) * np.sin(phi1))) -
            (np.tan(theta) / ((np.sin(phi1)) ** 2)) * (e43 - (np.cos(phi1) * e42))
    )
    st4 = -(st1 + st2 + st3)
    return [st1, st2, st3, st4]

# Routes the number of indices given to the corr. IC type
wilson_func_dict = {
    2: wilson_bond_vec,
    3: wilson_angle_vec,
    4: wilson_dihedral_vec
}

def get_sts(posns, idcs):
    """
    :param posns: List of all positions (n,3)
    :param idcs: List of indices of atoms of interest (len 2 -> 4)
    :return sts: List of vectors for each atom's component of internal coordinate
    """
    try:
        func = wilson_func_dict[len(idcs)]
    except Exception as e:
        print(e)
    sts = func(posns, idcs)
    return sts


def ic_in_cart(posns, idcs):
    """
    :param posns: List of all positions
    :param idcs: List of indices of atoms of interest
    :return St: (n,3) vector describing internal coordinate in cartesian
    """
    try:
        sts = get_sts(posns, idcs)
    except Exception as e:
        print(e)
    St = np.zeros(tuple([len(posns), 3]))
    for i in range(len(idcs)):
        St[idcs[i]] = sts[i]
    return St