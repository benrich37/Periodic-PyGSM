import numpy as np
import copy
from funcs import geom
from pickle import load as pload
bond_ref = pload(open('funcs/refs/refs.txt', 'rb'))
############################################################

def is_bonded(posn1, posn2, num1, num2, cell, pbc, margin=0.1):
    """ Returns if the two atoms are close enough to be considered
    bonded, and the trans indices of the closest cell for posn2
    to belong to
    :param (np.ndarray) posn1:
    :param (np.ndarray) posn2:
    :param (int) num1:
    :param (int) num2:
    :param (np.ndarray) cell:
    :param (list[bool]) pbc:
    :param (float) margin:
    :return (bool), (list[int]) :
    """
    dist, img_idcs = geom.closest_img(posn1, posn2, pbc, cell)
    ref1, ref2 = bond_ref[num1 - 1], bond_ref[num2 - 1]
    cutoff_dist = ref1[0] + ref2[0] - abs(ref1[1] - ref2[1]) * 0.09
    cutoff_dist = (1 + margin) * cutoff_dist
    return cutoff_dist >= dist, img_idcs

def atoms_is_bonded(atoms_obj, id1, id2):
    nums = atoms_obj.get_atomic_numbers()
    posns = atoms_obj.get_positions()
    return is_bonded(posns[id1], posns[id2], nums[id1], nums[id2],
                     atoms_obj.cell, atoms_obj.pbc)[0]

def update_bonds_dict(bonds_dict, pair, trans_idcs):
    bonds_dict[pair[0]][0].append(pair[1])
    bonds_dict[pair[0]][1].append(trans_idcs)
    bonds_dict[pair[1]][0].append(pair[0])
    bonds_dict[pair[1]][1].append(invert_trans_idcs(trans_idcs))

def update_bonds_dict_all(bonds_dict, main_idx, paired_to, trans_idcs_to):
    """ Updates the bonds_dict for the main atom and all found bonded atoms
    :param (dict) bonds_dict:
    :param (int) main_idx: index for main atom of interest
    :param (list[int]) paired_to: indices of all atoms main atom is bonded to
    :param (list[list[int]]) trans_idcs_to: translate indices for each bonded atom
    :return:
    """
    for i in range(len(paired_to)):
        update_bonds_dict(bonds_dict,
                          [main_idx, paired_to[i]],
                          trans_idcs_to[i])

def get_bonds(idx, posns, nums, cell, pbc):
    n = len(nums)
    atom_idcs = []
    trans_idcs = []
    main_posn = posns[idx]
    main_num = nums[idx]
    for j in range(idx + 1, n):
        bond_bool, trans_vec = is_bonded(main_posn, posns[j], main_num,
                                         nums[j], cell, pbc)
        if bond_bool:
            atom_idcs.append(j)
            trans_idcs.append(trans_vec)
    return atom_idcs, trans_idcs

def bonds_dict(atoms_obj, con_frag=False):
    """ Returns dictionary mapping atom index to list of indices for bonded
    atoms, and list of trans indices in case of inter-cell bonds
    :param (ase.Atoms) atoms_obj:
    :param (bool) con_frag: If True, artificial bonds will be created until all fragments are conjoined
    :return (dict) bond_dict: Dictionary with atom indices for key values, and
    returns a list of indices for all atoms bonded to that atom, and a list of
    the trans indices in case of inter-cell bonds
    """
    pbc = atoms_obj.pbc
    cell = atoms_obj.cell
    nums = atoms_obj.get_atomic_numbers()
    posns = atoms_obj.get_positions()
    n = len(nums)
    return_dict = {}
    for i in range(n):
        atom_idcs, trans_idcs = get_bonds(i, posns, nums, cell, pbc)
        return_dict[i] = [atom_idcs, trans_idcs]
    if con_frag:
        return_dict = bonds_dict_con_frags(return_dict, posns, cell, pbc)
    return return_dict
############################################################

def grow_bool_edge(bonds_dict, bool_edge):
    """ Takes an edge and return a list of bool_edges, each with True if the front
    atom may have un-traversed atoms
    :param (dict) bonds_dict:
    :param (list[list[int] | bool]) bool_edge:
    :return (list[list[int] | bool]) bool_edges:
    """
    bool_edges = [[bool_edge[0], False]]
    # the front is the most recently added atom to an edge
    front = bool_edge[0][-1]
    # for all atoms bonded to the front atom of the edge
    # (specifying [0] here as bonds_dict[front] returns bonded atoms AND their
    # trans indices)
    for a in bonds_dict[front][0]:
        # Second element of a bool_edge is a boolean, so specifying [0] to get
        # only the list of atom indices
        if not a in bool_edge[0]:
            new_bool_edge = copy.copy(bool_edge[0])
            new_bool_edge.append(a)
            # Setting second element to True as bonds for 'a' have not been
            # checked yet
            bool_edges.append([new_bool_edge, True])
    return bool_edges

# Edges will be a list of edge, where each edge has the list of contiguous
# bonded atoms as the first element, and the second element will be a bool
# which is True is the edge has not yet terminated
def grow_bool_edges(bonds_dict, bool_edges):
    """ Adds edges to list of edges until all edges have no un-traversed atoms
    :param (dict) bonds_dict:
    :param (list[list[int], bool]) bool_edges:
    :return (list[list[int], bool]) new_bool_edges:
    """
    cont_bool = False
    new_bool_edges = []
    for e in bool_edges:
        # The first element of an edge is the actual list of indices
        # the second element is a bool, which is False once the edge has
        # finished growing
        if e[1]:
            cont_bool = True
            bool_edges_from_e = grow_bool_edge(bonds_dict, e)
            # Using + here as bool_edges_from_e will be a list of edges
            new_bool_edges = new_bool_edges + bool_edges_from_e
        else:
            # Using append here as e is an edge
            new_bool_edges.append(e)
    # Cont_bool will only remain False once grow_edges is fed a list of edges
    # with all second options set to False
    if cont_bool:
        new_bool_edges = grow_bool_edges(bonds_dict, new_bool_edges)
    return new_bool_edges

def get_edges_a(bonds_dict, a):
    """ Returns all non-bool edges originating with atom of index a
    :param (dict) bonds_dict:
    :param (int) a:
    :return:
    """
    bool_edges = grow_bool_edges(bonds_dict, [[[a], True]])
    edges = []
    for e in bool_edges:
        edges.append(e[0])
    return edges

def a_in_edges(edges, a):
    """ Returns True if index 'a' appears in any edge in edges
    :param (list[list[int]]) edges: List of list of contiguous bonded atoms
    :param (int) a: Index for atom of interest
    :return (bool) return_bool:
    """
    # Edges shouldn't have bools anymore at this point
    return_bool = False
    for e in edges:
        if a in e:
            return_bool = True
    return return_bool

def get_edges(bonds_dict):
    return get_all_edges(bonds_dict)

def get_all_edges(bonds_dict):
    """ Goes through all atom indices, finds all edges starting with each atom
    index, and appends them all to the return list
    :param (dict) bonds_dict:
    :return (list[list[int]]) all_edges:
    """
    all_edges = []
    for a in range(len(bonds_dict)):
        all_edges += get_edges_a(bonds_dict, a)
    return all_edges


def filter_for_ic_edges(edges):
    """ Takes a list of edges and organizes them into a dict by their length,
    discarding any edge too long to be a useful internal coordinate
    :param (list[list[int]]) edges:
    :return (dict) edges_by_length:
    """
    edges_by_length = {0: [],
                       1: [],
                       2: [],
                       3: [],
                       4: []}
    for e in edges:
        if len(e) < 5:
            edges_by_length[len(e)].append(e)
    return edges_by_length

def remove_redundant_edges(edges):
    """ Takes a list of edges of the same length, and returns only unique edges
    :param (list[list[int]]) edges: MUST ALL BE OF SAME LENGTH
    :return (list[list[int]]) unique_edges:
    """
    unique_edges = []
    for e in edges:
        unique = True
        if len(unique_edges) > 0:
            for o in unique_edges:
                same = same_edge(e, o)
                unique = unique and not same
        if unique:
            unique_edges.append(e)
    return unique_edges

def get_all_ic_edges(bonds_dict):
    """ Traverses through bonds of bonds_dict to return all unique bonds,
    angles, and dihedrals
    :param (dict) bonds_dict:
    :return (list[list[int]]) ic_edges:
    """
    all_edges = get_all_edges(bonds_dict)
    edge_dict = filter_for_ic_edges(all_edges)
    ic_edges = []
    for collection in [edge_dict[2], edge_dict[3], edge_dict[4]]:
        ic_edges += remove_redundant_edges(collection)
    return ic_edges

def flip_edge(edge):
    """ Takes an edge and just reverses the order for uniqueness testing
    :param (list[int]) edge:
    :return (list[int]) output:
    """
    flipped_edge = []
    for i in range(len(edge)):
        flipped_edge.append(edge[-(1+i)])
    return flipped_edge

def same_all(edge1, edge2):
    """ Returns True if all atom indices are the same in both edges
    :param (list[int]) edge1:
    :param (list[int]) edge2:
    :return (bool) same:
    """
    same = True
    if not edge1[0] in edge2:
        return False
    shift_idx = edge2.index(edge1[0])
    shift_mod = len(edge2)
    for i in range(len(edge2)):
        same = same and edge1[i] == edge2[(i + shift_idx) % shift_mod]
    return same

def same_edge(edge1, edge2):
    """ Returns True if edge1 and edge2 have the same atoms in the same order
    in either direction
    :param (list[int]) edge1:
    :param (list[int]) edge2:
    :return (bool) same:
    """
    # Assuming of same length
    edge1_flip = flip_edge(edge1)
    same = same_all(edge1, edge2) or same_all(edge1_flip, edge2)
    return same

def grow_frag(bonds_dict, frag):
    """ Iterates through each atom in the fragment and adds
    all atoms bonded to that atom but not yet in the fragment list
    :param (dict) bonds_dict:
    :param (list[int]) frag:
    :return (list[int]) frag:
    """
    len1 = len(frag)
    for a in frag:
        a_sees = bonds_dict[a][0]
        for b in a_sees:
            if not b in frag:
                frag.append(b)
    len2 = len(frag)
    if len2 > len1:
        frag = grow_frag(bonds_dict, frag)
    return frag

def get_frags(bonds_dict):
    """ Returns a list of atom index lists, where each list contains all
    atom indices for a fragment
    :param (dict) bonds_dict:
    :return (list[list[int]]) frags:
    """
    n = len(bonds_dict)
    frags = []
    for i in range(n):
        if len(frags) == 0:
            frags.append(grow_frag(bonds_dict, [i]))
        else:
            grow = True
            for f in frags:
                if i in f:
                    grow = False
            if grow:
                frags.append(grow_frag(bonds_dict, [i]))
    return frags

def invert_trans_idcs(trans_idcs):
    """ Inverts translate indices by sign
    :param (list[int]) trans_idcs: list of (-1, 0, or 1) of length 3
    :return (list[int]) : The same list but with inverted signs
    """
    inv_trans_idcs = []
    for idx in trans_idcs:
        inv_trans_idcs.append(int(idx * (-1)))
    return inv_trans_idcs

def connect_frags(bonds_dict, frags, posns, cell, pbc):
    """ Finds the two closest atoms not belonging to the same fragment and creates an artificial bond
    Calls bonds_dict_con_frags at the end on the updated bonds_dict to check for more disconnected fragments
    :param (dict) bonds_dict:
    :param (list[list[int]]) frags:
    :param (list[np.ndarray]) posns:
    :param (np.ndarray) cell:
    :param (list[bool]) pbc:
    :return:
    """
    fs = len(frags)
    mdist = 1E10
    mpair = [None, None]
    mtrans_idcs = [None, None, None]
    for i in range(fs):
        for j in range(fs)[i + 1:]:
            pair, dist, trans_idcs = geom.closest_pair(frags[i], frags[j],
                                                       posns, cell, pbc)
            print('closest pair between frags ' + str(i) + ' and ' + str(
                j) + ' is through atoms ' + str(pair[0]) + ' and ' + str(
                pair[1]))
            if dist < mdist:
                mpair, mdist, mtrans_idcs = pair, dist, trans_idcs
    print('adjoining atoms ' + str(mpair[0]) + ' and ' + str(mpair[1]))
    update_bonds_dict(bonds_dict, mpair, mtrans_idcs)
    return bonds_dict_con_frags(bonds_dict, posns, cell, pbc)

def bonds_dict_con_frags(bonds_dict, posns, cell, pbc):
    """ Dispatches connect_frags if disconnected fragments are found
    :param (dict) bonds_dict:
    :param (list[np.ndarray]) posns:
    :param (np.ndarray) cell:
    :param (list[bool]) pbc:
    :return:
    """
    frags = get_frags(bonds_dict)
    if len(frags) > 1:
        bonds_dict = connect_frags(bonds_dict, frags, posns, cell, pbc)
    return bonds_dict
