import numpy as np
import copy
from funcs import geom


############################################################
radii_dict = {
    1: 0.31,
    2: 0.25,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06
}
def is_bonded(posn1, posn2, num1, num2, margin=0.1):
    dist = abs(np.linalg.norm(posn2 - posn1))
    maxdist = radii_dict[num1] + radii_dict[num2]
    maxdist = maxdist + margin * maxdist
    return maxdist >= dist and dist != 0.
def atoms_is_bonded(atoms_obj, id1, id2):
    nums = atoms_obj.get_atomic_numbers()
    posns = atoms_obj.get_positions()
    return is_bonded(posns[id1], posns[id2], nums[id1], nums[id2])
def get_bonds(idx, posns, nums):
    n = len(nums)
    output = []
    main_posn = posns[idx]
    main_num = nums[idx]
    for i in range(n):
        if is_bonded(main_posn, posns[i], main_num, nums[i]):
            output.append(i)
    return output
def bonds_dict(atoms_obj):
    return_dict = {}
    nums = atoms_obj.get_atomic_numbers()
    posns = atoms_obj.get_positions()
    n = len(nums)
    for i in range(n):
        bonds = get_bonds(i, posns, nums)
        return_dict[i] = bonds
    return return_dict
############################################################

def grow_edge(bonds_dict, edge):
    new_edges = []
    front = edge[0][-1]
    for a in bonds_dict[front]:
        if a in edge[0]:
            new_edges.append([edge[0], False])
        else:
            new_edge_0 = copy.copy(edge[0])
            new_edge_0.append(a)
            new_edges.append([new_edge_0, True])
    return new_edges

# Edges will be a list of edge, where each edge has the list of contiguous
# bonded atoms as the first element, and the second element will be a bool
# which is True is the edge has not yet terminated
def grow_edges(bonds_dict, edges):
    cont_bool = False
    new_edges = []
    for e in edges:
        if e[1]:
            cont_bool = True
            es = grow_edge(bonds_dict, e)
            # Using + here as es will be a list of edges
            new_edges = new_edges + es
        else:
            # Using append here as e is just an edge still
            new_edges.append(e)
    if cont_bool:
        return grow_edges(bonds_dict, new_edges)
    else:
        return new_edges

def get_edges_a(bonds_dict, a):
    edges = grow_edges(bonds_dict, [[[a], True]])
    output = []
    for e in edges:
        output.append(e[0])
    return output

def a_in_edges(edges, a):
    # Edges shouldnt have bools anymore at this point
    for e in edges:
        if a in e:
            return True
    return False

# def get_edges(bonds_dict):
#     output = []
#     for a in range(len(bonds_dict)):
#         if not a_in_edges(output, a):
#             output = output + get_edges_a(bonds_dict, a)
#     return output

def get_all_edges(bonds_dict):
    all_edges = []
    for a in range(len(bonds_dict)):
        all_edges = all_edges + get_edges_a(bonds_dict, a)
    return all_edges

def sget_edges(bonds_dict):
    all_edges = get_all_edges(bonds_dict)
    edges = []
    for e in all_edges:
        if len(e) < 5:
            edges.append(e)
    all_bonds = []
    all_angles = []
    all_dihedrals = []
    for e in edges:
        l = len(e)
        if l == 2:
            all_bonds.append(e)
        if l == 3:
            all_angles.append(e)
        if l == 4:
            all_dihedrals.append(e)
    bonds = remove_redundant_edges(all_bonds)
    angles = remove_redundant_edges(all_angles)
    dihedrals = remove_redundant_edges(all_dihedrals)
    return bonds + angles + dihedrals
    # return bonds, angles, dihedrals

def remove_redundant_edges(edges):
    output = []
    for e in edges:
        unique = True
        if len(output) > 0:
            for o in output:
                same = same_edge(e, o)
                unique = unique and not same
        if unique:
            output.append(e)
    return output

def flip_edge(edge):
    output = []
    for i in range(len(edge)):
        output.append(edge[-(1+i)])
    return output

def same_all(edge1, edge2):
    same = True
    for i in range(len(edge1)):
        same = same and edge1[i] == edge2[i]
    return same

def same_edge(edge1, edge2):
    # Assuming of same length
    edge1_flip = flip_edge(edge1)
    same = same_all(edge1, edge2) or same_all(edge1_flip, edge2)
    return same

def grow_frag(bonds_dict, frag):
    len1 = len(frag)
    for a in frag:
        a_sees = bonds_dict[a]
        for b in a_sees:
            if not b in frag:
                frag.append(b)
    len2 = len(frag)
    if len2 > len1:
        return grow_frag(bonds_dict, frag)
    else:
        return frag

def get_frags(bonds_dict):
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


def bonds_dict_con_frags(bonds_dict, posns):
    frags = get_frags(bonds_dict)
    fs = len(frags)
    if fs > 1:

        mdist = 1E10
        mpair = [None, None]
        for i in range(fs):
            for j in range(fs)[i+1:]:
                pair, dist = geom.closest_pair(frags[i], frags[j], posns)
                print('closest pair between frags ' + str(i) + ' and ' + str(j) + ' is through atoms ' + str(pair[0]) + ' and ' + str(pair[1]))
                if dist < mdist:
                    mpair, mdist = pair, dist
        print('adjoining atoms ' + str(mpair[0]) + ' and ' + str(mpair[1]))
        bonds_dict[mpair[0]].append(mpair[1])
        bonds_dict[mpair[1]].append(mpair[0])
        return bonds_dict_con_frags(bonds_dict, posns)
    else:
        return bonds_dict




# def bonds_dict_con_frags(atoms_obj):
#     uncon_bonds_dict = bonds_dict(atoms_obj)
#     frags = get_frags



# def group_frags(bonds_dict):
#     n = len(bonds_dict)
#     frags = [[]]
#     for i in range(n):
#         for f in frags:
#             if not i in f:

