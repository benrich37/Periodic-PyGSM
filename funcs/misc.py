import numpy as np
import copy

radii_dict = {
    1: 0.31,
    2: 0.25,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66
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

def grow_edge(bonds_ref, edge):
    new_edges = []
    front = edge[0][-1]
    for a in bonds_ref[front]:
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
def grow_edges(bonds_ref, edges):
    cont_bool = False
    new_edges = []
    for e in edges:
        if e[1]:
            cont_bool = True
            es = grow_edge(bonds_ref, e)
            # Using + here as es will be a list of edges
            new_edges = new_edges + es
        else:
            # Using append here as e is just an edge still
            new_edges.append(e)
    if cont_bool:
        return grow_edges(bonds_ref, new_edges)
    else:
        return new_edges

def get_edges_a(bonds_ref, a):
    edges = grow_edges(bonds_ref, [[[a], True]])
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

def get_edges(bonds_ref):
    output = []
    for a in range(len(bonds_ref)):
        if not a_in_edges(output, a):
            output = output + get_edges_a(bonds_ref, a)
    return output

def same_edge(edge1, edge2):
    # Assuming of same length
    same = True
    with_shift = False
    for a in edge1:
        same = same and a in edge2
    if same:
        with_shift = same_edge_shift(edge1, edge2)
    return same, with_shift

def same_edge_shift(edge1, edge2):
    n = len(edge1)
    shift = edge2.index(edge1[0])
    with_shift = True
    for a in range(n):
        aligned = edge1[a] == edge2[(a + shift) % n]
        with_shift = with_shift and aligned
    return with_shift




#def get_all_edge_n(edges)


# def get_n_strings_from_a(bonds_ref, a):
#     next_as = bonds_ref[a]
#     branches = len(next_as)


# def get_n_strings(atoms_obj, n):
#     # followup function for bonds_dict, takes the bonds_dict and finds all contiguous strings of n bonded atoms
#     bonds_ref = bonds_dict(atoms_obj)
#     output = []
#     for a in range(len(bonds_ref)):
#         get_n_strings_from_a(bonds_ref, )




def coordination_numbers(atoms_obj, sort=True):
    bond_dict = bonds_dict(atoms_obj)
    return_list_of_tuples = []
    for i in range(len(bond_dict)):
        return_list_of_tuples.append(tuple([i, len(bond_dict[i])]))
    if sort:
        return_list_of_tuples = sorted(return_list_of_tuples, key=lambda x: -x[1])
    return return_list_of_tuples


def get_bond_disp_vec(posns, idcs):
    posn1 = posns[idcs[0]]
    posn2 = posns[idcs[1]]
    bvec = posn2 - posn1
    posn2_disp = posn2 - (bvec / 2)
    posn1_disp = posn1 + (bvec / 2)
    posns_new = copy.copy(posns)
    posns_new[idcs[0]] = posn1_disp
    posns_new[idcs[1]] = posn2_disp
    posns_new = posns - posns_new
    return posns_new / np.linalg.norm(posns_new)


def stretch_posn_vec(posn_vec):
    n = len(posn_vec)
    output = np.empty(n * 3)
    for i in range(n):
        for j in range(3):
            output[3 * i + j] = posn_vec[i][j]
    return output


def bond_vec(posns, idcs):
    vec = get_bond_disp_vec(posns, idcs)
    return stretch_posn_vec(vec)


def get_unique_bonds(atoms_obj):
    bond_dict = bonds_dict(atoms_obj)
    return_dict = {}
    tmp_list = []
    for i in range(len(bond_dict)):
        return_dict[i] = []
        bonds = bond_dict[i]
        for a in bonds:
            if a < i:
                if not i in return_dict[a]:
                    return_dict[i].append(a)
            else:
                return_dict[i].append(a)
    return return_dict


def unique_bonds(atoms_obj):
    u_bond_dict = get_unique_bonds(atoms_obj)
    output = []
    for i in range(len(u_bond_dict)):
        for j in u_bond_dict[i]:
            output.append(tuple([i, j]))
    return output


def get_empty_vec(n):
    return np.zeros(n)


def bond_disp_vecs_sq(atoms_obj):
    posns = atoms_obj.get_positions()
    u_bonds = unique_bonds(atoms_obj)
    n_atoms = len(atoms_obj.get_atomic_numbers())
    m_bonds = len(u_bonds)
    output = np.empty(tuple([3 * n_atoms, 3 * n_atoms]))
    for m in range(m_bonds):
        output[m] = bond_vec(posns, [u_bonds[m][0], u_bonds[m][1]])
    for s in range(3 * n_atoms - m_bonds):
        output[m_bonds + s] = get_empty_vec(3 * n_atoms)
    return output


def bond_disp_vecs(atoms_obj):
    posns = atoms_obj.get_positions()
    u_bonds = unique_bonds(atoms_obj)
    n_atoms = len(atoms_obj.get_atomic_numbers())
    m_bonds = len(u_bonds)
    output = np.empty(tuple([m_bonds, 3 * n_atoms]))
    for m in range(m_bonds):
        output[m] = bond_vec(posns, [u_bonds[m][0], u_bonds[m][1]])
    return output