from funcs import wilson
from funcs import misc
from funcs import geom
from funcs import connec
import numpy as np

# returns a list of atoms lists, signifying bonds, angles, and dihedrals
def all_prim_coords(atoms_obj):
    con_dict = connec.bonds_dict(atoms_obj, con_frag=True)
    prim_coords = connec.get_edges(con_dict)
    return prim_coords



def all_prim_carts(atoms_obj):
    """
    :param (ase.Atoms) atoms_obj:
    :return:
    """
    posns = atoms_obj.get_positions()
    bond_dict = connec.bonds_dict(atoms_obj, con_frag=True)
    prim_coords = connec.get_edges(bond_dict)
    output = []
    for pc in prim_coords:
        all_trans_idcs = geom.gen_all_trans_idcs(pc, bond_dict)
        wilson_pc = wilson.ic_in_cart(posns, pc,
                                      all_trans_idcs, atoms_obj.cell)
        output.append(wilson_pc)
    return output

def get_n_pairs(atom_list, i, m):
    output = []
    for j in range(m):
        output.append(atom_list[i+j])
    return tuple(output)

def get_n_pair(atom_list, n):
    output = []
    # number of times we can fit an n_string into our atom list
    m = len(atom_list) + 1 - n
    for i in range(m):
        output.append(get_n_pairs(atom_list, i, n))
    return output
# naive approach
def get_ic_idcs(atom_list):
    bonds = get_n_pair(atom_list, 2)
    angles = get_n_pair(atom_list, 3)
    dihedrals = get_n_pair(atom_list, 4)
    return bonds, angles, dihedrals

def add_trans_idcs(t_idcs):
    output = np.zeros(3)
    for t in t_idcs:
        output += np.array(t)
    return list(output)

def get_ic_trans_idcs(ic_idcs, bond_dict):
    trans_idcs = [[0,0,0]]
    for i in range(len(ic_idcs) - 1):
        ref = bond_dict[ic_idcs[i]]
        get_idx = ref[0].index(ic_idcs[i + 1])
        trans = ref[1][get_idx]
        trans_idcs.append(add_trans_idcs([trans_idcs[-1], trans]))
    return trans_idcs

def get_all_ic_trans_idcs(ics, bond_dict):
    output = []
    for ic in ics:
        output.append(get_ic_trans_idcs(ic, bond_dict))
    return output

def get_ic_vecs(posns):
    atom_list = range(len(posns))
    bonds, angles, dihedrals = get_ic_idcs(atom_list)
    idcs_list = bonds + angles + dihedrals
    trans_idcs_list = get_all_ic_trans_idcs(idcs_list, bond_dict)
    vecs = []
    for idcs in idcs_list:
        vecs.append(misc.stretch_posn_vec(wilson.ic_in_cart(posns, idcs)))
    return vecs

def project_out(vec, vecs):
    vec = vec/np.linalg.norm(vec)
    output = []
    for v in vecs:
        v = v/np.linalg.norm(v)
        overlap=np.dot(vec, v)
        newv = v - overlap*vec
        newv = newv/np.linalg.norm(newv)
        output.append(newv)
    return output