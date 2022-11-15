from funcs import wilson
from funcs import misc
from funcs import geom
from funcs import connec
import numpy as np

# returns a list of atoms lists, signifying bonds, angles, and dihedrals
def all_prim_coords(atoms_obj):
    init_dict = connec.bonds_dict(atoms_obj)
    con_dict = connec.bonds_dict_con_frags(init_dict, atoms_obj.get_positions())
    prim_coords = connec.get_edges(con_dict)
    return prim_coords

def all_prim_carts(atoms_obj):
    posns = atoms_obj.get_positions()
    prim_coords = all_prim_coords(atoms_obj)
    output = []
    for pc in prim_coords:
        wilson_pc = wilson.ic_in_cart(posns, pc)
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

def get_ic_vecs(posns):
    atom_list = range(len(posns))
    bonds, angles, dihedrals = get_ic_idcs(atom_list)
    idcs_list = bonds + angles + dihedrals
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