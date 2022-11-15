from funcs import wilson
from funcs import misc
from funcs import geom
from funcs import connec
from funcs import ic_gen
import math
import numpy as np

def get_Bprim(atoms_obj):
    prim_carts = ic_gen.all_prim_carts(atoms_obj)
    prim_carts_mat = []
    for v in prim_carts:
        next = geom.stretch_vec(v)
        prim_carts_mat.append(next / np.linalg.norm(next))
    B_prim = np.array(prim_carts_mat)
    return B_prim

def get_U(atoms_obj):
    B_prim = get_Bprim()
    BBT = np.dot(B_prim, B_prim.T)
    # # Eigh used instead of eig due to problems of complex leakage
    solve = np.linalg.eigh(BBT)
    non_red_idcs = []
    for i in range(len(solve[0])):
        if not math.isclose(abs(solve[0][i]), 0.0, rel_tol=1e-09, abs_tol=1e-09):
            non_red_idcs.append(i)
    U = []
    for idx in non_red_idcs:
        U.append(solve[1][idx])
    U = np.array(U).T
    return U

def get_B_and_BTinv(atoms_obj):
    B_prim = get_Bprim()
    U = get_U(atoms_obj)
    B = np.dot(U.T, B_prim)
    B = misc.normalize(B)
    # you have to manually orthogonalize the set
    B = np.linalg.qr(B.T)[0].T
    BTinv = np.dot(np.linalg.inv(np.dot(B, B.T)),B)
    return B, BTinv