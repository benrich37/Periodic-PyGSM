import numpy as np
import copy


def add_trans_idcs(t1, t2):
    """ Returns two lists added together
    :param (list[int]) t1:
    :param (list[int]) t2:
    :return (list[int]) new_trans_idcs:
    """
    new_trans_idcs = []
    for i in range(len(t1)):
        new_trans_idcs.append(int(t1[i] + t2[i]))
    return new_trans_idcs

def gen_all_trans_idcs(atom_idcs, bond_dict):
    """ Returns a list of trans_idcs to tell the cartesian vector generator
    where the origin should be shifted for each atom position

    (ie if atom1 bonds to atom2 with trans_idcs [1, 0, 0], and
    atom2 bonds to atom3 with trans_idcs [0, 1, 0], the returned
    list would be [[0,0,0], [1,0,0], [1,1,0]])
    :param (list[int]) atom_idcs:
    :param (dict) bond_dict:
    :return (list[list[int]]) all_trans_idcs:
    """
    all_trans_idcs = [[0, 0, 0]]
    for i in range(len(atom_idcs) - 1):
        a_i = atom_idcs[i]
        a_i_step_idx = bond_dict[a_i][0].index(atom_idcs[i+1])
        trans_step = bond_dict[a_i][1][a_i_step_idx]
        all_trans_idcs.append(add_trans_idcs(all_trans_idcs[-1], trans_step))
    return all_trans_idcs

def posns_trans(posns, cell, all_trans_idcs):
    """
    :param (list[np.ndarray]) posns:
    :param (np.ndarray) cell:
    :param (list[list[int]]) all_trans_idcs:
    :return list[np.ndarray]):
    """
    output = []
    for i in range(len(posns)):
        output.append(posn_trans(posns[i], cell, all_trans_idcs[i]))
    return output

def posn_trans(posn, cell, trans_idcs):
    # Cell is actually a 3x3, where each entry is a lattice vector
    """
    :param posn: [x, y, z] (coords in origin image)
    :param cell: 3 lattice vectors of length 3
    :param trans_idcs: [i, j, k] (indices of desired image neighbor)
                     (ie the origin image is [0, 0, 0])
    :return: posn_trans: [x, y, z] (translated posn]
    """
    posn_trans = np.zeros(3)
    for i in range(3):
        posn_trans += cell[i] * trans_idcs[i]
    posn_trans += posn
    return posn_trans

def set_iterate(bool):
    """
    :param bool: Taken from pbc condition for a lattice vector
    :return iterate list: Either [0] (if bool False) or [-1, 0, 1]
    """
    if bool:
        return [-1, 0, 1]
    else:
        return [0]

def closest_img(posn1, posn2, pbc, cell):
    """ Finds the closest cell for posn2 given cell vectors and pbc criteria,
    returns the closest distance and trans indices for best cell
    :param (np.ndarray) posn1: xyz array
    :param (np.ndarray) posn2: xyz array
    :param (list[bool]) pbc: len 3 list of bools
    :param (np.ndarray) cell: lattice vectors defining cell in cartesian
    :return (float) cur_min: posn1 to posn2 closest dist
    :return (list[int]) cur_min_trans_idcs: translate idcs of closest img for posn2
    """
    # Strategy - keep posn1 the same, but iterate posn2 through all neighboring
    #            images, and save the distance of the closest neighbor and a
    #            list to remember which neighbor that was (ie the cell image
    #            that's up one unit cell along x, down one along y, and aligned
    #            along z would be [1, -1, 0]
    cur_min = 1E10
    cur_min_trans_idcs = [0, 0, 0]
    for i in set_iterate(pbc[0]):
        for j in set_iterate(pbc[1]):
            for k in set_iterate(pbc[2]):
                trans_posn2 = posn_trans(posn2, cell, [i, j, k])
                trans_dist = abs(np.linalg.norm(trans_posn2 - posn1))
                if trans_dist < cur_min:
                    cur_min = trans_dist
                    cur_min_trans_idcs = [i, j, k]
    return cur_min, cur_min_trans_idcs

def closest_pair(frag1, frag2, posns, cell, pbc):
    """ Returns min_a1_a2, min_dist, a2_trans_idcs
    :param frag1: list of atom indices
    :param frag2: another list of atom indices
    :param posns: ref list of atom cartesian positions
    :param cell: lattice vectors
    :param pbc: list of bools for whether system is periodic for each lattice vector
    :return min_a1_a2: list of ints of length 2
    :returns min_dist: float
    :returns a2_trans_idcs: list of (-1, 0, or 1) of length 3
    """
    min_dist = 1E10
    min_a1_a2 = [None, None]
    a2_trans_idcs = [None, None, None]
    for a1 in frag1:
        for a2 in frag2:
            dist, trans_idcs = closest_img(posns[a1], posns[a2], pbc, cell)
            if dist < min_dist:
                min_dist = dist
                a2_trans_idcs = trans_idcs
                min_a1_a2 = [a1, a2]
    return min_a1_a2, min_dist, a2_trans_idcs

def stretch_vec(vec):
    # assumes an n by 3 vec, as usually used to describe molecule geometry
    n = len(vec)
    output = list(np.zeros(3*n))
    for i in range(n):
        for j in range(3):
            output[3*i + j] = vec[i][j]
    return np.array(output)

def squish_vec(vec):
    # assumed a 1 by 3N vec
    n = len(vec)
    output = []
    for i in range(int(n/3)):
        output.append([0, 0, 0])
    for i in range(int(n/3)):
        for j in range(3):
            output[i][j] = vec[3*i+j]
    return np.array(output)

def stretch_vecs(vecs):
    output = []
    for v in vecs:
        output.append(stretch_vec(v))
    return output

def squish_vecs(vecs):
    output = []
    for v in vecs:
        output.append(squish_vec(v))
    return output

def normalize_basis(vecs):
    output = []
    for v in vecs:
        output.append(v/np.linalg.norm(v))
    return output

# Functions below this line are currently unused
################################

def get_vec_in_basis(vec, basis):
    cs = []
    for b in basis:
        cs.append(np.dot(vec, b))
    return cs

def get_vec_from_cs(cs, basis):
    vec = np.zeros(np.shape(basis[0]))
    for i in range(len(cs)):
        vec += basis[i]*cs[i]
    return vec

def get_cart_mom_n(posns, n):
    mom = np.zeros(np.shape(posns[0]))
    l = len(posns)
    for p in posns:
        mom += p*(np.linalg.norm(p)**n)
    mom = mom/l
    return mom

def center_moment(posns, mom_c, mom_t):
    # mom_c is current moment of posns, mom_t is what we want the posns moment to be
    dif = mom_t - mom_c
    posns_return = copy.copy(posns)
    for i in range(len(posns_return)):
        posns_return[i] += dif
    return posns_return

def get_rot_matrix_ax_2(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mat2 = [
        [c,-s,0],
        [s,c,0],
        [0,0,1]
    ]
    return mat2

def get_tot_rot_matrix(angles):
    a = angles[0]
    cs = np.cos(a)
    ss = np.sin(a)
    b = angles[1]
    cb = np.cos(b)
    sb = np.sin(b)
    c = angles[2]
    cc = np.cos(c)
    sc = np.sin(c)
    mat = [
        [],
        [],
        []
    ]

def get_rot_matrix(angle, axis):
    mat = get_rot_matrix_ax_2(angle)
    dif = (axis - 2) % 3
    if not dif == 0:
        mat = next_rot_matrix(mat, step=dif)
    return mat

def next_rot_matrix(rot_matrix, step=1):
    next = np.zeros(tuple([3, 3]))
    for i in range(3):
        for j in range(3):
            next[(i+step)%3][(j+step)%3] = rot_matrix[i][j]
    return next

def rotate_posns(posns, rot_matrix):
    newps = []
    for p in posns:
        newps.append(np.dot(rot_matrix, p))
    return newps

def rotate(posns, angles):
    for i in range(3):
        rot_matrix = get_rot_matrix(angles[i], i)
        posns = rotate_posns(posns, rot_matrix)
    return posns

def get_angle(vec, axis):
    return np.arctan(vec[(axis + 2) % 3]/vec[(axis + 1) % 3])

def get_angles(vec):
    angles = []
    for a in range(3):
        angles.append(get_angle(vec, a))
    # in order theta_x (yz plane), theta_y (xz plane), theta_z (xy plane)
    return angles

def center_cart_mean(posns):
    mean = get_cart_mom_n(posns, 0)
    posns += -mean
    return posns

def get_d_angles(vec1, vec2):
    vec1_angles = np.array(get_angles(vec1))
    vec2_angles = np.array(get_angles(vec2))
    d_angles = vec1_angles - vec2_angles
    return d_angles

def project_out_vec_from_vec(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    p_out = vec2*np.dot(vec1, vec2)
    vec1 = vec1 - p_out
    return vec1

def align(posns1, posns2):
    # 1. align cartesian means
    posns1 = center_cart_mean(posns1)
    posns2 = center_cart_mean(posns2)
    # 2. Find vectors for first cartesian moment (sum posn*norm)
    posns1_mom1 = get_cart_mom_n(posns1, 1)
    posns2_mom1 = get_cart_mom_n(posns2, 1)
    d_angles1 = get_d_angles(posns2_mom1, posns1_mom1)
    # 3. Rotate posns2 along xy/yz/zx planes so first cartesian moment is aligned
    posns2 = rotate(posns2, d_angles1)
    # 4. Find second cartesian moment (sum posn*norm**2) of both posns with first moment projected out
    posns1_mom2 = get_cart_mom_n(posns1, 2)
    posns2_mom2 = get_cart_mom_n(posns2, 2)
    posns1_mom2_p = project_out_vec_from_vec(posns1_mom2, posns1_mom1)
    posns2_mom2_p = project_out_vec_from_vec(posns2_mom2, posns2_mom1)
    # 5. Rotate posns2 along plane orthogonal to first moment to align projected second moments
    d_angles2 = get_d_angles(posns2_mom2_p, posns1_mom2_p)
    posns2 = rotate(posns2, d_angles2)
    return posns1, posns2