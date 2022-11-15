import numpy as np
import copy

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

def closest_pair(frag1, frag2, posns):
    min = 1E10
    closest_pair = None
    for a1 in frag1:
        for a2 in frag2:
            dist = np.linalg.norm(posns[a2] - posns[a1])
            if dist < min:
                min = dist
                closest_pair = [a1, a2]
    return closest_pair, min