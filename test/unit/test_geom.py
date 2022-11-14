import unittest
import os
import sys
mainpath = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(mainpath)
from funcs import geom
import numpy as np

M = 3
N = 5
class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.i_modes = list(np.eye(3*N))
        cls.ex_squish_vec = np.random.random(tuple([N, 3]))
        tmp_squish_vecs = []
        for i in range(M):
            tmp_squish_vecs.append(np.random.random(tuple([N, 3])))
        cls.ex_squish_vecs = tmp_squish_vecs
        cls.mom_0_0_vec = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
        cls.mom_0_100_vec = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        cls.mom_0_010_vec = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
        cls.mom_0_001_vec = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])


    def test_stretch_vec(self):
        shape = np.shape(self.ex_squish_vec)
        output = geom.stretch_vec(self.ex_squish_vec)
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.assertEqual(output[shape[1]*i + j],
                                 self.ex_squish_vec[i][j])

    def test_stretch_vecs(self):
        output = geom.stretch_vecs(self.ex_squish_vecs)
        for n in range(len(output)):
            shape = np.shape(self.ex_squish_vecs[n])
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.assertEqual(output[n][shape[1] * i + j],
                                     self.ex_squish_vecs[n][i][j])

    def test_squish_vec(self):
        shape = np.shape(self.ex_squish_vec)
        output = geom.squish_vec(geom.stretch_vec(self.ex_squish_vec))
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.assertEqual(output[i][j], self.ex_squish_vec[i][j])

    def test_normalize_basis(self):
        basis = geom.stretch_vecs(self.ex_squish_vecs)
        output = geom.normalize_basis(basis)
        for b in output:
            self.assertAlmostEqual(np.linalg.norm(b), 1.0)

    def test_get_vec_in_basis(self):
        vec = geom.stretch_vec(self.ex_squish_vec)
        output = geom.get_vec_in_basis(vec, self.i_modes)
        for i in range(len(vec)):
            self.assertEqual(output[i], vec[i])

    def test_get_vec_from_cs(self):
        vec = geom.stretch_vec(self.ex_squish_vec)
        cs = geom.get_vec_in_basis(vec, self.i_modes)
        output = geom.get_vec_from_cs(cs, self.i_modes)
        for i in range(len(vec)):
            self.assertAlmostEqual(vec[i], output[i])

    def test_get_cart_mom_0(self):
        mom_0_0 = geom.get_cart_mom_n(self.mom_0_0_vec,0)
        for v in mom_0_0:
            self.assertAlmostEqual(v, 0)
        mom_0_100 = geom.get_cart_mom_n(self.mom_0_100_vec, 0)
        self.assertAlmostEqual(mom_0_100[0]*len(self.mom_0_100_vec), 1)
        self.assertAlmostEqual(mom_0_100[1]*len(self.mom_0_100_vec), 0)
        self.assertAlmostEqual(mom_0_100[2]*len(self.mom_0_100_vec), 0)
        mom_0_010 = geom.get_cart_mom_n(self.mom_0_010_vec, 0)
        self.assertAlmostEqual(mom_0_010[0]*len(self.mom_0_010_vec), 0)
        self.assertAlmostEqual(mom_0_010[1]*len(self.mom_0_010_vec), 1)
        self.assertAlmostEqual(mom_0_010[2]*len(self.mom_0_010_vec), 0)
        mom_0_001 = geom.get_cart_mom_n(self.mom_0_001_vec, 0)
        self.assertAlmostEqual(mom_0_001[0]*len(self.mom_0_001_vec), 0)
        self.assertAlmostEqual(mom_0_001[1]*len(self.mom_0_001_vec), 0)
        self.assertAlmostEqual(mom_0_001[2]*len(self.mom_0_001_vec), 1)

    def test_get_cart_mom_n(self):
        testvec1 = np.array([[5, 0, 0], [0, 1, 0]])
        testvec2 = np.array([[2, 0, 0], [0, 1, 0]])
        testvec1_mom1 = geom.get_cart_mom_n(testvec1, 1)
        testvec1_mom2 = geom.get_cart_mom_n(testvec1, 2)
        testvec2_mom1 = geom.get_cart_mom_n(testvec2, 1)
        testvec2_mom2 = geom.get_cart_mom_n(testvec2, 2)
        self.assertTrue(testvec1_mom1[0] > testvec1_mom1[1])
        self.assertTrue(testvec1_mom2[0] > testvec1_mom2[1])
        self.assertTrue(testvec2_mom1[0] > testvec2_mom1[1])
        self.assertTrue(testvec2_mom2[0] > testvec2_mom2[1])
        self.assertTrue(testvec1_mom1[0] > testvec2_mom1[0])
        self.assertTrue(testvec1_mom2[0] > testvec2_mom2[0])
        self.assertTrue(testvec1_mom1[2] == testvec2_mom1[2])
        self.assertTrue(testvec1_mom2[2] == testvec2_mom2[2])

    def test_center_cart_mean(self):
        centered = geom.center_cart_mean(self.ex_squish_vec)
        mom0 = geom.get_cart_mom_n(centered,0)
        for v in mom0:
            self.assertAlmostEqual(v, 0)

    def test_project_out_vec_from_vec(self):
        vec1 = geom.stretch_vec(np.random.random(tuple([N, 3])))
        vec2 = geom.stretch_vec(np.random.random(tuple([N, 3])))
        vec1_p = geom.project_out_vec_from_vec(vec1, vec2)
        overlap = np.dot(vec1_p, vec2)
        self.assertAlmostEqual(overlap, 0)

    def test_align(self):
        vec1 = geom.stretch_vec(np.random.random(tuple([N, 3])))
        vec2 = geom.stretch_vec(np.random.random(tuple([N, 3])))
        vec1, vec2 = geom.align(vec1, vec2)




if __name__ == '__main__':
    unittest.main()