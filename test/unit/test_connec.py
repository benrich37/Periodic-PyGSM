import unittest
import os
import sys
mainpath = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(mainpath)
from funcs import connec
import numpy as np

M = 3
N = 5
class TestUtils(unittest.TestCase):
    def test_flip_edge(self):
        ex = [0,1,2,3]
        ex_out = [3,2,1,0]
        flip_ex = connec.flip_edge(ex)
        for i in range(len(ex)):
            self.assertEqual(ex_out[i], flip_ex[i])
    def test_same_all(self):
        ex = [0, 1, 2, 3]
        ex1 = [0, 2, 2, 3]
        self.assertTrue(connec.same_all(ex,ex))
        self.assertTrue(not connec.same_all(ex, ex1))
    def test_same_edge(self):
        ex = [0, 1, 2, 3]
        ex_out = [3, 2, 1, 0]
        self.assertTrue(connec.same_edge(ex, ex_out))
    def test_pbc_trans(self):
        posn_og = list(np.random.random(3))
        dummy_pbcs = list(np.random.random(3))
        dummy_idcs = list(np.random.randint(-1, 1, 3))
        self.assertListEqual(list(posn_og),
                             list(connec.pbc_trans(posn_og,
                                          [0, 0, 0],
                                          dummy_idcs
                                          ))
                         )
        self.assertListEqual(list(posn_og),
                             list(connec.pbc_trans(posn_og,
                                                   dummy_pbcs,
                                                   [0, 0, 0]
                                                   ))
                             )
        self.assertListEqual([1.5, 1.5, 1.5],
                             connec.pbc_trans([1.0, 1.0, 1.0],
                                              [0.5, 0.5, 0.5],
                                              [1, 1, 1]))
        self.assertListEqual([0.5, 0.5, 0.5],
                             connec.pbc_trans([1.0, 1.0, 1.0],
                                              [0.5, 0.5, 0.5],
                                              [-1, -1, -1]))
    def test_pbc_dist(self):
        pbc = [5.0, 5.0, 5.0]
        posn1 = np.array([0, 0, 0])
        posn2 = np.array([4.9, 4.9, 4.9])
        self.assertListEqual([0, 0, 0],
                             connec.pbc_dist(posn1, posn1, pbc)[1])
        self.assertListEqual([-1, -1, -1],
                             connec.pbc_dist(posn1, posn2, pbc)[1])








if __name__ == '__main__':
    unittest.main()