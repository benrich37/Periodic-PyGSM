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








if __name__ == '__main__':
    unittest.main()