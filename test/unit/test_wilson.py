import unittest
from ase.io import read,write
import numpy as np
import os
import sys
mainpath = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(mainpath)
from funcs import wilson, ic_gen, misc
#print(mainpath)
proponal = read('../toy_data/proponal.xyz')
ics = np.array(ic_gen.get_ic_vecs(proponal.get_positions()))
# output = []
# output.append(ics[0])
# output.append(ics[1])
# output.append(ics[2])
for ic in ics:
    print(ic)
#print(ics[0])
# print(np.shape(ics))
BBT = np.dot(ics,ics.T)
solve = np.linalg.eig(BBT)
# print(solve[0])
# print(len(solve[0]))
# print(solve[1])
choose_v = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
newvs = np.array(ic_gen.project_out(choose_v, solve[1]))
newsolve = np.linalg.eig(newvs)
#print(newvs)
# print(newsolve[0])
# print(newsolve[1][0])

# print(len(ics[2]) + len(ics[1]) + len(ics[0]))
# print(3*len(proponal.get_positions()) - 6)

