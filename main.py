from basis_sets.gaussian import GaussianBasis
from structure import atom, molecule
from hartree_fock.scf import scf_hf
import typing
import numpy as np 

from pyscf import gto, scf

#Basic program structure:
# define basis sets for the atoms
# put atoms into the molecule
# calculate the overlap matrix etc.
# perform the SCF calculation

H2 = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr')
print(H2.atom_coords())

scf = scf_hf(H2)

initial_guess = np.zeros((scf.Nbas, scf.Nbas))
energy, c, total_energy = scf(initial_guess, tol=1E-10, max_iter=100000000000)

print('Energy')
print(energy)
print('total energy')
print(total_energy + H2.energy_nuc())


#
#CO = gto.M(atom='C 0 0 0; O 0 0 2.132', basis='sto-3g', unit='Bohr')
#scf = scf_hf(CO)
#initial_guess = np.zeros((scf.Nbas, scf.Nbas))
#energy, c, total_energy = scf(initial_guess, tol=1e-10, max_iter=1000)
#print('Energy')
#print(energy)
#print('total electronic energy')
#print(total_energy)
#print('total energy')
#print(total_energy + CO.energy_nuc())
#


#N2 = gto.M(atom='N 0 0 0; N 0 0 2.074', basis='sto-3g', unit='Bohr')
#
#scf = scf_hf(N2)
#initial_guess = np.zeros((scf.Nbas, scf.Nbas))
#energy, c, total_energy = scf(initial_guess, tol=1e-200, max_iter=1000)
#print('Energy')
#print(energy)
#print('total electronic energy')
#print(total_energy)
#print('total energy')
#print(total_energy + N2.energy_nuc())
