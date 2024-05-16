from basis_sets.gaussian import GaussianBasis
from structure import atom, molecule
from hartree_fock.scf import RHF
from dft.scf import RKSDFT
import typing
import numpy as np 

from pyscf import gto, scf, dft

from pyscf.dft.libxc import eval_xc
from pyscf.dft import numint

#DFT

H20 = gto.M(atom='O 0 0 0; H 0 0 1.809; H 1.809 0 0', basis='sto-3g', unit='Bohr')
grid = dft.gen_grid.Grids(H20)
grid.build(with_non0tab=True)
print(grid.coords)
DFT_scf = RKSDFT(H20, grid=grid.coords)
initial_c = np.zeros((DFT_scf.Nbas, DFT_scf.Nbas))
DF_energy, DF_KSwave = DFT_scf(initial_c=initial_c, tol=1E-1, max_iter=10)
print('Energy')
print(DF_energy)


##############################################################
#hartree_fock:

#H20 = gto.M(atom='O 0 0 0; H 0 0 1.809; H 1.809 0 0', basis='sto-3g', unit='Bohr')
#mf = dft.RKS(H20)
#mf.xc = 'LDA'
#mf = mf.newton()
#mf.kernel()
#
#print(eval_xc('LDA', np.array([0.1, 0.4])))
#print(eval_xc('LDA', np.array([0.1, 0.4]))[1][0] * [2, 2])

#print(eval_xc('LDA', np.array([0.1])))
#print(eval_xc('LDA', np.array([0.4])))
#
#mol = gto.M(atom='O 0 0 0', basis='sto-3g', unit='Bohr')
#grid = dft.gen_grid.Grids(mol)
#grid.build(with_non0tab=True)
#print(grid.coords)
#coords = np.random.random((4, 3))
#iny = numint.eval_ao(mol, coords, deriv=0)
#print(coords)
#print(iny)

#density = numint.eval_rho(mol, iny, mol.make_rdm1())


#Basic program structure:
# define basis sets for the atoms
# put atoms into the molecule
# calculate the overlap matrix etc.
# perform the SCF calculation
#
#H2 = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr')
#print(H2.atom_coords())
#
#scf = RHF(H2)
#
#initial_guess = np.zeros((scf.Nbas, scf.Nbas))
#energy, c, total_elec_energy = scf(initial_guess, tol=1E-10, max_iter=100000000000)
#
#print('Energy')
#print(energy)
#print('c')
#print(c)
#print('total electronic energy')
#print(total_elec_energy)
#print('total energy')
#print(total_elec_energy + H2.energy_nuc())
#
#
#H = gto.M(atom='He 0 0 0', basis='sto-3g', unit='Bohr')
#scf = RHF(H)
#print('Nbas')
#print(scf.Nbas)
#initial_guess = np.zeros((scf.Nbas, scf.Nbas))
#energy, c, total_elec_energy = scf(initial_guess, tol=1e-10, max_iter=1000)
#print('Energy')
#print(energy)
#print('total electronic energy')
#print(total_elec_energy)
#print('total energy')
#print(total_elec_energy + CO.energy_nuc())
#
#CO = gto.M(atom='C 0 0 0; O 0 0 2.166', basis='sto-3g', unit='Bohr')
#scf = RHF(CO)
#initial_guess = np.zeros((scf.Nbas, scf.Nbas))
#energy, c, total_elec_energy = scf(initial_guess, tol=1e-9, max_iter=1000)
#print('Energy')
#print(energy)
#print('total electronic energy')
#print(total_elec_energy)
#print('total energy')
#print(total_elec_energy + CO.energy_nuc())
#print('Nbas')
#print(scf.Nbas)
#print('number of electrons')
#print(CO.nelectron)
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
#
#H2O = gto.M(atom='O 0 0 0; H 0 0 1.809; H 1.809 0 0', basis='sto-3g', unit='Bohr')
#scf = RHF(H2O)
#print('Nbas')
#print(scf.Nbas)
#print('overlapp')
#print(scf.overlap)
#initial_guess = np.zeros((scf.Nbas, scf.Nbas))
#energy, c, total_elec_energy = scf(initial_guess, tol=1e-10, max_iter=1000)
#print('Energy')
#print(energy)
#print('total electronic energy')
#print(total_elec_energy)
#print('total energy')
#print(total_elec_energy + H2O.energy_nuc())
#
