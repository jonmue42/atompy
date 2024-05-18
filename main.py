from basis_sets.gaussian import GaussianBasis
from structure import atom, molecule
from hartree_fock.scf import RHF
from dft.scf import RKSDFT
from dft.scf2 import RKSDFT2
import typing
import numpy as np 

from pyscf import gto, scf, dft

from pyscf.dft.libxc import eval_xc
from pyscf.dft import numint

#DFT

H20 = gto.M(atom='O 0 0 0; H 0 0 1.809; H 1.809 0 0', basis='sto-3g', unit='Bohr')
#H20 = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr')
#H20 = gto.M(atom='C 0 0 0; O 0 0 2.166', basis='sto-3g', unit='Bohr')
grid = dft.gen_grid.Grids(H20)
grid.build(with_non0tab=True)
print('Grid')
print(grid.weights)
#grid = np.linspace(-5, 5, 1000)
#print(grid)
#print(np.stack((grid, grid, grid), axis=1))
#print(grid.coords)
#print(len(grid.coords))
#print(numint.eval_ao(H20, grid.coords, deriv=0).shape)
#libxc_return = eval_xc('lda', np.array([0.1, 0.4]))
#print(libxc_return)
#print(libxc_return[0] + libxc_return[1] * np.array([0.1, 0.4]))


DFT_scf = RKSDFT2(H20, grid=grid)
initial_c = np.zeros((DFT_scf.Nbas, DFT_scf.Nbas))
mf = dft.RKS(H20)
mf.xc = 'LDA'
initial_c = mf.get_init_guess()
DF_energy, DF_KSwave, total_elec_energy = DFT_scf(initial_c=initial_c, tol=1E-6, max_iter=100)
print('Energy')
print(total_elec_energy)
print('Total energy')
print(total_elec_energy + H20.energy_nuc())
print('mo_energy')
print(DF_energy)
print('mo_coeff')
print(DF_KSwave)

print('######################################################################################################################')

def density_matrix_func(molecule, mf):
    Nbas = molecule.nao
    nelectron = molecule.nelectron
    density_matrix = np.zeros((Nbas, Nbas))
    for i in range(Nbas):
        for j in range(Nbas):
                for k in range(int(nelectron/2)):
                    density_matrix[i, j] += mf.mo_coeff[i, k] * mf.mo_coeff[j, k] 
    return density_matrix

def density_func(molecule, grid, density_matrix):
    weights = grid.weights
    grid = grid.coords
    Nbas = molecule.nao
    
    ao_vals = numint.eval_ao(molecule, grid, deriv=0)
    density = numint.eval_rho(molecule, ao_vals, density_matrix)
    return density

def coulomb(molecule, density_matrix):
    Nbas = molecule.nao
    two_electron = molecule.intor('int2e')
    coulomb = np.zeros((Nbas, Nbas))
    for i in range(Nbas):
        for j in range(Nbas):
            for k in range(Nbas):
                for l in range(Nbas):
                    coulomb[i, j] += density_matrix[k, l] * two_electron[i, j, k, l]
    return coulomb

def XCenergy(molecule, grid, density):
    weights = grid.weights
    grid = grid.coords
    XC = eval_xc('LDA', density)
    epsilon = XC[0]
    first_derivative = XC[1][0]
    XC_pot = epsilon + first_derivative * density
    print('XC_pot')
    print(XC_pot)
    Exc = np.sum(epsilon * weights * density)
    Exc -= np.sum(XC_pot * weights * density)
    return Exc

def total_energy(molecule, mf, grid):
    density_matrix = density_matrix_func(molecule, mf)
    density = density_func(molecule, grid, density_matrix)
    coulomb_matrix = coulomb(molecule, density_matrix)
    Exc = XCenergy(molecule, grid, density)

    total_energy = np.sum(mf.mo_energy) 
    for i in range(molecule.nao):
        for j in range(molecule.nao):
            total_energy += -0.5 * density_matrix[i, j] * coulomb_matrix[i, j]
    print('Exc')
    print(Exc)
    total_energy += Exc

    return total_energy


mf = dft.RKS(H20)
mf.xc = 'LDA'
res = mf.kernel()
print(res)
e_tot = mf.e_tot
mo_energy = mf.mo_energy
mo_coeff = mf.mo_coeff


print('total Energy')
print(e_tot)
print('mo_energy')
print(mo_energy)
print('mo_coeff')
print(mo_coeff)
print('initial')
mf.dump_scf_summary()
#total_energy = total_energy(H20, mf, grid)
#print('Total energy')
#print(total_energy)

#grid = dft.gen_grid.Grids(H20)

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
