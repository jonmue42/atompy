from basis_sets.gaussian import GaussianBasis
from structure import atom, molecule
from hartree_fock.scf import scf_hf
import typing
import numpy as np 

#define a STO-3G basis set for 1s of H
#alpha values for STO-3G: 0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00
#coeff values for STO-3G: 0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00

STO_3G = GaussianBasis([0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00], [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], 0)

H1_atom = atom.atom(1, [0.0, 0.0, 0.0], orbs=['1s'], basis_sets=[STO_3G])
H2_atom = atom.atom(1, [0.0, 0.0, 0.74], orbs=['1s'], basis_sets=[STO_3G])

molecule = molecule.molecule([H1_atom, H2_atom])
print(molecule.atomic_orbs)
print(molecule.basic_sets)
print(molecule.overlap())


#Basic program structure:
# define basis sets for the atoms
# put atoms into the molecule
# calculate the overlap matrix etc.
# perform the SCF calculation

