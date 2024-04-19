"""
Code used to define a molecule.
"""
import typing
import numpy as np
from basis_sets.gaussian import GaussianBasis
from structure import atom

class molecule():

    def __init__(self, atoms):
        self.atoms = atoms
        self.atomic_orbs = np.array([atom.orbs for atom in self.atoms])[:,0]
        self.basic_sets = np.array([atom.basis_sets for atom in self.atoms])[:,0]

    def distances(self):
        """
        Calculate the distances between all atoms in the molecule.
        """
        pass

    def distance_vecs(self):
        """
        Calculate the vectors between all atoms in the molecule.
        """
        pass
    
    def overlap(self):
        """
        Calculate the overlap matrix of the molecule.
        """
        num_AOs = len(self.atomic_orbs)
        S = np.zeros([num_AOs, num_AOs])
        for mu in range(num_AOs):
            for nu in range(num_AOs):
                S[mu, nu] = 2
        return S

                        
                
