import numpy as np
from scipy.linalg import eigh

from pyscf.dft import libxc
from pyscf.dft import numint

class RKSDFT():
    def __init__(self, molecule):
        self.molecule = molecule
        self.Nbas = molecule.nao

        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.Hcore = self.kinetic + self.nuclear

        self.two_electron = molecule.intor('int2e')

    def _calculate_cycle(self, Fock, KSwave):
        """perfrom single cycle of scf
        """
        energy, KSwave = eigh(Fock, overlap)
        return energy, KSwave



    def _density_coeff(self, KSwave):
        """get the coefficents of the density matrix
        """
        density_coeff = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(int(self.nelectron/2)):
                    density_coeff[i, j] += 2 * KSwave[i, k] * KSwave[j, k]
        return density_coeff


    def _density(self, grid):
        """calculate the density matrix on a grid
        """
        ao_vals = numint.eval_ao(self.molecule, grid)
        density, dx, dy, dz = numint.eval_rho(self.molecule, ao_vals, self._density_coeff())
        
        return density


    def _Fock(self, density_coeff):
        """calculate the Fock matrix
        F_mu,nu = Hcore_mu,nu + J_mu,nu + XC_mu,nu
        """
        Hcore = self.Hcore
        coulomb = self._coulomb(density_coeff)

        Fock = np.zeros((self.Nbas, self.Nbas))
        Fock = Hcore + coulomb + XC

    def _coulomb(self, density_coeff):
        """calculate the coulomb matrix
        """
        coulomb = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(self.Nbas):
                    for l in range(self.Nbas):
                        coulomb[i, j] += density_coeff[k, l] * self.two_electron[i, j, k, l]
        return coulomb



    def _XC(self, grid, density):
        """calculate the exchange correlation energy
        """
        # Get the exchange correlation evaluation from libxc
        libxc = libxc.eval_xc('lda', density)
        # First return value is the energy per particle
        epsilon_xc = libxc[0]
        # Second return value is the first derivative of the energy per particle
        first_deriv_eps = libxc[1][0]
        # The exchange correlation potential for LDA gets calculated as follows:
        # V_xc = epsilon_xc + density * d(epsilon_xc)/d(density)
        XC_pot = epsilon_xc + first_deriv_eps*density

        #get the values for the atomic orbitals for each grid point
        ao_vals = numint.eval_ao(self.molecule, grid)

        XC = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(len(grid)):
                    XC[i, j] += ao_vals[k, i] * XC_pot[k] * ao_vals[k, j]
        return XC




    
