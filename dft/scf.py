import numpy as np
from scipy.linalg import eigh

from pyscf.dft import libxc
from pyscf.dft import numint

#TODO total electronic energy, total energy 

class RKSDFT():
    def __init__(self, molecule, grid):
        self.grid = grid

        self.molecule = molecule
        self.Nbas = molecule.nao
        self.nelectron = molecule.nelectron

        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.Hcore = self.kinetic + self.nuclear

        self.two_electron = molecule.intor('int2e')

    def __call__(self, initial_c, tol, max_iter):
        # perform a first cycle of scf
        energy, KSwave = self._calculate_cycle(self.overlap, initial_c)
        for iteration in range(max_iter):
            energy_new, KSwave_new = self._calculate_cycle(self.overlap, KSwave)
            if np.allclose(energy, energy_new, atol=tol, rtol=0.0) and np.allclose(KSwave, KSwave_new, atol=tol, rtol=0.0):
                print('Converged')
                density_matrix = self._density_coeff(KSwave_new)
                density = self._density(self.grid, density_matrix)
                Fock = self._Fock(self._coulomb(density_matrix), self._XC(self.grid, density))
                total_elec_energy = self._total_elec_energy(density_matrix, Fock)
                return energy_new, KSwave_new, total_elec_energy
            else:
                energy = energy_new
                KSwave = KSwave_new
        print('Did not converge')

    def _calculate_cycle(self, overlap, KSwave):
        """perfrom single cycle of scf
        """
        # get the initial density matrix
        density_matrix = self._density_coeff(KSwave)
        # get the initial density on a grid
        density = self._density(self.grid, density_matrix)
        #print('Density')
        #print(density)
        #calculate the coulomb matrix
        coulomb = self._coulomb(density_matrix)
        # calculate the exchange correlation potential
        XC = self._XC(self.grid, density)
        # calculate the initial Fock matrix
        Fock = self._Fock(coulomb, XC)
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


    def _density(self, grid, density_matrix):
        """calculate the density matrix on a grid
        """
        ao_vals = numint.eval_ao(self.molecule, grid)
        #density, dx, dy, dz = numint.eval_rho(self.molecule, ao_vals, density_matrix)
        density = numint.eval_rho(self.molecule, ao_vals, density_matrix)
        
        return density


    def _Fock(self, coulomb, XC):
        """calculate the Fock matrix
        F_mu,nu = Hcore_mu,nu + J_mu,nu + XC_mu,nu
        """
        Hcore = self.Hcore

        Fock = Hcore + coulomb + XC
        return Fock

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
        libxc_return = libxc.eval_xc('lda', density)
        # First return value is the energy per particle
        epsilon_xc = libxc_return[0]
        # Second return value is the first derivative of the energy per particle
        first_deriv_eps = libxc_return[1][0]
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

    def _total_elec_energy(self, density_matrix, Fock):
        total_elec_energy = 0
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                total_elec_energy += 0.5 * density_matrix[i, j] * (self.Hcore[j, i] + Fock[j, i])
        print('Fock')
        print(Fock)
        return total_elec_energy


    
