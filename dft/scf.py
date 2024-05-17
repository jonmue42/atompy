import numpy as np
from scipy.linalg import eigh

from pyscf.dft import libxc
from pyscf.dft import numint

#TODO total electronic energy, total energy 

class RKSDFT():
    def __init__(self, molecule, grid):
        self.grid = grid.coords
        self.weights = grid.weights

        self.molecule = molecule
        self.Nbas = molecule.nao
        self.nelectron = molecule.nelectron

        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.Hcore = self.kinetic + self.nuclear

        self.two_electron = molecule.intor('int2e')

        self.cycle_iter = 0

    def __call__(self, initial_c, tol, max_iter):
        # perform a first cycle of scf
        energy, KSwave = self._calculate_cycle(self.overlap, initial_c)
        for iteration in range(max_iter):
            energy_new, KSwave_new = self._calculate_cycle(self.overlap, KSwave)
            print('total energy: ', self._total_elec_energy(energy_new, self._density_matrix(KSwave_new), self._density(self.grid, self._density_matrix(KSwave_new)), self._coulomb(self._density_matrix(KSwave_new))))
            if np.allclose(energy, energy_new, atol=tol, rtol=0.0) and np.allclose(KSwave, KSwave_new, atol=tol, rtol=0.0):
                print('Converged')
                density_matrix = self._density_matrix(KSwave_new)
                density = self._density(self.grid, density_matrix)
                coulomb = self._coulomb(density_matrix)
                total_elec_energy = self._total_elec_energy(energy_new, density_matrix, density, coulomb)
                return energy_new, KSwave_new, total_elec_energy
            else:
                energy = energy_new
                KSwave = KSwave_new
        print('Did not converge')
        density_matrix = self._density_matrix(KSwave_new)
        density = self._density(self.grid, density_matrix)
        coulomb = self._coulomb(density_matrix)
        total_elec_energy = self._total_elec_energy(energy_new, density_matrix, density, coulomb)
        return energy_new, KSwave_new, total_elec_energy



    def _calculate_cycle(self, overlap, KSwave):
        """perfrom single cycle of scf
        """
        print('Calculating cycle: ', self.cycle_iter)
        self.cycle_iter += 1
        # get the initial density matrix
        density_matrix = self._density_matrix(KSwave)
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



    def _density_matrix(self, KSorbs):
        """get the coefficents of the density matrix
        """
        density_coeff = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(int(self.nelectron/2)):
                    #print('nelectron: ', self.nelectron)
                    #print('electron: ', k)
                    #print('KSorbs[i, k]', KSorbs)
                    density_coeff[i, j] += KSorbs[i, k] * KSorbs[j, k]
        return density_coeff


    def _density(self, grid, density_matrix):
        """calculate the density matrix on a grid
        """
        ao_vals = numint.eval_ao(self.molecule, grid)
        #density, dx, dy, dz = numint.eval_rho(self.molecule, ao_vals, density_matrix)
        #density = numint.eval_rho(self.molecule, ao_vals, density_matrix)
        density = np.zeros(len(grid))
        for p in range(len(density)):
            for i in range(self.Nbas):
                for j in range(self.Nbas):
                    density[p] += density_matrix[i, j] * ao_vals[p, i] * ao_vals[p, j]
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
                for p in range(len(grid)):
                    XC[i, j] += ao_vals[p, i] * XC_pot[p] * ao_vals[p, j] * self.weights[p]
        return XC

    def _total_elec_energy(self, orbs_energy, density_matrix, density, coulomb):
        """calculate the total electronic energy given by
        E = sum_i^N epsilon_i - 0.5 
        """
        #initiate total energy with sum of orbital energies
        total_elec_energy = np.sum(orbs_energy)
        #add coulomb part to total energy
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                #total_elec_energy += -0.5 * density_matrix[i, j] * coulomb[i, j]
                total_elec_energy += -density_matrix[i, j] * coulomb[i, j]
        #add exchange correlation part to total energy
        Exc = self._XCenergy(self.grid, density)
        total_elec_energy += Exc

        return total_elec_energy

    def _XCenergy(self, grid, density):
        """calculate the exchange correlation energy
        """
        libxc_return = libxc.eval_xc('lda', density)
        epsilon_xc = libxc_return[0]

        first_deriv_eps = libxc_return[1][0]
        XC_pot = epsilon_xc + first_deriv_eps*density

        Exc = np.sum(epsilon_xc * density * self.weights)
        #subtract potential term
        Exc -= np.sum(XC_pot * density * self.weights)
        return Exc


    
