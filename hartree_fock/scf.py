import numpy as np
from scipy.linalg import eigh

class RHF():

    def __init__(self, molecule):
        self.molecule = molecule
        self.Nbas = molecule.nao 
        self.nelectron = molecule.nelectron
        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.Hcore = self.kinetic + self.nuclear
        self.two_electron = molecule.intor('int2e')
        print('Initialized RHF')
        print('Overlap')
        print(self.overlap)
        print('Hcore')
        print(self.Hcore)

    def __call__(self, initial_c, tol, max_iter):
        energy, c = self._calculate_cycle(initial_c, self.overlap)
        for iteration in range(max_iter):
            energy_new, c_new = self._calculate_cycle(c, self.overlap)
            if np.allclose(energy, energy_new, atol=tol, rtol=0.0) and np.allclose(c, c_new, atol=tol, rtol=0.0):
                print('Converged')
                density = self._density(c_new)
                total_elec_energy = self._total_elec_energy(density, self._Fock(density))
                print('Fock')
                print(self._Fock(density))
                return energy_new, c_new, total_elec_energy
            else:
                energy = energy_new
                c = c_new
        print('Did not converge')

    def _calculate_cycle(self, c, S):
        density = self._density(c)
        Fock = self._Fock(density)
        overlap = S
        energy, c = eigh(Fock, overlap) 
        return energy, c 

    def _Fock(self, density):
        #F_mu,nu = Hcore_mu,nu + G_mu,nu
        Hcore = self.Hcore
        
        Fock = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                Fock[i, j] += Hcore[i, j]
                for k in range(int(self.Nbas)):
                    for l in range(int(self.Nbas)):
                        Fock[i, j] += density[k, l] * (self.two_electron[i, j, k, l] - 0.5 * self.two_electron[i, k, j, l])
        return Fock

    def _density(self, c):
        density = np.zeros((self.Nbas, self.Nbas))
        norm = np.zeros(self.Nbas)
        for k in range(int(self.Nbas)):
            for p in range(self.Nbas):
                for q in range(self.Nbas):
                    norm[k] += c[p, k] * c[q, k] * self.overlap[p, q]
        print('Norm')
        print(norm)
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(int(self.nelectron/2)):
                    density[i, j] +=2 * c[i, k] * c[j, k]

        return density
    
    def _total_elec_energy(self, density, Fock):
        total_elec_energy = 0
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                total_elec_energy += 0.5 * density[i, j] * (self.Hcore[j, i] + Fock[j, i])
        return total_elec_energy

    def _total_energy(self):
        return 0




