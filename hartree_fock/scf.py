import numpy as np
from scipy.linalg import eigh

class RHF():

    def __init__(self, molecule):
        self.molecule = molecule
        self.Nbas = molecule.nao 
        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.Hcore = self.kinetic + self.nuclear
        self.two_electron = molecule.intor('int2e')
        print('Initialized RHF')
        print('Overlap')
        print(self.overlap)

    def __call__(self, initial_c, tol, max_iter):
        density = self._density(initial_c)
        Fock = self._Fock(density)
        energy, c = self._calculate_cycle(initial_c, self.overlap)
        for iteration in range(max_iter):
            energy_new, c_new = self._calculate_cycle(c, self.overlap)
            if np.allclose(energy, energy_new, atol=tol, rtol=0.0):
                print('Converged')
                density = self._density(c_new)
                total_elec_energy = self._total_elec_energy(density, self._Fock(density))
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
                        Fock[i, j] += density[k, l] * (self.two_electron[i, j, k, l] - 0.5 * self.two_electron[i, l, k, j])
        return Fock

    def _density(self, c):
        density = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(int(self.Nbas/2)):
                    density[i, j] += 2 * c[i, k] * c[j, k]
        return density
    
    def _total_elec_energy(self, density, Fock):
        total_elec_energy = 0
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                total_elec_energy += 0.5 * density[i, j] * (self.Hcore[j, i] + Fock[j, i])
        return total_elec_energy

    def _total_energy(self):
        return 0




class scf_hf():

    def __init__(self, molecule):
        self.molecule = molecule
        self.Nbas = molecule.nao 
        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.Hcore = self.kinetic + self.nuclear
        self.two_electron = molecule.intor('int2e')

    def _calculate_cycle(self, c, S):
        density = self._calculate_density(c)
        Fock = self._Fock_matrix(density)
        overlap = S
        energy, c = eigh(Fock, overlap) 
        return energy, c 

    def _Fock_matrix(self, density):
        #_Fock_matrix = T + V + Vcoulomb - Vexchange
        # F_mu,nu = Hcore_mu,nu + G_mu,nu
        #Fock = self._calculate_Hcore()

        coulomb = self._calculate_coulomb(density)
        exchange = self._calculate_exchange(density)
        Fock = self.kinetic + self.nuclear + coulomb - 0.5 * exchange
        return Fock

    def _calculate_Hcore(self):
        return self.kinetic + self.nuclear

    def _calculate_coulomb(self, density):
        coulomb = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(self.Nbas):
                    for l in range(self.Nbas):
                        coulomb[i, j] += density[k, l] * self.two_electron[i, j, k, l]
        return coulomb
    
    def _calculate_exchange(self, density):
        exchange = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(self.Nbas):
                    for l in range(self.Nbas):
                        exchange[i, j] += density[k, l] * self.two_electron[i, l, k, j]
        return exchange

    def _calculate_density(self, c):
        print('Calculating density')
        print('c')
        print(c)
        density = np.zeros((self.Nbas, self.Nbas))
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(int(self.Nbas/2)):
                    density[i, j] += 2 * c[i, k] * c[j, k]
        return density

    def _calculate_total_energy(self, density, Hcore, Fock):
        total_energy = 0
        for mu in range(int(self.Nbas)):
            for nu in range(int(self.Nbas)):
                total_energy += 0.5 * density[nu, mu] * (Hcore[mu, nu] + Fock[mu, nu])
        return total_energy

    def __call__(self, initial_c, tol, max_iter):
        print('Called SCF')
        print('#################################################')
        density = self._calculate_density(initial_c)
        print('Initial density')
        print(density)
        energy, c = self._calculate_cycle(initial_c, self.overlap)
        for iteration in range(max_iter):
            energy_new, c_new = self._calculate_cycle(c, self.overlap)
            if np.allclose(energy, energy_new, atol=tol, rtol=0.0):
                print('Converged')
                density = self._calculate_density(c_new)
                total_energy = self._calculate_total_energy(density, self.Hcore, self._Fock_matrix(density))
                return energy_new, c_new, total_energy
            else:
                energy = energy_new
                c = c_new
        density = self._calculate_density(c)
        total_energy = self._calculate_total_energy(density, self.Hcore, self._Fock_matrix(density))
        return energy, c, total_energy
            







       
