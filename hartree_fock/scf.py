import numpy as np
from scipy.linalg import eigh

class scf_hf():

    def __init__(self, molecule):
        self.molecule = molecule
        self.Nbas = molecule.nao 
        self.overlap = molecule.intor('int1e_ovlp')
        self.kinetic = molecule.intor('int1e_kin')
        self.nuclear = molecule.intor('int1e_nuc')
        self.two_electron = molecule.intor('int2e')

    def _calculate_cycle(self, c, S):
        density = self._calculate_density(c)
        Fock = self._Fock_matrix(density)
        overlap = S
        energy, c = eigh(Fock, overlap) 
        return energy, c 

    def _Fock_matrix(self, density):
        #_Fock_matrix = T + V + Vcoulomb - Vexchange
        coulomb = self._calculate_coulomb(density)
        print('coulomb')
        print(coulomb)
        exchange = self._calculate_exchange(density)
        print('exchange')
        print(exchange)
        Fock = self.kinetic + self.nuclear + coulomb - 0.5 * exchange
        print('Fock')
        print(Fock)
        return Fock

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
       # print('entered _calculate_density')
        density = np.zeros((self.Nbas, self.Nbas))
       # print('density')
        #print(density)
       # print('c')
       # print(c)
        #density = np.matmul(c, c)
        for i in range(self.Nbas):
            for j in range(self.Nbas):
                for k in range(int(self.Nbas * 0.5)):
                    density[i, j] += 2 * c[i, k] * c[j, k]
       # print('left _calculate_density')
        return density

    def __call__(self, initial_c, tol, max_iter):
        density = self._calculate_density(initial_c)
        print('Called SCF')
        print('################################')
        print('initial_c')
        print(initial_c)
        #print('initial_density')
        #print(initial_density)
        energy, c = self._calculate_cycle(initial_c, self.overlap)
        for iteration in range(max_iter):
            energy_new, c_new = self._calculate_cycle(c, self.overlap)
            if np.allclose(energy, energy_new, atol=0.0, rtol=0.0):
                return energy_new, c_new
            else:
                energy = energy_new
                c = c_new
        return energy, c
            







       
