import typing
from typing import List
import numpy as np

class GaussianBasis():

    def __init__(self, exponents, coeffs, L):
        '''
        '''
        self.exponents = np.array(exponents)
        self.coeffs = coeffs
        self.num_prim = len(coeffs)
        self.L = L
        self.norm = self.normalize(self.exponents, L)

    def normalize(self, exponents, L):
        '''normalize the basis set
        '''
        norm = (2.0 * exponents /np.pi)**0.75
        return norm






