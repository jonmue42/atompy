"""
Code used to define the properties of an atom.
"""

class atom():
    def __init__(self, atomic_num, coords, orbs, basis_sets):
        self.atomic_num = atomic_num
        self.coords = coords
        self.orbs = orbs
        self.basis_sets = basis_sets
