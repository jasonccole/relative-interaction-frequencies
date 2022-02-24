'''
Return geometric parameters for pi systems.
'''

########################################################################################################################

import __future__
from ccdc.descriptors import MolecularDescriptors

########################################################################################################################


class PlaneDistanceCalculator(object):
    
    def __init__(self, plane_atom, contact_atom):
        self.central_atom = plane_atom
        self.contact_atom = contact_atom
        
    def return_plane_distance(self):
        plane_atoms = self._return_plane_atoms()
        plane = MolecularDescriptors.atom_plane(plane_atoms[0], plane_atoms[1], plane_atoms[2])
        distance = abs(plane.point_distance(self.contact_atom.coordinates))
        return distance
    
    def _return_plane_atoms(self):
        '''

        :return:
        >>> from ccdc import io, protein
        >>> p = protein.Protein.from_entry(io.EntryReader('testdata/4A5S_002.mol2')[0])
        >>> plane_atom = p.atoms[65]
        >>> contact_atom = p.atoms[142]
        >>> plane = PlaneDistanceCalculator(plane_atom, contact_atom)
        >>> plane_atoms = plane._return_plane_atoms()
        >>> len(plane_atoms)
        3
        >>> plane_atoms[0] == plane_atoms[1]
        False
        >>> plane_atoms[1] == plane_atoms[2]
        False
        >>> plane_atoms[0] == plane_atoms[2]
        False
        >>> p = protein.Protein.from_entry(io.EntryReader('testdata/4AGQ_011.mol2')[0])
        >>> plane_atom = p.atoms[84]
        >>> contact_atom = p.atoms[133]
        >>> plane = PlaneDistanceCalculator(plane_atom, contact_atom)
        >>> plane_atoms = plane._return_plane_atoms()
        >>> len(plane_atoms)
        3
        >>> plane_atoms[0] == plane_atoms[1]
        False
        >>> plane_atoms[1] == plane_atoms[2]
        False
        >>> plane_atoms[0] == plane_atoms[2]
        False
        '''
        if len(self.central_atom.neighbours) < 2:
            plane_atoms = list(self.central_atom.neighbours)
            for second_degree_neighbour in self.central_atom.neighbours[0].neighbours:
                if second_degree_neighbour != self.central_atom:
                    plane_atoms.append(second_degree_neighbour)
                    break
            plane_atoms.append(self.central_atom)
        else:
            plane_atoms = list(self.central_atom.neighbours[0:2])
            plane_atoms.append(self.central_atom)
        return plane_atoms


class TorsionAngleCalculator(object):

    def __init__(self, torsion_atom, contact_atom):
        self.central_atom = torsion_atom
        self.contact_atom = contact_atom

    def return_torsion_angle(self):
        '''
        Calculate O-C-O...X torsion angle for carboxylate.
        :return: Torsion angle
        '''

        torsion_atoms = self._return_torsion_atoms()
        tau = MolecularDescriptors.atom_torsion_angle(self.contact_atom, *torsion_atoms)
        return abs(tau)

    def _return_torsion_atoms(self):
        a1 = self.central_atom
        a2 = self.central_atom.neighbours[0]
        a3 = [a for a in a2.neighbours if a.atomic_symbol == 'O' and a.partial_charge!=a1.partial_charge][0]
        return [a1, a2, a3]


