
########################################################################################################################

import __future__
import numpy as np
import pandas as pd
from ccdc_roche import lookup_files, atom_types
from ccdc_roche.python.total_rf_calculator import TotalRf
from ccdc_roche.python import atom_geometries
from pathlib import Path

########################################################################################################################

def return_central_ligand_atoms(_protein):
    '''

    :return: list of atoms in central ligand
    '''
    for ligand in _protein.ligands:
        if '_Z' in ligand.atoms[0].label:
            ligand.remove_hydrogens()
            return ligand.atoms
    for nucleotide in _protein.nucleotides:
        if '_Z' in nucleotide.atoms[0].label:
            return nucleotide.atoms
    for cofactor in _protein.cofactors:
        if '_Z' in cofactor.atoms[0].label:
            cofactor.remove_hydrogens()
            return cofactor.atoms

    central_ligand_atoms = []
    for atom in _protein.atoms:
        if '_Z' in atom.label:
            central_ligand_atoms.append(atom)
    return central_ligand_atoms


def return_atom_by_partial_charge(partial_charge, protein_):
    for atom in protein_.atoms:
        if atom.partial_charge == partial_charge:
            return atom


class RfAnalysis(object):

    def __init__(self, input_structure, expected=10):
        atom_type_path = Path(atom_types.__path__[0])
        lookup_files_path = Path(lookup_files.__path__[0])
        self.db = input_structure
        self.output = '.'

        self.protein_atom_types = pd.read_csv(atom_type_path / 'protein_atom_types.csv', sep='\t')

        self.protein = input_structure

        self.global_ligand_lookup_alpha = pd.read_csv(lookup_files_path / 'global_ligand_lookup_alpha.csv',
                                                      sep='\t').astype({'interaction_types': str})
        self.global_ligand_lookup_alpha = self.global_ligand_lookup_alpha[
            self.global_ligand_lookup_alpha['expected'] >= expected]
        self.global_ligand_lookup_alpha = self.global_ligand_lookup_alpha.groupby(by=['ligand_atom_type', 'atom_type'])

        self.global_ligand_lookup_h = pd.read_csv(lookup_files_path / 'global_ligand_lookup_h.csv',
                                                  sep='\t').astype({'interaction_types': str})
        self.global_ligand_lookup_h = self.global_ligand_lookup_h[
            self.global_ligand_lookup_h['expected'] >= expected]
        self.global_ligand_lookup_h = self.global_ligand_lookup_h.groupby(by=['ligand_atom_type', 'atom_type'])

        self.global_protein_lookup_alpha = pd.read_csv(lookup_files_path / 'global_protein_lookup_alpha.csv',
                                                       sep='\t').astype({'interaction_types': str})
        self.global_protein_lookup_alpha = self.global_protein_lookup_alpha[
            self.global_protein_lookup_alpha['expected'] >= expected]
        self.global_protein_lookup_alpha = self.global_protein_lookup_alpha.groupby(by=['ligand_atom_type', 'atom_type']
                                                                                    )

        self.global_protein_lookup_h = pd.read_csv(lookup_files_path / 'global_protein_lookup_h.csv',
                                                   sep='\t').astype({'interaction_types': str})
        self.global_protein_lookup_h = self.global_protein_lookup_h[
            self.global_protein_lookup_h['expected'] >= expected]
        self.global_protein_lookup_h = self.global_protein_lookup_h.groupby(by=['ligand_atom_type', 'atom_type'])

        self.ligand_atom_types_df = pd.read_csv(atom_type_path / 'ligand_atom_types.csv', sep='\t')
        self.ligand_plane_atoms = self.ligand_atom_types_df[self.ligand_atom_types_df['pi_atom'] == True][
            'ligand_atom_type'].values
        self.total_rf = TotalRf(self.protein, self.protein_atom_types, self.global_ligand_lookup_alpha,
                                self.global_ligand_lookup_h, self.global_protein_lookup_alpha,
                                self.global_protein_lookup_h)

    def return_ligand_h(self, ligand_atom_type, not_plane_atom, plane_atom):
        from ccdc_roche.python import atom_geometries
        if ligand_atom_type in self.ligand_plane_atoms:
            h = atom_geometries.PlaneDistanceCalculator(plane_atom, not_plane_atom).return_plane_distance()
        else:
            h = np.nan

        return h

    def return_protein_h(self, not_plane_atom, plane_atom):
        h = atom_geometries.PlaneDistanceCalculator(plane_atom, not_plane_atom).return_plane_distance()
        return h


def main():
    identifier = '1H1S_001'
    rf_analyzer = RfAnalysis(f'{identifier}.mol2')
    los_pairs_df = rf_analyzer.run_rf_analysis()
    los_pairs_df.to_csv(f'rf_assignments_{identifier}.csv', index=False, sep='\t')


if __name__ == "__main__":
    main()
