#!/usr/bin/env python
#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2019-08-14: created by Andreas Tosstorff, CCDC, Roche
#
'''
This module allows to plug-in RDKit SMARTS matching into the line of sight framework.
'''

########################################################################################################################

import __future__
from rdkit import Chem
import pandas as pd

########################################################################################################################


# def extract_proasis_ligand(rdkit_mol):
#
#     frags = Chem.rdmolops.GetMolFrags(rdkit_mol, asMols=True)
#     for frag in frags:
#         if '_Z' in frag.GetAtomWithIdx(0).GetProp('_TriposAtomName'):
#             frag.SetProp('_Name', rdkit_mol.GetProp('_Name'))
#             return frag


# def assign_index_to_atom_label(rdkit_mol):
#     for index, at in enumerate(rdkit_mol.GetAtoms()):
#         old_label = at.GetProp('_TriposAtomName')
#         new_label = old_label[:2] + str(index)
#         at.SetProp('_TriposAtomName', new_label)
#     return rdkit_mol


class RdkitSubstructure(object):

    def __init__(self, db):
        self.db = db

    def z_filter(self, rdkit_molecule, substructure):
        '''
        Remove all entries, that do not have a match on the central ligand or that have HET group that matches the
        substructure
        :param match: RDKit molecule to be searched
        :param substructure: RDKit molecule substructure
        :return:
        >>> substructure = Chem.MolFromSmarts('[CX4](-F)(-F)(-F)')
        >>> mol = Chem.MolFromMol2File('testdata/1A29_001.mol2')
        >>> searcher = RdkitSubstructure(mol)
        >>> searcher.z_filter(mol, substructure)
        False
        >>> mol = Chem.MolFromMol2File('testdata/1AD8_001.mol2')
        >>> searcher = RdkitSubstructure(mol)
        >>> searcher.z_filter(mol, substructure)
        True
        '''
        match_indices = rdkit_molecule.GetSubstructMatches(substructure)
        match_atoms = [rdkit_molecule.GetAtomWithIdx(match_index[0]) for match_index in match_indices]
        atom_labels = [atom.GetProp('_TriposAtomName') for atom in match_atoms]
        z_label = False
        for atom_label in atom_labels:
            if atom_label.startswith('_U'):
                return False
            if atom_label.startswith('_Z'):
                z_label = True
        return z_label

    def rdkit_substructure_search(self, smarts, return_identifiers=False):
        '''

        :param smarts:
        :param return_identifiers:
        :return:

        >>> import pickle
        >>> from ccdc_roche.python import los_rdkit_utilities
        >>> mol = Chem.MolFromMol2File('testdata/13GS_013.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[c]-[OX1]'
        >>> searcher.rdkit_substructure_search(smarts)
        [['_Z2', '_Z2']]

        One can also use recursive SMARTS:
        >>> mol = Chem.MolFromMol2File('testdata/2WPO_001.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[C&!$([C]~[#7,#8,#9])]-[CX1]'
        >>> searcher.rdkit_substructure_search(smarts)
        [['_Z1', '_Z1'], ['_Z1', '_Z1'], ['_Z1', '_Z1'], ['_Z2', '_Z2'], ['_Z2', '_Z2'], ['_Z2', '_Z2']]

        >>> mol = Chem.MolFromMol2File('testdata/4AGQ_011.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[c,$(C@=[*])]-[OD1]'
        >>> searcher.rdkit_substructure_search(smarts)
        [['_Z1291', '_Z1291']]

        >>> mol = Chem.MolFromMol2File('testdata/1AD8_001.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[C]@-[ND2]-[C]=[O]'
        >>> searcher.rdkit_substructure_search(smarts)
        []

        >>> mol = Chem.MolFromMol2File('testdata/4ELE_001.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[N][CD3](~[N])(~[N])'
        >>> searcher.rdkit_substructure_search(smarts)
        []

        >>> mol = Chem.MolFromMol2File('testdata/1NY2_001.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[N][CD3](~[N])(~[N])'
        >>> searcher.rdkit_substructure_search(smarts)
        [['_ZN33', '_ZC141', '_ZN34', '_ZN35']]

        >>> mol = Chem.MolFromMol2File('testdata/4ELE_001.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[C][CD3](~[N])(~[N])'
        >>> searcher.rdkit_substructure_search(smarts)
        []

        >>> mol = Chem.MolFromMol2File('testdata/1C4U_001.mol2')
        >>> searcher = los_rdkit_utilities.RdkitSubstructure(mol)
        >>> smarts = '[C][CD3](~[N])(~[N])'
        >>> searcher.rdkit_substructure_search(smarts)
        [['_ZC195', '_ZC198', '_ZN48', '_ZN49']]
        '''

        substructure = Chem.MolFromSmarts(smarts)
        if substructure:
            try:
                substructure.UpdatePropertyCache()
            except Chem.rdchem.AtomValenceException:
                pass
        else:
            raise Exception('SMARTS error.')
        Chem.rdmolops.FastFindRings(substructure)  # this is necessary to identify rings
        if isinstance(self.db, Chem.Mol):
            matched_mols = [self.db]
        elif isinstance(self.db, str) and self.db.split('.')[-1] == 'mol2':
            matched_mols = [Chem.MolFromMol2File(self.db)]
        elif isinstance(self.db, Chem.rdSubstructLibrary.SubstructLibrary):
            matched_mol_indices = self.db.GetMatches(substructure, maxResults=10000000)
            matched_mols = [self.db.GetMol(i) for i in matched_mol_indices]

        matches = []
        for mol in matched_mols:
            try:
                mol.UpdatePropertyCache()
                Chem.rdmolops.FastFindRings(mol)  # this is necessary to identify rings
                if self.z_filter(mol, substructure):
                    matches.append(mol)
            except AttributeError:
                continue
        if return_identifiers:
            identifiers_df = pd.DataFrame({'identifier': [(match.GetProp('_Name')) for match in matches],
                                           'rdkit_mol': matches})
            return identifiers_df

        if not return_identifiers:
            match_atoms = []
            for match in matches:
                matches = match.GetSubstructMatches(substructure)
                for match_indices in matches:
                    if '_Z' in match.GetAtomWithIdx(match_indices[0]).GetProp('_TriposAtomName'):
                        match_labels = [match.GetAtomWithIdx(i).GetProp('_TriposAtomName') for i in match_indices]
                        match_atoms.append(match_labels)
            return match_atoms

    def rdkit_get_query_atoms(self, smarts, smarts_index):
        '''

        :param smarts: SMARTS string
        :param smarts_index: index of query atom in SMARTS string
        :return: indices of all atoms that match the atom specified by SMARTS and SMARTS_index, as specified in partial
        charge
        >>> smarts = '[CX4](-F)(-F)(-F)'
        >>> smarts_index = 1
        >>> mol = Chem.MolFromMol2File('testdata/1A29_001.mol2')
        >>> searcher = RdkitSubstructure(mol)
        >>> sorted(searcher.rdkit_get_query_atoms(smarts, smarts_index))
        ['_U154', '_U154', '_U154', '_Z153', '_Z153', '_Z153']
        >>> smarts = '[c][C]'
        >>> smarts_index = 1
        >>> sorted(searcher.rdkit_get_query_atoms(smarts, smarts_index))
        ['_F141', '_F92', '_U154', '_Z153']
        '''
        db = self.db
        if isinstance(db, Chem.Mol):
            db = db
        elif isinstance(db, str) and db.split('.')[-1] == 'mol2':
            db = Chem.MolFromMol2File(db)
        substructure = Chem.MolFromSmarts(smarts)
        try:
            substructure.UpdatePropertyCache()
        except Chem.rdchem.AtomValenceException:
            pass
        Chem.rdmolops.FastFindRings(substructure)  # this is necessary to identify rings
        matching_atom_indices = list(set([match[smarts_index] for match in db.GetSubstructMatches(substructure, uniquify=False) if match[smarts_index]]))
        matching_atom_labels = [db.GetAtomWithIdx(i).GetProp('_TriposAtomName') for i in matching_atom_indices]
        return matching_atom_labels


def main():
    rd_searcher = RdkitSubstructure('testdata/1A29_001.mol2')
    match_atoms = rd_searcher.rdkit_get_query_atoms('[c][C]', 1)


if __name__ == '__main__':
    main()
