
'''
Assign RF values to intramolecular protein contacts.
'''

########################################################################################################################

import __future__
from ccdc_roche.python import los_descriptors, los_utilities, rf_structure_analysis, total_rf_calculator
from ccdc import io, protein, descriptors, molecule
import os
import pandas as pd
import numpy as np
from scipy.stats import gmean

########################################################################################################################


def _intramolecular_protein_contact_df(los_contacts_df, ligand_atom_type, los_atom_ligand_atom_type, contact_angle_atoms, los_atom, central_atom, structure_analyzer,
                identifier, ligand_smiles, los_home):
    ligand_alphas = []
    for contact_angle_atom in contact_angle_atoms:
        ligand_alpha_i = descriptors.GeometricDescriptors.point_angle(contact_angle_atom.coordinates,
                                                                      central_atom.coordinates,
                                                                      los_atom.coordinates)
        ligand_alphas.append(180 - ligand_alpha_i)

    protein_alphas = []
    for neighbour in los_atom.neighbours:
        protein_alpha_i = descriptors.GeometricDescriptors.point_angle(neighbour.coordinates,
                                                                       los_atom.coordinates,
                                                                       central_atom.coordinates)
        protein_alphas.append(180 - protein_alpha_i)

    ligand_h = structure_analyzer.return_ligand_h(ligand_atom_type, los_atom,
                                                  central_atom)
    if hasattr(los_atom, 'los_at') and 'pi' in los_atom.los_at:
        protein_h = structure_analyzer.return_protein_h(central_atom, los_atom)

    else:
        protein_h = np.nan

    vdw_distance = descriptors.MolecularDescriptors().atom_distance(central_atom, los_atom) - central_atom.vdw_radius -\
                   los_atom.vdw_radius

    rf_dictionary_1 = structure_analyzer.total_rf.return_rf(los_atom, ligand_alphas, protein_alphas,
                                                          ligand_atom_type, ligand_h, protein_h)

    rf_dictionary_2 = structure_analyzer.total_rf.return_rf(central_atom, protein_alphas, ligand_alphas,
                                                            los_atom_ligand_atom_type, protein_h, ligand_h)

    rf_ligands = [rf_dictionary_1['rf_ligand'], rf_dictionary_2['rf_ligand']]
    rf_ligands_errors = [rf_dictionary_1['rf_ligand_error'], rf_dictionary_2['rf_ligand_error']]
    rf_total = gmean(rf_ligands)
    rf_total_error = total_rf_calculator.error_propagation(rf_ligands, rf_ligands_errors)

    rf_dictionary = rf_dictionary_1.copy()
    rf_dictionary['rf_protein'] = rf_dictionary_2['rf_ligand']
    rf_dictionary['rf_protein_error'] = rf_dictionary_2['rf_ligand_error']
    rf_dictionary['rf_total'] = rf_total
    rf_dictionary['rf_total_error'] = rf_total_error

    los_pair_dic = {'identifier': identifier,
                    'ligand_atom_index': central_atom.partial_charge,
                    'los_atom_index': los_atom.partial_charge,
                    'ligand_atom_label': central_atom.label,
                    'los_atom_label': los_atom.label,
                    'ligand_atom_residue_label': central_atom.residue_label,
                    'los_atom_residue_label': los_atom.residue_label,
                    'ligand_atom_is_acceptor': central_atom.is_acceptor,
                    'los_atom_is_acceptor': los_atom.is_acceptor,
                    'ligand_atom_is_donor': central_atom.is_donor,
                    'los_atom_is_donor': los_atom.is_donor,
                    'ligand_alphas': ';'.join(map(str, ligand_alphas)),
                    'protein_alphas': ';'.join(map(str, protein_alphas)),
                    'ligand_h': str(ligand_h),
                    'protein_h': str(protein_h),
                    'vdw_distance': vdw_distance,
                    'ligand_smiles': ligand_smiles,
                    'ligand_atom_type': ligand_atom_type,
                    'protein_atom_type': los_atom.los_at}

    los_pair_dic.update(rf_dictionary)
    los_pair_dic['interaction_type'] = rf_dictionary['interaction_type']
    return los_pair_dic


class ProteinDescriptors(object):
    def __init__(self, csd_protein, csd_ligand, rdkit_protein, pdb=False,
                 los_home=''):
        '''
        Class to assign RF values to intramolecular protein atom contacts.
        :param protein_file:
        :param input_type: 'standard', 'Proasis', 'GOLD'
        :param los_home: Path to RF lookup files, atom type definitions, Proasis database
        '''
        self.los_home = los_home
        self.csd_protein = csd_protein
        self.csd_ligand = csd_ligand
        self.rdkit_protein = rdkit_protein
        # if input_type == 'standard' and '.mol2' in input_protein or '.pdb' in input_protein:
        #     if '.pdb' in input_protein:
        #         pdb = True
        #     with io.EntryReader(input_protein) as rdr:
        #         self.csd_protein = protein.Protein.from_entry(rdr[0])
        #         self.csd_protein.remove_hydrogens()
        #         self.csd_protein = los_utilities.assign_index_to_atom_partial_charge(self.csd_protein)
        #         # take largest ligand
        #         ligands = list(self.csd_protein.ligands)
        #         ligands.sort(key=lambda x: len(x.atoms))
        #         self.csd_ligand = ligands[-1]
        #         self.rdkit_protein = Chem.MolFromPDBFile(input_protein)
        #
        # if input_type == 'Proasis':
        #     with io.EntryReader(input_protein) as rdr:
        #         self.csd_protein = protein.Protein.from_entry(rdr[0])
        #         self.csd_protein.remove_hydrogens()
        #         self.csd_protein = los_utilities.assign_index_to_atom_partial_charge(self.csd_protein)
        #         # take central ligand
        #         self.csd_ligand = [c for c in self.csd_protein.components if '_Z' in c.atoms[0].label][0]
        #         self.rdkit_protein = rdkit_protein

        self.csd_ligand.remove_hydrogens()
        self.ligand_atom_types_df = pd.read_csv(os.path.join(los_home, 'ligand_atom_types.csv'), sep='\t')
        self.protein_atom_types_df = pd.read_csv(os.path.join(los_home, 'protein_atom_types.csv'), sep='\t')
        self.protein_atoms = self.csd_protein.atoms
        self.protein_df = los_descriptors.rdkit_match_protein_atoms(self.rdkit_protein, self.protein_atoms,
                                                                    self.protein_atom_types_df, pdb=pdb)
        self.pseudo_ligand_df = los_descriptors.match_ligand_atoms(self.rdkit_protein, self.protein_atoms)

    def return_protein_intramolecular_contact_df(self):
        structure_analyzer = rf_structure_analysis.RfAnalysis(self.csd_protein, los_home=self.los_home)
        los_contacts_df = pd.DataFrame(columns=['ligand_atom_index', 'los_atom_index'])
        ligand_smiles = ''
        ligand_atoms = self.csd_ligand.atoms
        query_atom = ligand_atoms[0]
        query_atom.ligand_index = 0
        # pseudo_ligand_atoms = los_utilities.return_ligand_contacts(self.protein_atoms, [ligand_atoms], query_atom,
        #                                                            self.csd_protein, interaction_cutoff=0.5,
        #                                                            interatomic_cutoff=15)
        pseudo_ligand_atoms = [sorted(list(c.atoms), key=lambda x: x.label) for c in self.csd_protein.contacts(distance_range=(-5.0, 0.5), path_length_range=(-1, 999)) if c.is_in_line_of_sight]
        pseudo_ligand_atoms = [contact_atoms for contact_atoms in pseudo_ligand_atoms if
                               self.csd_protein.shortest_path(self.csd_protein.atom(contact_atoms[0].label), self.csd_protein.atom(contact_atoms[1].label)) == 0
                               or self.csd_protein.shortest_path(self.csd_protein.atom(contact_atoms[0].label), self.csd_protein.atom(contact_atoms[1].label)) >= 4]
        pseudo_ligand_atoms = set([a[0] for a in pseudo_ligand_atoms if a[0].protein_atom_type != 'ligand' and a[1].protein_atom_type != 'ligand'])

        los_contacts_dict_list = []
        for central_atom in pseudo_ligand_atoms:
            if '_Z' in central_atom.label:
                continue
            if '_U' in central_atom.label:
                central_atom.los_at = 'other_ligand'
            if central_atom.protein_atom_type in ['Cofactor', 'Ligand', 'Nucleotide', 'Metal']:
                continue
            if central_atom.partial_charge in self.pseudo_ligand_df['ligand_atom_index'].unique():
                los_contact_atoms = los_utilities.return_los_contacts(central_atom, self.csd_protein,
                                                                      self.protein_atoms,
                                                                      pseudo_ligand_atoms)
                ligand_atom_type = \
                    self.pseudo_ligand_df[self.pseudo_ligand_df['ligand_atom_index'] == central_atom.partial_charge][
                        'ligand_atom_type'].to_numpy()[0]
                contact_angle_atoms = central_atom.neighbours
                for los_atom in los_contact_atoms:
                    if sorted((los_atom.partial_charge, central_atom.partial_charge)) in \
                            [sorted(t) for t in
                             zip(los_contacts_df['los_atom_index'], los_contacts_df['ligand_atom_index'])
                             ]:
                        continue
                    if '_Z' in los_atom.label:
                        continue
                    elif '_U' in los_atom.label:
                        los_atom.los_at = 'other_ligand'
                        continue
                    elif los_atom.protein_atom_type in ['Cofactor', 'Ligand', 'Nucleotide', 'Metal']:
                        continue
                    elif los_atom.atomic_symbol == 'H':
                        continue
                    else:
                        los_atom.los_at = \
                            self.protein_df[self.protein_df['protein_atom_index'] == los_atom.partial_charge][
                                'protein_atom_type'].to_numpy()[0]
                    # los_atom_ligand_atom_type = self.pseudo_ligand_df[
                    #     self.pseudo_ligand_df['ligand_atom_index'] == los_atom.partial_charge][
                    #     'ligand_atom_type'].to_numpy()[0]
                    # los_pair_dict = _intramolecular_protein_contact_df(los_contacts_df, ligand_atom_type,
                    #                                                      los_atom_ligand_atom_type, contact_angle_atoms,
                    #                                                      los_atom, central_atom, structure_analyzer,
                    #                                                      self.csd_protein.identifier, ligand_smiles,
                    #                                                      self.los_home)
                    los_pair_dict = los_descriptors._contact_df(ligand_atom_type, contact_angle_atoms, los_atom,
                                                                central_atom, structure_analyzer,
                                                                self.csd_protein.identifier, ligand_smiles)
                    los_contacts_dict_list.append(los_pair_dict)
        los_contacts_df = pd.DataFrame.from_dict(los_contacts_dict_list)
        los_contacts_df = los_descriptors._return_primary_contacts(los_contacts_df, self.csd_protein)
        los_contacts_df['is_intramolecular'] = 1
        return los_contacts_df


def main():
    protein_describer = ProteinDescriptors('4mk8.pdb')
    df = protein_describer.return_protein_intramolecular_contact_df()


if __name__ == '__main__':
    main()
