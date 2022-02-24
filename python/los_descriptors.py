#!/usr/bin/env python

'''
Generate descriptors for protein-ligand complexes.
'''

########################################################################################################################

import __future__

from ccdc import io, protein, descriptors, docking, molecule
from ccdc_roche.python import los_utilities, rf_structure_analysis
from ccdc_roche import atom_types
from rdkit import Chem
from pathlib import Path
import os
import pandas as pd
import numpy as np
import subprocess as sp
import matplotlib as mpl
import random
import string

mpl.use("agg")


########################################################################################################################

def is_metal(a):
    atomic_num = a.GetAtomicNum()
    if atomic_num <= 4 or 11 <= atomic_num <= 13 or 19 <= atomic_num <= 32 or 37 <= atomic_num <= 52:
        return True
    else:
        return False


def return_assignation_df(rdkit_ligand, ligand_atom_types_df, atoms_df):
    assignation_df = pd.DataFrame(columns=['ligand_atom_index', 'ligand_atom_type'])
    for index, row in ligand_atom_types_df.iterrows():
        smarts = row['RDKit_SMARTS']
        smarts_index = int(row['RDKit_SMARTS_index'])
        substructure = Chem.MolFromSmarts(smarts)
        substructure.UpdatePropertyCache()
        Chem.rdmolops.FastFindRings(substructure)
        matches = rdkit_ligand.GetSubstructMatches(substructure, maxMatches=100000, uniquify=False)
        if len(matches) > 0:
            matches = [match[smarts_index] for match in matches if
                       match[smarts_index] not in assignation_df['ligand_atom_index'].to_list()]
            atom_indices = atoms_df[atoms_df['atom_index'].isin(matches)]['atom_partial_charge'].to_list()
            atom_labels = atoms_df[atoms_df['atom_index'].isin(matches)]['atom_label'].to_list()
            ligand_atom_types = [row['ligand_atom_type'] for _ in atom_labels]
            pi_atoms = [row['pi_atom'] for _ in atom_labels]
            df = pd.DataFrame({'ligand_atom_index': atom_indices, 'ligand_atom_label': atom_labels,
                               'ligand_atom_type': ligand_atom_types, 'pi_atom': pi_atoms})
            assignation_df = assignation_df.append(df, ignore_index=True)
            assignation_df = assignation_df.drop_duplicates(subset='ligand_atom_index')
        if len(assignation_df['ligand_atom_index']) == rdkit_ligand.GetNumAtoms():
            break
    return assignation_df


def match_ligand_atoms(rdkit_ligand, csd_ligand_atoms, freq_threshold=1500):
    '''
    This assumes that the docked ligand is not labeled as a cofactor or nucleotide.
    :return:
    >>> rdr = io.MoleculeReader('testdata/1ny2_001_ligand.sdf')
    >>> csd_ligand = rdr[0]
    >>> rdkit_lig = Chem.MolFromMol2Block(csd_ligand.to_string())
    >>> for a in csd_ligand.atoms:
    ...     a.partial_charge = a.index
    >>> csd_ligand_ats = csd_ligand.atoms
    >>> assignation_df = match_ligand_atoms(rdkit_lig, csd_ligand_ats)
    >>> assignation_df.shape
    (29, 4)
    '''

    atoms_df = pd.DataFrame.from_records(
        [(at.label, cnt, cnt) for cnt, at in enumerate(csd_ligand_atoms)],
        columns=['atom_label', 'atom_partial_charge', 'atom_index'])
    atom_type_path = Path(atom_types.__path__[0])
    ligand_atom_types_df = pd.read_csv(atom_type_path / 'ligand_atom_types.csv', sep='\t')

    frequent_ligand_atom_types_df = ligand_atom_types_df[ligand_atom_types_df['combined_occurrences_filtered'] >= freq_threshold]
    assignation_df = return_assignation_df(rdkit_ligand, frequent_ligand_atom_types_df, atoms_df)

    if len(assignation_df['ligand_atom_index']) != rdkit_ligand.GetNumAtoms():
        infrequent_ligand_atom_types_df = ligand_atom_types_df[
            ligand_atom_types_df['combined_occurrences_filtered'] < freq_threshold]
        complementary_assignation_df = return_assignation_df(rdkit_ligand, infrequent_ligand_atom_types_df, atoms_df)
        complementary_assignation_df = complementary_assignation_df[
            ~complementary_assignation_df.ligand_atom_index.isin(assignation_df['ligand_atom_index'])]
        assignation_df = assignation_df.append(complementary_assignation_df, ignore_index=True)

    return assignation_df


def rdkit_match_protein_atoms(rdkit_protein, csd_protein_atoms, protein_atom_types_df, pdb=False,
                              index_is_in_atom_name=False):

    assignation_df = pd.DataFrame(columns=['protein_atom_index', 'protein_atom_type'])

    atoms_df = pd.DataFrame.from_records(
        [(at.label, int(at.partial_charge), cnt) for cnt, at in enumerate(csd_protein_atoms)],
        columns=['atom_label', 'atom_partial_charge', 'atom_index'])
    rdkit_protein.UpdatePropertyCache()
    Chem.rdmolops.FastFindRings(rdkit_protein)
    contact_dict = {'protein_atom_index': [], 'protein_atom_label': [], 'protein_atom_type': []}
    for index, row in protein_atom_types_df.iterrows():
        smarts = row['RDKit_SMARTS']
        smarts_index = 0
        substructure = Chem.MolFromSmarts(smarts)
        substructure.UpdatePropertyCache()
        Chem.rdmolops.FastFindRings(substructure)
        matches = rdkit_protein.GetSubstructMatches(substructure, maxMatches=100000, uniquify=False)
        if len(matches) > 0:
            protein_atom_type = row['protein_atom_type']
            if pdb:
                if protein_atom_type == 'Water':
                    matches = [match[smarts_index] for match in matches if
                               match[smarts_index] not in assignation_df[
                                   'protein_atom_index'].to_list()]
                else:
                    matches = [match[smarts_index] for match in matches if
                               match[smarts_index] not in assignation_df['protein_atom_index'].to_list() and not rdkit_protein.GetAtomWithIdx(match[smarts_index]).GetPDBResidueInfo().GetIsHeteroAtom()]
            else:
                matches = [match[smarts_index] for match in matches if
                           match[smarts_index] not in assignation_df['protein_atom_index'].to_list() and '_Z' not in rdkit_protein.GetAtomWithIdx(match[smarts_index]).GetProp('_TriposAtomName')]
            if index_is_in_atom_name:
                matches = [rdkit_protein.GetAtomWithIdx(match_index).GetPDBResidueInfo().GetName().strip() for match_index in matches]
                filtered_atoms_df = atoms_df[atoms_df['atom_label'].isin(matches)]
            else:
                filtered_atoms_df = atoms_df[atoms_df['atom_index'].isin(matches)]
            atom_indices = filtered_atoms_df['atom_partial_charge'].to_list()
            atom_labels = filtered_atoms_df['atom_label'].to_list()
            protein_atom_types = [protein_atom_type for _ in atom_indices]
            contact_dict['protein_atom_index'] = contact_dict['protein_atom_index'] + atom_indices
            contact_dict['protein_atom_label'] = contact_dict['protein_atom_label'] + atom_labels
            contact_dict['protein_atom_type'] = contact_dict['protein_atom_type'] + protein_atom_types

    assignation_df = pd.DataFrame(contact_dict)
    assignation_df = assignation_df.drop_duplicates(subset='protein_atom_index')
    # if len(assignation_df['protein_atom_index']) == rdkit_protein.GetNumAtoms():
    #     break

    if len(assignation_df['protein_atom_index']) != rdkit_protein.GetNumAtoms():
        assigned_indices = assignation_df['protein_atom_index'].unique()
        unassigned_indices = [i for i in range(rdkit_protein.GetNumAtoms()) if i not in assigned_indices]
        for i in unassigned_indices:
            if is_metal(rdkit_protein.GetAtomWithIdx(i)):
                df = pd.DataFrame({'protein_atom_index': [i],
                                   'protein_atom_label': atoms_df[atoms_df['atom_index'] == i]['atom_label'].to_list(),
                                   'protein_atom_type': ['metal']})

            else:
                if i in atoms_df['atom_index'].to_list():
                    df = pd.DataFrame(
                        {'protein_atom_index': [i], 'protein_atom_label': atoms_df[atoms_df['atom_index'] == i]['atom_label'].to_list(),
                         'protein_atom_type': ['other_ligand']})
            assignation_df = assignation_df.append(df, ignore_index=True)

    return assignation_df


def match_protein_atoms(ccdc_protein, protein_atom_types_df):
    from ccdc import search

    pat_dict = {'protein_atom_index': [], 'protein_atom_type': []}

    # assignation_df = pd.DataFrame()
    # for protein_atom_type, group_df in protein_atom_types_df.groupby('protein_atom_type'):
    #     substructures = [search.SMARTSSubstructure(pat_smarts) for pat_smarts in group_df['CCDC_SMARTS'].values]
    #     # smarts_index = int(row['CCDC_SMARTS_index'])
    #     # protein_atom_type = row['protein_atom_type']
    #     hits = []
    #     for substructure in substructures:
    #         searcher = search.SubstructureSearch()
    #     # substructure = search.SMARTSSubstructure(smarts)
    #     # for substructure in substructures:
    #         searcher.add_substructure(substructure)
    #         hits = hits + searcher.search(database=ccdc_protein)
    #     hit_atoms = set([hit.match_atoms(indices=True)[0] for hit in hits])
    #     assignation_df = assignation_df.append(
    #         pd.DataFrame(
    #             {'protein_atom_index': hit_atoms, 'protein_atom_type': [protein_atom_type for i in hit_atoms]}
    #         ), ignore_index=True)
    # assignation_df = assignation_df.drop_duplicates(subset='protein_atom_index')
    # return assignation_df

    protein_elements = ['C', 'O', 'N']
    unmatched_atoms = ccdc_protein.atoms
    for atomic_symbol in protein_elements:
        protein_atom_types = protein_atom_types_df[protein_atom_types_df['atomic_symbol'] == atomic_symbol]
        atoms_to_match = [at for at in unmatched_atoms if at.atomic_symbol == atomic_symbol and
                          at.protein_atom_type == 'Amino_acid' or at.protein_atom_type == 'Water' and
                          '_Z' not in at.label]
        substructures = [search.SMARTSSubstructure(pat_smarts) for pat_smarts in protein_atom_types['CCDC_SMARTS'].values]
        pats = protein_atom_types['protein_atom_type'].values
        for i, substructure in enumerate(substructures):
            for at in atoms_to_match[:]:
                if substructure.match_atom(at):
                    pat_dict['protein_atom_index'].append(at.index)
                    pat_dict['protein_atom_type'].append(pats[i])
                    atoms_to_match.remove(at)
                    continue

    for at in [at for at in unmatched_atoms if at.atomic_symbol == 'S' and at.protein_atom_type == 'Amino_acid']:  # this is to match sulfur which could be covalently bound
        protein_atom_type = los_utilities.return_los_pat(at, ccdc_protein, protein_atom_types_df)
        pat_dict['protein_atom_index'].append(at.index)
        pat_dict['protein_atom_type'].append(protein_atom_type)

    # for at in ccdc_protein.atoms:
    #     if at.protein_atom_type == 'Amino_acid':
    #         protein_atom_type = los_utilities.return_los_pat(at, ccdc_protein, protein_atom_types_df)
    #         pat_dict['protein_atom_index'].append(at.index)
    #         pat_dict['protein_atom_type'].append(protein_atom_type)
    #     elif at.protein_atom_type == 'Water':
    #         pat_dict['protein_atom_index'].append(at.index)
    #         pat_dict['protein_atom_type'].append('Water')
    assignation_df = pd.DataFrame(pat_dict)
    return assignation_df


def _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas, protein_alphas,
                      ligand_h, protein_h):
    if ligand_h != ligand_h or ligand_h is None or ligand_atom_type_df[f'{interaction_type}_h_min'].to_numpy()[0] <= ligand_h <= \
            ligand_atom_type_df[f'{interaction_type}_h_max'].to_numpy()[0]:
        if protein_h != protein_h or protein_h is None or protein_atom_type_df[f'{interaction_type}_h_min'].to_numpy()[0] <= protein_h <= \
                protein_atom_type_df[f'{interaction_type}_h_max'].to_numpy()[0]:
            for ligand_alpha in ligand_alphas:
                if ligand_atom_type_df[f'{interaction_type}_alpha_min'].to_numpy()[0] <= ligand_alpha <= \
                        ligand_atom_type_df[f'{interaction_type}_alpha_max'].to_numpy()[0]:
                    for protein_alpha in protein_alphas:
                        if protein_atom_type_df[f'{interaction_type}_alpha_min'].to_numpy()[0] <= protein_alpha \
                                <= protein_atom_type_df[f'{interaction_type}_alpha_max'].to_numpy()[0]:
                            return True
    return False


def return_interaction_type(ligand_atom_type, protein_atom_type, rf_total, rf_total_error, ligand_alphas,
                            protein_alphas, ligand_h,
                            protein_h, los_home=''):
    protein_atom_type_df = pd.read_csv(os.path.join(los_home, 'protein_atom_types.csv'), sep='\t')
    protein_atom_type_df = protein_atom_type_df.loc[protein_atom_type_df['protein_atom_type'] == protein_atom_type]
    ligand_atom_type_df = pd.read_csv(os.path.join(los_home, 'ligand_atom_types.csv'), sep='\t')
    ligand_atom_type_df = ligand_atom_type_df.loc[ligand_atom_type_df['ligand_atom_type'] == ligand_atom_type]

    try:
        if type(ligand_alphas) == str:
            ligand_alphas = [float(ligand_alpha) for ligand_alpha in ligand_alphas.split(';')]
        if type(protein_alphas) == str:
            protein_alphas = [float(protein_alpha) for protein_alpha in protein_alphas.split(';')]
        if ligand_h == 'None' or ligand_h != ligand_h:
            ligand_h = np.nan
        if protein_h == 'None' or protein_h != protein_h:
            protein_h = np.nan
        if type(ligand_h) == str:
            ligand_h = float(ligand_h)
        if type(protein_h) == str:
            protein_h = float(protein_h)

        if protein_atom_type == 'other_ligand':
            return None

        elif rf_total - rf_total_error > 1:
            interaction_type = 'halogen'
            if ligand_atom_type_df['halogen_don'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['halogen_acc'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            if ligand_atom_type_df['halogen_acc'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['halogen_don'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            interaction_type = 'ionic'
            if ligand_atom_type_df['ionic_pos'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['ionic_neg'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            if ligand_atom_type_df['ionic_neg'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['ionic_pos'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            interaction_type = 'hbond'
            if ligand_atom_type_df['hbond_acc'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hbond_don'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    if ligand_atom_type_df['hbond_classic'].to_numpy()[0] == 1 and \
                            protein_atom_type_df['hbond_classic'].to_numpy()[0] == 1:
                        return 'hbond_classic'
                    else:
                        return 'hbond_weak'

            if ligand_atom_type_df['hbond_don'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hbond_acc'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    if ligand_atom_type_df['hbond_classic'].to_numpy()[0] == 1 and \
                            protein_atom_type_df['hbond_classic'].to_numpy()[0] == 1:
                        return 'hbond_classic'
                    else:
                        return 'hbond_weak'

            interaction_type = 'multipolar'
            if ligand_atom_type_df['multipolar_pos'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['multipolar_neg'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            if ligand_atom_type_df['multipolar_neg'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['multipolar_pos'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            interaction_type = 'pi'
            if ligand_atom_type_df['pi'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['pi'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            interaction_type = 'hydrophobic'
            if ligand_atom_type_df['hydrophobic'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hydrophobic'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, interaction_type, ligand_alphas,
                                     protein_alphas, ligand_h,
                                     protein_h):
                    return interaction_type

            return 'uncat_competitive'

        elif rf_total + rf_total_error < 1:
            if ligand_atom_type_df['ionic_pos'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['ionic_pos'].to_numpy()[0] == 1:
                return 'electrostatic_repulsion'

            if ligand_atom_type_df['ionic_neg'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['ionic_neg'].to_numpy()[0] == 1:
                return 'electrostatic_repulsion'

            if ligand_atom_type_df['hydrophobic'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hydrophobic'].to_numpy()[0] == 0:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, 'hydrophobic', ligand_alphas,
                                     protein_alphas, ligand_h, protein_h):
                    return 'desolvation'

            if ligand_atom_type_df['hydrophobic'].to_numpy()[0] == 0 and \
                    protein_atom_type_df['hydrophobic'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, 'hydrophobic', ligand_alphas,
                                     protein_alphas, ligand_h, protein_h):
                    return 'desolvation'

            if ligand_atom_type_df['hydrophobic'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hydrophobic'].to_numpy()[0] == 1:
                if not _geometries_match(ligand_atom_type_df, protein_atom_type_df, 'hydrophobic', ligand_alphas,
                                         protein_alphas, ligand_h, protein_h):
                    return 'desolvation'

            if ligand_atom_type_df['multipolar_pos'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['multipolar_pos'].to_numpy()[0] == 1:
                return 'electrostatic_repulsion'

            if ligand_atom_type_df['multipolar_neg'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['multipolar_neg'].to_numpy()[0] == 1:
                return 'electrostatic_repulsion'

            if ligand_atom_type_df['hbond_acc'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hbond_acc'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, 'hbond', ligand_alphas,
                                     protein_alphas, ligand_h, protein_h):
                    return 'electrostatic_repulsion'
            if ligand_atom_type_df['hbond_don'].to_numpy()[0] == 1 and \
                    protein_atom_type_df['hbond_don'].to_numpy()[0] == 1:
                if _geometries_match(ligand_atom_type_df, protein_atom_type_df, 'hbond', ligand_alphas,
                                     protein_alphas, ligand_h, protein_h):
                    return 'electrostatic_repulsion'

            return 'uncat_non_competitive'

        else:
            return 'non_sig.'

    except:
        print(ligand_atom_type, protein_atom_type)
        return 'error'


def _is_primary(index, _protein, contact_atom1, central_atom, df, mode='ligand'):
    for index2, row2 in df.iterrows():
        if index2 == index:
            continue
        if mode == 'ligand':
            contact_atom2 = _protein.atom(row2['los_atom_label'])
        else:
            contact_atom2 = _protein.atom(row2['ligand_atom_label'])
        angle = descriptors.MolecularDescriptors.atom_angle(contact_atom1, central_atom, contact_atom2)
        if angle < 25:
            return False
    return True


def _return_primary_contacts(entry_contact_df, _protein):
    # find ligand sided primary contacts
    entry_contact_df = entry_contact_df.sort_values(by='vdw_distance', ascending=False)
    _protein_atoms = _protein.atoms
    ligand_atom_labels = entry_contact_df['ligand_atom_label']
    for ligand_atom_label in ligand_atom_labels.unique():

        central_atom = _protein.atom(ligand_atom_label)
        df = entry_contact_df.loc[ligand_atom_labels == ligand_atom_label].reset_index(drop=False)
        for index, row in df.iterrows():
            contact_atom1 = _protein.atom(row['los_atom_label'])
            entry_contact_df.loc[row['index'], 'is_primary'] = _is_primary(index, _protein, contact_atom1,
                                                                           central_atom, df[index + 1:])
    # find protein sided primary contacts
    filtered_df = entry_contact_df[entry_contact_df['is_primary'] == True]
    los_atom_labels = filtered_df['los_atom_label']
    for los_atom_label in los_atom_labels.unique():
        central_atom = _protein.atom(los_atom_label)
        df = filtered_df.loc[los_atom_labels == los_atom_label].reset_index(drop=False)
        for index, row in df.iterrows():
            contact_atom1 = _protein.atom(row['ligand_atom_label'])
            entry_contact_df.loc[row['index'], 'is_primary'] = _is_primary(index, _protein, contact_atom1,
                                                                           central_atom, df[index + 1:], mode='protein')
    return entry_contact_df


def _contact_df(ligand_atom_type, contact_angle_atoms, los_atom, central_atom, structure_analyzer,
                identifier, ligand_smiles):
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

    ligand_h = structure_analyzer.return_ligand_h(ligand_atom_type, los_atom, central_atom)
    if hasattr(los_atom, 'los_at') and 'pi' in los_atom.los_at:
        protein_h = structure_analyzer.return_protein_h(central_atom, los_atom)
    else:
        protein_h = np.nan

    distance = descriptors.MolecularDescriptors().atom_distance(central_atom, los_atom)
    vdw_distance = distance - central_atom.vdw_radius -\
                   los_atom.vdw_radius

    rf_dictionary = structure_analyzer.total_rf.return_rf(los_atom, ligand_alphas, protein_alphas,
                                                          ligand_atom_type, ligand_h, protein_h)

    los_pair_dic = {'identifier': identifier,
                    'ligand_atom_index': central_atom.partial_charge,
                    'los_atom_index': los_atom.partial_charge,
                    'ligand_atom_label': central_atom.label,
                    'ligand_atom_symbol': central_atom.atomic_symbol,
                    'los_atom_label': los_atom.label,
                    'los_atom_symbol': los_atom.atomic_symbol,
                    'los_atom_buriedness': los_atom.buriedness,
                    'ligand_atom_residue_label': central_atom.residue_label,
                    'los_atom_residue_label': los_atom.residue_label,
                    'ligand_atom_is_acceptor': central_atom.is_acceptor,
                    'los_atom_is_acceptor': los_atom.is_acceptor,
                    'ligand_atom_is_donor': central_atom.is_donor,
                    'los_atom_is_donor': los_atom.is_donor,
                    'ligand_alphas': ';'.join(map(str, ligand_alphas)),
                    'protein_alphas': ';'.join(map(str, protein_alphas)),
                    'ligand_h': ligand_h,
                    'protein_h': protein_h,
                    'vdw_distance': vdw_distance,
                    'distance': distance,
                    'ligand_smiles': ligand_smiles,
                    'ligand_atom_type': ligand_atom_type,
                    'protein_atom_type': los_atom.los_at}

    los_pair_dic.update(rf_dictionary)

    # interaction_type = return_interaction_type(ligand_atom_type, los_atom.los_at, los_pair_dic['rf_total'],
    #                                            los_pair_dic['rf_total_error'], ligand_alphas, protein_alphas, ligand_h,
    #                                            protein_h, los_home)
    los_pair_dic['interaction_type'] = rf_dictionary['interaction_type']
    # los_pair = pd.DataFrame(los_pair_dic)

    return los_pair_dic


def rf_count_df(contact_df, csd_ligand):
    '''
    Calculate metrics related to RF values.
    :param contact_df: DataFrame that contains RF descriptors for each contact in a protein-ligand-complex
    :param csd_ligand: Ligand as CSD molecule
    :return: DataFrame with RF statistics for a protein-ligand-complex
    '''
    rf_total = contact_df['rf_total']
    rf_total_error = contact_df['rf_total_error']
    non_competitive_contact_count = rf_total + rf_total_error
    non_competitive_contact_count = non_competitive_contact_count[non_competitive_contact_count < 1.0].count()
    competitive_contact_count = rf_total - rf_total_error
    competitive_contact_count = competitive_contact_count[competitive_contact_count > 1.0].count()

    non_competitive_contact_count_d01 = rf_total + rf_total_error
    non_competitive_contact_count_d01 = non_competitive_contact_count_d01[
        non_competitive_contact_count_d01 < 0.9].count()
    competitive_contact_count_d01 = rf_total - rf_total_error
    competitive_contact_count_d01 = competitive_contact_count_d01[competitive_contact_count_d01 > 1.1].count()

    non_competitive_contact_count_d03 = rf_total + rf_total_error
    non_competitive_contact_count_d03 = non_competitive_contact_count_d03[
        non_competitive_contact_count_d03 < 0.7].count()
    competitive_contact_count_d03 = rf_total - rf_total_error
    competitive_contact_count_d03 = competitive_contact_count_d03[competitive_contact_count_d03 > 1.3].count()

    non_competitive_contact_count_d05 = rf_total + rf_total_error
    non_competitive_contact_count_d05 = non_competitive_contact_count_d05[
        non_competitive_contact_count_d05 < 0.5].count()
    competitive_contact_count_d05 = rf_total - rf_total_error
    competitive_contact_count_d05 = competitive_contact_count_d05[competitive_contact_count_d05 > 1.5].count()

    heavy_atom_count = len(csd_ligand.atoms)
    rf_max = rf_total.max()
    rf_min = rf_total.min()
    clash_count_non_hbond = contact_df[(contact_df['is_intramolecular']==False) & (contact_df['interaction_type'] != 'hbond_classic')]['vdw_distance'][contact_df['vdw_distance'] < -0.5].count()
    clash_count_hbond = contact_df[(contact_df['is_intramolecular']==False) & (contact_df['interaction_type'] == 'hbond_classic')]['vdw_distance'][contact_df['vdw_distance'] < -0.7].count()
    clash_count = clash_count_hbond + clash_count_non_hbond

    if 'ligand_file' in contact_df.columns:
        ligand_file = contact_df.loc[0, 'ligand_file']
    else:
        ligand_file = np.nan
    if 'identifier' in contact_df.columns:
        identifier = contact_df.loc[0, 'identifier']
    else:
        identifier = np.nan

    rf_count_df = pd.DataFrame({'non_competitive_contact_count': [non_competitive_contact_count],
                                'competitive_contact_count': [competitive_contact_count],
                                'non_competitive_contacts_per_atom': [non_competitive_contact_count / heavy_atom_count],
                                'competitive_contacts_per_atom': [competitive_contact_count / heavy_atom_count],

                                'non_competitive_contact_count_d01': [non_competitive_contact_count_d01],
                                'competitive_contact_count_d01': [competitive_contact_count_d01],
                                'non_competitive_contacts_per_atom_d01': [
                                    non_competitive_contact_count_d01 / heavy_atom_count],
                                'competitive_contacts_per_atom_d01': [competitive_contact_count_d01 / heavy_atom_count],

                                'non_competitive_contact_count_d03': [non_competitive_contact_count_d03],
                                'competitive_contact_count_d03': [competitive_contact_count_d03],
                                'non_competitive_contacts_per_atom_d03': [
                                    non_competitive_contact_count_d03 / heavy_atom_count],
                                'competitive_contacts_per_atom_d03': [competitive_contact_count_d03 / heavy_atom_count],

                                'non_competitive_contact_count_d05': [non_competitive_contact_count_d05],
                                'competitive_contact_count_d05': [competitive_contact_count_d05],
                                'non_competitive_contacts_per_atom_d05': [
                                    non_competitive_contact_count_d05 / heavy_atom_count],
                                'competitive_contacts_per_atom_d05': [competitive_contact_count_d05 / heavy_atom_count],
                                'identifier': [identifier], 'ligand_file': [ligand_file],
                                'rf_max': [rf_max], 'rf_min': [rf_min], 'clash_count': [clash_count], 'smiles': [csd_ligand.smiles]
                                })

    def binned_rf_contacts(rf_total):
        bins = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
        binned_contacts = pd.cut(rf_total, bins).value_counts().to_frame().sort_index().transpose().reset_index(drop=True)
        return binned_contacts

    binned_contacts = binned_rf_contacts(rf_total)
    rf_count_df = rf_count_df.join(binned_contacts)
    return rf_count_df


def calculate_delta_apolar_sasa(bound_ligand_atoms, unbound_ligand_atoms):
    '''
    Calculate difference in apolar surface area of unbound and bound ligand.
    :param bound_ligand_atoms:
    :param unbound_ligand_atoms:
    :return:
    '''
    bound_apolar_sasas = []
    unbound_apolar_sasas = []

    for a in bound_ligand_atoms:
        if not a.is_acceptor or not a.is_donor:
            bound_apolar_sasas.append(a.solvent_accessible_surface())
    for a in unbound_ligand_atoms:
        if not a.is_acceptor or not a.is_donor:
            unbound_apolar_sasas.append(a.solvent_accessible_surface())

    bound_apolar_sasa = np.sum(bound_apolar_sasas)
    unbound_apolar_sasa = np.sum(unbound_apolar_sasas)

    delta_apolar_sasa = unbound_apolar_sasa - bound_apolar_sasa

    return delta_apolar_sasa


def get_hbond_df(mol, intermolecular_only: bool = True) -> pd.DataFrame:
    hbond_criterion = molecule.Molecule.HBondCriterion()
    hbond_criterion.require_hydrogens = True
    hbond_criterion.angle_tolerance = 110
    hbond_criterion.donor_types['-sulphur'] = False
    hbond_criterion.donor_types['+sulphur'] = False
    hbond_criterion.donor_types['aromatic C'] = True
    hbond_criterion.donor_types['sp3 C'] = True
    hbond_criterion.donor_types['sp2 C'] = True
    hbond_criterion.acceptor_types['+sulphur'] = False
    hbond_criterion.acceptor_types['metal bound S'] = False
    hbond_criterion.acceptor_types['terminal S'] = False
    hbond_criterion.acceptor_types['-sulphur'] = False
    hbond_criterion.acceptor_types['+fluorine'] = False
    hbond_criterion.acceptor_types['metal bound F'] = False
    hbond_criterion.acceptor_types['fluoride ion (F-)'] = False
    hbond_criterion.acceptor_types['-fluorine'] = False
    hbond_criterion.acceptor_types['metal bound Cl'] = False
    hbond_criterion.acceptor_types['chloride ion (Cl-)'] = False
    hbond_criterion.acceptor_types['unclassified Cl'] = False
    hbond_criterion.acceptor_types['+bromine'] = False
    hbond_criterion.acceptor_types['metal bound Br'] = False
    hbond_criterion.acceptor_types['unclassified Br'] = False
    hbond_criterion.acceptor_types['-bromine'] = False
    hbond_criterion.acceptor_types['+iodine'] = False
    hbond_criterion.acceptor_types['metal bound I'] = False
    hbond_criterion.acceptor_types['iodide ion (I-)'] = False
    hbond_criterion.acceptor_types['unclassified I'] = False
    hbond_criterion.acceptor_types['-iodine'] = False

    if intermolecular_only:
        hbond_criterion.path_length_range = (-1, 999)
        hbond_criterion.intermolecular = 'intermolecular'
        hbonds = mol.hbonds(path_length_range=(-1, 999), distance_range=(-5, 0.2), hbond_criterion=hbond_criterion)
    else:
        hbonds = mol.hbonds(path_length_range=(4, 999), distance_range=(-5, 0.2), hbond_criterion=hbond_criterion)
    atom1_labels = []
    atom2_labels = []
    atom1_res_labels = []
    atom2_res_labels = []
    h_atom_labels = []
    distances = []
    for hbond in hbonds:
        if hbond.is_in_line_of_sight:
            hbond_atoms = hbond.atoms
            atom1_labels.append(hbond_atoms[0].label)
            atom2_labels.append(hbond_atoms[2].label)
            atom1_res_labels.append(hbond_atoms[0].residue_label)
            atom2_res_labels.append(hbond_atoms[2].residue_label)
            distances.append(hbond.da_distance)
            h_atom_labels.append(hbond_atoms[1].label)
    hbond_df = pd.DataFrame({'atom1_label': atom1_labels, 'atom2_label': atom2_labels,
                             'atom1_residue_label': atom1_res_labels, 'atom2_residue_label': atom2_res_labels,
                             'h_atom_label': h_atom_labels, 'distance': distances})
    hbond_df = hbond_df.sort_values('distance', ascending=True).drop_duplicates('h_atom_label')
    return hbond_df


def get_hbond_mismatch_df(csd_protein, interaction_cutoff):
    csd_ligand = csd_protein.ligands[-1]
    csd_ligand.remove_unknown_atoms()
    central_ligand_atoms = csd_ligand.atoms
    for at in central_ligand_atoms:
        at.ligand_index = -1
    close_contact_atoms = los_utilities.return_ligand_contacts([central_ligand_atoms], central_ligand_atoms[0],
                                                               csd_protein,
                                                               interaction_cutoff=interaction_cutoff)
    hbond_mismatches = {'atom1_label': [], 'atom2_label': [], 'atom1_residue_label': [], 'atom2_residue_label': []}
    for central_atom in central_ligand_atoms:
        los_contact_atoms = los_utilities.return_los_contacts(central_atom, csd_protein, list(close_contact_atoms),
                                                              central_ligand_atoms,
                                                              interaction_cutoff=interaction_cutoff)
        if central_atom.is_acceptor:
            acceptor_los_atoms = [a for a in los_contact_atoms if a.is_acceptor]
            for acceptor_los_atom in acceptor_los_atoms:
                hbond_mismatches['atom1_label'].append(central_atom.label)
                hbond_mismatches['atom1_residue_label'].append(central_atom.residue_label)
                hbond_mismatches['atom2_label'].append(acceptor_los_atom.label)
                hbond_mismatches['atom2_residue_label'].append(acceptor_los_atom.residue_label)
        if central_atom.atomic_symbol == 'H':
            central_heavy_atom = central_atom.neighbours[0]
            if central_heavy_atom.is_donor:
                h_los_atoms = [a for a in los_contact_atoms if a.atomic_symbol == 'H']
                for h_los_atom in h_los_atoms:
                    if not h_los_atom.is_in_line_of_sight(central_heavy_atom):
                        donor_heavy_atoms = [a.neighbours[0] for a in h_los_atoms if a.is_donor]
                        for donor_heavy_atom in donor_heavy_atoms:
                            hbond_mismatches['atom1_label'].append(central_heavy_atom.label)
                            hbond_mismatches['atom1_residue_label'].append(central_heavy_atom.residue_label)
                            hbond_mismatches['atom2_label'].append(donor_heavy_atom.label)
                            hbond_mismatches['atom2_residue_label'].append(donor_heavy_atom.residue_label)
    hbond_mismatch_df = pd.DataFrame(hbond_mismatches)
    return hbond_mismatch_df


def _is_clash(atom1, atom2):
    if descriptors.MolecularDescriptors.atom_distance(atom1, atom2) - atom1.vdw_radius - atom2.vdw_radius < -0.5:
        return True
    else:
        False


def _contact_df_loop(central_ligand_atoms, ligand_df, close_contact_atoms, protein_df, structure_analyzer,
                     ligand_smiles, protein, identifier, csd_ligand_entry=None, ligand_file=None,
                     interaction_cutoff=0.5):

    los_contacts_dict_list = []

    for central_atom in central_ligand_atoms:
        if central_atom.label in ligand_df['ligand_atom_label'].to_list():
            los_contact_atoms = los_utilities.return_los_contacts(central_atom, protein, list(close_contact_atoms),
                                                                  central_ligand_atoms,
                                                                  interaction_cutoff=interaction_cutoff)
            los_contact_atoms.extend([a for a in list(close_contact_atoms) if _is_clash(central_atom, a)])
            los_contact_atoms = list(set(los_contact_atoms))
            ligand_atom_type = \
                ligand_df[ligand_df['ligand_atom_label'] == central_atom.label][
                    'ligand_atom_type'].to_numpy()[0]
            contact_angle_atoms = central_atom.neighbours
            for los_atom in los_contact_atoms:
                if los_atom in central_ligand_atoms or los_atom.atomic_symbol == 'H':
                    continue
                los_atom.los_at = protein_df[protein_df['protein_atom_label'] == los_atom.label][
                    'protein_atom_type'].to_numpy()[0]
                los_atom.buriedness = los_utilities.return_protein_buriedness(central_ligand_atoms, los_atom, protein)
                los_pair_dict = _contact_df(ligand_atom_type, contact_angle_atoms, los_atom, central_atom,
                                            structure_analyzer, identifier, ligand_smiles)
                los_contacts_dict_list.append(los_pair_dict)

    los_contacts_df = pd.DataFrame.from_dict(los_contacts_dict_list)
    if csd_ligand_entry is not None:
        for attribute in csd_ligand_entry.attributes:
            los_contacts_df.loc[:, attribute] = csd_ligand_entry.attributes[attribute]
    if ligand_file is not None:
        los_contacts_df.loc[:, 'ligand_file'] = str(Path(ligand_file).absolute())
    # los_contacts_df = _return_primary_contacts(los_contacts_df, protein)
    los_contacts_df['is_intramolecular'] = 0
    los_contacts_df['is_clash'] = ((los_contacts_df['interaction_type'] != 'hbond_classic') &
                                   (los_contacts_df['vdw_distance'] < -0.5)) |\
                                  ((los_contacts_df['interaction_type'] == 'hbond_classic') &
                                   (los_contacts_df['vdw_distance'] < -0.7))
    return los_contacts_df


def _cut_out_binding_site_by_distance(prot, ligand):
    bs = prot.BindingSiteFromMolecule(prot, ligand, 4.0, whole_residues=True)
    bs_residues = bs.residues
    for r in bs.protein.residues:
        if r not in bs_residues:
            prot.remove_residue(r.identifier)
    return prot


def _cut_out_binding_site_by_residue_df(prot, residues='binding_site_residues.csv'):
    if Path(residues).is_dir():
        residues = Path(residues) / Path('binding_site_residues.csv')
    binding_site_residues = pd.read_csv(residues)['residue_identifier'].values

    met713 = False
    met703 = False

    for r in prot.residues:
        residue_identifier = r.identifier
        if 'MET703' in residue_identifier:
            met703 = residue_identifier
        if 'MET713' in residue_identifier:
            met713 = True
        if residue_identifier.split(':')[1] not in binding_site_residues:
            prot.remove_residue(residue_identifier)

    return prot


def get_b_factors(pdb_protein_file):
    pdb_protein = protein.Protein.from_file(str(pdb_protein_file))
    bfactors = []
    labels = []
    chain_labels = []
    residue_labels = []
    for atom in pdb_protein.heavy_atoms:
        bfactors.append(atom.displacement_parameters.temperature_factor)
        labels.append(atom.label)
        chain_labels.append(atom.chain_label)
        residue_labels.append(atom.residue_label)
    df = pd.DataFrame({'label': labels, 'residue_label': residue_labels, 'chain_label': chain_labels, 'bfactor': bfactors})
    df['relative_bfactor'] = np.log(df['bfactor'] / df['bfactor'].mean())
    return df


class RfDescriptors(object):
    def __init__(self, csd_protein, csd_ligand):
        self.csd_protein = csd_protein
        self.csd_ligand = csd_ligand
        atom_type_path = Path(atom_types.__path__[0])
        self.protein_atom_types_df = pd.read_csv(atom_type_path / 'protein_atom_types.csv', sep='\t')
        self.ligand_atom_types_df = pd.read_csv(atom_type_path / 'ligand_atom_types.csv', sep='\t')

        self.csd_protein.remove_hydrogens()
        self.rdkit_ligand = Chem.MolFromMol2Block(self.csd_ligand.to_string(), removeHs=False)
        self.rdkit_protein = Chem.MolFromMol2Block(self.csd_protein.to_string())
        self.rdkit_ligand.UpdatePropertyCache()
        Chem.rdmolops.FastFindRings(self.rdkit_ligand)

        self.csd_protein = los_utilities.assign_index_to_atom_partial_charge(self.csd_protein)
        protein_df = rdkit_match_protein_atoms(self.rdkit_protein, self.csd_protein.atoms, self.protein_atom_types_df)
        self.ligand_df = match_ligand_atoms(self.rdkit_ligand, self.csd_ligand.atoms)
        self.protein_df = protein_df[
            protein_df['protein_atom_label'].isin(self.ligand_df['ligand_atom_label']) == False]

        #assign RF_total
        structure_analyzer = rf_structure_analysis.RfAnalysis(self.csd_protein)
        central_ligand_atoms = [self.csd_protein.atom(at.label) for at in self.csd_ligand.heavy_atoms]
        ligand_smiles = self.csd_ligand.smiles
        for at in central_ligand_atoms:
            at.ligand_index = los_utilities.ligand_index([central_ligand_atoms], at)
        close_contact_atoms = los_utilities.return_ligand_contacts([central_ligand_atoms], central_ligand_atoms[0],
                                                                   self.csd_protein)
        los_contacts_df = _contact_df_loop(central_ligand_atoms, self.ligand_df, close_contact_atoms, self.protein_df,
                                           structure_analyzer, ligand_smiles, self.csd_protein,
                                           self.csd_protein.identifier)
        self.los_contacts_df = los_contacts_df

        return


class CsdDescriptorsFromMol2(object):

    def __init__(self, input_protein):

        if type(input_protein) == str:
            self.protein = protein.Protein.from_file(input_protein)
        else:
            self.protein = input_protein
        self.protein.assign_bond_types()
        self.protein.remove_hydrogens()
        self.protein.normalise_labels()
        self.protein = los_utilities.assign_index_to_atom_partial_charge(self.protein)
        self.csd_ligand = self.protein.ligands[0]

        if type(input_protein) == str:
            if '.mol2' in input_protein:
                self.csd_ligand.remove_hydrogens()
                self.rdkit_ligand = Chem.MolFromMol2Block(self.csd_ligand.to_string())
                self.rdkit_protein = Chem.MolFromMol2Block(self.protein.to_string())
                self.rdkit_ligand.UpdatePropertyCache()
                Chem.rdmolops.FastFindRings(self.rdkit_ligand)
        describer = RfDescriptors(self.protein, self.csd_ligand)
        self.los_contacts_df = describer.los_contacts_df


class CsdDescriptorsFromProasis(object):
    '''
    Creates an object with RF descriptors from a mol2 file in Proasis format. Central ligand atoms start with _Z, other
    HET atoms with _U and amino acids with _[one letter code].
    '''

    def __init__(self, input_protein):

        if type(input_protein) == str:
            self.protein = protein.Protein.from_file(input_protein)
        else:
            self.protein = input_protein

        for c in self.protein.components:
            if c.heavy_atoms[0].label.startswith('_Z'):
                self.csd_ligand = c.copy()
                break

        describer = RfDescriptors(self.protein, self.csd_ligand)
        self.los_contacts_df = describer.los_contacts_df
        self.ligand_df = describer.ligand_df
        self.protein_df = describer.protein_df


class CsdDescriptorsFromGold(object):

    def __init__(self, input_ligand, gold_conf='gold.conf', docking_settings=None,
                 only_binding_site=False, bfactor=False, strucid=None, interaction_cutoff=0.5, pdb_file=None):
        '''

        :param input_ligand:
        :param gold_conf:
        :param docking_settings:
        :param waters:
        :param only_binding_site:
        :param bfactor If True, a dataframe containing B-factors will be retrieved:
        :param strucid:
        :param interaction_cutoff:
        '''
        atom_type_path = Path(atom_types.__path__[0])
        self.protein_atom_types_df = pd.read_csv(atom_type_path / 'protein_atom_types.csv', sep='\t')
        self.ligand_atom_types_df = pd.read_csv(atom_type_path / 'ligand_atom_types.csv', sep='\t')

        self.bfactor = bfactor
        self.interaction_cutoff = interaction_cutoff

        # deal with possible bug in GOLD that uses
        if Path('gold_protein.mol2').is_file():
            Path('gold_protein.mol2').unlink()

        if docking_settings is None:
            docking_settings = docking.Docker.Settings().from_file(gold_conf)

        docking_results = docking.Docker.Results(docking_settings)
        docked_ligand_reader = docking_results.DockedLigandReader(input_ligand, docking_settings)
        self.csd_ligand_entry = docked_ligand_reader[0]
        self.input_ligand = input_ligand

        # setup protein-ligand-complex
        self.protein = docking_results.make_complex(self.csd_ligand_entry)
        self.protein.remove_unknown_atoms()
        self.protein.kekulize()
        if strucid:
            self.protein.identifier = strucid

        if 'template_strucid' in self.csd_ligand_entry.attributes:
            self.protein.identifier = self.csd_ligand_entry.attributes['template_strucid']
        else:
            self.protein.identifier = str(Path(self.input_ligand).absolute().parent).split('_')[-2]

        if only_binding_site:
            self.protein = _cut_out_binding_site_by_residue_df(self.protein, str(Path(self.bfactor) / Path('binding_site_residues.csv')))

        if self.bfactor and pdb_file:
            self.bfactor_df = get_b_factors(pdb_file)

            # Getting Water B-factors, accounting for the fact that waters can be turned off.
            docked_water_residue_labels = [i.heavy_atoms[0].residue_label for i in
                                           docking_results.ligands[0].docked_waters]
            docked_water_indices = [int(i.split('HOH_AM')[1])-1 for i in docked_water_residue_labels]
            water_cnt = 0
            for atom in self.protein.atoms:
                if atom.protein_atom_type == 'Water' and atom.atomic_symbol != 'H' and docked_water_residue_labels:
                    docked_water_index = docked_water_indices[water_cnt]
                    atom_label = docking_settings.waters[docked_water_index].heavy_atoms[0].label
                    atom_residue_label = docking_settings.waters[docked_water_index].heavy_atoms[0].residue_label
                    atom_chain_label = docking_settings.waters[docked_water_index].heavy_atoms[0].chain_label
                    water_cnt += 1
                else:
                    atom_label = atom.label
                    atom_residue_label = atom.residue_label
                    atom_chain_label = atom.chain_label
                self.bfactor_df.loc[(self.bfactor_df['label'] == atom_label) &
                                    (self.bfactor_df['residue_label'] == atom_residue_label) &
                                    (self.bfactor_df['chain_label'] == atom_chain_label), 'atom_index'] = atom.index

        self.protein.normalise_labels()

        if self.bfactor:
            for atom in self.protein.atoms:
                self.bfactor_df.loc[self.bfactor_df['atom_index'] == atom.index, 'normalised_atom_label'] = atom.label
        self.hbond_df = get_hbond_df(self.protein)
        self.hbond_mismatch_df = get_hbond_mismatch_df(self.protein, self.interaction_cutoff)

        temp_protein = self.protein.copy()
        self.protein.remove_hydrogens()

        self.protein = los_utilities.assign_index_to_atom_partial_charge(self.protein)
        self.protein_atoms = self.protein.atoms

        # setup ligand
        self.csd_ligand = self.protein.ligands[-1]
        self.csd_ligand.remove_unknown_atoms()
        self.rdkit_ligand = Chem.MolFromMol2Block(self.csd_ligand.to_string('mol2'), removeHs=True)
        self.csd_ligand_atoms = self.csd_ligand.atoms
        self.rdkit_protein = Chem.MolFromMol2Block(temp_protein.to_string('mol2'))

        self.ligand_df = match_ligand_atoms(self.rdkit_ligand, self.csd_ligand_atoms)
        self.protein_df = rdkit_match_protein_atoms(self.rdkit_protein, self.protein_atoms, self.protein_atom_types_df,
                                                    pdb=False)
        self.protein_df = self.protein_df[self.protein_df['protein_atom_label'].isin(self.ligand_df['ligand_atom_label']) == False]
        self.csd_ligand.remove_hydrogens()

    def contact_df(self):
        structure_analyzer = rf_structure_analysis.RfAnalysis(self.protein)
        protein_atoms = self.protein_atoms
        central_ligand_atoms = [protein_atoms[int(at.partial_charge)] for at in self.csd_ligand_atoms if
                                at.atomic_symbol != 'H']
        ligand_smiles = self.csd_ligand.smiles
        for at in central_ligand_atoms:
            at.ligand_index = -1
        close_contact_atoms = los_utilities.return_ligand_contacts([central_ligand_atoms], central_ligand_atoms[0],
                                                                   self.protein,
                                                                   interaction_cutoff=self.interaction_cutoff)

        los_contacts_df = _contact_df_loop(central_ligand_atoms, self.ligand_df, close_contact_atoms, self.protein_df,
                                           structure_analyzer, ligand_smiles, self.protein,
                                           identifier=self.protein.identifier,
                                           csd_ligand_entry=self.csd_ligand_entry, ligand_file=self.input_ligand,
                                           interaction_cutoff=self.interaction_cutoff)

        # Assign hbonds with explicit hydrogen
        hbond_list_1 = self.hbond_df[['atom1_label', 'atom1_residue_label',  'atom2_label', 'atom2_residue_label']].agg(','.join, axis=1).to_list()
        hbond_list_2 = self.hbond_df[['atom2_label', 'atom2_residue_label',  'atom1_label', 'atom1_residue_label']].agg(','.join, axis=1).to_list()
        los_contacts_df['is_hbond'] = \
            (los_contacts_df[['los_atom_label', 'los_atom_residue_label', 'ligand_atom_label', 'ligand_atom_residue_label']].agg(','.join, axis=1).isin(hbond_list_1)) | \
            (los_contacts_df[['los_atom_label', 'los_atom_residue_label', 'ligand_atom_label', 'ligand_atom_residue_label']].agg(','.join, axis=1).isin(hbond_list_2))
        # hbond mismatch: donor...donor or acceptor...acceptor contact
        # los_contacts_df['is_hbond_mismatch'] = (los_contacts_df['ligand_atom_is_acceptor']) & \
        #                                        (los_contacts_df['los_atom_is_acceptor']) & \
        #                                        (los_contacts_df['is_hbond'] == False) | \
        #                                        (los_contacts_df['ligand_atom_is_donor']) & \
        #                                        (los_contacts_df['los_atom_is_donor']) & \
        #                                        (los_contacts_df['is_hbond'] == False)
        hbond_mismatch_list_1 = self.hbond_mismatch_df[['atom1_label', 'atom1_residue_label', 'atom2_label', 'atom2_residue_label']].agg(
            ','.join, axis=1).to_list()
        hbond_mismatch_list_2 = self.hbond_mismatch_df[['atom2_label', 'atom2_residue_label', 'atom1_label', 'atom1_residue_label']].agg(
            ','.join, axis=1).to_list()
        los_contacts_df['is_hbond_mismatch'] = \
            (los_contacts_df[
                 ['los_atom_label', 'los_atom_residue_label', 'ligand_atom_label', 'ligand_atom_residue_label']].agg(
                ','.join, axis=1).isin(hbond_mismatch_list_1)) | \
            (los_contacts_df[
                 ['los_atom_label', 'los_atom_residue_label', 'ligand_atom_label', 'ligand_atom_residue_label']].agg(
                ','.join, axis=1).isin(hbond_mismatch_list_2))
        return los_contacts_df

    def delta_apolar_sasa(self):
        bound_ligand_atoms = [self.protein.atom(l) for l in [a.label for a in self.csd_ligand_atoms]]
        delta_apolar_sasa = calculate_delta_apolar_sasa(bound_ligand_atoms, self.csd_ligand_atoms)
        return delta_apolar_sasa


class CsdDescriptorsFromPDB(object):

    def __init__(self, target_file, ligand=None, only_binding_site=False, bfactor=False, strucid=None, project_home=''):
        '''
        :param target_file:
        :param ligand:
        :param only_binding_site:
        :param bfactor: If True, a dataframe containing bfactors will be retrieved
        '''

        self.ligand = ligand
        self.bfactor = bfactor
        self.bfactor_df = None
        self.project_home = project_home

        # setup protein-ligand-complex
        if Path(target_file).is_file():
            self.protein = protein.Protein.from_file(str(target_file))
            self.protein.kekulize()

        waters = [w for w in self.protein.waters]
        [self.protein.remove_water(w) for w in waters]
        if strucid:
            self.protein.identifier = strucid
        self.protein.remove_all_waters()

        # setup ligand
        if ligand is None:
            self.csd_ligand = self.protein.ligands[-1]
            self.csd_ligand.remove_unknown_atoms()
            self.csd_ligand.remove_hydrogens()
            self.ligand_file = ligand
            self.rdkit_ligand = Chem.MolFromMol2Block(self.csd_ligand.to_string(), removeHs=True, sanitize=True)
            self.protein.remove_ligand(self.csd_ligand.identifier)

        elif type(ligand) == str and Path(ligand).is_file():
            with io.MoleculeReader(ligand) as rdr:
                self.csd_ligand = rdr[0]
            self.rdkit_ligand = Chem.MolFromMol2Block(self.csd_ligand.to_string('mol2'))

        elif type(ligand) == molecule.Molecule:
            self.csd_ligand = ligand
            self.rdkit_ligand = Chem.MolFromMol2Block(self.csd_ligand.to_string('mol2'))

        if only_binding_site:
            self.protein = _cut_out_binding_site_by_residue_df(self.protein, self.project_home)

        self.protein.add_molecule(self.csd_ligand)

        for w in waters:
            try:
                self.protein.add_molecule(w)
            except:
                continue

        if self.bfactor:
            pdb_protein_file = \
            list((Path(self.project_home) / Path('tmp_aligned_for_MOE_sanitized')).glob(f'{molecule.identifier}*.pdb'))[
                0]
            self.bfactor_df = get_b_factors(pdb_protein_file)
            for atom in self.protein.atoms:
                self.bfactor_df.loc[(self.bfactor_df['label'] == atom.label) &
                                    (self.bfactor_df['residue_label'] == atom.residue_label) &
                                    (self.bfactor_df['chain_label'] == atom.chain_label), 'atom_index'] = atom.index
        self.protein.normalise_labels()

        if self.bfactor:
            for atom in self.protein.atoms:
                self.bfactor_df.loc[self.bfactor_df['atom_index'] == atom.index, 'normalised_atom_label'] = atom.label
        self.protein = los_utilities.assign_index_to_atom_partial_charge(self.protein)

        self.csd_ligand = self.protein.ligands[-1]
        self.csd_ligand_atoms = self.csd_ligand.atoms

        self.protein_atoms = self.protein.atoms

        self.hbond_df = get_hbond_df(self.protein)

        describer = RfDescriptors(self.protein, self.csd_ligand)
        los_contacts_df = describer.los_contacts_df

        #hbonds
        hbond_list_1 = self.hbond_df[['atom1_label', 'atom1_residue_label', 'atom2_label', 'atom2_residue_label']].agg(
            ','.join, axis=1).to_list()
        hbond_list_2 = self.hbond_df[['atom2_label', 'atom2_residue_label', 'atom1_label', 'atom1_residue_label']].agg(
            ','.join, axis=1).to_list()
        los_contacts_df['is_hbond'] = \
            (los_contacts_df[
                 ['los_atom_label', 'los_atom_residue_label', 'ligand_atom_label', 'ligand_atom_residue_label']].agg(
                ','.join, axis=1).isin(hbond_list_1)) | \
            (los_contacts_df[
                 ['los_atom_label', 'los_atom_residue_label', 'ligand_atom_label', 'ligand_atom_residue_label']].agg(
                ','.join, axis=1).isin(hbond_list_2))
        # hbond mismatch: donor...donor or acceptor...acceptor contact
        los_contacts_df['is_hbond_mismatch'] = (los_contacts_df['ligand_atom_is_acceptor']) & \
                                               (los_contacts_df['los_atom_is_acceptor']) & \
                                               (los_contacts_df['is_hbond'] == False) | \
                                               (los_contacts_df['ligand_atom_is_donor']) & \
                                               (los_contacts_df['los_atom_is_donor']) & \
                                               (los_contacts_df['is_hbond'] == False)
        self.los_contacts_df = los_contacts_df


    def delta_apolar_sasa(self):
        try:
            bound_ligand_atoms = [self.protein.atom(a.label) for a in self.csd_ligand_atoms]
        except RuntimeError:
            ligand_res_at_labels = [(a.residue_label, a.label) for a in self.csd_ligand_atoms]
            bound_ligand_atoms = [a for a in self.protein_atoms if (a.residue_label, a.label) in ligand_res_at_labels]

        delta_apolar_sasa = calculate_delta_apolar_sasa(bound_ligand_atoms, self.csd_ligand_atoms)
        return delta_apolar_sasa


class PlotDescriptors(object):

    def __init__(self, input_file):

        self.los_home = ''

        self.all_targets_df = []
        self.all_targets_contacts_df = []
        self.all_targets_top500_df = []
        for target in ['ampc', 'akt1', 'cp3a4', 'gcr', 'cxcr4', 'kif11', 'hivrt', 'hivpr']:  # 'ampc', 'cp3a4', 'gcr', ','cxcr4 'kif11', 'hivrt', 'hivpr'
            target_df = pd.read_parquet(os.path.join(target, 'GOLD', input_file))
            if 'gold_ligand_name' not in target_df.columns:
                target_df['gold_ligand_name'] = target_df.apply(lambda row: row['ligand_file'].split('_')[-2], axis=1)
                target_df.loc[:, 'target'] = target
                target_df.to_parquet(os.path.join(target, 'GOLD', input_file), compression='gzip')

            target_contact_df = pd.read_parquet(os.path.join(target, 'GOLD', 'combined_contacts.gzip'))
            self.fitness_str = [s for s in target_contact_df.columns if 'Fitness' in s][0]
            if 'gold_ligand_name' not in target_contact_df.columns:
                target_contact_df['gold_ligand_name'] = target_contact_df.apply(
                    lambda row: row['ligand_file'].split('_')[-2], axis=1)
                target_contact_df.loc[:, 'target'] = target
                target_contact_df.to_parquet(os.path.join(target, 'GOLD', 'combined_contacts.gzip'), compression='gzip')
            target_df = target_df.join(target_contact_df.drop_duplicates('ligand_file')[['ligand_file', self.fitness_str]].set_index('ligand_file'), on='ligand_file')
            target_df = target_df[(target_df['competitive_contacts_per_atom'] > 0) |
                                  (target_df['non_competitive_contacts_per_atom'] > 0)].sort_values(by=self.fitness_str,
                                                                                                    ascending=False)
            for delta in ['', '_d01', '_d03', '_d05']:
                target_df[f'% non-competitive{delta}'] = target_df[f'non_competitive_contacts_per_atom{delta}'] / \
                                                         (target_df[f'competitive_contacts_per_atom'] + target_df[f'non_competitive_contacts_per_atom'])

            top_500_df = target_df.reset_index(drop=True).loc[:500, :]
            self.all_targets_top500_df.append(top_500_df)
            self.all_targets_df.append(target_df)
            self.all_targets_contacts_df.append(target_contact_df)
        self.all_targets_top500_df = pd.concat(self.all_targets_top500_df, ignore_index=True).astype({self.fitness_str: float})
        self.all_targets_df = pd.concat(self.all_targets_df, ignore_index=True).astype({self.fitness_str: float})
        self.all_targets_contacts_df = pd.concat(self.all_targets_contacts_df, ignore_index=True).astype({self.fitness_str: float})
        columns = ['identifier', 'resolution', 'non_competitive_contact_count', 'competitive_contact_count',
                   'non_competitive_contacts_per_atom', 'competitive_contacts_per_atom', 'rf_max', 'rf_min', 'rf_total']
        self.in_house_df = pd.read_parquet(os.path.join(self.los_home, 'full_p2cq_roche_oct2019_rf_extended.gzip'),
                                           columns = columns).drop_duplicates('identifier')
        self.in_house_df = self.in_house_df[self.in_house_df['resolution'] <= 2.5]
        self.in_house_df = self.in_house_df[(self.in_house_df['non_competitive_contacts_per_atom'] > 0) |
                                            (self.in_house_df['competitive_contacts_per_atom'] > 0)]
        self.in_house_df.loc[:, 'category'] = 'in_house'
        self.in_house_df.loc[:, 'target'] = 'in_house'

        self.public_df = pd.read_parquet(os.path.join(self.los_home, 'full_p2cq_pub_oct2019_rf_extended.gzip'),
                                         columns=columns + ['ligand_rscc', 'ligand_altcode', 'ligand_avgoccu',]
                                         ).drop_duplicates('identifier')
        self.public_df = self.public_df[(self.public_df['resolution'] <= 2.5) & (self.public_df['ligand_rscc'] >= 0.8) &
                                        (self.public_df['ligand_altcode'] == ' ') &
                                        (self.public_df['ligand_avgoccu'] == 1)]
        self.public_df = self.public_df[(self.public_df['non_competitive_contacts_per_atom'] > 0) |
                                        (self.public_df['competitive_contacts_per_atom'] > 0)]
        self.public_df.loc[:, 'category'] = 'public'
        self.public_df.loc[:, 'target'] = 'public'

    def data_distribution(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        for target in self.all_targets_df['target'].unique():
            plot_df = pd.concat(
                [self.all_targets_df[self.all_targets_df['target'] == target], self.public_df, self.in_house_df])
            long_df = pd.melt(plot_df, id_vars=['category'],
                              value_vars=['non_competitive_contact_count', 'competitive_contact_count'])
            boxplot = sns.boxplot(x='category', y='value', hue='variable', data=long_df, showfliers=False)
            boxplot.set_title(target)
            boxplot.figure.savefig(f'boxplot_{target}.png', dpi=600)
            plt.clf()

        plot_df = pd.concat([self.public_df.loc[:, ['target', 'rf_total', 'category']],
                             self.in_house_df.loc[:, ['target', 'rf_total', 'category']],
                             self.all_targets_contacts_df.loc[:, ['target', 'rf_total', 'category']]],
                            ignore_index=True)

        # long_df = pd.melt(plot_df, id_vars=['target'], value_vars=['rf_total'])
        violinplot = sns.violinplot(x='target', y='rf_total', hue='category', data=plot_df, showfliers=False)
        violinplot.figure.savefig(f'violinplot.png', dpi=600)
        plt.clf()

        plot_df = pd.concat([self.all_targets_df, self.public_df, self.in_house_df])

        long_df = pd.melt(plot_df, id_vars=['target', 'category'],
                          value_vars=['non_competitive_contacts_per_atom', 'competitive_contacts_per_atom'])
        long_df['interaction_category'] = long_df.apply(lambda row: row['variable'] + '_' + row['category'], axis=1)
        boxplot = sns.boxplot(x='target', y='value', hue='interaction_category', data=long_df, showfliers=False)
        boxplot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), prop={'size': 6})
        boxplot.figure.savefig('boxplot_top500.png', dpi=600, bbox_inches='tight')
        plt.clf()

        plot_df = pd.concat([self.all_targets_df, self.public_df, self.in_house_df])
        plot_df = plot_df.astype({'rf_max': 'float64', 'rf_min': 'float64'})

        for delta in ['', '_d01', '_d03', '_d05']:
            boxplot = sns.boxplot(x='target', y=f'% non-competitive{delta}', hue='category', data=plot_df,
                                  showfliers=False)
            boxplot.figure.savefig(f'boxplot_top500_ratio{delta}.png', dpi=600)
            plt.clf()

        boxplot = sns.boxplot(x='target', y='rf_max', hue='category', data=plot_df, showfliers=False)
        boxplot.figure.savefig('boxplot_top500_rf_max.png', dpi=600)
        plt.clf()

        boxplot = sns.boxplot(x='target', y='rf_min', hue='category', data=plot_df, showfliers=False)
        boxplot.figure.savefig('boxplot_top500_rf_min.png', dpi=600)
        plt.clf()

    def _rescored_fitness(self, target_contact_df, weight=10):

        rescored_fitness_list = []
        rf_score_list = []
        ligand_file_list = []
        target_contact_df[f'{weight}xlg_rf_total'] = weight * np.log(target_contact_df['rf_total'])
        grouped = target_contact_df.groupby(['ligand_file', self.fitness_str])[f'{weight}xlg_rf_total'].sum()
        for ligand_file, plp_fitness in grouped.keys():
            rf_score = grouped[(ligand_file, plp_fitness)]
            rescored_fitness = plp_fitness + rf_score
            rf_score_list.append(rf_score)
            rescored_fitness_list.append(rescored_fitness)
            ligand_file_list.append(ligand_file)
        rescored_fitness_df = pd.DataFrame({'ligand_file': ligand_file_list, 'rescored_fitness': rescored_fitness_list,
                                            'rf_score': rf_score_list})

        return rescored_fitness_df

    def _enrichment_factor(self, target_df, score_label):
        total_dataset_size = target_df.drop_duplicates(subset='ligand_file').shape[0]
        hits_total_dataset = target_df.drop_duplicates(subset='ligand_file')['category'].str.count('actives').sum()
        denominator = hits_total_dataset / total_dataset_size
        target_df = target_df.sort_values(by=score_label, ascending=False).drop_duplicates(subset='gold_ligand_name')
        target_df_filtered = target_df['category']
        top_1pc = target_df_filtered[0:int(0.01 * total_dataset_size)]
        numerator = (top_1pc.str.count('actives').sum() / top_1pc.shape[0])
        ef_1pc = numerator / denominator
        return ef_1pc

    def enrichment_barplot(self, weight):
        import seaborn as sns
        from matplotlib import pyplot as plt

        ef_1pc_df = []
        all_targets_contacts_df = self.all_targets_contacts_df[['ligand_file', 'category', 'target', self.fitness_str,
                                                                'gold_ligand_name', 'rf_total', 'is_primary']].copy()

        meta = {'ligand_file': np.str_, 'category': np.str_, 'target': np.str_, self.fitness_str: np.float64,
                'gold_ligand_name': np.str_, 'rf_total': np.float64, 'is_primary': np.bool_}
        all_targets_contacts_df = all_targets_contacts_df.astype(meta)
        for target in all_targets_contacts_df['target'].unique():
            target_contact_df = all_targets_contacts_df[all_targets_contacts_df['target'] == target].copy()
            target_df = self.all_targets_df[self.all_targets_df['target'] == target].copy()

            rescored_fitness = self._rescored_fitness(target_contact_df, weight)
            target_df = target_df.join(rescored_fitness.set_index('ligand_file'), on='ligand_file')

            mask = (target_df['Red'] > 1) | (target_df['Orange'] > 3)
            target_df['ChemPLP_RF_torsion_rescore'] = target_df['rescored_fitness']
            target_df.loc[mask, 'ChemPLP_RF_torsion_rescore'] = 0
            # target_df['RF_torsion_score'] = target_df['rf_score']
            # target_df.loc[mask, 'RF_torsion_score'] = 0

            ef_1pc = self._enrichment_factor(target_df, self.fitness_str)
            ef_1pc_rescored = self._enrichment_factor(target_df, 'rescored_fitness')
            ef_1pc_RF_torsion_rescore = self._enrichment_factor(target_df, 'ChemPLP_RF_torsion_rescore')
            # ef_1pc_RF_score = self._enrichment_factor(target_df, 'rf_score')
            # ef_1pc_RF_torsion_score = self._enrichment_factor(target_df, 'RF_torsion_score')
            ef_1pc_df.append(pd.DataFrame({'EF_1%': [ef_1pc, ef_1pc_rescored, ef_1pc_RF_torsion_rescore],
                                           'target': [target, target, target],
                                           'Scoring_method': ['ChemPLPScore', 'ChemPLPScore_RF_rescore',
                                                              'ChemPLP_RF_torsion_rescore']}))
        ef_1pc_df = pd.concat(ef_1pc_df, ignore_index=True)

        print('plotting...')
        ef_plot = sns.barplot(x='target', y='EF_1%', hue='Scoring_method', data=ef_1pc_df)
        ef_plot.set(ylabel='EF 1%')
        # ef_plot.legend(loc='center', bbox_to_anchor=(1.2, 0.5), prop={'size': 8})
        ef_plot.figure.savefig(f'ef_1pc_barplot_weight_{weight}.png', dpi=600, bbox_inches='tight')
        plt.clf()

    def enrichment_plot(self, unfav_parameter=None, fav_parameter=None, fav_parameter_max=1, unfav_parameter_max=1):
        import seaborn as sns
        from matplotlib import pyplot as plt
        if unfav_parameter:
            if unfav_parameter_max == 1:
                thresholds = np.linspace(0, 1, 20)
            else:
                thresholds = range(unfav_parameter_max)
        if fav_parameter:
            if fav_parameter_max == 1:
                thresholds = np.linspace(0, 1, 20)
            else:
                thresholds = range(0, fav_parameter_max)

        ef_1pc_df = pd.DataFrame({'threshold': thresholds})
        for target in self.all_targets_df['target'].unique():
            target_df = self.all_targets_df[self.all_targets_df['target'] == target].sort_values(by='fitness',
                                                                                                 ascending=False)
            total_dataset_size = target_df.drop_duplicates(subset='ligand_file').shape[0]
            hits_total_dataset = target_df.drop_duplicates(subset='ligand_file')['category'].str.count('actives').sum()
            denominator = hits_total_dataset / total_dataset_size
            for index, threshold in enumerate(thresholds):
                filtered_df = target_df.copy()
                if unfav_parameter:
                    filtered_df.loc[filtered_df[unfav_parameter] > threshold, 'fitness'] = 0
                    # filtered_df.loc[filtered_df['Red'] > 1, 'fitness'] = 0
                if fav_parameter:
                    filtered_df.loc[filtered_df[fav_parameter] < thresholds[len(thresholds) - index - 1], 'fitness'] = 0

                target_df_filtered = filtered_df.sort_values(by='fitness', ascending=False).drop_duplicates(
                    subset='gold_ligand_name')
                sample_number = target_df_filtered[target_df_filtered['fitness'] > 0].shape[0]
                target_df_filtered = target_df_filtered['category']
                if sample_number > 0.01 * target_df_filtered.shape[0]:
                    top_1pc = target_df_filtered[0:int(0.01 * total_dataset_size)]
                    numerator = (top_1pc.str.count('actives').sum() / top_1pc.shape[0])
                    ef_1pc = numerator / denominator
                    ef_1pc_df.loc[ef_1pc_df['threshold'] == threshold, target] = ef_1pc
                    ef_1pc_df.loc[ef_1pc_df['threshold'] == threshold, 'total_count_top_1pc'] = len(top_1pc)
                else:
                    ef_1pc_df.loc[ef_1pc_df['threshold'] == threshold, target] = np.nan
                    ef_1pc_df.loc[ef_1pc_df['threshold'] == threshold, 'total_count_top_1pc'] = np.nan

        plot_df = pd.melt(ef_1pc_df, id_vars=['threshold', 'total_count_top_1pc'])
        ef_plot = sns.scatterplot(x='threshold', y='value', hue='variable', data=plot_df)
        ef_plot.set(ylabel='EF 1%')
        ef_plot.legend(loc='center', bbox_to_anchor=(1.2, 0.5), prop={'size': 8})
        ef_plot.figure.savefig(f'ef_1pc_{unfav_parameter}_{fav_parameter}.png', dpi=600, bbox_inches='tight')
        plt.clf()


def main():
    # csd_describer = CsdDescriptorsFromGold('gold_soln_decoys_final_m3841_8.sdf')
    # contact_df = csd_describer.contact_df()
    # rf_counts = rf_count_df(contact_df, csd_describer.csd_ligand)
    # torsion_df = torsion_df('gold_soln_decoys_final_m20578_3.sdf')

    plotter = PlotDescriptors('combined_descriptors.gzip')
    for weight in [5]: # [1, 5, 10, 20, 50, 100]:
        plotter.enrichment_barplot(weight)
    # plotter.data_distribution()

    # plotter.enrichment_plot(unfav_parameter='Red', unfav_parameter_max=10)
    #
    # for delta in ['', '_d01', '_d03', '_d05']:
    #     plotter.enrichment_plot(unfav_parameter=f'% non-competitive{delta}')
    #
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contacts_per_atom')
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contact_count', unfav_parameter_max=20)
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contacts_per_atom_d01')
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contact_count_d01', unfav_parameter_max=20)
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contacts_per_atom_d03')
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contact_count_d03', unfav_parameter_max=20)
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contacts_per_atom_d05')
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contact_count_d05', unfav_parameter_max=20)
    #
    # plotter.enrichment_plot(fav_parameter='competitive_contacts_per_atom')
    # plotter.enrichment_plot(fav_parameter='competitive_contact_count', fav_parameter_max=10)
    # plotter.enrichment_plot(fav_parameter='competitive_contacts_per_atom_d01')
    # plotter.enrichment_plot(fav_parameter='competitive_contact_count_d01', fav_parameter_max=10)
    # plotter.enrichment_plot(fav_parameter='competitive_contacts_per_atom_d03')
    # plotter.enrichment_plot(fav_parameter='competitive_contact_count_d03', fav_parameter_max=10)
    # plotter.enrichment_plot(fav_parameter='competitive_contacts_per_atom_d05')
    # plotter.enrichment_plot(fav_parameter='competitive_contact_count_d05', fav_parameter_max=10)
    #
    # plotter.enrichment_plot(unfav_parameter='non_competitive_contacts_per_atom_d01',
    #                         fav_parameter='competitive_contacts_per_atom_d01')
    # print('fulano')


if __name__ == '__main__':
    main()
