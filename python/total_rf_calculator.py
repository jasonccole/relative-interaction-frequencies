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
########################################################################################################################

import __future__
from scipy.stats import gmean
import numpy as np
from collections import Counter

########################################################################################################################


def derivative(c, v):
    '''

    :param c: List of constant elements from inner derivative
    :param v: variable part
    :return: derivative of geometric mean
    >>> c = []
    >>> v = 0.5
    >>> derivative(c, v)
    1.0
    >>> c = [1]
    >>> v = 4
    >>> derivative(c, v)
    0.25
    >>> c = [8, 2]
    >>> v = 4
    >>> np.round(derivative(c, v), 2)
    0.33
    '''
    exponent = (1 / (len(c) + 1))
    return (np.prod(c) * (exponent) * np.power((np.prod(c) * v), exponent - 1))


def error_propagation(rf_list, rf_errors):
    '''

    :return:
    '''
    error = 0
    for index, rf in enumerate(rf_list):
        b = [rf2 for index2, rf2 in enumerate(rf_list) if index2 != index]
        error += derivative(b, rf) * rf_errors[index]
    return error


def return_interaction_type(rf_dict):
    interaction_types = rf_dict['protein_interactions'] + rf_dict['ligand_interactions']
    try:
        # pi-pi interactions have low priority for in-plane...in-plane contacts+
        if rf_dict['rf_total'] + rf_dict['rf_total_error'] <= 1:
            interaction_priority = ['electrostatic_repulsion', 'desolvation']

        else:
            interaction_priority = ['ionic', 'hbond_classic', 'hbond_weak', 'halogen_bond', 'multipolar', 'pi',
                                    'hydrophobic']

        interaction_type = None
        for i in interaction_priority:
            if i in interaction_types.keys():
                interaction_type = i
                break
    except:
        interaction_type = None

    return interaction_type


class TotalRf(object):

    def __init__(self, protein, protein_atom_types, global_ligand_lookup_alpha, global_ligand_lookup_h,
                 global_protein_lookup_alpha, global_protein_lookup_h, expected=10):
        '''

        :param lookup_dfs: Dictionary of lookup dataframes.
        :param protein: Protein object
        :param protein_atom_types: Protein atom types DataFrame
        :param expected: Only Rf values with "Expected" values >= expected
        '''
        self.expected = expected
        self.protein = protein
        self.protein_atom_types = protein_atom_types
        self.global_ligand_lookup_alpha = global_ligand_lookup_alpha
        self.global_ligand_lookup_h = global_ligand_lookup_h
        self.global_protein_lookup_alpha = global_protein_lookup_alpha
        self.global_protein_lookup_h = global_protein_lookup_h

    def rf_alpha(self, lookup, alphas, ligand_atom_type, pat):
        '''
        Return alpha constrained RF values.
        :param lookup: Lookup filename.
        :param alphas:
        :param pat: Protein atom type, as defined in protein_atom_types.csv
        :return: Rf_ligand_alpha or Rf_protein_alpha
        '''

        rf_alphas = []
        rf_alpha_errors = []
        it_alphas = []
        alpha_df = lookup
        try:
            alpha_df = alpha_df.get_group((ligand_atom_type, pat))
            for alpha in alphas:
                _filtered_alpha_df = alpha_df[(alpha_df['alpha_max'] > alpha) & (alpha_df['alpha_min'] <= alpha)].squeeze()
                if _filtered_alpha_df.shape[0] != 0:
                    rf_alpha = [_filtered_alpha_df['rf']]
                    rf_low = [_filtered_alpha_df['rf_low']]
                    rf_high = [_filtered_alpha_df['rf_high']]
                    interaction_types = _filtered_alpha_df['interaction_types'].split(';')
                    if len(interaction_types) == 0:
                        interaction_types = []

                if _filtered_alpha_df.shape[0] == 0 or len(rf_alpha) == 0 or pat == 'other_ligands':
                    rf_alpha = [1]
                    rf_high = [0]
                    rf_low = [0]
                    interaction_types = []

                rf_alpha_errors.append((rf_high[0] - rf_low[0]) / 2)
                rf_alphas.append(rf_alpha[0])
                it_alphas.append(interaction_types)

        except KeyError:
            for alpha in alphas:
                rf_alpha = [1]
                rf_high = [0]
                rf_low = [0]
                interaction_types = []
                rf_alpha_errors.append((rf_high[0] - rf_low[0]) / 2)
                rf_alphas.append(rf_alpha[0])

            rf_alpha_errors.append((rf_high[0] - rf_low[0]) / 2)
            rf_alphas.append(rf_alpha[0])
            it_alphas.append(interaction_types)

        rf_alpha_error = error_propagation(rf_alphas, rf_alpha_errors)
        rf_alpha = gmean(rf_alphas)
        it_alphas = np.sum([Counter(i) for i in it_alphas])
        return rf_alpha, rf_alpha_error, it_alphas

    def rf_h(self, lookup, h, ligand_atom_type, pat):
        '''

        :param lookup: Lookup filename, has to be filtered already by "Expected" threshold.
        :param h: Distance to plane. Applies only to pi-systems.
        :param pat: Protein atom type, as defined in protein_atom_types.csv
        :return: Rf_ligand_h or Rf_protein_h
        '''

        h_df = lookup
        try:
            h_df = h_df.get_group((ligand_atom_type, pat))
            h_df = h_df[(h_df['h_max'] > h) & (h_df['h_min'] <= h)].squeeze()
            if h_df.shape[0] != 0:
                rf_h = [h_df['rf']]
                rf_h_high = [h_df['rf_high']]
                rf_h_low = [h_df['rf_low']]
                interaction_types = h_df['interaction_types'].split(';')
                if len(interaction_types) == 0:
                    interaction_types = []

            if h_df.shape[0] == 0 or len(rf_h) == 0 or pat == 'other_ligands':
                rf_h = [1]
                rf_h_high = [0]
                rf_h_low = [0]
                interaction_types = []

        except KeyError:
            rf_h = [1]
            rf_h_high = [0]
            rf_h_low = [0]
            interaction_types = []

        rf_h_error = (rf_h_high[0] - rf_h_low[0]) / 2
        rf_h = gmean(rf_h)
        interaction_types = Counter(interaction_types)
        return rf_h, rf_h_error, interaction_types

    def rf_ligand(self, ligand_alphas, ligand_atom_type, pat, ligand_h=None):
        '''

        :param ligand_alphas:
        :param SMARTS: SMARTS string for ligand query atom
        :param SMARTS_index: SMARTS index for ligand query atom
        :param pat: Protein atom type, as defined in protein_atom_types.csv
        :param ligand_h: Distance to ligand plane. Applies only to pi systems.
        :return: Rf_ligand
        '''
        rf_ligand = []
        rf_ligand_errors = []
        if len(ligand_alphas) == 0:
            rf_ligand_alpha, rf_ligand_alpha_error = 1, 0
        else:
            rf_ligand_alpha, rf_ligand_alpha_error, alpha_interaction_types = self.rf_alpha(self.global_ligand_lookup_alpha,
                                                                                      ligand_alphas, ligand_atom_type,
                                                                                      pat)
        rf_ligand.append(rf_ligand_alpha)
        rf_ligand_errors.append(rf_ligand_alpha_error)

        if ligand_h is not None and ligand_h == ligand_h:
            rf_ligand_h, rf_ligand_h_error, h_interaction_types = self.rf_h(self.global_ligand_lookup_h, ligand_h,
                                                                          ligand_atom_type, pat)
            ligand_interactions = h_interaction_types + alpha_interaction_types
            rf_ligand.append(rf_ligand_h)
            rf_ligand_errors.append(rf_ligand_h_error)

        else:
            rf_ligand_h, rf_ligand_h_error = np.nan, np.nan
            ligand_interactions = alpha_interaction_types

        rf_ligand_error = error_propagation(rf_ligand, rf_ligand_errors)
        if rf_ligand_error != rf_ligand_error:
            rf_ligand_error = 0
        rf_ligand = gmean(rf_ligand)
        rf_ligand_dict = {'rf_ligand': rf_ligand, 'rf_ligand_error': rf_ligand_error,
                          'rf_ligand_alpha': rf_ligand_alpha, 'rf_ligand_alpha_error': rf_ligand_alpha_error,
                          'rf_ligand_h': rf_ligand_h, 'rf_ligand_h_error': rf_ligand_h_error,
                          'ligand_interactions': ligand_interactions}
        return rf_ligand_dict

    def rf_protein(self, protein_alphas, ligand_atom_type, pat, protein_h=None):

        '''
        :param protein_alphas: List of contact angles for protein atom
        :param SMARTS: SMARTS string for ligand query atom
        :param SMARTS_index: SMARTS index for ligand query atom
        :param pat: Protein atom type, as defined in protein_atom_types.csv
        :param protein_h: Distance to protein plane. Applies only to pi systems.
        :return: Rf_protein
        '''
        rf_protein = []
        rf_protein_errors = []
        if len(protein_alphas) == 0:
            rf_protein_alpha, rf_protein_alpha_error = 1, 0
            alpha_interaction_types = Counter()
        else:
            rf_protein_alpha, rf_protein_alpha_error, alpha_interaction_types = self.rf_alpha(self.global_protein_lookup_alpha,
                                                                                        protein_alphas,
                                                                                        ligand_atom_type, pat)
        rf_protein.append(rf_protein_alpha)
        rf_protein_errors.append(rf_protein_alpha_error)

        if protein_h is not None and protein_h == protein_h:
            rf_protein_h, rf_protein_h_error, h_interaction_types = self.rf_h(self.global_protein_lookup_h, protein_h,
                                                                              ligand_atom_type, pat)
            protein_interactions = h_interaction_types + alpha_interaction_types
            rf_protein.append(rf_protein_h)
            rf_protein_errors.append(rf_protein_h_error)

        else:
            rf_protein_h, rf_protein_h_error = np.nan, np.nan
            protein_interactions = alpha_interaction_types

        rf_protein_error = error_propagation(rf_protein, rf_protein_errors)
        if rf_protein_error != rf_protein_error:
            rf_protein_error = 0
        rf_protein = gmean(rf_protein)
        rf_protein_dict = {'rf_protein': rf_protein, 'rf_protein_error': rf_protein_error,
                           'rf_protein_alpha': rf_protein_alpha, 'rf_protein_alpha_error': rf_protein_alpha_error,
                           'rf_protein_h': rf_protein_h, 'rf_protein_h_error': rf_protein_h_error,
                           'protein_interactions': protein_interactions}
        return rf_protein_dict

    def return_rf(self, los_atom, ligand_alphas, protein_alphas, ligand_atom_type, ligand_h=np.nan, protein_h=np.nan):
        '''

        :param los_atom: Atom which is in line of sight contact with ligand query atom
        :param ligand_alphas: List of contact angles for ligand atom
        :param protein_alphas: List of contact angles for protein atom
        :param ligand_atom:
        :param ligand_h: Distance to ligand plane
        :param protein_h: Distance to protein plane. Applies only to pi systems.
        :return: A dictionary with Rf values for Rf_total, Rf_protein, Rf_ligand
        '''

        if not hasattr(los_atom, 'los_at'):
            # Assign protein atom type to LoS contact. Perform SMARTs matching and for sulfur, check for disulfide bonds.
            # If two cystein sulfide atoms are within 2.25 Angstrom, they are classified as S_apol.
            from ccdc_roche.python.los_utilities import return_atom_type
            pat = return_atom_type(los_atom, hit_protein=self.protein, protein_atom_types_df=self.protein_atom_types)
        else:
            pat = los_atom.los_at

        rf_protein_dict = self.rf_protein(protein_alphas, ligand_atom_type, pat, protein_h)
        rf_ligand_dict = self.rf_ligand(ligand_alphas, ligand_atom_type, pat, ligand_h)


        rf_total_error = error_propagation([rf_protein_dict['rf_protein'], rf_ligand_dict['rf_ligand']],
                                           [rf_protein_dict['rf_protein_error'], rf_ligand_dict['rf_ligand_error']])
        rf_total = gmean([rf_protein_dict['rf_protein'], rf_ligand_dict['rf_ligand']])

        rf_dictionary = {'rf_total': rf_total, 'rf_total_error': rf_total_error}

        rf_dictionary.update(rf_ligand_dict)
        rf_dictionary.update(rf_protein_dict)

        interaction_type = return_interaction_type(rf_dictionary)
        rf_dictionary['interaction_type'] = interaction_type

        del rf_ligand_dict['ligand_interactions']
        del rf_protein_dict['protein_interactions']

        del rf_dictionary['protein_interactions']
        del rf_dictionary['ligand_interactions']
        return rf_dictionary
