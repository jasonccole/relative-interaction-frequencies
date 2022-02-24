#!/usr/bin/env python


# A. Tosstorff, B. Kuhn
# 15-APR-2020

########################################################################################################################

import __future__
from ccdc_roche.python.los_postprocessing import distance_dependency
from ccdc_roche.python import total_rf_calculator
import os
import pandas as pd
from scipy.stats import gmean
import itertools

########################################################################################################################


class RfPlanePlane(object):

    def __init__(self, protein_atom_type, ligand_atom_type, los_home='.', protein_plane=True, ligand_plane=True):
        self.ligand_atom_type = ligand_atom_type
        self.protein_atom_type = protein_atom_type
        self.los_home = os.path.abspath(los_home)
        self.cwd = os.getcwd()
        self.protein_plane = protein_plane
        self.ligand_plane = ligand_plane

    def _calculate_ligand_plane(self):
        os.chdir(os.path.join('output', self.ligand_atom_type, 'query_atom'))
        # if not os.path.isfile('statistics_h_plane_dependency.csv'):
        distance_dependency('h', 'ligand', self.los_home, self.ligand_atom_type, bin_size=1, bins=[0, 2.5],
                                    filename_ext='_plane_dependency')
        os.chdir(self.cwd)

    def _calculate_protein_plane(self):
        os.chdir(os.path.join('output', self.ligand_atom_type, self.protein_atom_type))
        # if not os.path.isfile('statistics_h_plane_dependency.csv'):
        distance_dependency('h', 'protein', self.los_home, self.ligand_atom_type, bin_size=1, bins=[0, 2.5],
                                    filename_ext='_plane_dependency')
        os.chdir(self.cwd)

    def _calculate_rf_total(self, rf_dict):
        rf_ligand_alpha = rf_dict['rf_ligand_alpha']
        rf_ligand_alpha_uncertainty = rf_dict['rf_ligand_alpha_uncertainty']
        rf_ligand_h = rf_dict['rf_ligand_h']
        rf_ligand_h_uncertainty = rf_dict['rf_ligand_h_uncertainty']
        rf_protein_alpha = rf_dict['rf_protein_alpha']
        rf_protein_alpha_uncertainty = rf_dict['rf_protein_alpha_uncertainty']
        rf_protein_h = rf_dict['rf_protein_h']
        rf_protein_h_uncertainty = rf_dict['rf_protein_h_uncertainty']

        rf_ligand = gmean([rf_ligand_alpha, rf_ligand_h])
        rf_ligand_uncertainty = total_rf_calculator.error_propagation([rf_ligand_alpha, rf_ligand_h],
                                                                      [rf_ligand_alpha_uncertainty,
                                                                       rf_ligand_h_uncertainty])
        rf_protein = gmean([rf_protein_alpha, rf_protein_h])
        rf_protein_uncertainty = total_rf_calculator.error_propagation([rf_protein_alpha, rf_protein_h],
                                                                       [rf_protein_alpha_uncertainty,
                                                                       rf_protein_h_uncertainty])
        rf_total = gmean([rf_ligand, rf_protein])
        rf_total_uncertainty = total_rf_calculator.error_propagation([rf_ligand, rf_protein],
                                                                     [rf_ligand_uncertainty, rf_protein_uncertainty])
        return rf_total, rf_total_uncertainty

    def _get_rf_df(self):
        if self.protein_plane:
            protein_plane_limits = [0, 2.5]
            protein_df = pd.read_csv(os.path.join('output', self.ligand_atom_type, self.protein_atom_type,
                                                  'statistics_h_plane_dependency.csv')
                                     )

            protein_df['rf_uncertainty'] = (protein_df['rf_high'] - protein_df['rf_low']) / 2

        else:
            protein_plane_limits = [None, None]


        protein_unconstrained_df = pd.read_csv(os.path.join('output', self.ligand_atom_type, self.protein_atom_type,
                                                            'rf.csv')
                                               )
        
        protein_unconstrained_df['rf_uncertainty'] = (protein_unconstrained_df['rf_high']
                                                      - protein_unconstrained_df['rf_low'])/2

        if self.ligand_plane:
            ligand_plane_limits = [0, 2.5]
            ligand_df = pd.read_csv(os.path.join('output', self.ligand_atom_type, 'query_atom',
                                                 'statistics_h_plane_dependency.csv')
                                    )

            ligand_df['rf_uncertainty'] = (ligand_df['rf_high'] - ligand_df['rf_low']) / 2

        else:
            ligand_plane_limits = [None, None]

        ligand_unconstrained_df = pd.read_csv(os.path.join('output', self.ligand_atom_type, 'query_atom',
                                                           'rf.csv')
                                              )

        ligand_unconstrained_df['rf_uncertainty'] = (ligand_unconstrained_df['rf_high']
                                                     - ligand_unconstrained_df['rf_low'])/2

        plane_df = pd.DataFrame()

        plane_limits = set(itertools.product(protein_plane_limits, ligand_plane_limits))

        for i in plane_limits:
            
            protein_h_min, ligand_h_min = i
            rf_ligand_alpha = ligand_unconstrained_df[
                ligand_unconstrained_df['atom_type'] == self.protein_atom_type]['rf'].to_numpy()[0]
    
            rf_ligand_alpha_uncertainty = ligand_unconstrained_df[
                ligand_unconstrained_df['atom_type'] == self.protein_atom_type]['rf_uncertainty'].to_numpy()[0]

            if self.ligand_plane:
                rf_ligand_h = ligand_df[
                    (ligand_df['atom_type'] == self.protein_atom_type) &
                    (ligand_df['h_min'] == ligand_h_min)
                    ]['rf'].to_numpy()[0]

                rf_ligand_h_uncertainty = ligand_df[
                    (ligand_df['atom_type'] == self.protein_atom_type) &
                    (ligand_df['h_min'] == ligand_h_min)
                    ]['rf_uncertainty'].to_numpy()[0]
            else:
                rf_ligand_h = 1
                rf_ligand_h_uncertainty = 0

            rf_protein_alpha = protein_unconstrained_df[
                protein_unconstrained_df['atom_type'] == 'query_match']['rf'].to_numpy()[0]

            rf_protein_alpha_uncertainty = protein_unconstrained_df[
                protein_unconstrained_df['atom_type'] == 'query_match']['rf_uncertainty'].to_numpy()[0]

            if self.protein_plane:
                rf_protein_h = protein_df[
                    (protein_df['atom_type'] == 'query_match') &
                    (protein_df['h_min'] == protein_h_min)
                    ]['rf'].to_numpy()[0]

                rf_protein_h_uncertainty = protein_df[
                    (protein_df['atom_type'] == 'query_match') &
                    (protein_df['h_min'] == protein_h_min)
                    ]['rf_uncertainty'].to_numpy()[0]

            else:
                rf_protein_h = 1
                rf_protein_h_uncertainty = 0

            rf_dict = {'rf_ligand_alpha': rf_ligand_alpha, 'rf_ligand_alpha_uncertainty': rf_ligand_alpha_uncertainty,
                       'rf_ligand_h': rf_ligand_h, 'rf_ligand_h_uncertainty': rf_ligand_h_uncertainty,
                       'rf_protein_alpha': rf_protein_alpha,
                       'rf_protein_alpha_uncertainty': rf_protein_alpha_uncertainty,
                       'rf_protein_h': rf_protein_h, 'rf_protein_h_uncertainty': rf_protein_h_uncertainty
                       }

            rf_total, rf_total_uncertainty = self._calculate_rf_total(rf_dict)
            
            rf_df = pd.DataFrame({'protein_h_min': [protein_h_min], 'ligand_h_min': [ligand_h_min],
                                  'rf_total': [rf_total], 'rf_total_uncertainty': [rf_total_uncertainty],
                                  'protein_atom_type': [self.protein_atom_type],
                                  'ligand_atom_type': [self.ligand_atom_type]}
                                 )
            plane_df = plane_df.append(rf_df, ignore_index=True)
            
        return plane_df

    def get_plane_dependency_df(self):
        if self.ligand_plane:
            self._calculate_ligand_plane()
        if self.protein_plane:
            self._calculate_protein_plane()
        plane_df = self._get_rf_df()
        
        return plane_df


def main():
    plane_dependency_df = pd.DataFrame()
    for ligand_atom_type in ['carbon_aromatic_pyrazine', 'carbon_aromatic_phenyl', 'nitrogen_pyrimidine', 'nitrogen_aromatic_pyridine', 'carbon_aromatic_pyridine_2',
                             'carbon_aromatic_pyridine_3', 'carbon_aromatic_pyridine_4', 'carbon_aromatic_pyrimidine_2',
                             'carbon_aromatic_pyrimidine_4', 'carbon_aromatic_pyrimidine_5']:
        for protein_atom_type in ['O_pi_acc', 'C_pi_carbonyl', 'N_pi_don']:
            plane_dependency_df = plane_dependency_df.append(RfPlanePlane(protein_atom_type, ligand_atom_type,
                                                                          los_home='.').
                                                   get_plane_dependency_df())
    plane_dependency_df.to_csv('heteroaromatics_amide_plane_geometry_dependency.csv', index=False)

    # plane_dependency_df = pd.DataFrame()
    # for ligand_atom_type in ['carbon_alkyl_terminal_unpolarized', 'carbon_alkyl_terminal_polarized_aryl',
    #                          'carbon_alkyl_terminal_polarized_C,S,P_N,O,F', 'carbon_alkyl_terminal_polarized_N,O,F',
    #                          'nitrogen_amine_acyclic_sec', 'nitrogen_amine_prim']:
    #     for protein_atom_type in ['O_pi_acc', 'C_pi_carbonyl', 'N_pi_don', 'C_pi_phenyl', 'C_pi_neg', 'C_pi_pos']:
    #         plane_dependency_df = plane_dependency_df.append(RfPlanePlane(protein_atom_type, ligand_atom_type,
    #                                                                       los_home='.', ligand_plane=False
    #                                                                       ).get_plane_dependency_df())
    # plane_dependency_df.to_csv('polarized_ch_plane_dependency.csv', index=False)
    #
    # plane_dependency_df = pd.DataFrame()
    # for ligand_atom_type in ['oxygen_ether_acyclic_Cany_Csp2', 'oxygen_aromatic_isoxazole',
    #                          'oxygen_ether_acyclic_Csp3_Csp3', 'oxygen_ether_cyclic', 'oxygen_aromatic_oxazole']:
    #     for protein_atom_type in ['O_pi_acc', 'C_pi_carbonyl', 'N_pi_don', 'N_pi_don_pos',
    #                               'C_pi_phenyl', 'C_pi_neg', 'C_pi_pos']:
    #         plane_dependency_df = plane_dependency_df.append(RfPlanePlane(protein_atom_type, ligand_atom_type,
    #                                                                       los_home='.', ligand_plane=False
    #                                                                       ).get_plane_dependency_df())
    # plane_dependency_df.to_csv('ether_plane_dependency.csv', index=False)


if __name__ == '__main__':
    main()
