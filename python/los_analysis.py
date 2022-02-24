#! /usr/bin/env python

# program analyzes output of line-of-sight searches and counts occurrences of
# interactions defined by distance and angle thresholds.
# Confidence intervals are calculated by bootstrapping
# A. Tosstorff, B. Kuhn
# 2014-2021

########################################################################################################################

import __future__
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from pathos.multiprocessing import ProcessingPool
from pathos.multiprocessing import freeze_support
from functools import partial
from ccdc_roche import atom_types

np.seterr(divide='ignore', invalid='ignore')

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''

    parser = argparse.ArgumentParser(
        description=
        """
    Calculate Rf values.
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-i',
        '-in',
        '--input',
        help='Path to input folder.',
        default=os.getcwd()
    )

    parser.add_argument(
        '--max_vdw_dist',
        help='',
        default='0.5',
        type=float
    )

    parser.add_argument(
        '--min_alpha_i',
        help='',
        default='0',
        type=float
    )

    parser.add_argument(
        '--max_alpha_i',
        help='',
        default='180',
        type=float
    )

    parser.add_argument(
        '-m',
        '--mode',
        help='ligand or protein mode',
        default='ligand',
        type=str
    )

    parser.add_argument(
        '-wm',
        '--watmap',
        help='If watmap argument is passed, the watmap column will be included in the calculations',
        action="store_true"
    )

    parser.add_argument(
        '--n_boot',
        help='',
        default='100',
        type=int
    )

    parser.add_argument(
        '-los',
        '--los',
        help='',
        default='los_filtered.csv'
    )

    parser.add_argument(
        '-complex',
        '--complex',
        help='',
        default='complex_filtered.csv'
    )

    parser.add_argument(
        '-o',
        '-out',
        '--output',
        help='Path to output file.',
        default='rf.csv'
    )

    parser.add_argument(
        '-pat',
        '--protein_atom_types',
        help='Comma separated list of protein atom types.',
        default=None
    )

    parser.add_argument(
        '--second_geometry_name',
        help='Pass to activate 2D Rf analysis. Specify the name of the query. E.g. Cl_amide',
        default=None
    )

    parser.add_argument(
        '-second_geometry_max',
        '--second_geometry_max',
        help='max threshold for second_geometry',
        default=None,
        type=float
    )

    parser.add_argument(
        '-second_geometry_min',
        '--second_geometry_min',
        help='min threshold for second_geometry',
        default=None,
        type=float
    )

    parser.add_argument(
        '-no_ex',
        '--no_export',
        help='Pass if you do not want to write out rf stats to csv.',
        action="store_true"
    )

    return parser.parse_args()


def _get_totcount(los_input, protein_atom_types=None):
    '''

    :param los_input:
    :param blacklist:
    :param mode:
    :return:
    '''

    los_input = los_input[los_input['los_atom_type'].isin(protein_atom_types)]
    totcount = los_input.groupby('molecule_name').size().to_dict()

    return totcount


def _calculate_hits_expected(it, identifiers, protein_atom_types, acount_df, scount_df, randomize=False,
                             random_samples=None, merged_acount_scount_df=None):
    '''
    :param it: dummy iterator required for multiprocessing
    :param identifiers: Binding site identifier
    :param acount_dict: Dictionary with protein atom type counts for each binding site.
    :param scount_dict: Dictionary with surface of each atom type for each binding site. Don't forget to normalize
     with totcount before passing it to this function.
    :param protein_atom_types: List of strings of protein_atom_types.
    :param randomize: Boolean, set to true for bootstrapping.
    :return: hits: Dictionary with total number of occurences of a protein_atom_type. expected: Dictionary with
     expected number of occurences for each protein atom type.
    '''
    if randomize:
        merged_acount_scount_df = merged_acount_scount_df[merged_acount_scount_df['identifier'].isin(identifiers)].sample(len(identifiers), replace=True)
        merged_acount_scount_df = merged_acount_scount_df.sum()

        hits = {}
        expected = {}

        for protein_atom_type_ in protein_atom_types:
            hits[protein_atom_type_] = merged_acount_scount_df[protein_atom_type_]
            expected[protein_atom_type_] = merged_acount_scount_df[f'surf_area_{protein_atom_type_}']

    else:
        acount_df_filtered = acount_df.sum()
        scount_df_filtered = scount_df.sum()

        hits = {}
        expected = {}

        for protein_atom_type_ in protein_atom_types:

            hits[protein_atom_type_] = acount_df_filtered[protein_atom_type_]
            expected[protein_atom_type_] = scount_df_filtered[f'surf_area_{protein_atom_type_}']

    return hits, expected


def _reweigh_scount(totcount, scount_df):
    scount_df = scount_df.join(pd.DataFrame.from_dict(totcount, orient='index', columns=['totcount']), on='identifier')
    scount_df['totcount'] = scount_df['totcount'].fillna(0)
    columns = [c for c in scount_df.columns if c not in ['identifier', 'totcount']]
    scount_df.loc[:, columns] = scount_df.loc[:, columns].mul(scount_df['totcount'], axis=0)

    return scount_df


class RfAtom(object):
    '''
    Calculate Rf values and statistics from los_filtered.csv.
    '''

    def __init__(self, input_path='', complex_input='complex_filtered.csv', los_input='los_filtered.csv',
                 max_vdw_dist=0.5, min_alpha_i=0, max_alpha_i=180, n_boot=500, watmap=False, output_path='rf.csv',
                 second_geometry_name=None, second_geometry_max=None, second_geometry_min=None, protein_atom_types=None,
                 no_export=False, mode='ligand', los_home='', protein_perspective_water=False,
                 angle_name='alpha_i'):
        '''
        Initialize LoSAnalysis
        :param input_path:
        :param complex_input: csv file or DataFrame
        :param los_input: csv file or DataFrame
        :param max_vdw_dist:
        :param min_alpha_i:
        :param max_alpha_i:
        :param n_boot:
        :param watmap:
        :param output_path:
        :param second_geometry_name:
        :param second_geometry_max:
        :param second_geometry_min:
        :param protein_atom_types:
        :param no_export:
        :param mode:


        Protein mode
        >>> complex_df = pd.DataFrame({'molecule_name': ['2P3T_010'], 'surf_area_other_ligands': [0], 'surf_area_other_central_ligand': [284.], 'surf_area_query_match': [15.],'surf_area_protein': [635]})
        >>> los_analyzer = RfAtom(complex_input=complex_df, mode='protein')
        >>> blacklist, db_all, db_list, scount_df = los_analyzer._calculate_scount()
        >>> np.testing.assert_almost_equal(scount_df[scount_df['identifier']=='2P3T_010']['surf_area_other_central_ligand'].values[0], 0.949833, decimal=2, verbose=True)
        >>> np.testing.assert_almost_equal(scount_df[scount_df['identifier']=='2P3T_010']['surf_area_query_match'].values[0], 0.0501672, decimal=2, verbose=True)
        >>> blacklist
        {}
        >>> db_all[0]
        '2P3T_010'
        >>> db_list['other_central_ligand']['2P3T_010']
        1
        >>> db_list['query_match']['2P3T_010']
        1

         Ligand mode
        >>> complex_df = pd.DataFrame({'molecule_name': ['6AOC_065'], 'surf_area_other_ligands': [86.9757345331545], 'surf_area_C_ali_apol': [0], 'surf_area_C_ali_weak_pol': [0], 'surf_area_C_ali_don': [0], 'surf_area_C_pi_carbonyl': [0], 'surf_area_C_pi_phenyl': [0], 'surf_area_C_pi_don': [12.3380312827159], 'surf_area_C_pi_neg': [3.76726386889833], 'surf_area_C_pi_pos': [0], 'surf_area_O_ali_mix': [0], 'surf_area_O_pi_mix': [0], 'surf_area_O_pi_acc': [0], 'surf_area_O_pi_acc_neg': [17.7664701562804], 'surf_area_N_don_pos': [0], 'surf_area_N_pi_don_pos': [0], 'surf_area_N_pi': [0], 'surf_area_N_pi_don': [0], 'surf_area_N_pi_mix': [10.7136749899675], 'surf_area_S_don': [0], 'surf_area_S_apol': [0], 'surf_area_metal': [0], 'surf_area_Water': [0], 'ligand_sasa': [215.619508302164]})
        >>> los_analyzer = RfAtom(complex_input=complex_df, mode='ligand')
        >>> blacklist, db_all, db_list, scount_df = los_analyzer._calculate_scount()
        >>> np.testing.assert_almost_equal(scount_df[scount_df['identifier']=='6AOC_065']['surf_area_other_ligands'].values[0], 0.66, decimal=2, verbose=True)
        >>> np.testing.assert_almost_equal(scount_df[scount_df['identifier']=='6AOC_065']['surf_area_C_ali_apol'].values[0], 0., decimal=2, verbose=True)
        >>> blacklist
        {}
        >>> db_all[0]
        '6AOC_065'
        >>> db_list['other_ligands']['6AOC_065']
        1
        >>> db_list['C_ali_apol']
        {}
        '''
        self.los_home = los_home
        self.input = input_path
        if type(complex_input) == str:
            self.complex = os.path.join(self.input, complex_input)
            self.complex_df = pd.read_csv(self.complex)
        else:
            self.complex_df = complex_input

        if type(complex_input) == str:
            self.los = os.path.join(self.input, los_input)
            self.los_df = pd.read_csv(self.los)
        else:
            self.los_df = los_input
        self.max_vdw_dist = max_vdw_dist
        self.min_alpha_i = min_alpha_i
        self.max_alpha_i = max_alpha_i
        self.n_boot = n_boot
        self.watmap = watmap
        self.protein_perspective_water = protein_perspective_water
        self.output = output_path
        self.second_geometry_name = second_geometry_name
        self.second_geometry_min = second_geometry_min
        self.second_geometry_max = second_geometry_max
        self.protein_atom_types = protein_atom_types
        self.no_export = no_export
        self.mode = mode
        if mode == 'ligand':
            atom_type_path = Path(atom_types.__path__[0])
            self.protein_atom_types = pd.read_csv(atom_type_path / 'protein_atom_types.csv', sep='\t')[
                    'protein_atom_type'].unique()
            self.surf_list = ['other_ligands', 'metal'] + list(self.protein_atom_types)
            self.a_list = self.surf_list.copy()
            if type(self.protein_atom_types) == list:
                self.protein_atom_types = self.protein_atom_types
            elif type(self.protein_atom_types) == str:
                self.protein_atom_types = self.protein_atom_types.split(',')
            else:
                self.protein_atom_types = None
            if self.watmap:
                self.surf_list.append('ligand_sasa')
                self.a_list.append('watmap')
                self.los = '_'.join([self.los.split('.')[0], 'with_wat.csv'])

        if mode == 'protein':
            self.surf_list = ['other_central_ligand', 'query_match']
            self.a_list = self.surf_list.copy()
            if type(self.protein_atom_types) == list:
                self.protein_atom_types = self.protein_atom_types
            elif type(self.protein_atom_types) == str:
                self.protein_atom_types = self.protein_atom_types.split(',')
            else:
                self.protein_atom_types = None

        if self.second_geometry_name is not None:
            self.two_dimensions = True
        else:
            self.two_dimensions = False

        if self.no_export:
            self.export = False
        else:
            self.export = True
        if angle_name == 'alpha':
            self.angle_name = 'alpha_i'
        else:
            self.angle_name = angle_name

    def _calculate_scount(self):
        """
        Calculate surface area ratios.
        :return: scount, acount, blacklist, db_all, db_list

        """

        scount_df = pd.DataFrame(columns=[f'surf_area_{protein_atom_type}' for protein_atom_type in self.a_list] + ['identifier']) # interaction surfaces of los atoms per atom type. F(db_id)
        s_temp = {}
        s_tot = {}  # total interaction surface of los protein atoms. F(db_id)
        db_list = {}  # set to 1 if non-zero contact surface for given atom type, db id
        blacklist = {}
        for i in self.a_list:
            db_list[i] = {}
        db_all = []  # all DB id's with non-zero interaction surface
        # read surface file
        for index, line in self.complex_df.iterrows():
            db_id = line['molecule_name']
            s_tot[db_id] = 0.0
            for cnt, protein_atom_type in enumerate(self.surf_list):
                if protein_atom_type != 'ligand_sasa':
                    s_tot[db_id] += float(line[f'surf_area_{protein_atom_type}'])
                    if float(line[f'surf_area_{protein_atom_type}']) > 0:
                        db_list[protein_atom_type][db_id] = 1

                else:
                    s_tot[db_id] += float(line[protein_atom_type])
                    if line.loc[protein_atom_type] > 0:
                        db_list['watmap'][db_id] = 1


                db_all.append(db_id)
            if s_tot[db_id] == 0.0:
                blacklist[db_id] = 1
                continue

            if self.watmap:
                s_temp[db_id] = line.loc[f'surf_area_{self.surf_list[0]}':f'surf_area_{self.surf_list[-2]}']
                s_temp[db_id] = s_temp[db_id].append(line.loc['ligand_sasa':'ligand_sasa'])
            else:
                s_temp[db_id] = line.loc[f'surf_area_{self.surf_list[0]}':f'surf_area_{self.surf_list[-1]}']

            scount_df = scount_df.append(s_temp[db_id].divide(s_tot[db_id]).append(pd.Series({'identifier': db_id})), ignore_index=True)
        db_all = list(set(db_all))
        return blacklist, db_all, db_list, scount_df

    def _update_acount_and_refkey(self, db_id, line, refkey, db_hit, acount_df):
        # count interactions per entry
        if db_id != refkey:
            sum_row = acount_df[acount_df['identifier'] == db_id].squeeze()[self.a_list] + line[self.a_list]
            acount_df.loc[acount_df['identifier'] == db_id, self.a_list] = sum_row[self.a_list].to_list()

            db_hit.append(db_id)
            refkey = db_id
        else:
            sum_row = acount_df[acount_df['identifier'] == db_id].squeeze()[self.a_list] + line[self.a_list]
            acount_df.loc[acount_df['identifier'] == db_id, self.a_list] = sum_row[self.a_list].to_list()

        return refkey, db_hit, acount_df

    # calculate confidence intervals by bootstrapping

    def _boots(self, data, protein_atom_type, num_samples, acount_df, scount_df):
        import numpy as np

        def run_los_multiprocessing():
            '''
            Multiprocessing Rf calculation for a single protein_atom_type.
            :return: a_sum: list with hits for each bootstrapping cycle. e_sum: list with expected values after
            each bootstrapping cycle.
            '''
            # Multiprocessing of hits
            calculate_hits_expected_parallel = partial(_calculate_hits_expected, identifiers=data,
                                                       protein_atom_types=[protein_atom_type],
                                                       acount_df=acount_df, scount_df=scount_df, randomize=True,
                                                       random_samples=None)
            pool = ProcessingPool(4)
            out = pool.map(calculate_hits_expected_parallel, range(num_samples))
            a_sum_ = [a[0][protein_atom_type] for a in out]  # List with a_temp for each bootstrapping loop
            e_sum_ = [e[1][protein_atom_type] for e in out]  # List with e_temp for each bootstrapping loop
            return a_sum_, e_sum_

        def run_single_processing():
            # test effect of expected on true rf
            # a_sum_ = []  # List with a_temp for each bootstrapping loop
            # e_sum_ = []  # List with e_temp for each bootstrapping loop
            # for random_samples in [10]:
            #     for n_sample in range(num_samples):
            #         a_temp, e_temp = _calculate_hits_expected('', data, acount_dict, scount_dict, [protein_atom_type],
            #                                                   randomize=True, random_samples=random_samples)
            #         a_sum_.append(a_temp[protein_atom_type])
            #         e_sum_.append(e_temp[protein_atom_type])
            # bootstrap_df = pd.DataFrame({'hits': a_sum_, 'expected': e_sum_,
            #                              'rf': [a / b for a, b in zip(a_sum_, e_sum_)]})
            # bootstrap_df.to_csv(f'{protein_atom_type}_samples.csv', index=False)

            # calculate true rf
            a_sum_ = []  # List with a_temp for each bootstrapping loop
            e_sum_ = []  # List with e_temp for each bootstrapping loop
            merged_acount_scount = acount_df.join(scount_df.set_index('identifier'), on='identifier')
            for n_sample in range(num_samples):
                a_temp, e_temp = _calculate_hits_expected('', data, [protein_atom_type],
                                                          acount_df, scount_df, randomize=True,
                                                          merged_acount_scount_df=merged_acount_scount)
                a_sum_.append(a_temp[protein_atom_type])
                e_sum_.append(e_temp[protein_atom_type])

            return a_sum_, e_sum_

        data = np.array(list(data))
        if num_samples >= 50000:
            print('Multiprocessing...')
            a_sum, e_sum = run_los_multiprocessing()
        else:
            a_sum, e_sum = run_single_processing()
        rf_boots = [a / b if b != 0 else np.nan for a, b in zip(a_sum, e_sum)]
        return rf_boots

    def _confidence_limits(self, data, protein_atom_type, num_samples, alpha, acount_df, scount_df):
        rf_boots = self._boots(data, protein_atom_type, num_samples, acount_df, scount_df)
        rf_sorted = np.sort(rf_boots, axis=0)
        return rf_sorted[int((alpha / 2.0) * num_samples)], rf_sorted[int((1 - alpha / 2.0) * num_samples)]

    def _print_rf(self, protein_atom_type, e_full, a_full, db_list, acount_df, scount_df):
        # a_full[protein_atom_type] is the number of contacts between the ligand query atom and the protein atom type
        rf = {}
        size = len(db_list[protein_atom_type].keys())  # The number of occurrences of a protein atom type within all
        # binding sites containing the ligand query atom

        if e_full[protein_atom_type] == 0.0 or size == 0:
            rf[protein_atom_type] = 'nan'
            low = 'nan'
            high = 'nan'
            effect = '?'

        else:
            rf[protein_atom_type] = a_full[protein_atom_type] / e_full[protein_atom_type]

            if size <= 50 or rf[protein_atom_type] == 'nan':  # consider minimum 50 entries to calculate Rf stats
                effect = '?'
                low = np.nan
                high = np.nan

            else:
                low, high = self._confidence_limits(db_list[protein_atom_type].keys(), protein_atom_type, self.n_boot,
                                                    0.05, acount_df, scount_df)
                if low > 1.0 and high > 1.0:
                    effect = '+'
                elif low < 1.0 and high < 1.0:
                    effect = '-'
                else:
                    effect = '0'

        output_df = pd.DataFrame(
            columns=['atom_type', 'size', 'hits', 'expected', 'rf', 'rf_low', 'rf_high', 'type'])
        output_df.loc[0, 'atom_type'] = protein_atom_type
        output_df.loc[0, 'size'] = size
        output_df.loc[0, 'hits'] = a_full[protein_atom_type]
        output_df.loc[0, 'expected'] = e_full[protein_atom_type]
        output_df.loc[0, 'rf'] = rf[protein_atom_type]
        output_df.loc[0, 'rf_low'] = low
        output_df.loc[0, 'rf_high'] = high
        output_df.loc[0, 'type'] = effect
        return output_df

    def _filter_los_input(self, los_input):
        """
        Filter los.csv input file by alpha and if passed, a second geometric constraint.
        :param los_input: los DataFrame
        :return: DataFrame with filtered los data
        """

        # filter out entries that are not within distance and angle range
        los_input = los_input[(los_input['vdw_distance'] <= self.max_vdw_dist) |
                              (self.watmap and los_input['res_name'] == 'SOL')]
        if not self.protein_perspective_water:
            los_input = los_input[
                ((self.min_alpha_i <= los_input[self.angle_name]) & (los_input[self.angle_name] <= self.max_alpha_i))
                | (self.watmap and los_input['res_name'] == 'SOL')]
        # drop duplicates if more than one angle within angle_range for the same contact atom
        los_input = los_input.drop_duplicates(subset=['res_name', 'molecule_name', 'distance', 'query_atom_id'],
                                              keep='last')
        if self.two_dimensions:
            query_name = f'{self.second_geometry_name}'
            los_input[query_name] = abs(los_input[query_name].astype(float))
            los_input = los_input[(self.second_geometry_min <= abs(los_input[query_name])) &
                                  (abs(los_input[query_name]) <= self.second_geometry_max)]
        return los_input

    def _get_rf_stats(self, e_full, a_full, db_list, acount_df, scount_df):
        """
        Call bootstrapping to get Rf statistics.
        :param e_full: Dictionary with expected values for each atom type.
        :param a_full: Dictionary with Line of sight occurences for each atom type.
        :param db_list: List of entries.
        :param acount: Dictionary of line of sight occurences for each entry.
        :param scount: Dictionary of expected values for each entry.
        :return: Write out csv file with Rf statistics, return DataFrame with Rf statistics.
        """
        if self.protein_atom_types is not None:
            list_of_atom_types_to_calculate_rf = self.protein_atom_types
        else:
            list_of_atom_types_to_calculate_rf = self.a_list

        output_df_all = pd.DataFrame()
        for protein_atom_type in list_of_atom_types_to_calculate_rf:
            output_df = self._print_rf(protein_atom_type, e_full, a_full, db_list, acount_df, scount_df)
            output_df_all = pd.concat([output_df_all, output_df])

        if 'alpha' in self.angle_name:
            output_df_all = output_df_all.assign(alpha_min=180 - self.max_alpha_i, alpha_max=180 - self.min_alpha_i)
        else:
            output_df_all[f'{self.angle_name}_max'] = self.max_alpha_i
            output_df_all[f'{self.angle_name}_min'] = self.min_alpha_i
        if self.second_geometry_name is not None:
            output_df_all = output_df_all.assign(h_min=self.second_geometry_min, h_max=self.second_geometry_max)
        if self.export:
            output_df_all.to_csv(self.output, index=False)
        return output_df_all

    def _get_rf_boots(self, db_list, acount_df, scount_df):
        '''

        :param db_list:
        :param acount:
        :param scount:
        :return:
        '''
        if self.protein_atom_types is not None:
            list_of_atom_types_to_calculate_rf = self.protein_atom_types
        else:
            list_of_atom_types_to_calculate_rf = self.a_list

        output_df_all = pd.DataFrame()
        for protein_atom_type in list_of_atom_types_to_calculate_rf:
            pseudo_rfs = self._boots(db_list[protein_atom_type].keys(), protein_atom_type, self.n_boot, acount_df,
                                     scount_df)
            output_df_all['pseudo_rf'] = pseudo_rfs
        if 'alpha' in self.angle_name:
            output_df_all = output_df_all.assign(alpha_min=180 - self.max_alpha_i, alpha_max=180 - self.min_alpha_i,
                                                 protein_atom_type=protein_atom_type)
        else:
            output_df_all[f'{self.angle_name}_min'] = self.min_alpha_i
            output_df_all[f'{self.angle_name}_max'] = self.max_alpha_i
            output_df_all[protein_atom_type] = protein_atom_type

        if self.second_geometry_name is not None:
            output_df_all = output_df_all.assign(h_min=self.second_geometry_min, h_max=self.second_geometry_max)
        output_df_all = output_df_all.reset_index()
        output_df_all = output_df_all.rename(columns={'index': 'boots_cycle'})
        return output_df_all

    def _count_los(self, los_input, blacklist, scount_df):
        acount_df = pd.DataFrame(columns=self.a_list + ['identifier'], index=scount_df.index).fillna(0)
        acount_df['identifier'] = scount_df['identifier']
        acount_df_sum = los_input[los_input['molecule_name'].isin(blacklist) == False][
            ['molecule_name', 'los_atom_type']
            ].rename(columns={'molecule_name': 'identifier'})
        if acount_df_sum.shape[0]:
            acount_df_sum = acount_df_sum.groupby('identifier')['los_atom_type'].value_counts().unstack().drop(columns=['0.0', 'same_ligand'], errors='ignore').\
                reset_index()

        # concatenate with entries that don't make any contact
        acount_df = pd.concat([acount_df_sum, acount_df[
            acount_df['identifier'].isin(acount_df_sum['identifier']) == False]], ignore_index=True).fillna(0)

        return acount_df

    def calculate_acount_scount(self):
        blacklist, db_all, db_list, scount_df = self._calculate_scount()
        los_df = self._filter_los_input(self.los_df)
        # convert all absolute values
        if self.second_geometry_name is not None:
            los_df[f'{self.second_geometry_name}'] = los_df[f'{self.second_geometry_name}'].apply(abs)
        acount_df = self._count_los(los_df, blacklist, scount_df)
        totcount = _get_totcount(los_df, self.a_list)
        scount_df = _reweigh_scount(totcount, scount_df)
        return db_all, db_list, acount_df, scount_df

    def calculate_rf(self):
        """
        Run Rf calculation with bootstrapping.
        :return: DataFrame with Rf values and 95% confidence interval.
        """
        db_all, db_list, acount_df, scount_df = self.calculate_acount_scount()
        a_full, e_full = _calculate_hits_expected('', db_all, self.a_list, acount_df, scount_df)
        rf_stats = self._get_rf_stats(e_full, a_full, db_list, acount_df, scount_df)
        return rf_stats

    def calculate_pseudo_rf(self):
        '''

        :return:
        '''
        db_all, db_list, acount_df, scount_df = self.calculate_acount_scount()
        pseudo_rfs = self._get_rf_boots(db_list, acount_df, scount_df)
        return pseudo_rfs


def main():
    args = parse_args()
    los_a = RfAtom(second_geometry_name=args.second_geometry_name, second_geometry_max=args.second_geometry_max,
                   second_geometry_min=args.second_geometry_min, complex_input=args.complex,
                   input_path=args.input, los_input=args.los, max_alpha_i=args.max_alpha_i,
                   min_alpha_i=args.min_alpha_i, max_vdw_dist=args.max_vdw_dist, n_boot=args.n_boot,
                   no_export=args.no_export, output_path=args.output, watmap=args.watmap,
                   protein_atom_types=args.protein_atom_types, mode=args.mode)
    rf_stats_dict = los_a.calculate_rf()
    print(rf_stats_dict)


if __name__ == "__main__":
    freeze_support()
    main()

########################################################################################################################
