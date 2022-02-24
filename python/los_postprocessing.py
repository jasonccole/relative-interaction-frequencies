########################################################################################################################

import __future__
import os
from glob import glob
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

from ccdc_roche.python import los_analysis, p2cq_filter, rf_plotter
from ccdc_roche import atom_types

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
        help='Path to input folder containing campaign data.',
        default=os.getcwd()
    )

    parser.add_argument(
        '-search',
        '-s',
        '--search',
        help='Path to folder from a specific search within a campaign.',
        default=None
    )

    parser.add_argument(
        '-sq',
        '-strq',
        '--structure_quality',
        help='Path to structure quality file.',
        default='structure_quality_ligand_rscc_avgoccu_1.1.csv'
    )

    parser.add_argument(
        '-angle',
        '--angle',
        help='If angle argument is passed, angle dependent Rf values will be calculated.',
        action="store_true"
    )

    parser.add_argument(
        '-wm',
        '--watmap',
        help='If watmap argument is passed, the watmap column will be included in the calculations',
        action="store_true"
    )

    parser.add_argument(
        '-gw',
        '--generate_watermaps',
        help='If generate_watermaps argument is passed, the watermap thresholds will be calculated.',
        action="store_true"
    )

    parser.add_argument(
        '-2d',
        '-2D',
        '--two_dimensional',
        help='Calculate 2D Rf values for two geometric attributes.',
        action="store_true"
    )

    parser.add_argument(
        '-m',
        '--mode',
        help='ligand or protein mode',
        default='ligand',
        type=str
    )

    parser.add_argument(
        '--second_geometry_name',
        help='Pass to activate 2D Rf analysis. Specify the name of the query. E.g. Cl_amide',
        default=None,
        type=str
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
        '--protein_atom_types',
        help='If you want to calculate Rf only for one protein atom type.',
        default=None,
        type=str
    )

    parser.add_argument(
        '--los_home',
        help='Path to folder that contains databases.',
        default=None,
        type=str
    )

    return parser.parse_args()


def distance_dependency(second_geometry_name, mode, los_home, ligand_atom_type, bin_size=0.5, bins=None,
                        filename_ext='', los_input='los_filtered.csv', complex_input='complex_filtered.csv'):
    """
    Wrapper that calculates Rf values for attractive and repulsive ranges. Assign unique ligand, protein and binding
     site counts.
    :return: csv file with angle ranges for each atom type, their Rf value with 95% confidence and unique ligand,
     protein and binding site counts.
    """
    if not bins:
        bins = np.arange(0, 3.5, bin_size)

    statistics_df = pd.DataFrame()
    for min_dist in bins:
        max_dist = min_dist + bin_size
        output = f'statistics_h_{bin_size}_{str(max_dist)}.csv'
        if mode == 'protein':
            protein_atom_types = 'query_match'
        else:
            protein_atom_types = None
        rf_analyzer = los_analysis.RfAtom(second_geometry_min=min_dist, second_geometry_max=max_dist,
                                          second_geometry_name=second_geometry_name,
                                          output_path=output,
                                          n_boot=500, mode=mode, protein_atom_types=protein_atom_types,
                                          no_export=True, los_home=los_home, los_input=los_input, complex_input=complex_input)
        statistics_df = statistics_df.append(rf_analyzer.calculate_rf())
    statistics_df = statistics_df.assign(ligand_atom_type=ligand_atom_type)
    statistics_df.to_csv(f'statistics_h{filename_ext}.csv', index=False)


def _filter_vdw_distance(contacts_to_keep='shortest'):
    import math

    if contacts_to_keep == 'shortest':
        ascending = True
    elif contacts_to_keep == 'longest':
        ascending = False

    def _drop_half(_df):
        _df = _df.sort_values(by='vdw_distance', ascending=ascending).reset_index()
        rows_to_keep = math.ceil(_df.shape[0] / 2)
        _df = _df.loc[:rows_to_keep-1, :]
        return _df
    import shutil
    shutil.copy('los_filtered.csv', 'los_filtered_.csv')
    df = pd.read_csv('los_filtered_.csv')
    sorted_df = df.groupby(['molecule_name', 'substructure_match'])
    sorted_df = sorted_df.apply(lambda x: _drop_half(x))
    sorted_df.reset_index(drop=True)
    sorted_df.to_csv('los_filtered.csv')


class Postprocessing(object):

    def __init__(self, db, input_folder='output', los_home='', structure_quality_file='structure_quality.csv',
                 angle=False, watmap=False, generate_watermaps=False, search=None, two_dimensional=False,
                 second_geometry_name=None, protein_atom_types=None, mode='ligand',
                 smarts_filters=None, protein_atom_types_file='protein_atom_types.csv',
                 ligand_atom_types_file='ligand_atom_types.csv',
                 angle_name = 'alpha'):
        self.input = input_folder
        self.los_home = los_home
        self.db = db
        self.structure_quality = structure_quality_file
        self.angle = angle
        self.watmap = watmap
        self.generate_watermaps = generate_watermaps
        self.search = search
        self.two_dim = two_dimensional
        self.second_geometry_name = second_geometry_name

        atom_type_path = Path(atom_types.__path__[0])
        self.protein_atom_types = pd.read_csv(atom_type_path / 'protein_atom_types.csv', sep='\t')
        self.protein_atom_types = list(self.protein_atom_types['protein_atom_type'].unique()) + [
            'other_ligands', 'metal']

        self.mode = mode
        self.smarts_filters = smarts_filters

        atom_type_path = Path(atom_types.__path__[0])
        ligand_atom_types_df = pd.read_csv(atom_type_path / 'ligand_atom_types.csv', sep='\t')

        self.ligand_atom_type = os.getcwd().split('/')[-1]
        self.rdkit_smarts = ligand_atom_types_df[ligand_atom_types_df['ligand_atom_type'] == self.ligand_atom_type][
            'RDKit_SMARTS'].to_numpy()[0]
        self.rdkit_smarts_index = \
            ligand_atom_types_df[ligand_atom_types_df['ligand_atom_type'] == self.ligand_atom_type][
                'RDKit_SMARTS_index'].to_numpy()[0]

        self.angle_name = angle_name

    def call_p2cq_filter(self, input_folder, resolution_thr=2.5, rscc_thr=0.8, output_extension='_filtered.csv'):
        """
        Execute p2cq_filter.
        :param resolution_thr: Consider only structures with a resolution <= 2.5.
        :param rscc_thr: Consider only structures with an RSCC >= 0.8.
        :return: Calls p2cq_filter.main()
        """
        filter = p2cq_filter.P2cqFilter(input=input_folder, resolution=resolution_thr, rscc=rscc_thr,
                                               structure_quality=self.structure_quality, filter_out_cofactor=True,
                                               mode=self.mode, output_extension=output_extension,
                                               los_home=self.los_home)
        filter.filter_files()

    def angle_dependency(self, n_boot=500, los_input='los_filtered.csv', complex_input='complex_filtered.csv',
                         filename_ext=''):
        """
        Wrapper that calculates Rf values for attractive and repulsive ranges. Assign unique ligand, protein and binding
         site counts.
        :return: csv file with angle ranges for each atom type, their Rf value with 95% confidence and unique ligand,
         protein and binding site counts.
        """
        statistics_df = pd.DataFrame()
        for min_alpha_i in range(0, 180, 20):
            max_alpha_i = min_alpha_i + 20
            output = ''.join([f'statistics_{self.angle_name}_', str(180 - min_alpha_i), '.csv'])
            if self.mode == 'protein':
                protein_atom_types = 'query_match'
            else:
                protein_atom_types = None
            rf_analyzer = los_analysis.RfAtom(min_alpha_i=min_alpha_i, max_alpha_i=max_alpha_i, output_path=output,
                                              n_boot=n_boot, mode=self.mode, protein_atom_types=protein_atom_types,
                                              no_export=True, los_home=self.los_home, los_input=los_input,
                                              complex_input=complex_input, angle_name=self.angle_name)
            statistics_df = statistics_df.append(rf_analyzer.calculate_rf())
        statistics_df = statistics_df.assign(RDKit_SMARTS=self.rdkit_smarts, RDKit_SMARTS_index=self.rdkit_smarts_index,
                                             ligand_atom_type=self.ligand_atom_type)
        statistics_df.to_csv(f'statistics_{self.angle_name}{filename_ext}.csv', index=False)

    def alpha_max(self, rf_angle_range_df, rf_values_p):
        """
        Get the alpha_i with maximum Rf value for a given range. Defined as midpoint between the two alpha values with
         highest Rf. alpha_max is appended to the input DataFrame and the updated DataFrame is returned.
        :param rf_angle_range_df:
        :param rf_values_p:
        :return: DataFrame with 'max_rf_alpha_i' column.
        """
        rf_values_p = pd.read_csv(rf_values_p)
        for index, row in rf_angle_range_df.iterrows():
            if row['type'] != '+':
                continue
            alpha_i_max = int(row['alpha_i_max'])
            alpha_i_min = int(row['alpha_i_min'])
            protein_atom_type = row['atom_type']
            full_row = rf_values_p[rf_values_p['atom_type'] == protein_atom_type]
            rf_in_range = full_row.loc[:, str(180 - alpha_i_min):str(180 - alpha_i_max + 10)]
            max_rf_alpha = rf_in_range.idxmax(axis=1).values[0]
            max_rf_alpha_index = full_row.columns.get_loc(max_rf_alpha)
            max_rf_alpha_neighbour_1 = full_row.iloc[:, max_rf_alpha_index - 1].name

            # catch error if at limit of Data Frame
            try:
                max_rf_alpha_neighbour_2 = full_row.iloc[:, max_rf_alpha_index + 1].name
            except IndexError:
                alpha_i_max = 180 - 5
                rf_angle_range_df.loc[index, 'max_rf_alpha_i'] = alpha_i_max
                continue

            # Calculate midpoint
            if full_row[max_rf_alpha_neighbour_1].values[0] >= full_row[max_rf_alpha_neighbour_2].values[0]:
                alpha_max = (float(max_rf_alpha) + float(full_row[max_rf_alpha_neighbour_1].name)) / 2
            else:
                alpha_max = (float(max_rf_alpha) + float(full_row[max_rf_alpha_neighbour_2].name)) / 2

            rf_angle_range_df.loc[index, 'max_rf_alpha_i'] = 180 - alpha_max
        return rf_angle_range_df

    def alpha_min(self, rf_angle_range_df, rf_values_p):
        rf_values_p = pd.read_csv(rf_values_p)
        for index, row in rf_angle_range_df.iterrows():
            if row['type'] != '-':
                continue
            alpha_i_max = int(row['alpha_i_max'])
            alpha_i_min = int(row['alpha_i_min'])
            protein_atom_type = row['atom_type']
            full_row = rf_values_p[rf_values_p['atom_type'] == protein_atom_type]
            rf_in_range = full_row.loc[:, str(180 - alpha_i_min):str(180 - alpha_i_max)]
            min_rf_alpha = rf_in_range.idxmin(axis=1).values[0]
            min_rf_alpha_index = full_row.columns.get_loc(min_rf_alpha)
            min_rf_alpha_neighbour_1 = full_row.iloc[:, min_rf_alpha_index - 1].name
            min_rf_alpha_neighbour_2 = full_row.iloc[:, min_rf_alpha_index + 1].name

            # Calculate midpoint
            if full_row[min_rf_alpha_neighbour_1].values[0] <= full_row[min_rf_alpha_neighbour_2].values[0]:
                alpha_min = (float(min_rf_alpha) + float(full_row[min_rf_alpha_neighbour_1].name)) / 2
            else:
                alpha_min = (float(min_rf_alpha) + float(full_row[min_rf_alpha_neighbour_2].name)) / 2

            rf_angle_range_df.loc[index, 'alpha_i_min'] = 180 - alpha_min
        return rf_angle_range_df

    def two_dimensional_angle_angle_rf(self, second_geometry_name, protein_atom_types):
        """
        Calculate Rf values as a function of alpha and a second angle.
        :param second_geometry_name: Name of the second geometry
        :param protein_atom_types: Restrain Rf calculation to a list of protein atom types to reduce calculation times.
        :return:
        """
        for min_angle_2 in range(0, 180, 10):
            max_angle_2 = min_angle_2 + 10
            for min_alpha_i in range(0, 180, 10):
                max_alpha_i = min_alpha_i + 10
                output = f'statistics_{str(180 - min_alpha_i)}_{second_geometry_name}_{min_angle_2}_{max_angle_2}.csv'
                rf_analyzer = los_analysis.RfAtom(min_alpha_i=min_alpha_i, max_alpha_i=max_alpha_i,
                                                  output_path=output, n_boot=500,
                                                  second_geometry_name=second_geometry_name,
                                                  second_geometry_max=max_angle_2,
                                                  second_geometry_min=min_angle_2,
                                                  protein_atom_types=protein_atom_types,
                                                  mode=self.mode)
                rf_analyzer.calculate_rf()

    def two_dimensional_angle_distance_rf(self, second_geometry_name):
        """
        Calculate Rf values as a function of alpha and a distance. Distance range is 0 to 4 Angstrom with bin size 0.5.
        :param second_geometry_name: Name of the second geometry
        :param protein_atom_types: Restrain Rf calculation to a list of protein atom types to reduce calculation times.
        :return:
        """
        statistics_df = pd.DataFrame()
        if self.mode == 'protein':
            protein_atom_types = 'query_match'
        else:
            protein_atom_types = None
        for min_distance in np.arange(0, 5, 0.25):
            max_distance = min_distance + 0.25
            for min_alpha_i in range(0, 180, 10):
                max_alpha_i = min_alpha_i + 10
                output = f'statistics_{str(180 - min_alpha_i)}_{second_geometry_name}_{min_distance}_{max_distance}.csv'
                rf_analyzer = los_analysis.RfAtom(min_alpha_i=min_alpha_i, max_alpha_i=max_alpha_i,
                                                  output_path=output,
                                                  n_boot=500, second_geometry_name=second_geometry_name,
                                                  second_geometry_max=max_distance,
                                                  second_geometry_min=min_distance,
                                                  protein_atom_types=protein_atom_types, mode=self.mode,
                                                  no_export=True)
                statistics_df = statistics_df.append(rf_analyzer.calculate_rf())
        statistics_df.to_csv('statistics_alpha_h.csv', index=False)

    def calculate_preferred_angle_rf(self, interaction_type='attractive', n_boot=500):
        """
        Calculate attractive or repulsive angle ranges.
        :param interaction_type: 'attractive' or 'repulsive'
        :param n_boot: Number of bootstrapping cycles.
        :return: DataFrame with angle ranges.
        """
        preferred_angles = pd.read_csv(f'pref_angle_range_{interaction_type}.csv')
        preferred_dfs = [pd.DataFrame()]
        for row in preferred_angles.iterrows():
            row = row[1]
            pat = row['contact_type']
            for pref_range in range(1, 5):
                max_alpha_i = 180 - row[f'alpha_range_{pref_range}_min']
                min_alpha_i = 180 - row[f'alpha_range_{pref_range}_max']
                if not pd.isnull(min_alpha_i) and not pd.isnull(max_alpha_i):
                    output = f'range_statistics_{pat}_{min_alpha_i}_{max_alpha_i}_.csv'
                    rf_analyzer = los_analysis.RfAtom(min_alpha_i=min_alpha_i, max_alpha_i=max_alpha_i,
                                                      output_path=output, protein_atom_types=pat, no_export=True,
                                                      n_boot=n_boot, mode=self.mode)
                    preferred_df = rf_analyzer.calculate_rf()
                    preferred_df.loc[0, 'alpha_i_max'] = max_alpha_i
                    preferred_df.loc[0, 'alpha_i_min'] = min_alpha_i
                    preferred_dfs.append(preferred_df)
        rf_angle_range_df = pd.concat(preferred_dfs)
        return rf_angle_range_df


    def plot_protein_angle_ranges(self):
        rf_angle_range_df_with_hitlist_list = []
        for folder in self.protein_atom_types:
            try:
                rf_angle_range_df_with_hitlist = pd.read_csv(os.path.join(folder, 'rf_angle_range_df_with_hitlist.csv'))
            except FileNotFoundError:
                rf_angle_range_df_with_hitlist = pd.DataFrame(index=[0])
            if len(rf_angle_range_df_with_hitlist.index) == 0:
                for column in rf_angle_range_df_with_hitlist.columns:
                    rf_angle_range_df_with_hitlist.loc[0, column] = np.nan
            rf_angle_range_df_with_hitlist.loc[:, 'atom_type'] = folder
            rf_angle_range_df_with_hitlist_list.append(rf_angle_range_df_with_hitlist)
        try:
            rf_angle_range_df_with_hitlist_combined = pd.concat(rf_angle_range_df_with_hitlist_list, sort=False)
            rf_plotter.favorable_unfavorable_angle_ranges_heatmap(rf_angle_range_df_with_hitlist_combined)
        except ValueError:
            print('No data available, plot cannot be generated.')

    def plot_protein_rf_bars(self, input_path='rf.csv', output_path='protein_rf'):
        rf_df_list = []
        for folder in self.protein_atom_types:
            try:
                rf_df = pd.read_csv(os.path.join(folder, input_path))
                rf_df = rf_df.drop(rf_df[rf_df['atom_type'] == 'other_central_ligand'].index)
            except FileNotFoundError:
                rf_df = pd.DataFrame(index=[0])
            if len(rf_df.index) == 0:
                for column in rf_df.columns:
                    rf_df.loc[0, column] = np.nan
            rf_df.loc[:, 'atom_type'] = folder
            rf_df_list.append(rf_df)
        try:
            protein_rf_df = pd.concat(rf_df_list, sort=False)
            protein_rf_df.to_csv(f'protein_{input_path}', index=False)
            rf_plotter.rf_bar_plot(protein_rf_df, output_path)
        except ValueError:
            print('No data available, plot cannot be generated.')

        try:
            ligand_rf_df = pd.read_csv(os.path.join('query_atom', 'rf.csv'))
            ligand_rf_df.loc[:, 'perspective'] = 'ligand'
            protein_rf_df.loc[:, 'perspective'] = 'protein'
            both_rf_df = pd.concat([protein_rf_df, ligand_rf_df])
            both_rf_df = both_rf_df[(both_rf_df['atom_type'] != 'other_ligands') &
                                    (both_rf_df['atom_type'] != 'don') &
                                    (both_rf_df['atom_type'] != 'pi') &
                                    (both_rf_df['atom_type'] != 'acc') &
                                    (both_rf_df['atom_type'] != 'pos') &
                                    (both_rf_df['atom_type'] != 'neg') &
                                    (both_rf_df['atom_type'] != 'apol')]
            rf_plotter.rf_bar_plot(both_rf_df, title='rf_protein_ligand_barplot.png', hue='perspective')

        except FileNotFoundError:
            print('Comparison plot not possible.')

    # def calculate_watermap(self, resolution_thr=1.5, rscc_thr=0.8):
    #     """
    #     Get Surface Area thresholds for Watermaps.
    #     :param resolution_thr: Only consider entries with resolution <= resolution_thr
    #     :param rscc_thr: Only consider entries with resolution >= rscc_thr
    #     :return:
    #     """
    #     stripped_path = os.path.join(self.input, r'without_explicit_water\output')
    #     explicit_path = os.path.join(self.input, r'with_explicit_water\output_kw')
    #     folder_names = next(os.walk(explicit_path))
    #     for folder in folder_names:
    #         os.chdir(os.path.join(self.input, folder))
    #         print(folder)
    #         print('Filtering data...')
    #         self.call_p2cq_filter(resolution_thr, rscc_thr)
    #         print('Done filtering.')
    #         print('Calculating Watermap...')
    #         water_analysis.main(['-s', stripped_path, '-e', explicit_path, '-at', folder])
    #         print('Done calculating Watermap.')
    #         os.chdir(self.input)

    def generate_protein_lookup_file(self, geometry='alpha'):
        folders = next(os.walk('.'))[1]
        ligand_lookup = pd.DataFrame()
        protein_lookup = pd.DataFrame()
        for folder in folders:
            os.chdir(folder)
            try:
                for bin_file in glob(f'statistics_{geometry}.csv'):
                    bin_df = pd.read_csv(bin_file)
                    if folder not in self.protein_atom_types:
                        ligand_lookup = ligand_lookup.append(bin_df, sort=True)
                    else:
                        bin_df = bin_df.assign(atom_type=folder)
                        protein_lookup = protein_lookup.append(bin_df, sort=True)
            except FileNotFoundError:
                os.chdir('..')
                continue
            os.chdir('..')
        protein_lookup = protein_lookup.assign(SMARTS=self.rdkit_smarts, SMARTS_index=self.rdkit_smarts_index)
        protein_lookup.to_csv(f'protein_lookup_{geometry}.csv', index=False)

        print('Rf values were combined')

    def run(self):

        self.calculate_rf(angle=self.angle, second_geometry_name=self.second_geometry_name,
                          watmap=self.watmap, two_dim=self.two_dim, protein_atom_types=self.protein_atom_types,
                              )


class ProteinFilter(Postprocessing):
    def __init__(self, db, input_folder, los_home, smarts_filters,
                 protein_atom_type,
                 structure_quality_file,
                 angle=False,
                 watmap=False, generate_watermaps=False, search=None,
                 two_dimensional=False,
                 ):
        super().__init__(db, input_folder=input_folder, los_home=los_home,
                         structure_quality_file=structure_quality_file,
                         angle=angle, watmap=watmap, generate_watermaps=generate_watermaps, search=search,
                         two_dimensional=two_dimensional,
                         second_geometry_name='h', mode='protein',
                         smarts_filters=smarts_filters,
                         protein_atom_types_file='protein_atom_types.csv')

        self.protein_atom_type = protein_atom_type

    def run(self):
        cwd = os.getcwd()
        root_dir = Path(self.input) / Path(self.protein_atom_type)
        os.chdir(root_dir)
        print('Filtering data...')
        self.call_p2cq_filter(self.protein_atom_type)
        print('Done filtering.')

        if self.protein_atom_type == 'C_ali_apol':
            # update entry count in ligand_atom_type.csv
            ligand_atom_type = Path().resolve().parents[0].stem
            ligand_atom_types = pd.read_csv(os.path.join(self.los_home, 'ligand_atom_types.csv'), sep='\t')
            if type(self.db) == list:
                occurrences_label = f'combined_C_ali_apol_occurrences_filtered'
            else:
                dbname = self.db.split('.')[0]
                occurrences_label = f'{dbname}_C_ali_apol_occurrences_filtered'
            occurrences = pd.read_csv('complex_filtered.csv').shape[0]
            ligand_atom_types.loc[
                ligand_atom_types['ligand_atom_type'] == ligand_atom_type, occurrences_label] = occurrences
            ligand_atom_types.to_csv(os.path.join(self.los_home, 'ligand_atom_types.csv'), sep='\t',
                                     index=False)
        os.chdir(cwd)


class ProteinDistance(Postprocessing):
    def __init__(self, db, input_folder, los_home, smarts_filters,
                 protein_atom_type,
                 structure_quality_file,
                 angle=False,
                 watmap=False, generate_watermaps=False, search=None,
                 two_dimensional=False,
                 ):
        super().__init__(db, input_folder=input_folder, los_home=los_home,
                         structure_quality_file=structure_quality_file,
                         angle=angle, watmap=watmap, generate_watermaps=generate_watermaps, search=search,
                         two_dimensional=two_dimensional,
                         second_geometry_name='h', mode='protein',
                         smarts_filters=smarts_filters,
                         protein_atom_types_file='protein_atom_types.csv')

        self.protein_atom_type = protein_atom_type

    def run(self):
        root_dir = os.path.join(self.input)
        os.chdir(root_dir)
        if 'pi' not in self.protein_atom_type:
            os.chdir(self.protein_atom_type)
            try:
                os.remove('statistics_h.csv')
            except FileNotFoundError:
                pass
            os.chdir('..')

        else:
            os.chdir(self.protein_atom_type)
            try:
                print('Calculating distance dependent Rf...')
                distance_dependency('h', self.mode, self.los_home, self.ligand_atom_type)
            except (pd.errors.EmptyDataError, KeyError):
                print('No data available.')
            os.chdir('..')

        self.generate_protein_lookup_file(geometry='h')
        rf_plotter.plot_protein_geometry_bins(input='protein_lookup_h.csv', geometry='h', mode=self.mode,
                                              smarts_filters=self.smarts_filters, smarts_index=self.rdkit_smarts_index,
                                              title=self.rdkit_smarts)


class ProteinAngle(Postprocessing):

    def __init__(self, db, input_folder, los_home, smarts_filters, structure_quality_file, protein_atom_type,
                 search=None):
        '''

        :param input_folder:
        :param los_home:
        :param smarts_filters:
        :param structure_quality_file:
        :param search:
        :param protein_atom_types:
        '''
        super().__init__(db=db, input_folder=input_folder, los_home=los_home,
                         structure_quality_file=structure_quality_file,
                         angle=True, search=search, mode='protein',
                         smarts_filters=smarts_filters,
                         protein_atom_types_file='protein_atom_types.csv')
        self.protein_atom_type = protein_atom_type

    def run(self):
        root_dir = os.path.join(self.input)
        os.chdir(root_dir)
        os.chdir(self.protein_atom_type)
        try:
            protein_perspective_water = False
            if self.protein_atom_type == 'Water':
                protein_perspective_water = True
            rf_analyzer = los_analysis.RfAtom(mode=self.mode, n_boot=500, los_home=self.los_home,
                                              protein_perspective_water=protein_perspective_water)
            rf_analyzer.calculate_rf()
            print('Calculating angle dependent Rf...')
            self.angle_dependency()
            if self.protein_atom_type == 'Water':
                statistics_df = pd.read_csv('statistics_alpha.csv')
                rf_df = pd.read_csv('rf.csv')
                statistics_df['rf'] = rf_df[rf_df['atom_type'] == 'query_match']['rf'].to_numpy()[0]
                statistics_df['rf_low'] = rf_df[rf_df['atom_type'] == 'query_match']['rf_low'].to_numpy()[0]
                statistics_df['rf_high'] = rf_df[rf_df['atom_type'] == 'query_match']['rf_high'].to_numpy()[0]
                statistics_df['expected'] = rf_df[rf_df['atom_type'] == 'query_match']['expected'].to_numpy()[0]
                statistics_df['hits'] = rf_df[rf_df['atom_type'] == 'query_match']['hits'].to_numpy()[0]
                statistics_df.to_csv('statistics_alpha.csv', index=False)
            print('Done calculating and plotting Rf.')

        except (pd.errors.EmptyDataError, KeyError):
            print('No data available.')
        os.chdir('..')
        self.plot_protein_rf_bars()
        self.generate_protein_lookup_file(geometry='alpha')
        rf_plotter.plot_protein_geometry_bins(input='protein_lookup_alpha.csv', mode=self.mode, title=self.rdkit_smarts,
                                              smarts_filters=self.smarts_filters, smarts_index=self.rdkit_smarts_index)


class LigandFilter(Postprocessing):

    def __init__(self, db, input_folder, los_home, smarts_filters, structure_quality_file, search=None,
                 protein_atom_types=None):
        super().__init__(db=db, input_folder=input_folder, los_home=los_home,
                         structure_quality_file=structure_quality_file,
                         angle=True, search=search,
                         protein_atom_types=protein_atom_types, mode='ligand',
                         protein_atom_types_file='protein_atom_types.csv')

    def run(self):
        root_dir = os.path.join(self.input, 'query_atom')
        os.chdir(root_dir)
        print('Filtering data...')
        self.call_p2cq_filter('query_atom')
        print('Done filtering.')

        # update entry count in ligand_atom_type.csv
        ligand_atom_type = Path().resolve().parents[0].stem
        if (Path(self.los_home) / 'ligand_atom_types.csv').is_file():
            ligand_atom_types = pd.read_csv(Path(self.los_home) / 'ligand_atom_types.csv', sep='\t')
        else:
            atom_type_path = Path(atom_types.__path__[0])
            ligand_atom_types = pd.read_csv(atom_type_path / 'ligand_atom_types.csv', sep='\t')

        if type(self.db) == list:
            occurrences_label = f'combined_occurrences_filtered'
        else:
            dbname = self.db.split('.')[0]
            occurrences_label = f'{dbname}_occurrences_filtered'
        occurrences = pd.read_csv('complex_filtered.csv').shape[0]
        ligand_atom_types.loc[
            ligand_atom_types['ligand_atom_type'] == ligand_atom_type, occurrences_label] = occurrences
        ligand_atom_types.to_csv(os.path.join(self.los_home, 'ligand_atom_types.csv'), sep='\t',
                                 index=False)


class LigandAngle(Postprocessing):

    def __init__(self, db, input_folder, los_home, smarts_filters, structure_quality_file, search=None,
                 protein_atom_types=None, angle_name='alpha'):
        super().__init__(db=db, input_folder=input_folder, los_home=los_home,
                         structure_quality_file=structure_quality_file,
                         angle=True, search=search,
                         protein_atom_types=protein_atom_types, mode='ligand',
                         smarts_filters=smarts_filters,
                         protein_atom_types_file='protein_atom_types.csv',
                         angle_name=angle_name)

    def run(self):
        root_dir = os.path.join(self.input, 'query_atom')
        os.chdir(root_dir)
        try:
            if self.angle_name == 'alpha':  # Calculate geometry independet RF only once
                rf_analyzer = los_analysis.RfAtom(mode=self.mode, n_boot=500, los_home=self.los_home)
                rf_analyzer.calculate_rf()
                rf_plotter.rf_bar_plot('rf.csv', 'query_atom')

            self.angle_dependency()

            try:
                rf_plotter.plot_protein_geometry_bins(input=f'statistics_{self.angle_name}.csv', expected_threshold=10,
                                                      geometry=self.angle_name, mode=self.mode, title=self.rdkit_smarts,
                                                      smarts_index=self.rdkit_smarts_index)

                ligand_lookup = pd.read_csv(f'statistics_{self.angle_name}.csv')
                ligand_lookup.to_csv(f'../ligand_lookup_{self.angle_name}.csv', index=False)

            except FileNotFoundError:
                print('Plot cannot be generated.')

            print('Done calculating and plotting Rf.')

        except (pd.errors.EmptyDataError, KeyError):
            print('No data available.')
        os.chdir('..')


class LigandDistance(Postprocessing):
    def __init__(self, db, input_folder, los_home, smarts_filters, second_geometry_name,
                 structure_quality_file, search=None,
                 protein_atom_types=None, pi_atom=True):
        self.pi_atom = pi_atom
        super().__init__(db, input_folder=input_folder, los_home=los_home,
                         structure_quality_file=structure_quality_file,
                         angle=False, search=search,
                         second_geometry_name=second_geometry_name, protein_atom_types=protein_atom_types,
                         mode='ligand',
                         smarts_filters=smarts_filters,
                         protein_atom_types_file='protein_atom_types.csv')

    def run(self):
        if self.pi_atom:
            root_dir = os.path.join(self.input, 'query_atom')
            os.chdir(root_dir)
            print('Calculating distance dependent Rf...')
            try:
                distance_dependency(self.second_geometry_name, self.mode, self.los_home, self.ligand_atom_type)
            except KeyError:
                print(f'No geometry with name {self.second_geometry_name}.')

            try:
                rf_plotter.plot_protein_geometry_bins(input='statistics_h.csv', expected_threshold=10, geometry='h',
                                                      title=self.rdkit_smarts, smarts_filters=self.smarts_filters,
                                                      smarts_index=self.rdkit_smarts_index,
                                                      mode=self.mode)
                ligand_lookup = pd.read_csv('statistics_h.csv')
                ligand_lookup = ligand_lookup[ligand_lookup['expected'] > 0]
                ligand_lookup.to_csv('../ligand_lookup_h.csv', index=False)

            except FileNotFoundError:
                print('Plot cannot be generated.')
            except (pd.errors.EmptyDataError, KeyError):
                print('No data available.')
            os.chdir('..')
        # delete all traces of a faulty distance to pi plane calculation
        else:
            try:
                os.remove('ligand_lookup_h.csv')
            except FileNotFoundError:
                pass
            try:
                os.remove(os.path.join('query_atom', 'statistics_h.csv'))
            except FileNotFoundError:
                pass


def main():
    args = parse_args()
    batch = Postprocessing(input_folder=args.input, los_home=args.los_home,
                           structure_quality_file=args.structure_quality, angle=args.angle,
                           watmap=args.watmap, generate_watermaps=args.generate_watermaps, search=args.search,
                           two_dimensional=args.two_dimensional, protein_atom_types=args.protein_atom_types,
                           second_geometry_name=args.second_geometry_name, mode=args.mode)
    batch.run()


if __name__ == "__main__":
    main()
