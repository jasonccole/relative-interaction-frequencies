from ccdc_roche.python import los_analysis
import pandas as pd
import numpy as np
from scipy.stats import gmean
import seaborn as sns
from matplotlib import pyplot as plt


class RfTotalBoots(object):
    def __init__(self, smarts='[*]:,=[#6][I]', smarts_index=2, mode='protein', protein_atom_types='O_pi_acc',
                 n_boot=1000):
        self.smarts = smarts
        self.smarts_index = smarts_index
        self.protein_atom_types = protein_atom_types
        self.n_boot = n_boot

    def angle_dependency(self, input_path='', mode='protein', protein_atom_types=None):
        """
        Wrapper that calculates Rf values for attractive and repulsive ranges. Assign unique ligand, protein and binding
         site counts.
        :return: csv file with angle ranges for each atom type, their Rf value with 95% confidence and unique ligand,
         protein and binding site counts.
        """
        pseudo_rfs = pd.DataFrame()
        for min_alpha_i in range(0, 180, 20):
            max_alpha_i = min_alpha_i + 20
            output = ''.join(['statistics_alpha_', str(180 - min_alpha_i), '.csv'])
            if mode == 'protein':
                protein_atom_types = 'query_match'
            else:
                protein_atom_types = protein_atom_types
            rf_analyzer = los_analysis.RfAtom(min_alpha_i=min_alpha_i, max_alpha_i=max_alpha_i, output_path=output,
                                              n_boot=self.n_boot, mode=mode, protein_atom_types=protein_atom_types,
                                              no_export=True, input_path=input_path)
            pseudo_rfs = pseudo_rfs.append(rf_analyzer.calculate_pseudo_rf())
        pseudo_rfs = pseudo_rfs.assign(SMARTS=self.smarts, SMARTS_index=self.smarts_index)
        return pseudo_rfs

    def distance_dependency(self, second_geometry_name, bin_size=0.5, input_path='', mode='protein'):
        """
        Wrapper that calculates Rf values for attractive and repulsive ranges. Assign unique ligand, protein and binding
         site counts.
        :return: csv file with angle ranges for each atom type, their Rf value with 95% confidence and unique ligand,
         protein and binding site counts.
        """
        pseudo_rfs = pd.DataFrame()
        for min_dist in np.arange(0, 3.5, bin_size):
            max_dist = min_dist + bin_size
            output = f'statistics_h_{bin_size}_{str(max_dist)}.csv'
            if mode == 'protein':
                protein_atom_types = 'query_match'
            else:
                protein_atom_types = None
            rf_analyzer = los_analysis.RfAtom(second_geometry_min=min_dist, second_geometry_max=max_dist,
                                              second_geometry_name=second_geometry_name,
                                              output_path=output,
                                              n_boot=self.n_boot, mode=mode, protein_atom_types=protein_atom_types,
                                              no_export=True)
            pseudo_rfs = pseudo_rfs.append(rf_analyzer.calculate_pseudo_rf())
        pseudo_rfs = pseudo_rfs.assign(SMARTS=self.smarts, SMARTS_index=self.smarts_index)
        return pseudo_rfs

    def return_rf_total(self, protein_alpha_min, protein_h_min, ligand_alpha_min, protein_atom_type='O_pi_acc'):
        protein_h_df = pd.read_csv('statistics_h.csv')
        protein_alpha_df = pd.read_csv('statistics_alpha.csv')
        ligand_alpha_df = pd.read_csv('../query_atom/statistics_alpha.csv')
        protein_h_df = protein_h_df.fillna(1)
        protein_alpha_df = protein_alpha_df.fillna(1)
        ligand_alpha_df = ligand_alpha_df.fillna(1)

        rf_protein_h = protein_h_df[protein_h_df['h_min'] == protein_h_min]['rf'].to_numpy()[0]

        rf_protein_alpha = protein_alpha_df[protein_alpha_df['alpha_min'] == protein_alpha_min]['rf'].to_numpy()[0]

        rf_ligand_alpha = ligand_alpha_df[(ligand_alpha_df['alpha_min'] == ligand_alpha_min) &
                                          (ligand_alpha_df['atom_type'] == protein_atom_type)]['rf'].to_numpy()[0]
        rf_protein = gmean([rf_protein_h, rf_protein_alpha])
        rf_ligand = gmean([1, rf_ligand_alpha])
        rf_total = gmean([rf_protein, rf_ligand])

        ligand_alpha_expected = ligand_alpha_df[(ligand_alpha_df['alpha_min'] == ligand_alpha_min) &
                                                (ligand_alpha_df['atom_type'] == protein_atom_type)][
            'expected'].to_numpy()[0]
        protein_alpha_expected = protein_alpha_df[protein_alpha_df['alpha_min'] == protein_alpha_min][
            'expected'].to_numpy()[0]
        protein_h_expected = protein_h_df[protein_h_df['h_min'] == protein_h_min]['expected'].to_numpy()[0]
        return rf_total, ligand_alpha_expected, protein_alpha_expected, protein_h_expected

    def rf_total_bootstrap(self, confidence_level=0.05):
        protein_alpha_pseudo_rfs = self.angle_dependency()
        protein_alpha_pseudo_rfs = protein_alpha_pseudo_rfs.fillna(1)
        protein_h_pseudo_rfs = self.distance_dependency('h')
        protein_h_pseudo_rfs = protein_h_pseudo_rfs.fillna(1)
        ligand_alpha_pseudo_rfs = self.angle_dependency(input_path='../query_atom', mode='ligand',
                                                        protein_atom_types=self.protein_atom_types)
        ligand_alpha_pseudo_rfs = ligand_alpha_pseudo_rfs.fillna(1)
        pseudo_rf_total = pd.DataFrame()
        for protein_alpha_min in protein_alpha_pseudo_rfs['alpha_min'].unique():
            for protein_h_min in protein_h_pseudo_rfs['h_min'].unique():
                for ligand_alpha_min in ligand_alpha_pseudo_rfs['alpha_min'].unique():
                    df = pd.DataFrame()
                    df['pseudo_rf_protein_alpha'] = \
                        protein_alpha_pseudo_rfs[protein_alpha_pseudo_rfs['alpha_min'] == protein_alpha_min][
                            'pseudo_rf']
                    df['pseudo_rf_protein_h'] = protein_h_pseudo_rfs[protein_h_pseudo_rfs['h_min'] == protein_h_min][
                        'pseudo_rf']
                    df['pseudo_rf_protein'] = df.apply(lambda x: gmean([x['pseudo_rf_protein_alpha'],
                                                                        x['pseudo_rf_protein_h']]), axis=1)

                    df['pseudo_rf_ligand_alpha'] = ligand_alpha_pseudo_rfs[ligand_alpha_pseudo_rfs['alpha_min'] == ligand_alpha_min]['pseudo_rf']
                    df['pseudo_rf_ligand_h'] = 1

                    df['pseudo_rf_ligand'] = df.apply(lambda x: gmean([x['pseudo_rf_ligand_alpha'],
                                                                        x['pseudo_rf_ligand_h']]), axis=1)

                    df['rf_total'] = df.apply(lambda x: gmean([x['pseudo_rf_protein'], x['pseudo_rf_ligand']]), axis=1)
                    rf_sorted = df['rf_total'].sort_values(ascending=True).reset_index(drop=True)
                    rf_tot_low = rf_sorted[int((confidence_level / 2.0) * len(rf_sorted))]
                    rf_tot_high = rf_sorted[int((1 - confidence_level / 2.0) * len(rf_sorted))]
                    rf_total, ligand_alpha_expected, protein_alpha_expected, protein_h_expected = self.return_rf_total(
                        protein_alpha_min, protein_h_min, ligand_alpha_min)
                    row = pd.Series({'protein_alpha_min': protein_alpha_min, 'protein_h_min': protein_h_min,
                                     'ligand_alpha_min': ligand_alpha_min,
                                     'rf_tot_low': rf_tot_low, 'rf_tot_high': rf_tot_high, 'rf_total': rf_total,
                                     'ligand_alpha_expected': ligand_alpha_expected,
                                     'protein_alpha_expected': protein_alpha_expected,
                                     'protein_h_expected': protein_h_expected})
                    pseudo_rf_total = pseudo_rf_total.append(row, ignore_index=True)
        pseudo_rf_total.to_csv(f'rf_total_bootstrapped_{self.n_boot}.csv', index=False)
        return pseudo_rf_total


class Comparator(object):

    def __init__(self, n_boot=1000):

        self.n_boot = n_boot
        self.bootstrapping_df = pd.read_csv(f'rf_total_bootstrapped_{self.n_boot}.csv')
        self.database_rf = pd.read_csv('../../full_p2cq_pub_oct2019_rf.csv')

    def propagation_vs_boots(self):
        bootstrapping_df = self.bootstrapping_df[(self.bootstrapping_df['ligand_alpha_expected'] > 10) &
                                                 (self.bootstrapping_df['protein_alpha_expected'] > 10) &
                                                 (self.bootstrapping_df['protein_h_expected'] > 10)]
        database_rf = self.database_rf[(self.database_rf['SMARTS'] == '[*]:,=[#6][Cl]') &
                                       (self.database_rf['protein_atom_type'] == 'O_pi_acc')]
        database_rf = database_rf.astype({'protein_h': float, 'protein_alphas': float, 'ligand_alphas': float})
        plot_df = pd.DataFrame()
        for index, row in bootstrapping_df.iterrows():
            try:
                protein_alpha_min = float(row['protein_alpha_min'])
                ligand_alpha_min = float(row['ligand_alpha_min'])
                protein_h_min = float(row['protein_h_min'])
                boots_rf_total = float(row['rf_total'])
                propagated_error = database_rf[(database_rf['ligand_alphas'] >= ligand_alpha_min) &
                                               (database_rf['ligand_alphas'] <= ligand_alpha_min + 20) &
                                               (database_rf['protein_alphas'] >= protein_alpha_min) &
                                               (database_rf['protein_alphas'] <= protein_alpha_min + 20) &
                                               (database_rf['protein_h'] >= protein_h_min) &
                                               (database_rf['protein_h'] <= protein_h_min + 0.5)
                                               ]['rf_total_error'].to_numpy()[0]
                propagated_rf_total = database_rf[(database_rf['ligand_alphas'] >= ligand_alpha_min) &
                                               (database_rf['ligand_alphas'] <= ligand_alpha_min + 20) &
                                               (database_rf['protein_alphas'] >= protein_alpha_min) &
                                               (database_rf['protein_alphas'] <= protein_alpha_min + 20) &
                                               (database_rf['protein_h'] >= protein_h_min) &
                                               (database_rf['protein_h'] <= protein_h_min + 0.5)
                                               ]['rf_total'].to_numpy()[0]
                plot_df.loc[index, 'rf_tot_high'] = row['rf_tot_high']-row['rf_total']
                plot_df.loc[index, 'rf_tot_low'] = row['rf_total'] - row['rf_tot_low']
                plot_df.loc[index, 'propagated_error'] = propagated_error
                plot_df.loc[index, 'rf_total'] = propagated_rf_total
                plot_df.loc[index, 'boots_rf_total'] = boots_rf_total
            except IndexError:
                continue

        plot = sns.scatterplot(data=plot_df, x='rf_tot_high', y='propagated_error', hue='rf_total',
                               edgecolor='black')
        plot2 = sns.lineplot(x=[0.035, 0.12], y=[0.035, 0.12], color='black')
        plt.savefig(f'propagated_error_vs_{self.n_boot}_boots_high.png', dpi=600)
        plt.clf()

        plot = sns.scatterplot(data=plot_df, x='rf_tot_low', y='propagated_error', hue='rf_total',
                               edgecolor='black')
        plot2 = sns.lineplot(x=[0.035, 0.12], y=[0.035, 0.12], color='black')
        plt.savefig(f'propagated_error_vs_{self.n_boot}_boots_low.png', dpi=600)
        plt.clf()

        plot = sns.scatterplot(data=plot_df, x='boots_rf_total', y='rf_total')
        plot2 = sns.lineplot(x=[0., 2], y=[0, 2])
        plt.savefig('propagated_rf_vs_boots_rf.png', dpi=600)
        plt.clf()


def main():
    total_bootstrapper = RfTotalBoots()
    total_bootstrapper.rf_total_bootstrap()
    comparator = Comparator()
    comparator.propagation_vs_boots()


if __name__ == '__main__':
    main()
