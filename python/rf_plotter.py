'''
Functions to plot RF data.
'''

########################################################################################################################

import __future__
import matplotlib as mpl

mpl.use("agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import pandas as pd
import numpy as np


########################################################################################################################

def alpha_second_geometry_rf_heatmap(second_geometry_name, protein_atom_type, expected_threshold=10, mode='protein'):
    """
    Make a heatmap with alpha on the x-axis and a second geometry on the y axis. Colored by Rf value from blue to red.
    :param second_geometry_name: Str type with the name of the second geometry as it is used for naming the
     'statistics_*.csv' input files.
    :param protein_atom_type: Str type with the protein atom type.
    :return: Writes out a png file and csv files with the input data.
    """

    rf_data = pd.read_csv('statistics_alpha_h.csv')

    from matplotlib.colors import LogNorm
    import math
    log_norm = LogNorm(vmin=0, vmax=10)
    # cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(0.000000001)), 1 + math.ceil(math.log10(5)))]
    cbar_ticks = [0.1, 0.5, 1, 2, 5]
    plot_df = pd.DataFrame()
    for index, row in rf_data.iterrows():
        h_min = row['h_min']
        alpha_min = row['alpha_min']
        if row['expected'] >= expected_threshold:
            rf = row['rf']
        elif row['expected'] == 0:
            rf = np.nan
        else:
            rf = np.nan
        plot_df.loc[h_min, alpha_min] = rf
    plot_df.to_csv(f'2D_alpha_h.csv', index=False)

    plot_df = plot_df.astype(float).sort_index().reindex(sorted(plot_df.columns), axis=1)
    plot = sns.heatmap(plot_df, cmap='coolwarm', norm=log_norm,
                       cbar_kws={'label': 'Rf', 'ticks': [0.1, 1, 2, 5]})

    x_label = r'$\alpha$'
    x_unit = '°'
    y_unit = r'$\AA$'
    y_label = 'h'

    xticks = range(19)
    xtick_labels = [int(i) for i in np.linspace(0, 180, 19)]
    yticks = range(21)
    ytick_labels = np.linspace(0, 5, 21)
    plot.set_yticks(yticks)
    plot.set_yticklabels(ytick_labels)
    plot.set_xlabel(f'{x_label} [{x_unit}]')
    plot.set_xticks(xticks)
    plot.set_xticklabels(xtick_labels)
    plot.set_ylabel(f'{y_label} [{y_unit}]')
    plot.text(0.5, 1.1, '', horizontalalignment='center', verticalalignment='center',
              transform=plot.transAxes, fontsize=16, weight='bold')
    greater_eq = r'$\geq$'
    plot.text(0.5, 1.04, f'Expected {greater_eq} {expected_threshold}   mode = {mode}', horizontalalignment='center',
              verticalalignment='center',
              transform=plot.transAxes, fontsize=8)

    plt.savefig(f'2D_a_min_vs_b_min_exp_{expected_threshold}.png', dpi=600, bbox_inches='tight')
    plt.clf()


def favorable_unfavorable_angle_ranges_heatmap(input='rf_angle_range_df_with_hitlist.csv', geometry='alpha'):
    """
    Generate a heatmap with angle on the x-axis, Protein atom type on the y-axis. Colored by Rf value from blue to red.
    :param filename: Input csv file or DataFrame with Rf angle range data. Required columns are 'atom_type', 'type', 'alpha_i_min',
     'alpha_i_max', 'rf'.
    :return: Writes out a png file.
    """
    if type(input) == str:
        df = pd.read_csv(input)
    else:
        df = input
    plot_df = pd.DataFrame(columns=range(0, 180, 10))
    for index, row in df.iterrows():
        if row['type'] != '+' and row['type'] != '-':
            for alpha in range(0, 180, 10):
                plot_df.loc[row['atom_type'], alpha] = np.nan
        elif row['type'] == '-' and row['rf_high'] > 0.8:
            for alpha in range(0, 180, 10):
                plot_df.loc[row['atom_type'], alpha] = np.nan
        elif row['type'] == '+' and row['rf_low'] < 1.0:
            for alpha in range(0, 180, 10):
                plot_df.loc[row['atom_type'], alpha] = np.nan
        elif row['atom_type'] == 'other_ligands':
            for alpha in range(0, 180, 10):
                plot_df.loc[row['atom_type'], alpha] = np.nan
        elif row['expected'] < 10:
            for alpha in range(0, 180, 10):
                plot_df.loc[row['atom_type'], alpha] = np.nan
        else:
            alpha_max = 180 - int(row['alpha_i_min'])
            alpha_min = 180 - int(row['alpha_i_max'])
            for alpha in range(alpha_min, alpha_max, 10):
                plot_df.loc[row['atom_type'], alpha] = row['rf']
    plot_df = plot_df.fillna(value=np.nan)
    plot_df = plot_df.sort_index()
    try:
        plot = sns.heatmap(plot_df, cmap='coolwarm_r', center=1, vmax=2, cbar_kws={'label': 'Rf'})
        plot.set_xlabel(geometry)
        plot.set_xticks(range(19))
        plot.set_xticklabels([int(i) for i in np.linspace(0, 180, 19)])
        plot.set_ylabel('Protein atom type')
        plt.savefig(f'atom_vs_{geometry}_rf_ranges_heatmap.png', dpi=600, bbox_inches='tight')
        plt.clf()
    except TypeError:
        print('DataFrame is empty. Heat-map will not be generated.')


def rf_bar_plot(input, title, hue=None):
    """
    Generate a bar plot with Rf values and 95% confidence interval against protein atom type.
    :param hue: Group bars by this category
    :param filename: Input csv file with Rf data for protein atom types. Required columns are 'type', 'atom_type',
     'rf_high', 'rf_low', 'rf'
    :param title: Title of the plot.
    :return: Writes out a png file.
    """
    if type(input) == str:
        data = pd.read_csv(input)
    else:
        data = input
    # Don't show data below expected threshold
    data.loc[data['type'] == '?', 'rf'] = np.nan
    data.loc[data['type'] == '?', 'rf_high'] = np.nan
    data.loc[data['type'] == '?', 'rf_low'] = np.nan
    data['rf_high'] = data['rf_high'].astype('float')
    data['rf_low'] = data['rf_low'].astype('float')
    data = data.sort_values(by=['rf'], ascending=False).reset_index(drop=True)
    x = data['atom_type']
    y = data['rf']
    yerr = (data['rf_high'] - data['rf'], data['rf'] - data['rf_low'])

    if len(x) != 0 and len(y) != 0 and len(yerr) != 0:
        if hue is not None:
            fig, bars = plt.subplots()
            colors = ['salmon', 'royalblue']
            # data = data[data['rf'].isna() == False]
            if data.shape[0] > 0:
                groups = data[hue].unique()
                categories = np.arange(len(data['atom_type'].unique()))
                sort_order = data[data[hue] == 'ligand'].sort_values('rf', ascending=False)['atom_type'].unique()
                data.atom_type = data.atom_type.astype('category')

                for index, group in enumerate(groups):
                    sub_data = data[data[hue] == group]
                    sub_data.assign(atom_type=sub_data.atom_type.astype('category'))
                    sub_data.atom_type.cat.set_categories(sort_order, inplace=True)
                    sub_data = sub_data.sort_values(["atom_type"])
                    yerr = (sub_data['rf_high'] - sub_data['rf'], sub_data['rf'] - sub_data['rf_low'])
                    barwidth = 0.4
                    width = barwidth * index
                    x = categories + width
                    bars.bar(x, sub_data['rf'], width=barwidth, yerr=yerr, color=colors[index], capsize=2,
                             error_kw={'elinewidth': 0.5, 'capthick': 0.5}, label=f'{group} perspective')
                bars.set_xticks(categories + width / len(groups))
                bars.set_xticklabels(sort_order)
                bars.legend()
        else:
            bars = sns.barplot(x=data['atom_type'], y=data['rf'], yerr=yerr, color='skyblue')
        bars.set_title(title)
        bars.set_xticklabels(bars.get_xticklabels(), rotation=90)
        bars.set_xlabel('protein atom type')
        bars.set_ylabel('$R_F$')
        ax1 = bars.axes
        ax1.axhline(1, color='black', linewidth=0.5)
        plt.savefig(f'{title}', dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        print(f'no data available for {title}')


def _make_heatmap_df(input='protein_lookup_alpha.csv', expected_threshold=10, geometry='alpha'):
    input_df = pd.read_csv(input)
    if geometry in ['alpha', 'tau']:
        columns = range(20, 180, 20)  # range(20, 180, 20)
    else:
        columns = np.arange(0.5, 3.5, 0.5)  # np.arange(0.25, 5, 0.25)
    plot_df = pd.DataFrame(columns=columns)
    mask_df = pd.DataFrame(columns=columns)
    for atom_type in input_df['atom_type'].unique():
        df = input_df[input_df['atom_type'] == atom_type]
        for index, row in df.iterrows():
            geometry_bin = row[f'{geometry}_max']
            # if geometry_bin > 120:
            #     continue
            if row['expected'] >= expected_threshold:
                if row['rf'] == 0:
                    plot_df.loc[atom_type, geometry_bin] = 0.00000001  # for logarithmic colorcoding
                else:
                    plot_df.loc[atom_type, geometry_bin] = row['rf']
                if row['type'] == '0':
                    mask_df.loc[atom_type, geometry_bin] = 1
                else:
                    mask_df.loc[atom_type, geometry_bin] = np.nan
            else:
                plot_df.loc[atom_type, geometry_bin] = np.nan
                mask_df.loc[atom_type, geometry_bin] = np.nan
            # if row['expected'] == 0:
            #     mask_df.loc[atom_type, geometry_bin] = 1
            # else:
            #     mask_df.loc[atom_type, geometry_bin] = np.nan

    plot_df = plot_df.astype('float64')
    plot_df = plot_df.sort_index()
    plot_df = plot_df.drop(['acc', 'apol', 'don', 'neg', 'other_ligands', 'pi', 'pos'], errors='ignore')
    return plot_df, mask_df


def plot_protein_geometry_bins(input='protein_lookup_alpha.csv', expected_threshold=10, geometry='alpha',
                               mode='protein', title='', smarts_filters='', smarts_index=None, extension=''):
    '''
    Plot atom type vs. geom. parameter. Regions that are sterically hindered (hits=0) are gray. Regions with expected <
    threshold are white.
    :param title:
    :param mode: 'protein' or 'ligand'
    :param input:
    :param expected_threshold:
    :param geometry: 'h' or 'alpha'
    :return:
    '''

    from matplotlib.colors import LogNorm
    from matplotlib import cm

    # input_df = pd.read_csv(input)
    # if geometry == 'alpha':
    #     columns = range(20, 180, 20)  # range(20, 180, 20)
    #     xticks = range(10)  # range(10)
    #     xtick_labels = [int(i) for i in np.linspace(0, 180, 10)]  # [int(i) for i in np.linspace(0, 180, 10)]
    # else:
    #     columns = np.arange(0.5, 3.5, 0.5)  # np.arange(0.25, 5, 0.25)
    #     xticks = range(8)  # range(21)
    #     xtick_labels = np.linspace(0, 3.5, 8)  # np.linspace(0, 5, 21)
    # plot_df = pd.DataFrame(columns=columns)
    # mask_df = pd.DataFrame(columns=columns)
    # for atom_type in input_df['atom_type'].unique():
    #     df = input_df[input_df['atom_type'] == atom_type]
    #     for index, row in df.iterrows():
    #         geometry_bin = row[f'{geometry}_max']
    #         # if geometry_bin > 120:
    #         #     continue
    #         if row['expected'] >= expected_threshold:
    #             if row['rf'] == 0:
    #                 plot_df.loc[atom_type, geometry_bin] = 0.00000001  # for logarithmic colorcoding
    #             else:
    #                 plot_df.loc[atom_type, geometry_bin] = row['rf']
    #             if row['type'] == '0':
    #                 mask_df.loc[atom_type, geometry_bin] = 1
    #             else:
    #                 mask_df.loc[atom_type, geometry_bin] = np.nan
    #         else:
    #             plot_df.loc[atom_type, geometry_bin] = np.nan
    #             mask_df.loc[atom_type, geometry_bin] = np.nan
    #         # if row['expected'] == 0:
    #         #     mask_df.loc[atom_type, geometry_bin] = 1
    #         # else:
    #         #     mask_df.loc[atom_type, geometry_bin] = np.nan
    # try:
    #     plot_df = plot_df.astype('float64')
    #     plot_df = plot_df.sort_index()
    #     plot_df = plot_df.drop(['acc', 'apol', 'don', 'neg', 'other_ligands', 'pi', 'pos'], errors='ignore')

    try:
        plot_df, mask_df = _make_heatmap_df(input, expected_threshold, geometry)
        cmap1 = colors.ListedColormap(['gray'])
        fig, ax = plt.subplots(ncols=1)
        vmin = 0.5
        vmax = 2
        log_norm = LogNorm(0.5, 2)
        cmap = cm.coolwarm_r

        plot = sns.heatmap(plot_df, cmap=cmap, norm=log_norm, vmin=vmin, vmax=vmax,
                           cbar_kws={'label': r'R$_F$', 'ticks': [0.5, 0.7, 1, 1.5, 2]})
        plot.figure.axes[-1].set_ylim((0.5, 2))
        plot.figure.axes[-1].set_yticklabels(['0.5', '0.7', '1', '1.5', '2'], minor=False)
        plot.figure.axes[-1].set_yticklabels([], minor=True)
        plot.figure.axes[-1].tick_params(which='minor', axis='y', size=2)

        # plot sterically unfavored (expected value = 0) region in gray
        mask_df = mask_df.astype('float64')
        mask_df = mask_df.sort_index()
        mask_df = mask_df.drop(['acc', 'apol', 'don', 'neg', 'other_ligands', 'pi', 'pos'], errors='ignore')
        # plot2 = sns.heatmap(mask_df, cmap=cmap1, norm=log_norm, vmin=vmin, vmax=vmax, ax=ax, cbar=False)
        plt.rcParams.update({'hatch.color': 'grey'})
        plot2 = plt.pcolor(np.arange(len(mask_df.columns) + 1), np.arange(len(mask_df.index) + 1), np.array(mask_df),
                           hatch='.....', alpha=0., vmin=vmin, vmax=vmax)

        if geometry == 'alpha':
            x_label = r'$\alpha$'
            unit = '°'
        if geometry == 'tau':
            x_label = r'$\tau$'
            unit = '°'
        elif geometry == 'h':
            unit = r'$\AA$'
            x_label = 'h'

        if geometry in ['alpha', 'tau']:
            xticks = range(10)  # range(10)
            xtick_labels = [int(i) for i in np.linspace(0, 180, 10)]  # [int(i) for i in np.linspace(0, 180, 10)]
        else:
            xticks = range(8)  # range(21)
            xtick_labels = np.linspace(0, 3.5, 8)  # np.linspace(0, 5, 21)

        plot.set_xlabel(f'{x_label} [{unit}]')
        plot.set_xticks(xticks)
        plot.set_xticklabels(xtick_labels)
        plot.set_ylabel('Protein atom type')
        title = title.replace('$', r'\$')
        plot.text(0.5, 1.1, title, horizontalalignment='center', verticalalignment='center',
                  transform=plot.transAxes, fontsize=16, weight='bold')
        greater_eq = r'$\geq$'
        plot.text(0.5, 1.04,
                  f'Expected {greater_eq} {expected_threshold}   mode: {mode}   smarts_index: {int(smarts_index)}',
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=plot.transAxes, fontsize=8)
        plt.savefig(f'atom_vs_{geometry}_rf_bins_exp_{expected_threshold}_heatmap{extension}.png', dpi=300,
                    bbox_inches='tight')
        plt.clf()

    except (TypeError, ValueError):
        print('DataFrame is empty. Heat-map will not be generated.')


def plot_rscc_vs_interaction_count(input_data='full_p2cq_pub_oct2019_rf.csv'):
    if type(input_data) == str:
        input_data = pd.read_csv(input_data)
    input_data = input_data.drop_duplicates(subset='identifier')
    rscc_bins = [[0., 0.8], [0.8, 0.9], [0.9, 1.0]]
    plot_df = pd.DataFrame()
    i = -1
    for rscc_bin in rscc_bins:
        for count_no in range(0, 8):
            i += 1
            entries_with_count_no = input_data[(input_data['ligand_rscc'] > rscc_bin[0]) &
                                               (input_data['ligand_rscc'] <= rscc_bin[1]) &
                                               (input_data['unfavorable_per_entry'] == count_no)
                                               ].shape[0]
            total_entries = input_data[(input_data['ligand_rscc'] > rscc_bin[0]) &
                                       (input_data['ligand_rscc'] <= rscc_bin[1])
                                       ].shape[0]
            pc_entries_with_count_no = entries_with_count_no / total_entries
            plot_df.loc[i, f'RSCC'] = f'{rscc_bin[0]} < RSCC <= {rscc_bin[1]}'
            plot_df.loc[i, f'unfavorable_contacts_per_entry'] = count_no
            plot_df.loc[i, f'% entries with n unfavorable contacts'] = pc_entries_with_count_no
    plot = sns.barplot(x=plot_df['RSCC'], y=plot_df['% entries with n unfavorable contacts'],
                       hue=plot_df['unfavorable_contacts_per_entry'])
    plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('barplot.png', dpi=300, bbox_inches='tight')


def plot_rf_count_vs_expected(input_data):
    import os
    if type(input_data) == str:
        input_data = pd.read_csv(input_data)
    true_df = pd.read_csv('rf.csv')
    true_rf = true_df[true_df['atom_type'] == 'query_match']['rf']
    true_expected = true_df[true_df['atom_type'] == 'query_match']['expected']
    input_data['delta_rf_true'] = input_data['rf'].apply(lambda x: abs(x - true_rf))
    true_rf_bins = [[0., 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.]]
    expected_bins = [(i, i + 1) for i in range(5, 26)]
    plot_df = pd.DataFrame()
    i = -1

    # make boxplot
    # bin dataframe by "expected"
    bins = pd.IntervalIndex.from_tuples(expected_bins)
    binned_data = input_data.copy()
    binned_data = binned_data.assign(expected=pd.cut(input_data['expected'], bins))
    binned_data = binned_data[binned_data['expected'].notna()]
    print(binned_data)
    boxplot = sns.boxplot(data=binned_data, x='expected', y='delta_rf_true', showfliers=False, color='royalblue')
    boxplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    subtitle_rf = true_rf.to_numpy()[0]
    subtitle_expected = true_expected.to_numpy()[0]
    subtitle = f'true Rf = {subtitle_rf:.2f}, true expected = {subtitle_expected:.2f}'
    ligand_atom = os.getcwd().split('/')[-2]
    protein_atom = os.getcwd().split('/')[-1]
    boxplot.text(0.5, 1.1, f'{ligand_atom} {protein_atom}', horizontalalignment='center', verticalalignment='center',
                 transform=boxplot.transAxes, fontsize=16, weight='bold')
    boxplot.text(0.5, 1.04,
                 f'{subtitle} protein_perspective',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=boxplot.transAxes, fontsize=8)
    sample_numbers = [len(binned_data[binned_data['expected'] == bin_]) for bin_ in bins]
    boxplot.set_xticklabels(
        [f'Bin: {label} Samples: {sample_numbers[index]}' for index, label in enumerate(expected_bins)], fontsize=8,
        rotation=90)
    plt.savefig(f'{ligand_atom}_{protein_atom}_rf_vs_expected_protein_perspective_boxplot.png',
                dpi=300, bbox_inches='tight')
    plt.clf()

    for true_rf_bin in true_rf_bins:
        for expected_bin in expected_bins:
            i += 1
            entries_with_count_no = input_data[(input_data['delta_rf_true'] > true_rf_bin[0]) &
                                               (input_data['delta_rf_true'] <= true_rf_bin[1]) &
                                               (input_data['expected'] > expected_bin[0]) &
                                               (input_data['expected'] <= expected_bin[1])
                                               ].shape[0]
            total_entries = input_data[(input_data['expected'] >= expected_bin[0]) &
                                       (input_data['expected'] <= expected_bin[1])
                                       ].shape[0]
            if total_entries == 0:
                continue
            pc_entries_with_expected = entries_with_count_no / total_entries
            plot_df.loc[i, f'delta_rf_true'] = f'{true_rf_bin[0]} < dRf <= {true_rf_bin[1]}'
            plot_df.loc[i, f'expected'] = f'{expected_bin[0]} < expected <= {expected_bin[1]}\nsamples: {total_entries}'
            plot_df.loc[i, f'% dRf'] = pc_entries_with_expected
    plot = sns.barplot(x=plot_df['expected'], y=plot_df['% dRf'],
                       hue=plot_df['delta_rf_true'])

    plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    subtitle_rf = true_rf.to_numpy()[0]
    subtitle_expected = true_expected.to_numpy()[0]
    subtitle = f'true Rf = {subtitle_rf:.2f}, true expected = {subtitle_expected:.2f}'

    ligand_atom = os.getcwd().split('/')[-2]
    protein_atom = os.getcwd().split('/')[-1]

    plot.text(0.5, 1.1, f'{ligand_atom} {protein_atom}', horizontalalignment='center', verticalalignment='center',
              transform=plot.transAxes, fontsize=16, weight='bold')

    plot.text(0.5, 1.04,
              f'{subtitle} protein_perspective',
              horizontalalignment='center',
              verticalalignment='center',
              transform=plot.transAxes, fontsize=8)

    plot.set_xticklabels(plot.get_xticklabels(), fontsize=8)
    plt.savefig(f'{ligand_atom}_{protein_atom}_rf_vs_expected_protein_perspective.png', dpi=300, bbox_inches='tight')


def plot_rf_total_vs_pat(input_data='full_p2cq_pub_oct2019_rf_extended.gzip', ligand_atom_type1='chlorine_aryl',
                         ligand_atom_type2='fluorine_aryl',
                         filename=''):
    '''
    Make a boxplot comparing Rf_total for two ligand atom types.
    :param ligand_atom_type2: SMARTS query of first ligand atom type.
    :param smarts1: SMARTS query of second ligand atom type.
    :param input_data: CSV file with Rf_total values for Database.
    :param name: Filename
    :return:
    '''
    plot_df = pd.read_parquet(input_data)
    plot_df = plot_df[plot_df['protein_atom_type'] != 'other_ligands']
    plot_df = plot_df[(plot_df['resolution'] <= 2.5) & (plot_df['ligand_rscc'] >= 0.8) &
                      (plot_df['ligand_avgoccu'] == 1.0) & (plot_df['ligand_altcode'] == ' ')]
    plot_df = plot_df[
        (plot_df['ligand_atom_type'] == ligand_atom_type1) | (plot_df['ligand_atom_type'] == ligand_atom_type2)]

    plot_df_no_uniprot = plot_df[plot_df['uniprot'].isna()].drop_duplicates(subset=['ligand_smiles', 'project'])
    plot_df_with_uniprot = plot_df[plot_df['uniprot'].isna() == False].drop_duplicates(
        subset=['ligand_smiles', 'uniprot'])
    plot_df = pd.concat([plot_df_no_uniprot, plot_df_with_uniprot])

    plot_df = plot_df[plot_df['protein_atom_type'] != 'C_pi_pos']
    plot_df = plot_df[plot_df['protein_atom_type'] != 'N_pi']
    plot_df = plot_df[plot_df['protein_atom_type'] != 'other_ligand']
    print(ligand_atom_type2, plot_df[plot_df['ligand_atom_type'] == ligand_atom_type2].shape)
    print(ligand_atom_type1, plot_df[plot_df['ligand_atom_type'] == ligand_atom_type1].shape)
    print(ligand_atom_type2, plot_df[
        (plot_df['ligand_atom_type'] == ligand_atom_type2) & (plot_df['protein_atom_type'] == 'O_pi_acc_neg')][
        'rf_total'].median())
    print(ligand_atom_type1, plot_df[
        (plot_df['ligand_atom_type'] == ligand_atom_type1) & (plot_df['protein_atom_type'] == 'O_pi_acc_neg')][
        'rf_total'].median())
    # generate the plot
    box = sns.boxplot(x=plot_df['protein_atom_type'], y=plot_df['rf_total'], hue=plot_df['ligand_atom_type'],
                      showfliers=False)
    box.axhline(1, ls='--', color='gray', zorder=0)
    box.set_xticklabels(box.get_xticklabels(), rotation=90)
    box.set_xlabel('Protein atom type')
    box.set_ylabel('$R_Ftotal$')
    box.legend().set_title('ligand atom type')
    plt.savefig(f'{filename}_boxplot.png', dpi=600, bbox_inches='tight')
    plt.clf()


def protein_atom_type_heatmap(expected_threshold=10):
    '''
    Plot an overview of an atom type's RF values for all ligand atom types. Protein and ligand perspective.
    :param expected_threshold:
    :return:
    '''
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    import os

    folders = next(os.walk('.'))[1]
    protein_atom_types = list(pd.read_csv(os.path.join(folders[0], f'ligand_lookup_alpha.csv'))['atom_type'].unique()) + ['metal']
    geometries = ['alpha', 'h']
    groups = ['acc', 'apol', 'don', 'neg', 'other_ligands', 'pi', 'pos']
    for geometry in geometries:
        for protein_atom_type in protein_atom_types:
            if protein_atom_type == 'metal':
                continue
            for mode in ['ligand', 'protein']:
                if mode == 'protein' and geometry == 'h' and 'pi' not in protein_atom_type:
                    continue
                if protein_atom_type in groups:
                    continue
                input_df = pd.DataFrame()
                for folder in folders:
                    lookup_file = os.path.join(folder, f'{mode}_lookup_{geometry}.csv')
                    if os.path.isfile(lookup_file):
                        lookup_df = pd.read_csv(lookup_file)
                        if lookup_df.shape[0] > 0:
                            lookup_df = lookup_df[lookup_df['atom_type'] == protein_atom_type]
                            input_df = input_df.append(lookup_df, ignore_index=True)

                if geometry == 'alpha':
                    columns = range(20, 180, 20)  # range(10, 180, 10)
                    xticks = range(10)  # range(19)
                    xtick_labels = [int(i) for i in
                                    np.linspace(0, 180, 10)]  # [int(i) for i in np.linspace(0, 180, 19)]
                else:
                    columns = np.arange(0.5, 3.5, 0.5)  # np.arange(0.25, 5, 0.25)
                    xticks = range(8)  # range(21)
                    xtick_labels = np.linspace(0, 3.5, 8)  # np.linspace(0, 5, 21)
                plot_df = pd.DataFrame(columns=columns)
                mask_df = pd.DataFrame(columns=columns)
                for ligand_atom_type in input_df['ligand_atom_type'].unique():
                    df = input_df[input_df['ligand_atom_type'] == ligand_atom_type]
                    for index, row in df.iterrows():
                        geometry_bin = row[f'{geometry}_max']
                        if row['expected'] >= expected_threshold:
                            if row['rf'] == 0:
                                plot_df.loc[ligand_atom_type, geometry_bin] = 0.00000001  # for logarithmic colorcoding
                            else:
                                plot_df.loc[ligand_atom_type, geometry_bin] = row['rf']
                            if row['type'] == '0':
                                mask_df.loc[ligand_atom_type, geometry_bin] = 1
                            else:
                                mask_df.loc[ligand_atom_type, geometry_bin] = np.nan
                        else:
                            plot_df.loc[ligand_atom_type, geometry_bin] = np.nan
                            mask_df.loc[ligand_atom_type, geometry_bin] = np.nan

                # try:
                plot_df = plot_df.astype('float64')
                plot_df = plot_df.sort_index()
                plot_df = plot_df.drop(['acc', 'apol', 'don', 'neg', 'other_ligands', 'pi', 'pos'], errors='ignore')
                if plot.shape[0] == 0:
                    continue
                cmap1 = colors.ListedColormap(['gray'])
                fig, ax = plt.subplots(ncols=1)
                vmin = 0.5
                vmax = 2
                log_norm = LogNorm(1 / 3, 3)
                cmap = cm.coolwarm_r
                dpi = 300
                plt.rcParams['ytick.labelsize'] = 10
                fontsize_pt = plt.rcParams['ytick.labelsize']

                # compute the matrix height in points and inches
                matrix_height_pt = fontsize_pt * plot_df.shape[0]
                matrix_height_in = matrix_height_pt / dpi

                # compute the required figure height
                top_margin = 0.04  # in percentage of the figure height
                bottom_margin = 0.04  # in percentage of the figure height
                figure_height = 5 * matrix_height_in / (1 - top_margin - bottom_margin)

                # build the figure instance with the desired height
                fig, ax = plt.subplots(
                    figsize=(6, figure_height),
                    gridspec_kw=dict(top=1 - top_margin, bottom=bottom_margin))

                plot = sns.heatmap(plot_df, cmap=cmap, norm=log_norm, vmin=vmin, vmax=vmax,
                                   cbar_kws={'label': 'Rf', 'ticks': [0.5, 0.7, 1, 1.5, 2]}, ax=ax)
                plot.figure.axes[-1].set_yticklabels(['0.5', '0.7', '1', '1.5', '2'], minor=False)
                plot.figure.axes[-1].set_yticklabels([], minor=True)
                plot.figure.axes[-1].tick_params(which='minor', axis='y', size=2)

                mask_df = mask_df.astype('float64')
                mask_df = mask_df.sort_index()
                mask_df = mask_df.drop(['acc', 'apol', 'don', 'neg', 'other_ligands', 'pi', 'pos'], errors='ignore')

                plt.rcParams.update({'hatch.color': 'grey'})
                plot2 = plt.pcolor(np.arange(len(mask_df.columns) + 1), np.arange(len(mask_df.index) + 1),
                                   np.array(mask_df),
                                   hatch='.....', alpha=0.)

                if geometry == 'alpha':
                    x_label = r'$\alpha$'
                    unit = '°'
                elif geometry == 'h':
                    unit = r'$\AA$'
                    x_label = 'h'
                plot.set_xlabel(f'{x_label} [{unit}]')
                plot.set_xticks(xticks)
                plot.set_xticklabels(xtick_labels)
                plot.set_ylabel('Ligand atom type')
                # plot.set_yticklabels(plot.get_ymajorticklabels(), size=fontsize_pt)

                plot.text(0.5, 1.1, protein_atom_type, horizontalalignment='center', verticalalignment='center',
                          transform=plot.transAxes, fontsize=16, weight='bold')
                greater_eq = r'$\geq$'
                plot.text(0.5, 1.04,
                          f'Expected {greater_eq} {expected_threshold}   mode: {mode}',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=plot.transAxes, fontsize=8)
                plt.savefig(f'{protein_atom_type}_{geometry}_heatmap_{mode}_perspective.png', dpi=dpi,
                            bbox_inches='tight')
                plt.clf()
                plt.close()

                # except (TypeError, ValueError):
                #     print('DataFrame is empty. Heat-map will not be generated.')


def main():
    protein_atom_type_heatmap()
    # plot_protein_geometry_bins(input='statistics_h_high_res.csv', expected_threshold=10,
    #                            mode='ligand', title=r'[C^3]-[CD3](~[OD1])(~[OD1])',
    #                            smarts_index=2, geometry='h', extension='_high_res')
    # plot_protein_geometry_bins()
    # plot_rf_total_vs_pat(filename='chlorine_vs_fluorine')


if __name__ == "__main__":
    main()
