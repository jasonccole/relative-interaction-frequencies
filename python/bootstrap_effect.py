import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

all_dfs = pd.DataFrame()
for i in [10, 50, 100, 500, 1000, 5000]:
    df = pd.read_csv(f'statistics_alpha_n_boot_{i}.csv')
    df.loc[:, 'No. of bootstrapping cycles'] = i
    df['Size of confidence interval'] = df['rf_high'] - df['rf_low']
    all_dfs = all_dfs.append(df, ignore_index=True)


low_expected = all_dfs[(all_dfs['atom_type'] == 'C_pi_neg') & (all_dfs['alpha_max'] == 40)]
low_label = low_expected['expected'].unique()[0]

mid_low_expected = all_dfs[(all_dfs['atom_type'] == 'C_pi_neg') & (all_dfs['alpha_max'] == 100)]
mid_low_label = mid_low_expected['expected'].unique()[0]

mid_high_expected = all_dfs[(all_dfs['atom_type'] == 'C_ali_don') & (all_dfs['alpha_max'] == 120)]
mid_high_label = mid_high_expected['expected'].unique()[0]

high_expected = all_dfs[(all_dfs['atom_type'] == 'C_ali_apol') & (all_dfs['alpha_max'] == 80)]
high_label = high_expected['expected'].unique()[0]

fig, ax = plt.subplots()
sns.scatterplot(data=low_expected, x='No. of bootstrapping cycles', y='Size of confidence interval',
                label=f'Expected = {low_label:.1f}', color=sns.color_palette("colorblind", 4)[0])
sns.scatterplot(data=mid_low_expected, x='No. of bootstrapping cycles', y='Size of confidence interval',
                label=f'Expected = {mid_low_label:.1f}', color=sns.color_palette("colorblind", 4)[1])
sns.scatterplot(data=mid_high_expected, x='No. of bootstrapping cycles', y='Size of confidence interval',
                label=f'Expected = {mid_high_label:.1f}', color=sns.color_palette("colorblind", 4)[2])
sns.scatterplot(data=high_expected, x='No. of bootstrapping cycles', y='Size of confidence interval',
                label=f'Expected = {high_label:.1f}', color=sns.color_palette("colorblind", 4)[3])
ax.set_xscale('log')
ax.set_xlabel('bootstrapping cycles')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)
plt.savefig('bootstrap_comparison', dpi=600, bbox_inches='tight')
plt.clf()
