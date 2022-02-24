import os
import pandas as pd
from ccdc import io, protein, descriptors
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('los_filtered.csv')
ligand_df = pd.read_csv('../query_atom/los_filtered.csv')
los_home = ''
db = 'full_p2cq_pub_oct2019.csdsql'
rdr = io.EntryReader(os.path.join(los_home, db))

df = df[df['query_match'] == 1]

alpha_mins = range(0, 120, 20)
alpha_mins_df = pd.DataFrame(columns=['alpha_min', 'alpha_max', 'tyr_count', 'ser/thr_count'])
alpha_mins_df['alpha_min'] = alpha_mins
alpha_mins_df['alpha_max'] = [a + 20 for a in alpha_mins]
alpha_mins_df.loc[:, 'tyr_count'] = 0
alpha_mins_df.loc[:, 'ser/thr_count'] = 0

contact_angle_df = pd.DataFrame()

for index, row in df.iterrows():
    identifier = row['molecule_name']
    e = rdr.entry(identifier)
    p = protein.Protein.from_entry(e)
    alpha_protein = 180 - row['alpha_i']

    p.remove_all_waters()
    O_mix_atom = p.atoms[row['query_atom_id']-1]
    O_mix_residue = O_mix_atom.residue_label

    if 'TYR' in O_mix_residue:
        column_name = 'tyr_count'

    elif 'THR' in O_mix_residue or 'SER' in O_mix_residue:
        column_name = 'ser/thr_count'
    else:
        continue

    carbon_atom = O_mix_atom.neighbours[0]
    chlorine_atom = p.atoms[row['los_atom_id']-1]
    chlorine_neighbour_atom = chlorine_atom.neighbours[0]
    alpha_ligand = 180 - descriptors.MolecularDescriptors.atom_angle(chlorine_neighbour_atom, chlorine_atom, carbon_atom)

    for alpha_min in reversed(alpha_mins):
        if alpha_protein > alpha_min:
            alpha_mins_df.loc[alpha_mins_df['alpha_min'] == alpha_min, column_name] = alpha_mins_df.loc[alpha_mins_df['alpha_min'] == alpha_min, column_name] + 1
            break

    contact_angle_df = contact_angle_df.append({'alpha_protein': alpha_protein, 'alpha_ligand': alpha_ligand,
                                                'residue': O_mix_residue[:3], 'identifier': identifier},
                                               ignore_index=True)

contact_angle_plot = sns.scatterplot(data=contact_angle_df, x='alpha_protein', y='alpha_ligand', hue='residue', palette=sns.color_palette("colorblind", len(contact_angle_df['residue'].unique())))
contact_angle_plot.set_xlabel(r'$\alpha$ C-O_mix...Cl')
contact_angle_plot.set_ylabel(r'$\alpha$ C-Cl...$C_{O_mix}$')
contact_angle_plot.figure.savefig('contact_angle_plot.png', dpi=300, bbox_inches='tight')
plt.close()

alpha_mins_df.to_csv('tyrosin_count.csv', sep='\t', index=False)
contact_angle_df.to_csv('O_mix_contact_angles_comparison.csv', sep='\t', index=False)