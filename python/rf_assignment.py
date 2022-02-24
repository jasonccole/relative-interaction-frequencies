#!/usr/bin/env python

########################################################################################################################

from ccdc_roche.python.rf_psp import los_protein_descriptors
from ccdc import io, molecule
import argparse
from ccdc_roche.python.los_descriptors import CsdDescriptorsFromProasis, CsdDescriptorsFromPDB, CsdDescriptorsFromMol2
from ccdc_roche.python.database_utilities import export_entry
from pathlib import Path

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Visualize RF values in PyMOL.
        rf_assignment.py -i ABCD_001
        rf_assignment.py -i ABCDE_001 -db internal 
        rf_assignment.py -i gold.sdf --gold gold.conf
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-db',
        '--database',
        help='Path to input database in csdsql format. The database entries must be proteins.',
        default='full_p2cq_pub_aug2021'
    )

    parser.add_argument(
        '-i',
        '--identifier',
        nargs='+',
        help='Entry identifier. You can also pass multiple identifiers.',
        type=str,
        default=''
    )

    parser.add_argument(
        '-p',
        '--protein',
        help='Protein file',
        type=str,
        default=''
    )

    parser.add_argument(
        '-l',
        '--ligands',
        help='ligands file',
        type=str,
        default=''
    )

    parser.add_argument(
        '--gold',
        help='Pass path to GOLD config file',
        type=str,
        default=''
    )

    parser.add_argument(
        '--los_home',
        help='los_home folder with lookup files, databases, protein_atom_types.csv and ligand_atom_types.csv',
        type=str,
        default=''
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help='Write out RF assignment data.',
        action='store_true'
    )

    return parser.parse_args()


def return_blue_red_color(rf):
    yellow = [255, 247, 0]
    red = [255, 34, 0]
    blue = [0, 0, 255]
    if rf >= 1.0:
        color = blue
    elif rf <= 1.0:
        color = red
    else:
        color = yellow
    return color


class RfAssigner(object):
    def __init__(self, input: str, gold=False, los_home='',
                 intramolecular_protein_contacts=False, verbose=False, only_binding_site=False, target_file=None,
                 bfactor=False, strucid=None, interaction_cutoff=0.5, pdb_file=None):
        '''

        :param input:
        :param gold Set to True if input is a GOLD Docking pose.:
        :param los_home:
        :param intramolecular_protein_contacts:
        :param verbose:
        :param only_binding_site:
        :param target_file:
        :param bfactor:
        :param strucid:
        :param interaction_cutoff:
        >>> rf_assignments = []
        >>> with io.MoleculeReader('testdata/4mk8_ligand.sdf') as rdr:
        ...     for cnt, ligand in enumerate(rdr):
        ...        assigner = RfAssigner(ligand, target_file='testdata/4mk8_apo.pdb')
        ...        rf_assignment_df = assigner.rf_assignments
        ...        rf_assignment_df['ligand_id'] = cnt
        ...        rf_assignments.append(assigner.rf_assignments)
        >>> import pandas as pd
        >>> rf_values_df = pd.concat(rf_assignments, ignore_index=True).sort_values(['ligand_atom_label', 'los_atom_label']).reset_index(drop=True).fillna('')
        >>> test_df = pd.read_csv('testdata/4mk8_rf_values.csv').sort_values(['ligand_atom_label', 'los_atom_label']).reset_index(drop=True).fillna('')
        >>> from pandas.testing import assert_series_equal
        >>> assert_series_equal(test_df['rf_total'], rf_values_df['rf_total'])
        >>> assert_series_equal(test_df['rf_total_error'], rf_values_df['rf_total_error'])
        '''
        self.los_home = los_home
        self.gold = gold
        self.verbose = verbose
        self.pdb = False
        self.bfactor = bfactor
        self.interaction_cutoff = interaction_cutoff

        if type(input) == str and len(input.split('.')) == 1:
            strucid = input.split('_')[0]
            if len(strucid) == 5:
                dbname = 'full_p2cq_roche_aug2021'
            if len(strucid) == 4:
                dbname = 'full_p2cq_pub_aug2021'
            dbformat = 'csdsql'

        if gold:
            from ccdc_roche.python import los_descriptors
            self.describer = los_descriptors.CsdDescriptorsFromGold(input, gold_conf=gold,
                                                                    only_binding_site=only_binding_site,
                                                                    bfactor=self.bfactor, strucid=strucid,
                                                                    interaction_cutoff=self.interaction_cutoff,
                                                                    pdb_file=pdb_file)
            self.identifier = input.split('.')[0].split('/')[-1]
            ligand_id = self.describer.csd_ligand_entry.identifier
            if '|' in ligand_id and 'docking_input_' in ligand_id:
                ligand_id = ligand_id.split('|')[1].split('docking_input_')[1]
            self.describer.csd_ligand_entry.attributes['ligand_id'] = ligand_id
            self.rf_assignments = self.describer.contact_df()
            self.rotatable_bonds_num = len([b for b in self.describer.csd_ligand.bonds if b.is_rotatable])
            self.frozen_bonds_num = self.return_frozen_bonds_num(self.describer.csd_ligand)
            if intramolecular_protein_contacts:
                self.rf_assignments = self.rf_assignments.append(
                    los_protein_descriptors.ProteinDescriptors(csd_protein=self.describer.protein,
                                                               csd_ligand=self.describer.csd_ligand,
                                                               rdkit_protein=self.describer.rdkit_protein
                                                               ).return_protein_intramolecular_contact_df(),
                    ignore_index=True)
            self.protein_assignment = self.describer.protein_df
            self.ligand_assignment = self.describer.ligand_df
            return

        if target_file:
            self.structure = target_file
            self.ligand_file = input
            if type(input) == str:
                self.identifier = Path(input).stem.split('.')[0]
            elif type(input) == molecule.Molecule:
                self.identifier = input.identifier
            self.describer = CsdDescriptorsFromPDB(self.structure, only_binding_site=only_binding_site,
                                                   ligand=self.ligand_file, bfactor=False, strucid=strucid)
            self.rf_assignments = self.describer.los_contacts_df
            self.rotatable_bonds_num = len([b for b in self.describer.csd_ligand.bonds if b.is_rotatable])
            self.frozen_bonds_num = self.return_frozen_bonds_num(self.describer.csd_ligand)
            if intramolecular_protein_contacts:
                self.rf_assignments = self.rf_assignments.append(
                    los_protein_descriptors.ProteinDescriptors(csd_protein=self.describer.protein,
                                                               csd_ligand=self.describer.csd_ligand,
                                                               rdkit_protein=self.describer.rdkit_protein
                                                               ).return_protein_intramolecular_contact_df(
                    ), ignore_index=True)
            if self.verbose:
                with io.MoleculeWriter('complex.mol2') as w:
                    w.write(self.describer.protein)
            self.structure = 'complex.mol2'

            self.pdb = True

        elif '.mol2' in input:  # currently broken
            self.structure = input
            self.identifier = input.split('.')[0]
            self.describer = CsdDescriptorsFromMol2(self.structure)
            self.rf_assignments = self.describer.contact_df()
            self.protein_assignment = self.describer.protein_df
            self.ligand_assignment = self.describer.ligand_df
            with io.EntryWriter(input) as w:
                w.write(self.describer.protein)

            # if intramolecular_protein_contacts:
            #     self.rf_assignments = self.rf_assignments.append(
            #         los_protein_descriptors.ProteinDescriptors(self.structure).return_protein_intramolecular_contact_df(
            #         ), ignore_index=True)
        else:
            self.identifier = input

            csd_db = str(Path(self.los_home) / Path(f'{dbname}.{dbformat}'))
            export_entry(input, csd_db, remove_waters=False)
            export_entry(input, csd_db)
            self.structure = f'{input}_wet.mol2'
            self.describer = CsdDescriptorsFromProasis(self.structure)
            self.rf_assignments = self.describer.los_contacts_df
            # if intramolecular_protein_contacts:
            #     self.rf_assignments = self.rf_assignments.append(
            #         los_protein_descriptors.ProteinDescriptors(self.structure, rdkit_protein=rdkit_mol,
            #                                                    los_home=self.los_home,
            #                                                    ).return_protein_intramolecular_contact_df(
            #         ), ignore_index=True)

            self.protein_assignment = self.describer.protein_df
            self.ligand_assignment = self.describer.ligand_df
        if self.verbose:
            self.rf_assignments.to_csv(f'{input}.csv', index=False)

    def return_frozen_bonds_num(self, ligand):
        rotatable_bonds = [b for b in ligand.bonds if b.is_rotatable]
        polar_favorable_interaction_atoms = \
            self.rf_assignments[(self.rf_assignments['rf_total'] >= 0.8) & (self.rf_assignments['interaction_type'
                                                                            ].isin(['hbond_weak', 'hbond_classic']))
                                ]['ligand_atom_label'].to_list()
        polar_hbond_atoms = self.describer.hbond_df['atom1_label'].to_list() + self.describer.hbond_df['atom2_label'].to_list()

        polar_favorable_interaction_atoms = list(set(polar_favorable_interaction_atoms + polar_hbond_atoms))
        frozen_bonds = [b for b in rotatable_bonds if b.atoms[0].label in polar_favorable_interaction_atoms or
                        b.atoms[1].label in polar_favorable_interaction_atoms]
        return len(frozen_bonds)

    def visualize_in_pymol(self):
        from ccdc import protein

        with io.EntryReader(self.structure) as r:
            p = protein.Protein.from_entry(r[0])
            p.remove_hydrogens()
            for a in p.atoms:
                a.partial_charge = 0

        with io.MoleculeWriter(self.structure) as w:
            w.write(p)

        from pymol import cmd
        cmd.load(self.structure)

        # create object for each ligand atom
        cmd.color('cyan', 'all')
        cmd.color('atomic', 'not elem C')
        cmd.color('green', f'all and elem C and not name _Z*')
        cmd.hide('lines', 'all')
        cmd.show('licorice', 'all')
        if self.gold or self.pdb:
            for index, row in self.rf_assignments.iterrows():
                protein_atom_type = row['protein_atom_type']
                protein_atom_index = row['los_atom_index']
                cmd.select('protein_atom', f'rank {protein_atom_index}')
                cmd.create(f'protein_atom_{protein_atom_index}', 'protein_atom')
                cmd.alter(f'protein_atom_{protein_atom_index}', f'p.atom_type = "{protein_atom_type}"')
                cmd.group('receptor', f'protein_atom_{protein_atom_index}')

                ligand_atom_type = row['ligand_atom_type']
                ligand_atom_index = row['ligand_atom_index']
                cmd.select('ligand_atom', f'rank {ligand_atom_index}')
                cmd.create(f'ligand_atom_{ligand_atom_index}', 'ligand_atom')
                cmd.alter(f'ligand_atom_{ligand_atom_index}', f'p.atom_type = "{ligand_atom_type}"')
                cmd.group('ligand', f'ligand_atom_{ligand_atom_index}')

        else:
            for index, row in self.ligand_assignment.iterrows():
                ligand_atom_type = row['ligand_atom_type']
                ligand_atom_index = row['ligand_atom_index']
                cmd.select('ligand_atom', f'rank {ligand_atom_index}')
                cmd.create(f'ligand_atom_{ligand_atom_index}', 'ligand_atom')
                cmd.alter(f'ligand_atom_{ligand_atom_index}', f'p.atom_type = "{ligand_atom_type}"')
                cmd.group('ligand', f'ligand_atom_{ligand_atom_index}')

            for index, row in self.protein_assignment.iterrows():
                protein_atom_type = row['protein_atom_type']
                protein_atom_index = row['protein_atom_index']
                cmd.select('protein_atom', f'rank {protein_atom_index}')
                cmd.create(f'protein_atom_{protein_atom_index}', 'protein_atom')
                cmd.alter(f'protein_atom_{protein_atom_index}', f'p.atom_type = "{protein_atom_type}"')
                cmd.group('receptor', f'protein_atom_{protein_atom_index}')

        intramolecular_groups = []

        for index, row in self.rf_assignments.iterrows():
            rf = row['rf_total']
            ligand_atom_index = row['ligand_atom_index']
            los_atom_index = row['los_atom_index']
            cmd.select('ligand_atom', f'rank {ligand_atom_index}')
            cmd.select('los_atom', f'rank {los_atom_index}')
            rf_color = return_blue_red_color(rf)  # list(self.return_blue_red_color(rf)[0:3])
            cmd.set_color(f'rf_color{index}', rf_color)
            cmd.distance(f'{index}_rf_{rf:.1f}', 'ligand_atom', 'los_atom')
            cmd.set('dash_width', '3')
            cmd.color(f'rf_color{index}', f'{index}_rf_{rf:.1f}')
            interaction_type = row['interaction_type']
            if interaction_type is not None and not row['is_intramolecular']:
                if row['is_primary']:
                    cmd.group(f'{interaction_type}_primary', f'{index}_rf_{rf:.1f}')
                else:
                    cmd.group(f'{interaction_type}_secondary', f'{index}_rf_{rf:.1f}')
            elif interaction_type is not None and row['is_intramolecular']:
                if row['is_primary']:
                    intramolecular_group = f'{interaction_type}_primary_intramolecular'
                    cmd.group(intramolecular_group, f'{index}_rf_{rf:.1f}')
                else:
                    intramolecular_group = f'{interaction_type}_secondary_intramolecular'
                    cmd.group(intramolecular_group, f'{index}_rf_{rf:.1f}')
                intramolecular_groups.append(intramolecular_group)

        cmd.group('intramolecular', ' or '.join(intramolecular_groups))

        cmd.group('primary_competitive', 'hbond_classic_primary or hbond_weak_primary or multipolar_primary or '
                                         'hydrophobic_primary or halogen_primary or pi_primary or '
                                         'uncat_competitive_primary or ionic_primary')
        cmd.group('secondary_competitive', 'hbond_classic_secondary or hbond_weak_secondary or multipolar_secondary or '
                                           'hydrophobic_secondary or halogen_secondary or pi_secondary or '
                                           'uncat_competitive_secondary or ionic_secondary')
        cmd.group('competitive', 'primary_competitive or secondary_competitive')

        cmd.group('primary_non-competitive', 'hbond_mismatch_primary or electrostatic_repulsion_primary or '
                                             'multipolar_mismatch_primary or desolvation_primary or '
                                             'pi_mismatch_primary or uncat_non_competitive_primary')
        cmd.group('secondary_non-competitive', 'hbond_mismatch_secondary or electrostatic_repulsion_secondary or '
                                               'multipolar_mismatch_secondary or desolvation_secondary or '
                                               'pi_mismatch_secondary or uncat_non_competitive_secondary')
        cmd.group('non-competitive', 'primary_non-competitive or secondary_non-competitive')
        cmd.delete('los_atom')
        cmd.delete('ligand_atom')
        cmd.delete('protein_atom')
        cmd.set('depth_cue', 0)
        cmd.set('ray_shadows', 0)
        cmd.set('valence', 0)
        cmd.zoom('all')
        cmd.show('lines', 'all')
        cmd.set('label_size', 30)
        cmd.set('dash_width', 5)
        cmd.bg_color('white')
        cmd.save(f'rf_scene_{self.identifier}.pse')
        cmd.delete('all')
        if self.gold:
            # clean up
            Path('temp.mol2').unlink()


def main():
    args = parse_args()
    if args.identifier != '':
        for identifier in args.identifier:
            assigner = RfAssigner(identifier, args.database, args.gold, args.los_home, verbose=args.verbose)
            assigner.visualize_in_pymol()
    elif args.protein != '' and args.ligands != '':
        assigner = RfAssigner(args.ligands, args.database, args.gold, args.los_home, verbose=args.verbose,
                              target_file=args.protein)
        assigner.visualize_in_pymol()


if __name__ == "__main__":
    main()
