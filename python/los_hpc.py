#!/usr/bin/env python


# A. Tosstorff
# 15-APR-2020

########################################################################################################################

import matplotlib
matplotlib.use('Agg')
from ccdc_roche.python import ligand_los_contacts, los_postprocessing, protein_los_contacts
import argparse

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Execute Line of sight contact scripts.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '--los_home',
        help='Path to LoS home folder.',
        default=''
    )

    parser.add_argument(
        '-db',
        '--database',
        nargs='*',
        help='Database name in LoS home folder.',
        default=['full_p2cq_pub_aug2021.csdsql', 'full_p2cq_roche_aug2021.csdsql']
    )

    parser.add_argument(
        '-m',
        '--mode',
        help='ligand or protein perspective.',
        default='ligand'
    )

    parser.add_argument(
        '-sq',
        '--structure_quality',
        help='File containing structure quality information.',
        default='data_quality_aug2021.csv'
    )

    parser.add_argument(
        '-a',
        '--annotation',
        help='File containing structure annotation.',
        default='all_annotated_aug2021.csv'
    )

    parser.add_argument(
        '-s',
        '--smarts',
        help='SMARTS string to define ligand atom type.',
        default=None
    )

    parser.add_argument(
        '-si',
        '--smarts_index',
        help='Index to define atom in SMARTS string.',
        default=None
    )

    parser.add_argument(
        '--angle_name',
        help='Angle name.',
        default='alpha'
    )

    parser.add_argument(
        '-np',
        help='Number of parallel processes for multiprocessing.',
        default=24
    )

    parser.add_argument(
        '-pi',
        '--pi_atom',
        help='Central atom is in pi system.',
        action='store_true'
    )

    parser.add_argument(
        '-exe',
        '--executable',
        choices=['contacts', 'angle', 'h', 'filter'],
        help='With \'contacts\', contact counts will be generated. \'angle\' and \'h\' will execute '
             'postprocessing for the corresponding geometry.',
        default=None
    )

    parser.add_argument(
        '-lat',
        '--ligand_atom_type',
        help='Ligand atom type',
        default=None
    )

    parser.add_argument(
        '-pat',
        '--protein_atom_type',
        help='Protein atom type',
        default=None
    )

    return parser.parse_args()


class RfWrapper(object):
    def __init__(self, los_home, db, annotation, structure_quality, smarts, smarts_index, ligand_atom_type,
                 pi, np, executable, mode, protein_atom_type=None, angle_name='alpha'):
        self.los_home = los_home
        self.db = db
        self.annotations = annotation
        self.output = '.'
        self.structure_quality = structure_quality
        self.smarts = smarts
        self.smarts_index = smarts_index
        self.ligand_atom_type = ligand_atom_type
        self.np = np
        self.pi = pi
        if ligand_atom_type:
            if 'oxygen' in ligand_atom_type and 'carboxylate' in ligand_atom_type:
                self.tau_atom = True
            else:
                self.tau_atom = False
        self.angle_name = angle_name
        self.mode = mode
        self.executable = executable
        self.protein_atom_type = protein_atom_type
        print('SMARTS:', self.smarts)
        if self.mode == 'ligand':
            if self.executable == 'contacts':
                self.process = ligand_los_contacts.LigandLoS(input_folder=self.los_home, output_folder=self.output,
                                                             dbname=self.db, annotations_file=self.annotations,
                                                             np=self.np, smarts=self.smarts,
                                                             smarts_index=self.smarts_index,
                                                             ligand_atom_type=self.ligand_atom_type,
                                                             smarts_filters=None, pi_atom=self.pi,
                                                             tau_atom=self.tau_atom, verbose=False)

            if self.executable == 'filter':
                self.process = los_postprocessing.LigandFilter(db=self.db, input_folder=self.output,
                                                               los_home=self.los_home, smarts_filters=None,
                                                               structure_quality_file=self.structure_quality,
                                                               search=None, protein_atom_types=None)

            if self.executable == 'angle':
                self.process = los_postprocessing.LigandAngle(db=self.db, input_folder=self.output, los_home=self.los_home,
                                                              smarts_filters=None,
                                                              structure_quality_file=self.structure_quality,
                                                              search=None, protein_atom_types=None,
                                                              angle_name=self.angle_name)

            if self.executable == 'h':
                self.process = los_postprocessing.LigandDistance(db=self.db, input_folder=self.output,
                                                                 los_home=self.los_home,
                                                                 smarts_filters=None,
                                                                 structure_quality_file=self.structure_quality,
                                                                 search=None, second_geometry_name='h', pi_atom=self.pi,
                                                                 protein_atom_types=None)

        if self.mode == 'protein':
            if self.executable == 'contacts':
                self.process = protein_los_contacts.ProteinLoS(input_folder=self.los_home, output_folder=self.output,
                                                               dbname=self.db, annotations_file=self.annotations,
                                                               smarts=self.smarts, smarts_index=self.smarts_index,
                                                               ligand_atom_type=self.ligand_atom_type, np=self.np,
                                                               smarts_filters=None, verbose=True)

            if self.executable == 'filter':
                self.process = los_postprocessing.ProteinFilter(db=self.db, input_folder=self.output,
                                                                los_home=self.los_home, smarts_filters=None,
                                                                structure_quality_file=self.structure_quality,
                                                                search=None, protein_atom_type=protein_atom_type)

            if self.executable == 'angle':
                self.process = los_postprocessing.ProteinAngle(db=self.db, input_folder=self.output,
                                                               los_home=self.los_home, smarts_filters=None,
                                                               structure_quality_file=self.structure_quality,
                                                               search=None, protein_atom_type=protein_atom_type)

            if self.executable == 'h':
                self.process = los_postprocessing.ProteinDistance(db=self.db, input_folder=self.output,
                                                                  los_home=self.los_home,
                                                                  smarts_filters=None,
                                                                  structure_quality_file=self.structure_quality,
                                                                  search=None, protein_atom_type=protein_atom_type)


def main():
    args = parse_args()
    rf_analysis = RfWrapper(los_home=args.los_home, db=args.database, annotation=args.annotation,
                            structure_quality=args.structure_quality, smarts=args.smarts,
                            smarts_index=args.smarts_index, ligand_atom_type=args.ligand_atom_type, pi=args.pi_atom,
                            np=args.np, executable=args.executable, mode=args.mode,
                            protein_atom_type=args.protein_atom_type, angle_name=args.angle_name)
    if rf_analysis.mode == 'ligand' and rf_analysis.executable == 'contacts':
        rf_analysis.process.query_smarts()

    elif rf_analysis.mode == 'protein' and rf_analysis.executable == 'contacts':
        rf_analysis.process.query_smarts()

    else:
        rf_analysis.process.run()


if __name__ == "__main__":
    main()
