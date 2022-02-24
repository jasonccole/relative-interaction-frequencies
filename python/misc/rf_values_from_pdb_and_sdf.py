#!/usr/bin/env python

########################################################################################################################

import __future__
from ccdc_roche.python.rf_assignment import RfAssigner
from ccdc import io
import pandas as pd
import argparse

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Assign RF values to a ligand SDF file and a target protein structure and write them to a CSV file.

        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
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
        '-o',
        '--output',
        help='Output file with RF values',
        type=str,
        default='rf_values.csv'
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


def main():
    '''

    :return:

    '''
    args = parse_args()
    rf_assignments = []
    with io.MoleculeReader(args.ligands) as rdr:
        for cnt, ligand in enumerate(rdr):
            assigner = RfAssigner(ligand, verbose=args.verbose, target_file=args.protein)
            rf_assignment_df = assigner.rf_assignments
            rf_assignment_df['ligand_id'] = cnt
            rf_assignments.append(rf_assignment_df)
    pd.concat(rf_assignments, ignore_index=True).to_csv(args.output, index=False)
    return


if __name__ == "__main__":
    main()
