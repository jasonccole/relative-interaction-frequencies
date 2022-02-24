#!/usr/bin/env python

########################################################################################################################

from ccdc import io
from ccdc_roche.python.ligand_los_contacts import process_ligands
from pathlib import Path
import argparse

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Generate CSDSQL database from proasis mol2 files.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-i',
        '--input',
        default='public',
        choices=['public', 'internal']
    )

    parser.add_argument(
        '-o',
        '--output',
        help='Output filename.',
        default='database.csdsql'
    )

    return parser.parse_args()


def proasis_to_csdsql(path_to_proasis_mol2, output='database_.csdsql', strucid_len=4):
    # mol2_files = list(Path(path_to_proasis_mol2).glob('*_out/*_protonate3d_temp_relabel.mol2'))
    mol2_files = Path('./binding_sites').glob('*.mol2')
    mol2_files = [str(f) for f in mol2_files if len(f.stem.split('_')[0]) == strucid_len]
    # mol2_files = ['2rfc.mol2']
    # mol2_files = [str(path_to_proasis_mol2 / Path(f)) for f in mol2_files]

    with io.MoleculeWriter(output) as w, io.MoleculeReader(mol2_files) as rdr:
        identifiers = []
        for ccdc_mol in rdr:
            if ccdc_mol.identifier in identifiers:
                continue

            w.write(ccdc_mol)
            identifiers.append(ccdc_mol.identifier)
    return


def main():
    args = parse_args()
    if args.input == 'public':
        args.input = 'PPMOL2_FULL'
        strucid_len = 4
        args.output = 'public.csdsql'

    if args.input == 'internal':
        args.input = 'PPMOL2_FULL'
        strucid_len = 5
        args.output = 'internal.csdsql'
    proasis_to_csdsql(args.input, args.output, strucid_len)


if __name__ == '__main__':
    main()
