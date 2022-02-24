#!/usr/bin/env python

'''
Extract central ligand from binding site.
'''

########################################################################################################################

from ccdc import io, protein, entry
from tqdm import tqdm
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
        default='ligands.sdf'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.input == 'internal':
        db = 'full_p2cq_roche_aug2021.csdsql'
    else:
        db = 'full_p2cq_pub_aug2021.csdsql'
    rdr = io.MoleculeReader(db)

    cofactor_list = protein.Protein.known_cofactor_codes() + ['AMP', 'ADP', 'ATP', 'GMP', 'GDP', 'GTP']

    with io.EntryWriter(args.output) as w:
        for ccdc_mol in rdr:
            central_ligand = [c for c in ccdc_mol.components if '_Z' in c.atoms[0].label]
            if central_ligand:
                central_ligand = central_ligand[0]
                if len(central_ligand.heavy_atoms) >= 10:
                    if central_ligand.heavy_atoms[0].residue_label[:3] not in cofactor_list:
                        central_ligand.identifier = ccdc_mol.identifier
                        central_ligand_entry = entry.Entry.from_molecule(central_ligand)
                        central_ligand_entry.attributes['strucid'] = ccdc_mol.identifier.split('_')[0]
                        w.write(central_ligand_entry)


if __name__ == '__main__':
    main()
