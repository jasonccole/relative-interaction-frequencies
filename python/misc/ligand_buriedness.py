#!/usr/bin/env python

'''
Calculate the percentage of ligand surface that is solvent exposed
'''

########################################################################################################################

from ccdc import io, protein
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool
from pathos.helpers import freeze_support
from functools import partial
import argparse

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Submit Calculate Rf values on Basel HPC
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-nproc',
        help='Number of parallel processes for multiprocessing.',
        default=24
    )

    parser.add_argument(
        '-db',
        help='DB internal or external',
        default='internal'
    )

    parser.add_argument(
        '--los_home',
        help='LoS home directory with lookup files and ligand_atom_types.csv and protein_atom_types.csv',
        default=''
    )

    return parser.parse_args()


def calculate_buried_surface(mol):
    '''
    imp_sasa: SASA in presence of water molecules.
    exp_sasa: SASA covered by explicit water molecules = no_water_sasa - imp_sasa
    :param mol:
    :return:
    '''
    try:
        identifier = mol.identifier
        ligand_atoms = [a for a in mol.atoms if '_Z' in a.label]

        implicit_sasa = np.sum([a.solvent_accessible_surface() for a in ligand_atoms])

        mol.remove_all_waters()

        bound_ligand_sasa = np.sum([a.solvent_accessible_surface() for a in ligand_atoms])

        remove_atoms = [a for a in mol.atoms if a not in ligand_atoms]
        mol.remove_atoms(remove_atoms)

        # ligand = [c for c in mol.components if '_Z' in c.atoms[0].label][0]
        unbound_ligand_sasa = np.sum([a.solvent_accessible_surface() for a in mol.atoms])
        pc_buried = (unbound_ligand_sasa-bound_ligand_sasa)/unbound_ligand_sasa

        pc_implicit_sasa = implicit_sasa/unbound_ligand_sasa

        print(identifier)
        return bound_ligand_sasa, unbound_ligand_sasa, pc_buried, identifier, implicit_sasa, pc_implicit_sasa
    except:
        return np.nan, np.nan, np.nan, identifier, np.nan, np.nan


def _mp_calculate_buried_surface(i):
    try:
        r = io.EntryReader('full_p2cq_pub_oct2019.csdsql')
        e = r[i]
        p = protein.Protein.from_entry(e)
        bound_ligand_sasa, unbound_ligand_sasa, pc_buried, identifier, implicit_sasa, pc_implicit_sasa = calculate_buried_surface(p)
        return bound_ligand_sasa, unbound_ligand_sasa, pc_buried, identifier, implicit_sasa, pc_implicit_sasa

    except:
        return np.nan, np.nan, np.nan, identifier, np.nan, np.nan


def single_processing():
    bound_ligand_sasa_list = []
    unbound_ligand_sasa_list = []
    pc_buried_list = []
    identifiers = []
    implicit_sasa_list = []
    pc_implicit_sasa_list = []
    for p in io.EntryReader('full_p2cq_pub_oct2019.csdsql'):
        p = protein.Protein.from_entry(p)
        bound_ligand_sasa, unbound_ligand_sasa, pc_buried, identifier, implicit_sasa, pc_implicit_sasa = calculate_buried_surface(p)
        bound_ligand_sasa_list.append(bound_ligand_sasa)
        unbound_ligand_sasa_list.append(unbound_ligand_sasa)
        pc_buried_list.append(pc_buried)
        identifiers.append(identifier)
        implicit_sasa_list.append(implicit_sasa)
        pc_implicit_sasa_list.append(pc_implicit_sasa)

    df = pd.DataFrame({'bound_ligand_sasa': bound_ligand_sasa_list,
                       'unbound_ligand_sasa_list': unbound_ligand_sasa_list,
                       'pc_buried_list': pc_buried_list, 'identifiers': identifiers,
                       'pc_implicit_sasa': pc_implicit_sasa_list, 'implicit_sasa': implicit_sasa_list})

    df.to_csv('ligand_buriedness.csv', index=False)


def multi_processing(db, nproc=24):
    parallel_buriedness = partial(_mp_calculate_buried_surface)
    pool = ProcessingPool(nproc)
    output = [i for i in pool.map(parallel_buriedness, range(len(io.EntryReader('full_p2cq_pub_oct2019.csdsql')))) if i is not None]
    bound_ligand_sasa_list = [o[0] for o in output]
    unbound_ligand_sasa_list = [o[1] for o in output]
    pc_buried_list = [o[2] for o in output]
    identifiers = [o[3] for o in output]
    implicit_sasa_list = [o[4] for o in output]
    pc_implicit_sasa_list = [o[5] for o in output]

    df = pd.DataFrame({'bound_ligand_sasa': bound_ligand_sasa_list,
                       'unbound_ligand_sasa_list': unbound_ligand_sasa_list,
                       'pc_buried_list': pc_buried_list, 'identifiers': identifiers,
                       'pc_implicit_sasa': pc_implicit_sasa_list, 'implicit_sasa': implicit_sasa_list})

    df.to_csv('ligand_buriedness.csv', index=False)


def main():
    args = parse_args()
    multi_processing(db=args.db, nproc=int(args.nproc))
    # single_processing()


if __name__ == '__main__':
    freeze_support()  # For multiprocessing in Windows.
    main()

