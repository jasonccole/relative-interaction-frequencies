#!/usr/bin/env python

########################################################################################################################

import _pickle as pickle

from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary
import time
from pathlib import Path
import argparse
from ccdc import io
from functools import partial
from pathos.multiprocessing import ProcessPool

# Pickle molecules contain properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        RDKit Supplier generator.
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
        '-n',
        '--nproc',
        help='Number of parallel processes for multiprocessing.',
        default=24,
        type=int
    )

    return parser.parse_args()


def get_rdkit_mol(file):
    '''

    :param file:
    :return:
    '''
    try:
        with io.EntryReader(str(file)) as rdr:
            for e in rdr:
                ccdc_mol = e.molecule
                m = Chem.MolFromMolBlock(ccdc_mol.to_string('sdf'))
                if m is None:
                    m = Chem.MolFromMolBlock(ccdc_mol.components[0].to_string('sdf'))
                    for c in ccdc_mol.components[1:]:
                        component = Chem.MolFromMolBlock(c.to_string('sdf'))
                        if component:
                            m = Chem.rdmolops.CombineMols(m, component)
                        else:
                            print(str(file))
                            return None

                for cnt, a in enumerate(e.molecule.heavy_atoms):
                    m.GetAtomWithIdx(cnt).SetProp('_TriposAtomName', a.label)

                m.SetProp('_Name', ccdc_mol.identifier)

        if type(m) == Chem.rdchem.Mol:
            return m
    except:
        return None


def multiprocessing(db, nproc):

    parallel_supplier = partial(get_rdkit_mol)
    pool = ProcessPool(nproc)
    mols = pool.map(parallel_supplier, db)
    return mols


def singleprocessing(db):

    mols = []
    for file in db:
        new_mol = get_rdkit_mol(file)
        mols.append(new_mol)
    return mols


def mol_supplier(db, nproc=1):
    if nproc == 1:
        mols = singleprocessing(db)
    else:
        mols = multiprocessing(db, nproc)
    return mols


def make_library(strucid_length, basename, nproc=1):
    mol2_files = Path('./binding_sites').glob('*.mol2')
    mol2_files = [str(f) for f in mol2_files if len(f.stem.split('_')[0]) == strucid_length]
    mols = mol_supplier(mol2_files, nproc)
    print('Generating Library...')
    library = rdSubstructLibrary.SubstructLibrary()
    for m in mols:
        if type(m) == Chem.rdchem.Mol:
            m.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(m)
            m = Chem.rdmolops.AddHs(m, explicitOnly=True)
            library.AddMol(m)
    print('Dumping subset...')
    with open(basename + f'.p', 'wb+') as rdkit_library:
        pickle.dump(library, rdkit_library, protocol=4)
    return


def main():
    args = parse_args()
    if args.input == 'public':
        basename = 'full_p2cq_pub_aug2021' #/home/tosstora/scratch/LoS
        strucid_length = 4
    elif args.input == 'internal':
        basename = '/home/tosstora/scratch/LoS/full_p2cq_roche_aug2021'
        strucid_length = 5
    t1 = time.time()
    make_library(strucid_length, basename, args.nproc)
    t2 = time.time()
    print("That took %.2f seconds." % (t2-t1))


if __name__ == '__main__':
    main()
