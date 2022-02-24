#!/usr/bin/env python

########################################################################################################################

import __future__
from ccdc import io, protein
from ccdc_roche.python.los_utilities import assign_index_to_atom_partial_charge
from ccdc_roche.python.p2cq_filter import is_glycol
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

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


def find_mutation_series():
    import pandas as pd
    platinum_db = pd.read_csv('../mutation_series/platinum_flat_file.csv')
    platinum_db = platinum_db[platinum_db['MUT.IN_BINDING_SITE'] == 'YES']
    platinum_db = platinum_db[(platinum_db['MUT.MT_PDB'] != 'NO') & (platinum_db['MUT.WT_PDB'] != 'NO')]
    platinum_db = platinum_db[(platinum_db['MUT.IS_SINGLE_POINT'] != 'NO')]
    platinum_db = platinum_db[(platinum_db['MUT.DISTANCE_TO_LIG'] <= 5)]
    platinum_db = platinum_db[(platinum_db['PROT.RESOLUTION'] <= 2.5)]
    platinum_db.to_csv('../mutation_series/platinum_flat_file_filtered.csv')
    rf_assignments = pd.read_csv('full_p2cq_pub_oct2019_rf.csv')
    rf_assignments = rf_assignments[rf_assignments['rf_total'] <= 0.8]
    rf_assignments = rf_assignments[(rf_assignments['identifier'].apply(lambda x: x.split('_')[0]).isin(platinum_db['MUT.MT_PDB'])) |
                                    (rf_assignments['identifier'].apply(lambda x: x.split('_')[0]).isin(platinum_db['MUT.WT_PDB']))]
    rf_assignments.to_csv('../mutation_series/mutation_series_entries.csv', index=False)


def find_ligand_series():
    '''
    Find entries that are in the BindingDB Protein-Ligand-Validation set.
    :return:
    '''
    import pandas as pd
    import itertools
    binding_db = pd.read_csv('../ligand_series/binding_db/binding_db.csv')['Series'].unique()
    binding_db = [series.split(' '[0]) for series in binding_db]
    binding_db = [series for series in binding_db if len(series) > 1]
    binding_db = list(itertools.chain(*binding_db))
    binding_db = [identifier.split('-')[0] for identifier in binding_db]
    rf_assignments = pd.read_csv('full_p2cq_pub_oct2019_rf.csv')
    rf_assignments = rf_assignments[rf_assignments['rf_total'] <= 0.8]
    rf_assignments = rf_assignments[rf_assignments['identifier'].apply(lambda x: x.split('_')[0]).isin(binding_db)]
    rf_assignments.to_csv('prot_lig_val.csv', index=False)


def export_entry(entry, database, output_folder='.', remove_waters=True):
    cwd = os.getcwd()
    os.chdir(output_folder)
    if remove_waters:
        output_name = f'{entry}.mol2'
    else:
        output_name = f'{entry}_wet.mol2'
    with io.EntryReader(database) as rdr, io.MoleculeWriter(output_name, format='mol2') as wrt:
            e = rdr.entry(entry)
            p = protein.Protein.from_entry(e)
            if remove_waters:
                p.remove_all_waters()
            p = assign_index_to_atom_partial_charge(p)
            wrt.write(p)
    os.chdir(cwd)


class DatabaseUtil(object):
    def __init__(self, los_home=''):
        self.los_home = los_home
        self.sq_df = pd.read_csv(Path(self.los_home) / 'data_quality_aug2021.csv')
        self.annotation_df = pd.read_csv(Path(self.los_home) / 'all_annotated_aug2021.csv')
        self.cofactor_list = protein.Protein.known_cofactor_codes() + ['AMP', 'ADP', 'ATP', 'GMP', 'GDP', 'GTP']

    def extend_rf_file(self, db_rf_df):
        '''
        Add additional information to the DataFrame, such as counts of unfavorable and favorable interactions,
        Uniprot ID.
        :return:
        '''

        # add RSCC, add resolution, Uniprot ID, project
        def return_struc_qual(identifier):
            rscc, ligand_avgoccu, ligand_altcode = np.nan, np.nan, np.nan
            is_cofactor = False
            if len(identifier) == 8:
                struc_qual = self.sq_df[self.sq_df['identifier'] == identifier].squeeze()
                if len(struc_qual) > 0:
                    rscc = struc_qual['ligand_rscc']
                    ligand_avgoccu = struc_qual['ligand_avgoccu']
                    ligand_altcode = struc_qual['ligand_altcode']
                    ligand_name = struc_qual['ligand_name'][:3]
                    if ligand_name in self.cofactor_list:
                        is_cofactor = True
            return rscc, ligand_avgoccu, ligand_altcode, is_cofactor

        def return_annotation(identifier):
            project, resolution, uniprot = np.nan, np.nan, np.nan
            id = identifier.split('_')[0].lower()
            annotation = self.annotation_df[self.annotation_df['STRUCID'] == id].squeeze()
            if len(annotation) > 0:
                project = annotation['PROA_PROJ']
                resolution = annotation['RESOLUTION']
                uniprot = annotation['UNIP_ID']

            return project, resolution, uniprot

        identifier = db_rf_df['identifier'].unique()[0]
        rscc, ligand_avgoccu, ligand_altcode, is_cofactor = return_struc_qual(identifier)
        project, resolution, uniprot = return_annotation(identifier)
        ligand_smiles = db_rf_df['ligand_smiles'].values[0]
        db_rf_df['uniprot'] = uniprot
        db_rf_df['ligand_rscc'] = rscc
        db_rf_df['project'] = project
        db_rf_df['resolution'] = resolution
        db_rf_df['ligand_altcode'] = ligand_altcode
        db_rf_df['ligand_avgoccu'] = ligand_avgoccu
        db_rf_df['is_cofactor'] = is_cofactor
        db_rf_df['is_glycol'] = is_glycol(ligand_smiles)

        return db_rf_df

    def concatenate_lookup_files(self, search_home):
        from pathlib import Path
        import os
        import pandas as pd
        ligand_atom_types = pd.read_csv(Path(search_home) / Path('ligand_atom_types.csv'), sep='\t')['ligand_atom_type'].unique()
        p = Path(search_home) / 'output'
        folders = next(os.walk(p))[1]
        lookup_types = set()
        for folder in folders:
            if folder not in ligand_atom_types:
                continue
            files = p.joinpath(folder).glob('*lookup*.csv')
            for file in files:
                lookup_types.add(os.path.basename(file).split('.')[0])

        for lookup_type in lookup_types:
            global_lookup = []
            for folder in folders:
                if folder not in ligand_atom_types:
                    continue
                lookup_file = p / Path(folder) / f'{lookup_type}.csv'
                if lookup_file.is_file():
                    global_lookup.append(pd.read_csv(lookup_file))
            global_lookup = pd.concat(global_lookup, ignore_index=True)
            global_lookup.to_csv(f'global_{lookup_type}.csv', sep='\t', index=False)

    def assign_rf_to_database(self, db='full_p2cq_pub_oct2019.csdsql', nproc=24):

        from ccdc import io, protein
        from ccdc_roche.python import los_descriptors
        from pathos.multiprocessing import ProcessingPool
        from functools import partial
        import pandas as pd

        def return_rf_structure_df(entry_index, dbpath):
            with io.EntryReader(dbpath) as rdr:
                entry = rdr[entry_index]
                p = protein.Protein.from_entry(entry)
            try:
                print(p.identifier)
                describer = los_descriptors.CsdDescriptorsFromProasis(p)
                contact_df = describer.los_contacts_df
                # csd_ligand = describer.csd_ligand
                # rf_count_df = los_descriptors.rf_count_df(contact_df, csd_ligand)
                # rf_count_df = pd.concat([rf_count_df]*contact_df.shape[0], ignore_index=True)
                # rf_structure_df = contact_df.join(rf_count_df.drop('identifier', axis=1))
                self.extend_rf_file(contact_df)
            except:
                contact_df = pd.DataFrame()

            return contact_df

        def multiprocessing(nproc, db):
            parallel_return_structure_df = partial(return_rf_structure_df, dbpath=db)
            pool = ProcessingPool(nproc)
            it = range(len(io.MoleculeReader(db)))
            output = pool.map(parallel_return_structure_df, it)
            output_df = pd.concat(output)
            return output_df

        def single_processing(db):
            output = []
            for entry_index in range(len(io.MoleculeReader(db))):
                output.append(return_rf_structure_df(entry_index, dbpath=db))
                if entry_index == 5:
                    break
            output_df = pd.concat(output)
            return output_df

        if db == 'public':
            db = (Path(self.los_home) / 'full_p2cq_pub_aug2021.csdsql').resolve()
        elif db == 'internal':
            db = ((Path(self.los_home) / 'full_p2cq_roche_aug2021.csdsql').resolve())
        print(db)
        output_name = db.stem

        if nproc > 1:
            output_df = multiprocessing(nproc, str(db))
        else:
            output_df = single_processing(str(db))

        output_df.columns = [str(c) for c in output_df.columns]
        output_df.to_parquet(f'{output_name}_rf_test.gzip', compression='gzip')
        output_df.to_csv(f'{output_name}_rf_test.csv', index=False, sep='\t')


def main():
    args = parse_args()
    database_utiliser = DatabaseUtil(los_home=args.los_home)
    database_utiliser.assign_rf_to_database(db=args.db, nproc=int(args.nproc))


if __name__ == "__main__":
    main()
