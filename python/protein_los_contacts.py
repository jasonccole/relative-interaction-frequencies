# !/usr/bin/env python
#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2019-08-14: created by the Cambridge Crystallographic Data Centre
#
'''
Generate los.csv and complex.csv files, which are input for Rf calculation.
los.csv contains information on Line of sight contact atoms, e.g. protein atom types.
complex.csv contains information on the entry, e.g. binding site surface area, resolution of crystal structure.
'''

###################################################################################################################

import __future__
from ccdc_roche.python.los_utilities import parse_args, Query, substructure_search, identify_covalent_query_atom, \
    _get_rdkit_dbs
from ccdc_roche.python.ligand_los_contacts import LigandLoS
import pandas as pd
import numpy as np
import datetime
import os
from los_utilities import return_los_contacts, return_ligand_contacts
from pathos.multiprocessing import ProcessingPool
from pathos.helpers import freeze_support
from functools import partial
from ccdc import io
from ccdc_roche import atom_types
import pickle
from pathlib import Path

###################################################################################################################


class ProteinLoS(object):

    def __init__(self, np=1, dbname='database.csdsql', input_folder='input', output_folder='output',
                 keep_waters=False, annotations_file='annotation.csv',
                 debug=False, verbose=False, smarts_queries_file=None, smarts=None, smarts_index=None,
                 smarts_filters=None, ligand_atom_type=None, rdkit=True, keep_good_waters=True):

        self.los_home = input_folder

        if smarts_queries_file is None and smarts is None:
            raise Exception('Specify SMARTS.')

        if smarts_queries_file is None:
            self.smarts_queries_file = None

        # Small molecule substructure
        if smarts_queries_file is not None:
            self.smarts_queries_file = smarts_queries_file
            self.smarts_queries_df = pd.read_csv(Path(self.los_home) / self.smarts_queries_file, na_values='0')
            self.smarts_queries = self.smarts_queries_df['SMARTS'].to_list()
            self.smarts_indices = [smarts_index.split(';') for smarts_index in self.smarts_queries_df['SMARTS_index']]

        if smarts is not None:
            if smarts_index is not None:
                self.smarts = smarts
                self.smarts_index = smarts_index
                self.smarts_queries = [smarts]
                self.smarts_indices = [smarts_index]
            else:
                raise Exception('Please specify a smarts index to define the contact atom.')
            if smarts_filters is not None:
                self.smarts_filters = smarts_filters
                self.filter_queries = [Query(smarts_filter, 'SMARTS', 0) for smarts_filter in self.smarts_filters]

            else:
                self.filter_queries = None
        self.rdkit = rdkit
        self.ligand_atom_type = ligand_atom_type
        self.np = int(np)
        if type(dbname) == str:
            self.dbname = [dbname]
            self.los_db = [Path(self.los_home) / Path(_dbname) for _dbname in self.dbname]
        if type(dbname) == list:
            self.dbname = dbname
            self.los_db = [Path(self.los_home) / Path(_dbname) for _dbname in self.dbname]
        if type(dbname) == 'Protein':
            self.dbname = dbname

        self.output = Path(output_folder)
        if not self.output.is_dir():
            self.output.mkdir()
        self.annotations_file = annotations_file
        self.annotations = pd.read_csv(Path(self.los_home) / annotations_file, dtype={'UNIP_ID': str},
                                       na_values='\"')
        self.keep_waters = keep_waters
        self.keep_good_waters = keep_good_waters

        # Protein atom types
        atom_type_path = Path(atom_types.__path__[0])
        self.protein_atom_types = pd.read_csv(atom_type_path / 'protein_atom_types.csv', sep='\t')
        if self.keep_waters:
            water_df = pd.DataFrame({'SMARTS': ['[OX0]'], 'protein_atom_type': ['Water'], 'SMARTS_index': ['0'],
                                     'atomic_symbol': ['O']})
            self.protein_atom_types = self.protein_atom_types.append(water_df, ignore_index=True, sort=False)
        if debug:
            self.debug = True
        else:
            self.debug = False
        if verbose:
            self.verbose = True
        else:
            self.verbose = False
        self.unique_pats = self.protein_atom_types['protein_atom_type'].drop_duplicates().to_list() + \
                           ['other_ligands', 'metal']
        self.los_columns = ['RDKit_SMARTS', 'RDKit_SMARTS_index', 'res_name', 'molecule_name', 'distance',
                            'vdw_distance', 'alpha_i', 'solv_acc_area', 'substructure_match', 'query_atom_id',
                            'los_atom_id', 'los_atom_type']
        #'other_ligands', 'metal', 'other_central_ligand', 'query_match', 'protein'
        self.complex_columns = ['uniprot_id', 'ligand_SMILES', 'molecule_name', 'resolution'] \
                               + [''.join(['surf_area_', unique_pat]) for unique_pat in ['other_ligands', 'metal',
                                                                                         'other_central_ligand',
                                                                                         'query_match',
                                                                                         'protein', 'n_lig_in_bs']]
        self.aa_labels = {'ARG', 'MET', 'SER', 'PHE', 'ALA', 'TYR', 'THR', 'LEU', 'ILE', 'VAL', 'CYS', 'GLU', 'GLN',
                          'ASN', 'PRO', 'ASP', 'GLY', 'LYS', 'HIS', 'TRP'}

        self.los = LigandLoS(dbname=self.dbname, input_folder=self.los_home, output_folder=self.output,
                             keep_waters=False, annotations_file=self.annotations_file, debug=self.debug,
                             verbose=self.verbose, smarts_queries_file=self.smarts_queries_file,
                             smarts=self.smarts, smarts_index=self.smarts_index, ligand_atom_type=self.ligand_atom_type)

        # settings
        self.interaction_cutoff = 0.5

    def write_complex_df(self, hit_protein, n_ligands, central_ligand_atoms, ligand_contacts):
        import pandas as pd
        complex_row = pd.DataFrame(columns=self.complex_columns)
        complex_row.loc[0] = 0
        complex_row.loc[0, 'uniprot_id'] = hit_protein.uniprot_id
        complex_row.loc[0, 'ligand_SMILES'] = hit_protein.ligand_SMILES
        complex_row.loc[0, 'molecule_name'] = hit_protein.identifier
        complex_row.loc[0, 'resolution'] = hit_protein.resolution
        complex_row.loc[0, 'n_lig_in_bs'] = n_ligands
        binding_site_atoms = central_ligand_atoms + list(ligand_contacts)
        for binding_site_atom in binding_site_atoms:
            if binding_site_atom.los_at in self.protein_atom_types['protein_atom_type'].unique() \
                    or binding_site_atom.los_at == 'other_ligands' or binding_site_atom.los_at == 'metal':
                if binding_site_atom.los_at in self.protein_atom_types['protein_atom_type'].unique():
                    pat = 'protein'
                else:
                    pat = binding_site_atom.los_at
                complex_row = self.los.add_to_complex_row(complex_row,
                                                          hit_protein.atom(binding_site_atom.label),
                                                          pat=pat)

            else:
                complex_row = self.los.add_to_complex_row(complex_row,
                                                          hit_protein.atom(binding_site_atom.label),
                                                          pat=binding_site_atom.los_at, ligand_atoms=ligand_contacts)

        return complex_row

    def add_to_los_df(self, query_string, query_index, los_atom, pat=None):
        '''
        Write row to los.csv Data Frame.
        :param query_index: Indices to define contact_atom and contact_angle_atom
        :param query_string: SMARTS string .
        :param los_df:
        :param los_atom:
        :param pat:
        :return: Data Frame
        '''
        import pandas as pd
        import numpy as np
        row = pd.DataFrame(columns=self.los_columns)
        row[['RDKit_SMARTS', 'RDKit_SMARTS_index', 'res_name', 'molecule_name', 'distance', 'vdw_distance',
             'substructure_match', 'query_atom_id', 'los_atom_type']] = \
            pd.DataFrame(
                [(query_string, query_index, los_atom.residue_label, los_atom.identifier, los_atom.distance,
                  los_atom.vdw_distance, los_atom.substructure_match, los_atom.id_lig + 1, pat)]
            )

        if los_atom.residue_label == 'SOL':
            row['solv_acc_area'] = [los_atom.SAS]
            row['los_atom_id'] = [-1]
            row['intramolecular_contact'] = [0]
        else:
            row['solv_acc_area'] = [0]
            row['los_atom_id'] = [los_atom.index + 1]
            row['intramolecular_contact'] = [1 - los_atom.het_group]
        if pat is not None:
            row['intramolecular_contact'] = [0]
        row['los_atom_label'] = [los_atom.label]
        row = row.fillna(0)
        row['alpha_i'] = [np.nan]
        try:
            row['h'] = [los_atom.h]
        except (ValueError, AttributeError):
            row['h'] = [np.nan]

        return row

    def make_output_dataframes(self):
        import pandas as pd
        output_complex_dataframes = {}
        output_los_dataframes = {}
        for pat in self.unique_pats:
            output_complex_dataframes[pat] = pd.DataFrame()
            output_los_dataframes[pat] = pd.DataFrame(columns=self.los_columns)
        return output_complex_dataframes, output_los_dataframes

    def write_los_contacts(self, los_atoms, los_df, query_string, query_index, central_atom):
        # Protein atom types SMARTS
        rows = []
        for los_atom in los_atoms:
            if los_atom.protein_atom_type == 'Ligand' and not hasattr(los_atom, 'los_at'):
                los_atom.residue_label = 'LIG'
                rows.append(self.add_to_los_df(query_string, query_index, los_atom))
            elif los_atom.protein_atom_type == 'Unknown' and not hasattr(los_atom, 'los_at'):
                rows.append(self.add_to_los_df(query_string, query_index, los_atom))
            elif hasattr(los_atom, 'ligand_index') and los_atom.ligand_index is not None and not hasattr(los_atom,
                                                                                                         'los_at'):
                rows.append(self.add_to_los_df(query_string, query_index, los_atom))
            else:
                if los_atom.los_at in self.protein_atom_types['protein_atom_type'].unique():
                    pat = 'protein'
                else:
                    pat = los_atom.los_at
                rows.append(self.add_to_los_df(query_string, query_index, los_atom, pat=pat))
        rows = pd.concat(rows, ignore_index=True)
        rows['query_atom_label'] = central_atom.label
        los_df = los_df.append(rows, ignore_index=True)
        return los_df

    def match_los_atoms(self, query_atom, ligands, hit_protein, ligand_contacts, output_complex_dataframes,
                        output_los_dataframes, query, rdkit_db=None):
        '''

        :return: Dictionary with DataFrame for each key.
        '''
        from ccdc_roche.python.los_utilities import _assign_los_atom_type
        from ccdc import descriptors
        substructure_match_dictionary = {}
        central_ligand_atoms = ligands[query_atom.ligand_index]
        central_ligand_atoms = _assign_los_atom_type(central_ligand_atoms, hit_protein, query, self.filter_queries,
                                                     mode='protein', protein_atom_types_df=self.protein_atom_types,
                                                     rdkit_db=rdkit_db)
        complex_matrices = self.make_complex_matrices(
            output_complex_dataframes[list(output_complex_dataframes.keys())[0]], hit_protein, central_ligand_atoms,
            ligand_contacts,
            ligands)
        for protein_atom_type in set(self.unique_pats):
            substructure_match_dictionary[protein_atom_type] = 0
        for contact_atom in ligand_contacts:  # contact_atom is a protein atom
            if '_Z' not in contact_atom.label:
                if identify_covalent_query_atom(contact_atom, hit_protein):
                    continue

                protein_atom_type = contact_atom.los_at  # return_los_pat(contact_atom, hit_protein, self.protein_atom_types)
                substructure_match_dictionary[protein_atom_type] += 1
                if substructure_match_dictionary[protein_atom_type] == 1:
                    output_complex_dataframes[protein_atom_type] = complex_matrices

                output_los_dataframes[protein_atom_type] = self.make_los_df(
                    output_los_dataframes[protein_atom_type], contact_atom, hit_protein,
                    ligands[query_atom.ligand_index], substructure_match_dictionary[protein_atom_type],
                    query, rdkit_db=rdkit_db)

                if len(contact_atom.neighbours) != 0:
                    new_df_list = []
                    df = output_los_dataframes[protein_atom_type]
                    for contact_angle_atom in contact_atom.neighbours:
                        for index, row in df[df['query_atom_label'] == contact_atom.label].iterrows():
                            if row['res_name'] == 'SOL':
                                row['alpha_i'] = np.nan
                            else:
                                row['alpha_i'] = descriptors.MolecularDescriptors.atom_angle(
                                    contact_angle_atom, contact_atom, hit_protein.atom(row['los_atom_label']))
                            new_df_list.append(row)
                    output_los_dataframes[protein_atom_type] = pd.concat(new_df_list, ignore_index=True, axis=1).transpose()

        return output_los_dataframes, output_complex_dataframes

    def make_complex_matrices(self, complex_matrices, hit_protein, central_ligand_atoms, ligand_contacts, ligands):
        '''
        Generate a Data Frame containing information for complex.csv.
        :param central_ligand_atoms:
        :param complex_matrices:
        :param ligand_contacts:
        :param query_atom:
        :param hit_protein:
        :param ligands:
        :return: Data Frame
        '''
        # Write complex matrix
        import pandas as pd
        complex_df = self.write_complex_df(hit_protein, len(ligands), central_ligand_atoms, list(ligand_contacts))
        complex_matrices = pd.concat([complex_matrices, complex_df], sort=False)
        return complex_matrices

    def update_atom_attributes(self, atom_list, query_atom, hit_protein, substructure_match):
        from ccdc import descriptors
        updated_atom_list = []
        for atom in atom_list:
            atom.het_group = self.los.het_group_identifier(atom)
            atom.id_lig = query_atom.index
            atom.identifier = hit_protein.identifier
            atom.substructure_match = substructure_match
            atom.distance = descriptors.MolecularDescriptors.atom_distance(query_atom, atom)
            updated_atom_list.append(atom)
        return updated_atom_list

    def make_los_df(self, los_df, central_atom, hit_protein, central_ligand_atoms, substructure_match, query,
                    rdkit_db=None):
        '''
        Generate Data Frame containing information for los.csv
        :param query:
        :param geom_hits:
        :param los_df:
        :param central_atom: Is part of the binding site. We calculate LoS contacts to this atom.
        :param contact_angle_atom:
        :param hit_protein:
        :param ligand_contacts:
        :param ligands:
        :param substructure_match:
        :return: Data Frame
        '''
        from ccdc_roche.python.atom_geometries import PlaneDistanceCalculator
        from ccdc_roche.python.los_utilities import ligand_index, _assign_los_atom_type
        # find LoS_contacts

        los_atoms = return_los_contacts(central_atom, hit_protein, hit_protein.atoms, [])
        for atom in los_atoms:
            atom.ligand_index = ligand_index([central_ligand_atoms], atom)
        los_atoms = self.update_atom_attributes(los_atoms, central_atom, hit_protein, substructure_match)

        los_atoms = _assign_los_atom_type(los_atoms, hit_protein, query, self.filter_queries, mode='protein',
                                          protein_atom_types_df=self.protein_atom_types, rdkit_db=rdkit_db)

        if 'pi' in central_atom.los_at:
            for los_atom in los_atoms:
                h = PlaneDistanceCalculator(central_atom, los_atom).return_plane_distance()
                los_atom.h = h

        los_atoms.append(self.los.create_dummy_sol(central_atom, hit_protein, substructure_match))

        los_df = self.write_los_contacts(los_atoms, los_df, query.query, query.index, central_atom)

        return los_df

    def los_complex_per_hit_wrapper(self, iterrow, query, entry_reader=None):
        '''
        Generate Data Frames for complex.csv and los.csv for each hit.
        :param query_string:
        :param substructure:
        :param entry_identifier: String
        :param contact_atom_index: SMARTS indices defining query atom and contact angle atom.
        :param smarts_query: SMARTS query as string
        :param entry_reader: Optional. If reader is not passed, it will be created upon execution.
        :return: los.csv Data Frame and complex.csv Data Frame
        '''
        import traceback
        import sys
        from ccdc_roche.python.los_utilities import identify_covalent_query_atom
        from ccdc_roche.python.los_utilities import get_query_atoms, _assign_los_atom_type, ligand_index, get_hit_protein
        self.los.los_columns = self.los_columns

        try:
            if self.rdkit:
                index, row = iterrow
                entry_identifier = row['identifier']
                rdkit_hit_protein = row['rdkit_mol']
            hit_protein = get_hit_protein(entry_reader, entry_identifier, self.los.rdr())

            hits = substructure_search(query=query, db=hit_protein, filter_queries=self.filter_queries,
                                       return_identifiers=False, rdkit=self.rdkit, rdkit_db=rdkit_hit_protein)
            # cleanup hits for multiple neighbours
            unique_query_atoms = []
            cleaned_hits = []
            for hit in hits:
                query_atoms = get_query_atoms(hit, query, hit_protein, rdkit=self.rdkit, rdkit_db=rdkit_hit_protein)
                if query_atoms not in unique_query_atoms:
                    unique_query_atoms.append(query_atoms)
                    cleaned_hits.append(hit)

            output_complex_dataframes, output_los_dataframes = self.make_output_dataframes()
            if len(hits) >= 1:
                query_atom = get_query_atoms(hits[0], query, hit_protein, rdkit=True, rdkit_db=rdkit_hit_protein)[0]

                query_atom, ligands = process_ligands(query_atom, hit_protein)

                # skip hit if query atom or its neighbour is covalently bound to protein
                if identify_covalent_query_atom(query_atom, hit_protein):
                    return

                hit_protein = self.los.process_hit(hit_protein, self.annotations, query_atom)

                # get all atoms in contact with ligand
                ligand_contacts = return_ligand_contacts(ligands, query_atom, hit_protein)
                ligand_contacts = _assign_los_atom_type(ligand_contacts, hit_protein, query, mode='protein',
                                                        protein_atom_types_df=self.protein_atom_types,
                                                        rdkit_db=rdkit_hit_protein)
                for ligand_contact_atom in ligand_contacts:
                    ligand_contact_atom.ligand_index = ligand_index(ligands, ligand_contact_atom)

                output_los_dataframes, output_complex_dataframes = self.match_los_atoms(query_atom,
                                                                                        ligands,
                                                                                        hit_protein,
                                                                                        ligand_contacts,
                                                                                        output_complex_dataframes,
                                                                                        output_los_dataframes,
                                                                                        query, rdkit_db=rdkit_hit_protein)
                return output_los_dataframes, output_complex_dataframes
            else:
                pass
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            # write out bad entry
            print('Entry {} caused a problem. Continuing to next hit.'.format(entry_identifier))
            if self.los.verbose:
                traceback.print_exc(file=sys.stdout)
                print(e)

    def run_los_single_processing(self, hit_identifiers, query):
        # No multiprocessing of hits.
        los_matrices = []
        complex_matrices = []
        entry_reader = io.EntryReader([str(db) for db in self.los_db])
        for hit_identifier in hit_identifiers.iterrows():
            list_of_tuples_of_dicts = self.los_complex_per_hit_wrapper(hit_identifier, query,
                                                                       entry_reader=entry_reader)
            if list_of_tuples_of_dicts is not None:
                los_matrices.append(list_of_tuples_of_dicts[0])
                complex_matrices.append(list_of_tuples_of_dicts[1])
        return los_matrices, complex_matrices  # dictionary of lists

    def run_los_multiprocessing(self, hit_identifiers, query):
        # Multiprocessing of hits
        parallel_los_complex_per_hit_wrapper = partial(self.los_complex_per_hit_wrapper, query=query)
        # parallel_los_complex_per_hit_wrapper = partial(self.profiler, query=query)
        pool = ProcessingPool(self.np)
        list_of_tuples_of_dicts = [i for i in pool.map(parallel_los_complex_per_hit_wrapper, hit_identifiers.iterrows())
                                   if
                                   i is not None]
        if list_of_tuples_of_dicts is not None:
            list_of_dicts = list(zip(*list_of_tuples_of_dicts))
            list_of_los_dicts = list_of_dicts[0]
            list_of_complex_dicts = list_of_dicts[1]
        return list_of_los_dicts, list_of_complex_dicts

    def run_los(self, query):
        '''
        Run line of sight calculation. If np >= 2, multiprocessing is used. Else, no multiprocessing but a simple for
        loop.
        :param substructure:
        :param contact_atom_index:
        :param smarts_query:
        :return: los_df, complex_df
        '''

        if self.debug:
            if not Path('Debug').is_dir():
                Path('Debug').mkdir()

        print(''.join(['The LoS calculation started at ', str(datetime.datetime.now())]))

        # get only entries with relevant ligand atom type
        if self.rdkit:
            rdkit_dbs = _get_rdkit_dbs(self.los_db, self.los_home)

            hit_identifiers = pd.DataFrame()
            for rdkit_db in rdkit_dbs:
                if '.mol2' != self.los_db[0].stem:
                    with open(rdkit_db, 'rb') as rdkit_db:
                        rdkit_db = pickle.load(rdkit_db)

                print('Running Substructure search...')
                print('Selecting only hits that match the _Z label and have only one HET group matching the query.')
                hit_identifiers = hit_identifiers.append(substructure_search(query, db=io.EntryReader([str(db) for db in self.los_db]),
                                                                        return_identifiers=True, rdkit=self.rdkit,
                                                                        rdkit_db=rdkit_db), ignore_index=True)

                print(''.join([str(len(hit_identifiers)), ' entries match the query.']))

        else:
            print('Running Substructure search...')
            print('Selecting only hits that match the _Z label and have only one HET group matching the query.')
            hit_identifiers = substructure_search(query, db=io.EntryReader([str(db) for db in self.los_db]), return_identifiers=True)
            print(''.join([str(len(hit_identifiers)), ' entries match the query.']))

        if self.np >= 2:
            los_matrices, complex_matrices = self.run_los_multiprocessing(hit_identifiers, query)
        else:
            los_matrices, complex_matrices = self.run_los_single_processing(hit_identifiers, query)
        try:
            global_los_dictionary = {}
            for query_name in los_matrices[0]:
                global_los_dictionary[query_name] = []
            for los_dictionary in los_matrices:
                for query_name in los_dictionary:
                    global_los_dictionary[query_name] = global_los_dictionary[query_name] + [los_dictionary[query_name]]
            for query_name in global_los_dictionary:
                global_los_dictionary[query_name] = pd.concat(global_los_dictionary[query_name], sort=False)

            global_complex_dictionary = {}
            for query_name in complex_matrices[0]:
                global_complex_dictionary[query_name] = []
            for complex_dictionary in complex_matrices:
                for query_name in complex_dictionary:
                    global_complex_dictionary[query_name] = global_complex_dictionary[query_name] + [
                        complex_dictionary[query_name]]
            for query_name in global_complex_dictionary:
                global_complex_dictionary[query_name] = pd.concat(global_complex_dictionary[query_name], sort=False)

            return global_los_dictionary, global_complex_dictionary

        except:
            return pd.DataFrame(), pd.DataFrame()

    def query_smarts_csv(self):
        for cnt, smarts_query in enumerate(self.smarts_queries):
            self.smarts = smarts_query
            self.query_smarts()

    def query_smarts(self):
        for smarts_index in self.smarts_indices:
            query = Query(self.smarts, 'SMARTS', smarts_index)
            dictionary_of_los_df, dictionary_of_complex_df = self.run_los(query)
            for query_name in dictionary_of_los_df:
                if not (Path(self.output) / query_name).is_dir():
                    (Path(self.output) / query_name).mkdir()
                dictionary_of_los_df[query_name] = dictionary_of_los_df[query_name].assign(
                    ligand_atom_type=self.ligand_atom_type)
                dictionary_of_complex_df[query_name] = dictionary_of_complex_df[query_name].assign(
                    ligand_atom_type=self.ligand_atom_type)
                dictionary_of_los_df[query_name].to_csv(os.path.join(self.output, query_name, f'{query_name}_los.csv'),
                                                        index=False, header=True, line_terminator='\n')
                dictionary_of_complex_df[query_name].to_csv(os.path.join(self.output, query_name,
                                                                         f'{query_name}_complex.csv'),
                                                            index=False, header=True, line_terminator='\n')
            print('Output files were generated')
        print(''.join(['The run ended at ', str(datetime.datetime.now())]))


def main():
    print('Getting LoS contacts for protein')
    print('The run started at ' + str(datetime.datetime.now()))
    print('Running...')
    args = parse_args()
    print(os.getcwd())
    protein_los = ProteinLoS(annotations_file=args.annotations, input_folder=args.input, dbname=args.database,
                             output_folder=args.output, smarts_queries_file=args.smarts_queries_file, np=int(args.np),
                             verbose=args.verbose, smarts=args.smarts,
                             smarts_index=args.smarts_index)

    if protein_los.smarts_queries is not None:
        protein_los.query_smarts_csv()


if __name__ == "__main__":
    freeze_support()  # For multiprocessing in Windows.
    main()
