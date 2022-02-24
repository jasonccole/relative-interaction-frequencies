#!/usr/bin/env python
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

########################################################################################################################

import os
from pathos.multiprocessing import ProcessingPool
from pathos.helpers import freeze_support
from functools import partial
from ccdc import io
from ccdc_roche import atom_types
from ccdc_roche.python.los_utilities import *
import _pickle as pickle
from rdkit import Chem
import pandas as pd
from pathlib import Path
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

###################################################################################################################


def process_ligands(query_atom, hit_protein):
    '''
    Return a list of lists of HET group and central ligand atoms and updated query atom.
    :param query_atom:
    :param hit_protein:
    :return:
    '''
    ligands = []
    for component in hit_protein.components:
        if component.atoms[0].label.startswith('_Z'):
            ligands.append(component.atoms)
            query_atom.ligand_index = len(ligands) - 1
            mol = molecule.Molecule()
            mol.add_molecule(component)
            mol.add_hydrogens(mode='missing')
            query_atom.smiles = mol.smiles
        elif component.atoms[0].label.startswith('_U'):
            ligands.append(component.atoms)

    return query_atom, ligands


class LigandLoS(object):

    def __init__(self, np=1, dbname='database.csdsql', input_folder='input', output_folder='output', keep_waters=False,
                 annotations_file='annotation.csv', debug=False, verbose=False, smarts_queries_file=None, smarts=None,
                 smarts_index=None, smarts_filters=None, ligand_atom_type=None, pi_atom=False, tau_atom=False,
                 rdkit=True, keep_good_waters=True):

        self.los_home = input_folder

        if smarts_queries_file is None:
            self.smarts_queries_df = None

        # Small molecule substructure
        if smarts_queries_file is not None:
            self.smarts_queries_df = pd.read_csv(Path(self.los_home) / Path(smarts_queries_file))
                
        if smarts is not None:
            if smarts_index is not None:
                self.smarts_queries = [smarts]
                self.smarts_indices = [smarts_index]
                self.pi_atom = pi_atom
                self.tau_atom = tau_atom
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
        self.annotations = pd.read_csv(os.path.join(self.los_home, annotations_file), dtype={'UNIP_ID': str}, na_values='\"')
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
        if verbose:  # print error messages
            self.verbose = True
        else:
            self.verbose = False
        self.unique_pats = self.protein_atom_types['protein_atom_type'].drop_duplicates().to_list()
        self.los_columns = ['RDKit_SMARTS', 'RDKit_SMARTS_index', 'res_name', 'molecule_name', 'distance', 'vdw_distance',
                            'alpha_i', 'solv_acc_area', 'substructure_match', 'query_atom_id', 'los_atom_id']
                            #'other_ligands', 'same_ligand', 'metal'] + self.unique_pats

        @property
        def los_columns(self):
            return self.los_columns

        @los_columns.setter
        def los_columns(self):
            self.los_columns = los_columns

        self.complex_columns = ['uniprot_id', 'ligand_SMILES', 'molecule_name', 'resolution',
                                'surf_area_other_ligands', 'surf_area_metal'] + [''.join(['surf_area_', unique_pat]) for unique_pat in
                                                              self.unique_pats] + ['ligand_sasa', 'n_lig_in_bs']
        self.aa_labels = {'ARG', 'MET', 'SER', 'PHE', 'ALA', 'TYR', 'THR', 'LEU', 'ILE', 'VAL', 'CYS', 'GLU', 'GLN',
                          'ASN', 'PRO', 'ASP', 'GLY', 'LYS', 'HIS', 'TRP'}

        # settings
        self.interaction_cutoff = 0.5

    def process_hit(self, hit_protein, annotations, query_atom):
        '''
         This is to extract the information required from the substructure search hits to generate the complex.csv file
        '''
        pdb_code = hit_protein.identifier.split('_')[0].lower()
        if not annotations[annotations['STRUCID'] == pdb_code].empty:
            hit_protein.uniprot_id = annotations[annotations['STRUCID'] == pdb_code]['UNIP_ID'].array[0]
            hit_protein.resolution = annotations[annotations['STRUCID'] == pdb_code]['RESOLUTION'].array[0]
        else:
            print(hit_protein.identifier, 'has no Uniprot ID')
            hit_protein.uniprot_id = None
            hit_protein.resolution = None
        if query_atom.label.startswith('_Z') or query_atom.label.startswith('_U'):
            hit_protein.ligand_SMILES = query_atom.smiles
        return hit_protein

    def ligand_surfaces(self, query_atom, hit_protein, ligands):
        '''
        Calculate surface areas for the central ligand and other het groups.
        :param ligand_contacts:
        :param query_atom:
        :param hit_protein:
        :param ligands:
        :return: Protein object with additional ligand_sasa and other_ligands_sasa attributes.
        '''
        import numpy as np
        ligand_sasas = []
        for atom in ligands[query_atom.ligand_index]:
            ligand_sasas.append(self.atom_sas(hit_protein.atoms[int(atom.partial_charge)]))

        hit_protein.ligand_sasa = np.sum(ligand_sasas)

        return hit_protein

    def atom_sas(self, atom, sas_probe_radius=1.4, filter_by_ligand=False, ligand_atoms=None):
        '''
        :param atom:
        :param sas_probe_radius:
        :param filter_by_ligand:
        :param ligand_atoms:
        :return: Solvent accessible surface area in Angstrom2 as float type.
        '''
        import math
        import os
        from ccdc import utilities
        npoints = 15000

        if self.debug:
            logger = utilities.Logger()
            logger.set_ccdc_log_level(42)
            logger.set_ccdc_minimum_log_level(42)
            if sas_probe_radius == 1.4:
                npoints = 5000
                ligstring = 'ligand'
            else:
                npoints = 1000
                ligstring = 'other_ligands'

        atom_sas = 4 * math.pi * (atom.vdw_radius + sas_probe_radius) * (atom.vdw_radius + sas_probe_radius) * \
                   atom.solvent_accessible_surface(probe_radius=sas_probe_radius, npoints=npoints,
                                                   filter_by_ligand=filter_by_ligand, ligand_atoms=ligand_atoms)
        if self.debug:
            if os.path.exists('Debug/' + ligstring + '-' + str(atom.index) + '.mol2'):
                os.remove('Debug/' + ligstring + '-' + str(atom.index) + '.mol2')
            os.rename('Atom' + atom.label + '_' + str(atom.index) + '_vdw.mol2', 'Debug/' + ligstring + '-'
                      + str(atom.index) + '.mol2')
        return atom_sas

    def ligand_sanity_check(self, hit_protein, ligands):
        '''
        Returns True if all atoms of a ligand are covalently connected. Returns false if atoms of a ligand are not
        covalently connected.
        :param hit_protein: Protein
        :param ligands: List of lists of atoms.
        :return: True if atoms belong to the same molecule, False if atom do not belong to the same molecule.
        '''
        for ligand in ligands:
            for atom in ligand:
                try:
                    length = hit_protein.shortest_path(hit_protein.atoms[int(ligand[0].partial_charge)],
                                                       hit_protein.atoms[int(atom.partial_charge)])
                    if length == 0 and hit_protein.atoms[int(ligand[0].partial_charge)] != hit_protein.atoms[
                        int(atom.partial_charge)]:
                        return False
                    for neighbour in atom.neighbours:
                        if neighbour not in ligand:
                            return False
                except:
                    return False
        return True

    def cofactor_info(self, cofactors, atom):
        '''
        Return Cofactor identifier and cofactor index.
        :param cofactors:
        :param atom:
        :return: Cofactor identifier as string, cofactor index as integer.
        '''
        if atom.protein_atom_type == 'Cofactor':
            if len(cofactors) == 1:
                return cofactors[0].identifier, 0
            else:
                cof_indices = [set(cofatom.partial_charge for cofatom in cofactor.atoms) for cofactor in cofactors]
                for cof_index, atom_indices in enumerate(cof_indices):
                    if atom.index in atom_indices:
                        return cofactors[cof_index].identifier, cof_index
                return None

    def het_group_identifier(self, atom):
        '''
        :param ligand_atoms:
        :param atom:
        :param query_atom:
        :return: 0 if atom belongs to protein or central ligand. 1 if atom belongs to HET which is not central ligand.
        '''

        if '_U' in atom.label:
            hetgroup = 1
        else:
            hetgroup = 0
        return hetgroup

    def add_to_complex_row(self, complex_row, binding_site_atom, pat=None, ligand_atoms=None):
        '''
        Write Data Frame row for complex.csv.
        :param complex_row:
        :param binding_site_atom:
        :param pat:
        :param ligand_atoms:
        :return:
        '''
        import math
        import os
        from ccdc import utilities
        npoints = 15000
        if self.debug:
            logger = utilities.Logger()
            logger.set_ccdc_log_level(42)
            logger.set_ccdc_minimum_log_level(42)
            npoints = 1000

        if pat is not None and ligand_atoms is not None:  # surface area with probe radius = 0
            filter_by_ligand = True
        else:
            filter_by_ligand = False
        if pat is not None:
            complex_row.loc[0, 'surf_area_' + pat] = complex_row.loc[0, 'surf_area_' + pat] + (
                4 * math.pi * (binding_site_atom.vdw_radius + 0) * (binding_site_atom.vdw_radius + 0)
                * binding_site_atom.solvent_accessible_surface(probe_radius=0, npoints=npoints,
                                                               filter_by_ligand=filter_by_ligand,
                                                               ligand_atoms=ligand_atoms))

        if self.debug:
            if os.path.exists(r'Debug/' + pat + '-' + str(binding_site_atom.index) + '.mol2'):
                os.remove(r'Debug/' + pat + '-' + str(binding_site_atom.index) + '.mol2')
            os.rename('Atom' + binding_site_atom.label + '_' + str(binding_site_atom.index) + '_vdw.mol2',
                      r'Debug/' + pat + '-' + str(binding_site_atom.index) + '.mol2')

        return complex_row

    def update_atom_attributes(self, atom_list, query_atom, central_ligand_atom, contact_angle_atom, ligands,
                               hit_protein, substructure_match):
        from ccdc import descriptors
        updated_atom_list = []
        for atom in atom_list:
            atom.het_group = self.het_group_identifier(atom)
            atom.id_lig = query_atom.index
            atom.identifier = hit_protein.identifier
            atom.substructure_match = substructure_match
            atom.distance = descriptors.MolecularDescriptors.atom_distance(query_atom, atom)
            atom.angle = descriptors.GeometricDescriptors.point_angle(contact_angle_atom.coordinates,
                                                                      query_atom.coordinates,
                                                                      atom.coordinates)
            updated_atom_list.append(atom)
        return updated_atom_list

    def create_dummy_sol(self, query_atom, hit_protein, substructure_match):
        '''
        Create a dummy atom to add the query atom information to los.csv.
        :param query_atom:
        :param hit_protein:
        :param substructure_match:
        :return:
        '''
        import math
        from ccdc import molecule
        dummy_SOL = molecule.Atom(atomic_symbol='Pb')
        dummy_SOL.identifier = hit_protein.identifier
        dummy_SOL.substructure_match = substructure_match
        dummy_SOL.distance = 'N/A'
        dummy_SOL.vdw_distance = 'N/A'
        dummy_SOL.angle = 'N/A'
        dummy_SOL.h = {}
        dummy_SOL.het_group = 0
        dummy_SOL.id_lig = query_atom.index
        dummy_SOL.residue_label = 'SOL'
        dummy_SOL.SAS = 4 * math.pi * (query_atom.vdw_radius + 1.4) * (
                query_atom.vdw_radius + 1.4) * query_atom.solvent_accessible_surface(probe_radius=1.4,
                                                                                     npoints=15000)
        return dummy_SOL

    def add_to_los_df(self, query_string, query_index, los_df, los_atom, pat=None):
        '''
        Write row to los.csv Data Frame.
        :param query_index: Indices to define contact_atom and contact_angle_atom
        :param query_string: SMARTS string.
        :param los_df:
        :param los_atom:
        :param pat:
        :return: Data Frame
        '''
        import pandas as pd
        import numpy as np

        row = pd.DataFrame(columns=self.los_columns)
        row.loc[0, 'RDKit_SMARTS'] = query_string
        row.loc[0, 'RDKit_SMARTS_index'] = query_index
        row.loc[0, 'res_name'] = los_atom.residue_label
        row.loc[0, 'molecule_name'] = los_atom.identifier
        row.loc[0, 'distance'] = los_atom.distance
        row.loc[0, 'vdw_distance'] = los_atom.vdw_distance
        row.loc[0, 'alpha_i'] = los_atom.angle

        row.loc[0, 'substructure_match'] = los_atom.substructure_match
        row.loc[0, 'query_atom_id'] = los_atom.id_lig + 1
        if los_atom.is_metal:
            pat = 'metal'

        elif los_atom.het_group == 1:
            pat = 'other_ligands'

        if los_atom.label.startswith('_Z'):
            pat = 'same_ligand'

        if los_atom.residue_label == 'SOL':
            row.loc[0, 'solv_acc_area'] = los_atom.SAS
            row.loc[0, 'los_atom_id'] = -1

        else:
            row.loc[0, 'solv_acc_area'] = 0
            row.loc[0, 'los_atom_id'] = los_atom.index + 1

        row.loc[0, 'los_atom_type'] = pat
        row = row.fillna(0)

        try:
            row.loc[0, 'h'] = los_atom.h
        except (ValueError, AttributeError):
            row.loc[0, 'h'] = np.nan
        try:
            row.loc[0, 'tau'] = los_atom.tau
        except (ValueError, AttributeError):
            row.loc[0, 'tau'] = np.nan

        los_df = los_df.append(row, ignore_index=True, sort=False)
        return los_df

    def write_los_contacts(self, los_atoms, los_df, query_string, query_index, hit_protein, substructure_match, query_atom):
        import numpy as np
        import math
        # Protein atom types SMARTS
        for los_atom in los_atoms:
            if los_atom.protein_atom_type == 'Ligand' and not hasattr(los_atom, 'los_at'):
                los_atom.residue_label = 'LIG'
                los_df = self.add_to_los_df(query_string, query_index, los_df, los_atom)
            elif los_atom.protein_atom_type == 'Unknown' and not hasattr(los_atom, 'los_at'):
                los_df = self.add_to_los_df(query_string, query_index, los_df, los_atom)
            elif hasattr(los_atom, 'ligand_index') and los_atom.ligand_index is not None and not hasattr(los_atom,
                                                                                                         'los_at'):
                los_df = self.add_to_los_df(query_string, query_index, los_df, los_atom)
            else:
                pat = los_atom.los_at
                los_df = self.add_to_los_df(query_string, query_index, los_df, los_atom, pat=pat)

        #append SOL line
        sol_df = pd.DataFrame(columns=los_df.columns)
        sol_df = sol_df.append(pd.Series(dtype=object), ignore_index=True)
        sol_df = sol_df.fillna(0)
        sol_df['distance'] = 'N/A'
        sol_df['vdw_distance'] = 'N/A'
        sol_df['alpha_i'] = 'N/A'
        sol_df['h'] = np.nan
        sol_df['tau'] = np.nan
        sol_df['molecule_name'] = hit_protein.identifier
        sol_df['substructure_match'] = substructure_match
        sol_df['query_atom_id'] = query_atom.index + 1
        sol_df['los_atom_id'] = -1
        sol_df['res_name'] = 'SOL'
        sol_df['RDKit_SMARTS'] = query_string
        sol_df['RDKit_SMARTS_index'] = query_index
        sol_df['solv_acc_area'] = 4 * math.pi * (query_atom.vdw_radius + 1.4) * (
                query_atom.vdw_radius + 1.4) * query_atom.solvent_accessible_surface(probe_radius=1.4, npoints=15000)
        los_df = los_df.append(sol_df)

        return los_df

    def write_complex_df(self, ligand_contacts, hit_protein, n_ligands, ligand_atoms):
        import pandas as pd
        complex_df = pd.DataFrame(columns=self.complex_columns)
        complex_row = pd.DataFrame(columns=self.complex_columns)
        complex_row.loc[0] = 0
        complex_row.loc[0, 'ligand_sasa'] = hit_protein.ligand_sasa
        complex_row.loc[0, 'uniprot_id'] = hit_protein.uniprot_id
        complex_row.loc[0, 'ligand_SMILES'] = hit_protein.ligand_SMILES
        complex_row.loc[0, 'molecule_name'] = hit_protein.identifier
        complex_row.loc[0, 'resolution'] = hit_protein.resolution
        complex_row.loc[0, 'n_lig_in_bs'] = n_ligands
        for ligand_contact_atom in ligand_contacts:
            complex_row = self.add_to_complex_row(complex_row,
                                                  hit_protein.atoms[int(ligand_contact_atom.partial_charge)],
                                                  pat=ligand_contact_atom.los_at, ligand_atoms=ligand_atoms)
        complex_df = complex_df.append(complex_row, ignore_index=True, sort=False)
        return complex_df

    def detect_molecule(self, hit_protein, atom):
        '''
        This is to deal with peptide ligands.
        :param hit_protein:
        :param atom:
        :return:  List of all atoms that are in the same molecule/residue as the input atom if the input atom is part
        of a HET group.
        '''
        import re
        from ccdc_roche.python.los_utilities import change_atom_type
        # check if the query atom is actually part of a ligand
        res_label = atom.residue_label
        if re.split(r'(\d+)', res_label)[0] not in self.aa_labels:
            atoms = [hit_protein.atoms[int(atom.partial_charge)]]
            atoms_indices = {atom.index}
            for new_atom in atoms:
                for neighbour in new_atom.neighbours:
                    if neighbour.index not in atoms_indices:
                        neighbour.ligand_index = 0
                        atoms_indices.add(neighbour.index)
                        atoms.append(neighbour)
            ligand_atoms = set(new_atom for new_atom in atoms)
            return [list(ligand_atoms)]

        else:
            change_atom_type(atom, 'Amino_acid')
            return

    def rdr(self):
        '''
        Return entry reader.
        :return:
        '''
        from ccdc import io
        entry_reader = io.EntryReader([str(db) for db in self.los_db])
        return entry_reader

    def run_los_multiprocessing(self, hit_identifiers, query):
        # Multiprocessing of hits
        parallel_los_complex_per_hit_wrapper = partial(self.los_complex_per_hit_wrapper, query=query)
        pool = ProcessingPool(self.np)
        output = [i for i in pool.map(parallel_los_complex_per_hit_wrapper, hit_identifiers.iterrows()) if i is not None]
        print('Multiprocessing finished.')
        import time
        t1 = time.time()
        output = list(zip(*output))
        los_df = pd.concat(output[0])
        complex_df = pd.concat(output[1])
        t2 = time.time()
        print(f'Appending all Dataframes took {(t2 - t1):.2f} seconds.')
        return los_df, complex_df

    def run_los_single_processing(self, hit_identifiers, query):
        los_df = pd.DataFrame()
        complex_df = pd.DataFrame()
        entry_reader = io.EntryReader([str(db) for db in self.los_db])
        for hit_identifier in hit_identifiers.iterrows():
            matrices = self.los_complex_per_hit_wrapper(hit_identifier, query, entry_reader=entry_reader)
            if matrices is not None:
                los_df = los_df.append(matrices[0], ignore_index=True)
                complex_df = complex_df.append(matrices[1], ignore_index=True)
        return los_df, complex_df

    def return_los_complex_dfs(self, query):
        '''
        Run line of sight calculation. If np >= 2, multiprocessing is used. Else, no multiprocessing but a simple for
        loop.
        :param substructure:
        :param contact_atom_index:
        :param smarts_query:
        :return: los_df, complex_df
        '''
        from ccdc_roche.python.los_utilities import substructure_search, _get_rdkit_dbs

        if self.debug:
            if not os.path.exists('Debug'):
                os.mkdir('Debug')

        print(''.join(['The LoS calculation started at ', str(datetime.datetime.now())]))
        if self.rdkit:
            rdkit_dbs = _get_rdkit_dbs(self.los_db, self.los_home)
            hit_identifiers = pd.DataFrame()

            for rdkit_db_name in rdkit_dbs:
                if not '.mol2' == self.los_db[0].suffix:
                    with open(rdkit_db_name, 'rb') as rdkit_db:
                        rdkit_db = pickle.load(rdkit_db)
                    ccdc_db = io.EntryReader([str(db) for db in self.los_db])
                else:
                    ccdc_db = io.EntryReader(*self.los_db)
                print('Running Substructure search...')
                print('Selecting only hits that match the _Z label and have only one HET group matching the query.')
                print('RDKit library:', rdkit_db_name, ' Size: ', len(rdkit_db))
                hit_identifiers = hit_identifiers.append(substructure_search(query, db=ccdc_db,
                                                                             return_identifiers=True, rdkit=self.rdkit,
                                                                             rdkit_db=rdkit_db), ignore_index=True)

                print(''.join([str(len(hit_identifiers)), ' entries match the query.']))

        else:
            print('Running Substructure search...')
            print('Selecting only hits that match the _Z label and have only one HET group matching the query.')
            hit_identifiers = substructure_search(query, db=io.EntryReader([str(db) for db in self.los_db]),
                                                  return_identifiers=True)
            print(''.join([str(len(hit_identifiers)), ' entries match the query.']))

        if self.np >= 2:
            print('hit_identifiers:', hit_identifiers.shape)
            los_df, complex_df = self.run_los_multiprocessing(hit_identifiers, query)

        else:
            print('hit_identifiers:', hit_identifiers.shape)
            los_df, complex_df = self.run_los_single_processing(hit_identifiers, query)
        return los_df, complex_df

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
        import pandas as pd
        from ccdc_roche.python.los_utilities import identify_covalent_query_atom
        from ccdc_roche.python.los_utilities import get_query_atoms, substructure_search, get_hit_protein
        from ccdc_roche.python import atom_geometries

        def make_complex_matrices(complex_matrices, ligand_contacts, query_atom, hit_protein, ligands):
            '''
            Generate a Data Frame containing information for complex.csv.
            :param complex_matrices:
            :param ligand_contacts:
            :param query_atom:
            :param hit_protein:
            :param ligands:
            :return: Data Frame
            '''
            hit_protein = self.ligand_surfaces(query_atom, hit_protein, ligands)
            # Write complex matrix
            complex_df = self.write_complex_df(ligand_contacts, hit_protein, len(ligands),
                                                       ligands[query_atom.ligand_index])
            complex_matrices = pd.concat([complex_matrices, complex_df], sort=False)
            return complex_matrices

        def make_los_df(los_df, query_atom, contact_angle_atom, hit_protein, ligand_contacts,
                        ligands, substructure_match):
            '''
            Generate Data Frame containing information for los.csv
            :param los_df:
            :param query_atom:
            :param contact_angle_atom:
            :param hit_protein:
            :param ligand_contacts:
            :param ligands:
            :param substructure_match:
            :return: Data Frame
            '''
            # find LoS_contacts

            los_atoms = return_los_contacts(query_atom, hit_protein, list(ligand_contacts),
                                            ligands[query_atom.ligand_index])
            los_atoms = self.update_atom_attributes(los_atoms, query_atom, query_atom, contact_angle_atom, ligands,
                                                    hit_protein, substructure_match)

            if self.pi_atom:
                for los_atom in los_atoms:
                    los_atom.h = atom_geometries.PlaneDistanceCalculator(query_atom, los_atom).return_plane_distance()

            if self.tau_atom:
                for los_atom in los_atoms:
                    los_atom.tau = atom_geometries.TorsionAngleCalculator(query_atom, los_atom).return_torsion_angle()

            los_df = self.write_los_contacts(los_atoms, los_df, query.query, query.index, hit_protein,
                                             substructure_match, query_atom)
            return los_df

        try:
            if self.rdkit:
                index, row = iterrow
                entry_identifier = row['identifier']
                rdkit_hit_protein = row['rdkit_mol']

            complex_matrices = pd.DataFrame(columns=self.complex_columns)
            los_df = pd.DataFrame(columns=self.los_columns)
            hit_protein = get_hit_protein(entry_reader, entry_identifier, self.rdr())
            hits = substructure_search(query, hit_protein, filter_queries=self.filter_queries, return_identifiers=False,
                                       rdkit=self.rdkit, rdkit_db=rdkit_hit_protein)

            # cleanup hits for multiple neighbours
            unique_query_atoms = []
            cleaned_hits = []
            for hit in hits:
                query_atoms = get_query_atoms(hit, query, hit_protein, rdkit=self.rdkit, rdkit_db=rdkit_hit_protein)
                if query_atoms not in unique_query_atoms:
                    unique_query_atoms.append(query_atoms)
                    cleaned_hits.append(hit)
            hits = cleaned_hits
            if len(hits) >= 1:
                substructure_match = 0

                for hit in hits:
                    query_atoms = get_query_atoms(hit, query, hit_protein, rdkit=self.rdkit,
                                                  rdkit_db=rdkit_hit_protein)
                    for query_atom in query_atoms:
                        substructure_match += 1  # count how often a query appears in the same entry
                        query_atom, ligands = process_ligands(query_atom, hit_protein)  ### we should have to do this only once per entry

                        # skip hit if query atom or its neighbour is covalently bound to protein
                        if identify_covalent_query_atom(query_atom, hit_protein):
                            continue

                        hit_protein = self.process_hit(hit_protein, self.annotations, query_atom)

                        # get all atoms in contact with ligand
                        ligand_contacts = return_ligand_contacts(ligands, query_atom, hit_protein)

                        # remove any covalently bound protein atoms and their neighbours
                        ligand_contacts = [at for at in ligand_contacts if not identify_covalent_query_atom(
                            at, hit_protein)]

                        for ligand_contact_atom in ligand_contacts:
                            ligand_contact_atom.los_at = return_atom_type(ligand_contact_atom, query=query,
                                                                          hit_protein=hit_protein,
                                                                          protein_atom_types_df=self.protein_atom_types)
                            # if ligand_contact_atom.los_at is None:
                            #     with io.EntryWriter('missing_atom_type.mol2') as w:
                            #         w.write(hit_protein)
                            ligand_contact_atom.ligand_index = ligand_index(ligands, ligand_contact_atom)

                        if substructure_match == 1:  # Process complex information for the first substructure match
                            complex_matrices = make_complex_matrices(complex_matrices, ligand_contacts, query_atom,
                                                                     hit_protein, ligands)
                        for contact_angle_atom in query_atom.neighbours:
                            los_df = make_los_df(los_df, query_atom, contact_angle_atom, hit_protein,
                                                 ligand_contacts, ligands, substructure_match)
                return los_df, complex_matrices
            else:
                pass
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            # write out bad entry
            print(f'Entry {entry_identifier} caused a problem. Continuing to next hit.')
            if self.verbose:
                traceback.print_exc(file=sys.stdout)
                print(e)
            # entry_name = hit_identifier
            # with entry_reader as rdr, io.MoleculeWriter('{0}_dumped.mol2'.format(entry_name),
            #                                             format='mol2') as wrt:
            #
            #     p = protein.Protein.from_entry(rdr.entry(entry_name))
            #     ##remove all water atoms
            #     for atom in p.atoms:
            #         if 'HOH' in atom.label:
            #             p.remove_atom(atom)
            #     p = rdr.entry(entry_name).molecule
            #     wrt.write(p)

    def query_smarts(self):
        smarts_query = self.smarts_queries[0]
        contact_atom_index = self.smarts_indices[0]
        los_name = 'query_atom_los.csv'
        complex_name = 'query_atom_complex.csv'
        folder_name = 'query_atom'
        if not os.path.exists(os.path.join(os.getcwd(), self.output, folder_name)):
            os.mkdir(os.path.join(os.getcwd(), self.output, folder_name))
        query = Query(smarts_query, 'SMARTS', contact_atom_index)
        self.execute_los(folder_name, los_name, complex_name, query)
        print(''.join(['The run ended at ', str(datetime.datetime.now())]))

    def execute_los(self, foldername, los_name, complex_name, query):
        '''
        :param foldername:
        :param los_name:
        :param complex_name:
        :param query: Query object
        :return:
        '''
        with open(os.path.join(os.getcwd(), self.output, foldername, los_name), 'w') as los_file, \
                open(os.path.join(os.getcwd(), self.output, foldername, complex_name), 'w') as complex_file:
            los_df, complex_df = self.return_los_complex_dfs(query)
            los_df = los_df.assign(ligand_atom_type=self.ligand_atom_type)
            complex_df = complex_df.assign(ligand_atom_type=self.ligand_atom_type)
            los_df.to_csv(los_file, mode='a', index=False, header=True, line_terminator='\n')
            complex_df.to_csv(complex_file, mode='a', index=False, header=True, line_terminator='\n')
        print('Output files were generated')


def main():
    print('The run started at ' + str(datetime.datetime.now()))
    print('Running...')
    args = parse_args()
    print(os.getcwd())
    los = LigandLoS(annotations_file=args.annotations, input_folder=args.input, dbname=args.database,
                    output_folder=args.output, smarts_queries_file=args.smarts_queries_file, np=int(args.np),
                    verbose=args.verbose, smarts=args.smarts,
                    smarts_index=args.smarts_index)

    if los.smarts_queries is not None and los.smarts_queries_df is None:
        los.query_smarts()


if __name__ == "__main__":
    freeze_support()  # For multiprocessing in Windows.
    main()
