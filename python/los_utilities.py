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
Utilities for line of sight contact scripts.
'''

########################################################################################################################

import __future__
import datetime
import argparse
from ccdc import molecule, protein
from pathlib import Path
from rdkit import Chem

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Generate los.csv and complex.csv files, which are input for Rf calculation.
        los.csv contains information on Line of sight contact atoms, e.g. protein atom types.
        complex.csv contains information on the entry, e.g. binding site surface area, resolution of
        crystal structure.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-db',
        '--database',
        help='Path to input database in csdsql format. The database entries must be proteins.',
        default='database.csdsql'
    )

    parser.add_argument(
        '-a',
        '--annotations',
        help='Path to annotations file in CSV format.',
        default=r'all_annotated_aug2021.csv'
    )

    parser.add_argument(
        '-pat',
        '--protein_atom_types',
        help='Path to file containing protein atom types in CSV format.',
        default=r'protein_atom_types.csv'
    )

    parser.add_argument(
        '-sqf',
        '--smarts_queries_file',
        help='Path to file containing query atom types in CSV format.',
        default=None
    )

    parser.add_argument(
        '-if',
        '--isostar_folder',
        help='Path to file containing isostar *.ini files.',
        default=None
    )

    parser.add_argument(
        '-cqf',
        '--conquest_folder',
        help='Path to folder containing Conquest query files in ".con" format.',
        default=r'Conquest'
    )

    parser.add_argument(
        '--debug',
        help='Turn on debug to write out surfaces as mol2.',
        action="store_true"
    )

    parser.add_argument(
        '-np',
        '--np',
        help='Number of processes for multiprocessing',
        default=1
    )

    parser.add_argument(
        '-out',
        '--output',
        help='Folder for output files.',
        default=r'output'
    )

    parser.add_argument(
        '-in',
        '--input',
        help='Folder for input files. Default \'input\' ',
        default=''
    )

    parser.add_argument(
        '--smarts',
        help='SMARTS string for search',
        default=''
    )

    parser.add_argument(
        '--smarts_index',
        help='Index to specify contact atom in SMARTS string',
        default='',
        type=int
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help='Turn on verbose output.',
        action="store_true"
    )

    parser.add_argument(
        '-kw',
        '--keep_waters',
        help='Do not remove explicit water molecules from binding site',
        action="store_true"
    )

    return parser.parse_args()


def _get_rdkit_dbs(los_db, los_home):
    rdkit_dbs = [Path(los_home) / Path(Path(dbname).stem + '.p') for dbname in los_db]
    return rdkit_dbs


def assign_index_to_atom_partial_charge(molecule):
    '''
    Assign the atom index to the atom partial charge. The atom.index depends on the atom's context. Storing the
    atom's index in the protein context can be exploited to compare atoms called from different contexts.
    E.g. For the same atom, ligand.atoms[*].index is not protein.atoms[*].index.
    :param molecule:
    :return:
    '''
    for a in molecule.atoms:
        a.partial_charge = int(a.index)
    return molecule


def assign_index_to_atom_label(molecule):
    '''
    Assign the atom index to the atom partial charge. The atom.index depends on the atom's context. Storing the
    atom's index in the protein context can be exploited to compare atoms called from different contexts.
    E.g. For the same atom, ligand.atoms[*].index is not protein.atoms[*].index.
    :param molecule:
    :return:
    '''
    for a in molecule.atoms:
        old_label = a.label[:2]
        a.label = f'{old_label}{int(a.index)}'
    return molecule


def ligand_index(ligand_atoms, atom):
    '''
    Return ligand index as Integer type.
    :param ligand_atoms:
    :param atom:
    :return: ligand index as Integer type.
    '''
    if atom.protein_atom_type == 'Ligand' and len(ligand_atoms) == 1:
        return 0
    else:
        lig_indices = [set(ligatom.partial_charge for ligatom in lig) for lig in ligand_atoms]
        for ligand_index, atom_indices in enumerate(lig_indices):
            if atom.partial_charge in atom_indices:
                return ligand_index
        return None


def coordinates_are_in_list(coord, coordlist):
    '''
    Check if coordinates are in a list of coordinates.
    :param coord:
    :param coordlist:
    :return: True if coordinates are in list, False if not.
    '''
    if repr([round(x, 2) for x in coord]) in [repr([round(x, 2) for x in coord_]) for coord_ in coordlist]:
        return True
    else:
        return False


def change_atom_type(atom, new_atom_type):
    '''
    Overwrite protein atom type.
    :param atom:
    :param new_atom_type:
    :return: Atom object with changed protein_atom_type.
    '''
    from ccdc import utilities
    with utilities._private_importer():
        import FileFormatsLib
    psd = atom._get_psd()
    if new_atom_type == 'Ligand':
        psd.set_type(FileFormatsLib.ProteinSubstructureData.LIGAND)
    if new_atom_type == 'Amino_acid':
        psd.set_type(FileFormatsLib.ProteinSubstructureData.AMINOACID)
    if new_atom_type == 'Nucleotide':
        psd.set_type(FileFormatsLib.ProteinSubstructureData.NUCLEOTIDE)
    if new_atom_type == 'Metal':
        psd.set_type(FileFormatsLib.ProteinSubstructureData.METAL)
    if new_atom_type == 'Cofactor':
        psd.set_type(FileFormatsLib.ProteinSubstructureData.COFACTOR)
    return atom


def identify_covalent_query_atom(query_atom, hit_protein):
    '''
    Skip entry if query atom or neighbouring atom is covalently bound to protein.
    :return: True if ligand is covalently bound, false if ligand is not covalently bound.
    '''

    from ccdc import descriptors

    def neighbour_is_central_ligand(_atom):
        for neighbour in _atom.neighbours:
            if '_Z' in neighbour.label:
                return True
        return False

    def central_ligand_atom_within_cov_dist(atom, _hit_protein):
        atom_distance_search = descriptors.MolecularDescriptors().AtomDistanceSearch(_hit_protein)
        close_contacts = atom_distance_search.atoms_within_range(atom.coordinates, 2.25)
        for close_contact in close_contacts:
            if '_Z' in close_contact.label:
                return True
        return False

    def neighbour_is_protein(_atom):
        for neighbour in _atom.neighbours:
            if not (neighbour.label.startswith('_U') or neighbour.label.startswith('_Z')):
                return True
        return False

    def protein_within_cov_dist(atom, _hit_protein):
        atom_distance_search = descriptors.MolecularDescriptors().AtomDistanceSearch(_hit_protein)
        close_contacts = atom_distance_search.atoms_within_range(atom.coordinates, 2.25)
        for close_contact in close_contacts:
            if not (close_contact.label.startswith('_U') or close_contact.label.startswith('_Z')):
                return True
        return False

    atoms_to_test = set([query_atom])
    # also keep neighbours
    for env in range(4):
        atoms_to_test.update([n for sublist in [a.neighbours for a in atoms_to_test] for n in sublist])

    if query_atom.label.startswith('_U') or query_atom.label.startswith('_Z'):
        for at in atoms_to_test:
            if neighbour_is_protein(at):
                return True
            if protein_within_cov_dist(at, hit_protein):
                return True
        return False

    else:
        for at in atoms_to_test:
            if neighbour_is_central_ligand(at):
                return True
            if central_ligand_atom_within_cov_dist(at, hit_protein):
                return True
        return False


def sulfur_is_part_of_disulfide(query_atom, hit_protein):
    from ccdc import descriptors
    '''
    Return True if S_atom is a disulfide
    :return: True if disulfide, false if not
    '''
    if query_atom.atomic_symbol != 'S':
        raise Exception('Only Sulfur atoms can form disulfides.')

    def sulfur_within_cov_dist(atom, _hit_protein):
        atom_distance_search = descriptors.MolecularDescriptors().AtomDistanceSearch(_hit_protein)
        close_contacts = atom_distance_search.atoms_within_range(atom.coordinates, 2.25)
        for close_contact in close_contacts:
            if close_contact.atomic_symbol == 'S' and close_contact.protein_atom_type == 'Amino_acid' and close_contact != atom:
                return True
        return False

    if sulfur_within_cov_dist(query_atom, hit_protein):
        print('Close contact to S atom', hit_protein.identifier)
        return True
    return False


def read_annotation_file(annotation_file):
    import pandas as pd
    annotations = pd.read_csv(annotation_file, dtype={'UNIP_ID': str}, na_values='\"')
    return annotations


def make_safe_filename(filename, stamp, extension):
    '''
    Generate a safe filename from a given string.
    :param filename: String
    :return: Safe filename as String
    '''

    add = stamp
    base_name = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in [' ', '_', '.']]).rstrip()
    safe_name = ''.join([base_name, '_', add, extension])
    return safe_name


def get_filenames(smarts_query, contact_atom_index, dbname):
    import numpy as np
    stamp = str(np.random.randint(1000))
    los_name = make_safe_filename(''.join([dbname, '_', smarts_query, '_', str(contact_atom_index), '_los_',
                                           datetime.datetime.now().strftime("%Y%m%d")]), stamp, '.csv')
    complex_name = make_safe_filename(''.join([dbname, '_', smarts_query, '_', str(contact_atom_index), '_complex_',
                                               datetime.datetime.now().strftime("%Y%m%d")]), stamp, '.csv')
    return los_name, complex_name


def _return_central_ligand_atom(hit_protein):
    for atom in hit_protein.atoms:
        if '_Z' in atom.label:
            return atom


def get_folder_name(smarts_query, dbname):
    import numpy as np
    stamp = str(np.random.randint(1000))
    folder_name = make_safe_filename(''.join([dbname, '_', smarts_query, '_',
                                              datetime.datetime.now().strftime("%Y%m%d")]), stamp, '')
    return folder_name


# def compare_los_connser(not_plane_atom, plane_atom, geom_hits_, entry_identifier):
#     from ccdc_roche.los_utilities import coordinates_are_in_list
#     import numpy as np
#     '''
#     Match geometric information from Connser search with LoS atoms
#     :param query_atom_coordinates:
#     :param los_atom_list:
#     :param geom_hits_:
#     :param entry_identifier:
#     :return: Dictionaries for each geometry.
#     '''
#     local_geom_hits = {}
#     h = {}
#     beta = {}
#     gamma = {}
#     delta = {}
#     for conquest_query in geom_hits_.keys():
#         local_geom_hits[conquest_query] = [geom_hit for geom_hit in geom_hits_[conquest_query] if
#                                            geom_hit['identifier'] == entry_identifier]
#         for conquest_query_ in geom_hits_.keys():
#             h[conquest_query_] = np.nan
#             beta[conquest_query_] = np.nan
#             gamma[conquest_query_] = np.nan
#             delta[conquest_query_] = np.nan
#             for geom_hit_ in geom_hits_[conquest_query_]:
#                 if coordinates_are_in_list(not_plane_atom.coordinates, geom_hit_['DIST1_measurement_object']) and \
#                         coordinates_are_in_list(plane_atom.coordinates, geom_hit_['plane_coordinates']):
#                     try:
#                         h[conquest_query_] = geom_hit_['measurements']['H']
#                     except (AttributeError, KeyError):
#                         pass
#                     try:
#                         beta[conquest_query_] = geom_hit_['measurements']['BETA']
#                     except (AttributeError, KeyError):
#                         pass
#                     if coordinates_are_in_list(not_plane_atom.coordinates,
#                                                     geom_hit_['DIST1_measurement_object']):
#                         try:
#                             gamma[conquest_query_] = geom_hit_['measurements']['GAMMA']
#                         except (AttributeError, KeyError):
#                             pass
#                         try:
#                             delta[conquest_query_] = geom_hit_['measurements']['DELTA']
#                         except (AttributeError, KeyError):
#                             pass
#     return h, beta, gamma, delta
#
#
# def conquest(folder, conquest_file, db):
#     '''
#     :param folder: Folder in which Connser files are stored.
#     :param conquest_file: Connser file in .con format.
#     :param db: Database to be searched.
#     :return: Dictionary of Conquest queries with plane, DIST1 and V2 coordinates.
#     '''
#     from ccdc import search, molecule
#     import os
#     substructure_search = search.SubstructureSearch()
#     connser = search.ConnserSubstructure(os.path.join(folder, conquest_file))
#     substructure_search.add_substructure(connser)
#     geo_hits = substructure_search.search(database=db)
#
#     def get_geom_hits_objects(geom_hits):
#         new_geom_hits = []
#         for geom_hit in geom_hits:
#             geom_hit_identifier = geom_hit.identifier
#             new_dict = {}
#             new_dict['identifier'] = geom_hit_identifier
#             try:
#                 new_dict['plane_coordinates'] = [X1_atom.coordinates for X1_atom in geom_hit.group_atoms('X1')]
#             except KeyError:
#                 new_dict['plane_coordinates'] = [X1_atom.coordinates for X1_atom in geom_hit.group_atoms('P1')]
#             try:
#                 new_dict['DIST1_measurement_object'] = [meas_obj.coordinates for meas_obj in
#                                                         geom_hit.measurement_objects('DIST1')
#                                                         if type(meas_obj) == molecule.Atom]
#             except KeyError:
#                 new_dict['DIST1_measurement_object'] = [meas_obj.coordinates for meas_obj in
#                                                         geom_hit.constraint_objects('DIST1')
#                                                         if type(meas_obj) == molecule.Atom]
#             try:
#                 new_dict['V2'] = [meas_obj.coordinates for meas_obj in geom_hit.group_objects('V2')
#                                   if type(meas_obj) == molecule.Atom]
#             except KeyError:
#                 pass
#             new_dict['measurements'] = geom_hit.measurements
#             new_dict['measurements'] = {k: abs(v) for k, v in new_dict['measurements'].items()}
#             new_geom_hits.append(new_dict)
#         geom_hits = new_geom_hits
#         return geom_hits
#
#     if geo_hits is not None:
#         if len(geo_hits) != 0:
#             geo_hits = get_geom_hits_objects(geo_hits)
#
#     return geo_hits


def get_query_atoms(match_atoms, query, hit_protein, rdkit=False, rdkit_db=None):
    '''
    Get the query atom and the covalently bound atom used to calculate contact angles.
    Get all chemically equivalent atoms thath match a query. E.g. in CF2, get both fluorine atoms.
    :param match_atoms: Index of the matched atoms.
    :return: Query atom.
    >>> from rdkit import Chem
    >>> from ccdc import io
    >>> from ccdc import protein
    >>> match_atoms = ['_ZC108', '_ZF1', '_ZF2', '_ZF3']
    >>> smarts = '[CX4](-F)(-F)(-F)'
    >>> smarts_index = 1
    >>> query = Query(smarts, 'SMARTS', smarts_index)
    >>> rdkit_db = Chem.MolFromMol2File('testdata/1A29_001.mol2', removeHs=False)
    >>> ccdc_mol = protein.Protein.from_entry(io.EntryReader('testdata/1A29_001.mol2')[0])
    >>> [int(at.index) for at in get_query_atoms(match_atoms, query, ccdc_mol, rdkit=True, rdkit_db=rdkit_db)]
    [338, 339, 340]
    '''

    query_atoms = []
    if not rdkit:
        substructure = query.substructure()
        for match_atom in match_atoms:
            if substructure.match_atom(hit_protein.atoms[match_atom], substructure.atoms[int(query.index)]):
                query_atoms.append(hit_protein.atoms[match_atom])
    else:
        from ccdc_roche.python.los_rdkit_utilities import RdkitSubstructure
        from ccdc import protein

        matched_atom_labels = RdkitSubstructure(rdkit_db).rdkit_get_query_atoms(query.query, query.index)
        query_atoms = [hit_protein.atom(match_atom) for match_atom in matched_atom_labels if
                       match_atom in match_atoms]
    return query_atoms


def return_los_pat(atom, hit_protein, protein_atom_types_df):
    """
    Assign protein atom type to LoS contact. Perform SMARTs matching and for sulfur, check for disulfide bonds.
    If two cysteine sulfide atoms are within 2.25 Angstrom, they are classified as S_apol.
    :param atom:
    :param hit_protein:
    :param protein_atom_types_df:
    :return:
    """
    # go through protein atom types until one is matched.
    from ccdc import search
    atomic_symbol = atom.atomic_symbol

    def _pat_iter(ccdc_smarts, ccdc_smarts_indices, pats):
        for i, pat_smarts in enumerate(ccdc_smarts):
            pat_smarts_indices = ccdc_smarts_indices[i].split(';')
            pat = pats[i]
            sub = search.SMARTSSubstructure(pat_smarts)
            for pat_smarts_index in pat_smarts_indices:
                if sub.match_atom(atom):  # , sub.atoms[int(pat_smarts_index)] this is faster
                    return pat

    if atomic_symbol != 'S':
        # for index, row in protein_atom_types_df[protein_atom_types_df['atomic_symbol'] == atomic_symbol].iterrows():
        #     pat_smarts_indices = row['CCDC_SMARTS_index'].split(';')
        #     pat_smarts = row['CCDC_SMARTS']
        #     pat = row['protein_atom_type']
        #     sub = search.SMARTSSubstructure(pat_smarts)
        #     for pat_smarts_index in pat_smarts_indices:
        #         if sub.match_atom(atom, sub.atoms[int(pat_smarts_index)]):
        #             return pat
        df = protein_atom_types_df[protein_atom_types_df['atomic_symbol'] == atomic_symbol]
        return _pat_iter(df['CCDC_SMARTS'].values, df['CCDC_SMARTS_index'].values, df['protein_atom_type'].values)

    else:

        sub = search.SMARTSSubstructure('[SX2]')
        if sub.match_atom(atom, sub.atoms[0]):
            return 'S_apol'
        if sulfur_is_part_of_disulfide(atom, hit_protein):
            return 'S_apol'
        else:
            return 'S_don'


def central_ligand_atom_matches_query(atom, query=None, filter_queries=None, rdkit=True, rdkit_db=None):
    '''

    :return: If central ligand atom matches substructure query, return 'query_match'.
    Else, return 'other_central_ligand'.
    >>> from rdkit import Chem
    >>> from ccdc import io
    >>> from ccdc import protein
    >>> smarts = '[CX4](-F)(-F)(-F)'
    >>> smarts_index = 1
    >>> query = Query(smarts, 'SMARTS', smarts_index)
    >>> rdkit_db = Chem.MolFromMol2File('testdata/1A29_001.mol2', removeHs=False)
    >>> ccdc_mol = protein.Protein.from_entry(io.EntryReader('testdata/1A29_001.mol2')[0])
    >>> for a in ccdc_mol.atoms:
    ...     a.partial_charge = a.index
    >>> ccdc_atom = ccdc_mol.atoms[338]
    >>> central_ligand_atom_matches_query(atom=ccdc_atom, query=query, filter_queries=None, rdkit=True, rdkit_db=rdkit_db)
    'query_match'
    >>> smarts = '[c][C]'
    >>> smarts_index = 1
    >>> query = Query(smarts, 'SMARTS', smarts_index)
    >>> ccdc_atom = ccdc_mol.atoms[288]
    >>> central_ligand_atom_matches_query(atom=ccdc_atom, query=query, filter_queries=None, rdkit=True, rdkit_db=rdkit_db)
    'other_central_ligand'
    >>> ccdc_atom = ccdc_mol.atoms[337]
    >>> central_ligand_atom_matches_query(atom=ccdc_atom, query=query, filter_queries=None, rdkit=True, rdkit_db=rdkit_db)
    'query_match'
    '''
    if '_Z' not in atom.label:
        raise Exception('only _Z labeled atoms are allowed.')
    if rdkit:
        from rdkit import Chem
        substructure = Chem.MolFromSmarts(query.query)
        substructure_matches = rdkit_db.GetSubstructMatches(substructure, uniquify=False)
        sub_match = [substructure_match[query.index] for substructure_match in substructure_matches]
        if int(atom.partial_charge) in sub_match:
            return 'query_match'
        else:
            return 'other_central_ligand'

    else:
        sub = query.substructure()
        if sub.match_atom(atom, sub.atoms[int(query.index)]):
            if filter_queries is None:
                return 'query_match'
            else:
                filter_substructures = [filter_query.substructure() for filter_query in filter_queries]
                if not atoms_match_filter([atom], filter_substructures):
                    return 'query_match'
                else:
                    return 'other_central_ligand'
        else:
            return 'other_central_ligand'


def return_atom_type(atom, hit_protein, query=None, filter_queries=None, mode='ligand', protein_atom_types_df=None,
                     rdkit=True, rdkit_db=None):
    atom_label = atom.label
    if atom_label[1] in ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                               'Y', 'V']:
        return return_los_pat(atom, hit_protein, protein_atom_types_df)
    elif 'HOH' in atom_label:
        return 'Water'
    else:
        if '_Z' in atom_label:
            if mode == 'protein':
                return central_ligand_atom_matches_query(atom, query, filter_queries, rdkit, rdkit_db)
            else:
                return None
        else:
            if atom.is_metal:
                return 'metal'
            else:
                return 'other_ligands'


def _assign_los_atom_type(atoms_list, hit_protein, query, filter_queries=None, mode='ligand',
                          protein_atom_types_df=None, rdkit_db=None):
    atoms_with_atom_type = []
    for atom in atoms_list:
        atom.los_at = return_atom_type(atom, hit_protein, query, filter_queries, mode=mode,
                                       protein_atom_types_df=protein_atom_types_df, rdkit_db=rdkit_db)
        atoms_with_atom_type.append(atom)
    return atoms_with_atom_type


def atoms_match_filter(match_atoms, filter_substructures):
    for match_atom in match_atoms:
        for filter_substructure in filter_substructures:
            for filter_atom in filter_substructure.atoms:
                if filter_substructure.match_atom(match_atom, filter_atom):
                    return True
    return False


def substructure_search(query, db, filter_queries=None, return_identifiers=False, rdkit=False, rdkit_db=None):
    '''
    Run substructure search on HET groups.
    add constraints to search object, because it won't work for ConnserSubstructures (https://jira.ccdc.cam.ac.uk/browse/PYAPI-2220)
    :param query: Substructure
    :param db: Entry Reader with input Database
    :param return_identifiers: True returns list of hit.identifier. False returns list of hit.match_atom().index.
    :return: List of hits. If return_identifiers is True, returns List identifiers. If return_identifiers false, return list of
    hit.match_atoms().index
    >>> from ccdc import io, protein
    >>> from ccdc_roche.python.los_utilities import Query
    >>> from rdkit import Chem
    >>> hit_protein = protein.Protein.from_entry(io.EntryReader('testdata/13GS_013.mol2')[0])
    >>> rdkit_mol = Chem.MolFromMol2Block(hit_protein.to_string())
    >>> smarts = '[c]-[OX1]'
    >>> smarts_index = '1'
    >>> query = Query(smarts, 'SMARTS', 1)
    >>> substructure_search(query, hit_protein, rdkit=True, rdkit_db=rdkit_mol)
    [['_Z2', '_Z2']]
    '''
    from ccdc import search
    if rdkit:
        from ccdc_roche.python import los_rdkit_utilities
        from ccdc import protein
        hits = los_rdkit_utilities.RdkitSubstructure(rdkit_db).rdkit_substructure_search(query.query,
                                                                                         return_identifiers)
        return hits

    if not return_identifiers:
        # Limit search to HET group. This will not pick up peptide ligands.
        substructure = query.substructure()
        substructure_search = search.SubstructureSearch()
        substructure_search.add_substructure(substructure)
        for query_atom in substructure_search.substructures[0].atoms:
            # query_atom.add_protein_atom_type_constraint('Ligand', 'Metal', 'Cofactor', 'Nucleotide', 'Unknown')
            query_atom.label_match = '_Z'
        hits = substructure_search.search(database=db)
        if filter_queries is None:
            hits = [[match_atom.index for match_atom in hit.match_atoms()] for hit in hits]
        else:
            filter_substructures = [filter_query.substructure() for filter_query in filter_queries]
            filtered_hits = []
            for hit in hits:
                if not atoms_match_filter(hit.match_atoms(), filter_substructures):
                    filtered_hits.append([match_atom.index for match_atom in hit.match_atoms()])
            hits = filtered_hits

    elif return_identifiers:
        # search all entries for HET groups with _Z label
        substructure_with_z = query.substructure()
        substructure_search_with_z = search.SubstructureSearch()
        substructure_search_with_z.add_substructure(substructure_with_z)
        for query_atom in substructure_search_with_z.substructures[0].atoms:
            # query_atom.add_protein_atom_type_constraint('Ligand', 'Metal', 'Cofactor', 'Nucleotide', 'Unknown')
            query_atom.label_match = '_Z'
        hits_with_z = substructure_search_with_z.search(database=db)

        # search all entries for HET groups without _Z label
        substructure_without_z = query.substructure()
        substructure_search = search.SubstructureSearch()
        substructure_search.add_substructure(substructure_without_z)
        for query_atom in substructure_search.substructures[0].atoms:
            # query_atom.add_protein_atom_type_constraint('Ligand', 'Metal', 'Cofactor', 'Nucleotide', 'Unknown')
            query_atom.label_match = '_U'  # non-central HET groups are marked with _U '^((?!_Z).)*$' # anything that does not match _Z label
        hits_without_z = substructure_search.search(database=db)
        hit_identifiers_without_z = [hit.identifier for hit in hits_without_z]
        # retain only entries where the search query is only present in the _Z ligand, but not on any other HET
        # group
        hit_identifiers = [hit.identifier for hit in hits_with_z if hit.identifier not in hit_identifiers_without_z]
        hits = set(hit_identifiers)
        print(hit_identifiers)

    return hits


def get_smiles_molecule(hit_protein):
    '''
    Generate smiles string from list of atoms of a molecule.
    :param hit_protein:
    :param ligand: List of ligand atoms.
    :return: SMILES string.
    '''
    mol = molecule.Molecule()
    for component in hit_protein.components:
        if '_Z' in component.atoms[0].label:
            mol.add_molecule(component)
    mol.add_hydrogens(mode='missing')
    return mol.smiles


class Query(object):
    '''
    Class that handles both SMARTS and Isostar inputs.
    '''

    def __init__(self, query, type, index):
        '''
        :param query: Isostar filename or SMARTS string.
        :param type: 'Isostar' or 'SMARTS'
        :param indices: Int or Str specifying the contact atom.
        '''
        self.query = query
        self.type = type
        self.index = int(index)

    def substructure(self):
        '''
        Method to generate a search.SMARTSSubstructure or search.ConnserSubstructure object.
        :return: Substructure
        '''
        if self.type == 'SMARTS':
            substructure = self._smarts_substructure(self.query)
        elif self.type == 'Isostar':
            substructure = self._connser_substructure(self.query)
        else:
            raise Exception('Substructure only for SMARTS and Isostar type supported.')
        return substructure

    def _smarts_substructure(self, smarts_query):
        '''
        Return substructure for substructure search from SMARTS string
        :param smarts_query: SMARTS string
        :return: substructure for substructure search
        '''
        from ccdc import search
        smarts_substructure = search.SMARTSSubstructure(smarts_query)
        return smarts_substructure

    def _connser_substructure(self, isostar_file):
        '''
        Return substructure for substructure search from isostar file
        :param isostar_file: Path to isostar file
        :return: substructure for substructure search
        '''
        from ccdc import search
        connser_substructure = search.ConnserSubstructure(isostar_file)
        return connser_substructure


def return_los_contacts(query_atom, hit_protein, close_contact_atoms, central_ligand_atoms,
                        interaction_cutoff=0.5):
    '''
    Return all atoms within LoS of the query atom, but not the query atom.
    Several convenient attributes are added to the LoS atoms.
    :param query_atom:
    :param contact_angle_atom:
    :param hit_protein:
    :param ligand_contacts:
    :param ligands:
    :param substructure_match:
    :return: List of atoms in contact with ligand
    '''
    from ccdc import descriptors
    los_contacts = []
    for atom in close_contact_atoms:
        if atom in query_atom.neighbours:
            continue
        if query_atom.is_in_line_of_sight(atom) and query_atom.partial_charge != atom.partial_charge:
            atom.vdw_distance = descriptors.MolecularDescriptors.atom_distance(query_atom, atom) \
                                - query_atom.vdw_radius - atom.vdw_radius

            if atom.vdw_distance <= interaction_cutoff:
                los_contacts.append(atom)
    return los_contacts


def return_ligand_contacts(ligand_atoms, query_atom, holo_protein, interaction_cutoff=0.5):
    '''
    :param interaction_cutoff:
    :param query:
    :param ligand_atoms:
    :param query_atom:
    :param hit_protein:
    :return: List of atoms in contact with the ligand.
    '''
    from ccdc import descriptors

    # use searcher for faster contact search
    interatomic_cutoff = 3.8 + interaction_cutoff
    ligand_contacts = []
    lig_atom_labels = [ligatom.label for ligatom in ligand_atoms[query_atom.ligand_index]]
    # search for binding site atoms in contact with ligand atoms
    searcher = descriptors.MolecularDescriptors.AtomDistanceSearch(holo_protein)
    for ligand_atom in ligand_atoms[query_atom.ligand_index]:
        ats = searcher.atoms_within_range(ligand_atom.coordinates, interatomic_cutoff)
        for contact_atom in ats:
            if contact_atom.label in lig_atom_labels:
                continue
            if contact_atom not in ligand_contacts:
                if descriptors.MolecularDescriptors.atom_distance(ligand_atom, contact_atom) \
                        - ligand_atom.vdw_radius - contact_atom.vdw_radius <= interaction_cutoff:
                    ligand_contacts.append(contact_atom)
    return ligand_contacts


def return_protein_buriedness(ligand_atoms, query_atom, protein):
    '''
    :param interaction_cutoff:
    :param query:
    :param ligand_atoms:
    :param query_atom:
    :param hit_protein:
    :return: List of atoms in contact with the ligand.
    '''
    from ccdc import descriptors

    interatomic_cutoff = 8
    # search for protein atoms within cutoff
    searcher = descriptors.MolecularDescriptors.AtomDistanceSearch(protein)
    ats = searcher.atoms_within_range(query_atom.coordinates, interatomic_cutoff)
    buriedness = len([a for a in ats if a not in ligand_atoms])
    return buriedness


def get_hit_protein(entry_reader, entry_identifier, rdr, keep_waters=True, keep_good_waters=False):
    '''
    Get a protein molecule from the entry reader.
    :param entry_reader:
    :param entry_identifier:
    :return: Protein
    '''
    if entry_reader is None:
        with rdr as entry_reader:
            entry = entry_reader.entry(entry_identifier)
            hit_protein = protein.Protein.from_entry(entry)
    else:
        entry = entry_reader.entry(entry_identifier)
        hit_protein = protein.Protein.from_entry(entry)
    if not keep_waters and not keep_good_waters:
        hit_protein.remove_all_waters()
    hit_protein.standardise_aromatic_bonds()
    hit_protein.standardise_delocalised_bonds()
    hit_protein.remove_hydrogens()
    hit_protein = assign_index_to_atom_partial_charge(hit_protein)  # store indices in partial_charge
    return hit_protein


def get_rdkit_protein_from_csd_protein(csd_protein):
    '''

    :return: RDKit molecule with Tripos atom types in properties.
    '''
    rdkit_protein = Chem.MolFromMol2Block(csd_protein.components[0].to_string())
    for c in csd_protein.components[1:]:
        if c.atoms[0].protein_atom_type == 'Amino_acid':
            rdkit_component = Chem.MolFromMol2Block(c.to_string())
        else:
            rdkit_component = Chem.MolFromMolBlock(c.to_string('sdf'))
            for atom in c.atoms:
                rdkit_atom = rdkit_component.GetAtomWithIdx(atom.index)
                rdkit_atom.SetProp('_TriposAtomName', atom.label)
        rdkit_protein = Chem.rdmolops.CombineMols(rdkit_protein, rdkit_component)
    return rdkit_protein

