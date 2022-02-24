#!/usr/bin/env python

########################################################################################################################

import __future__
from ccdc import io, descriptors, protein, molecule, entry, search
import os
import pandas as pd
from ccdc_roche.python import los_utilities
from pathlib import PurePath
from pathos.multiprocessing import ProcessingPool
from functools import partial
import argparse
import traceback
import sys
from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import BRICS
import shelve
from ccdc_roche.python import los_rdkit_utilities
import re

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Align ligands filtered by RF value and geometric constraints.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '--protein_atom_type',
        help='Protein atom type for which to find high RF contacts.',
        type=str,
        default=''
    )

    parser.add_argument(
        '--protein_file',
        help='Protein file.',
        type=str,
        default='mol.sdf'
    )

    parser.add_argument(
        '--ligand_file',
        help='Ligand file.',
        type=str,
        default='ligand.sdf'
    )

    parser.add_argument(
        '--alignment_coordinates',
        help='Coordinates towards which to align.',
        type=str,
        default=[0, 0, 0]
    )

    parser.add_argument(
        '--protein_atom_index',
        help='Index of the protein atom type for which to find high RF contacts.',
        type=int,
        default=''
    )

    parser.add_argument(
        '--ligand_atom_index',
        help='Index of the ligand atom type for which to find high RF contacts.',
        type=int,
        default=''
    )

    parser.add_argument(
        '--search_limit',
        help='Maximum number of hits to be returned',
        type=int,
        default='50'
    )

    parser.add_argument(
        '-nproc',
        help='Number of parallel processes for multiprocessing.',
        type=int,
        default=1
    )

    parser.add_argument(
        '-db',
        '--database',
        help='Database name in LoS home folder.',
        type=str,
        default='public'
    )

    return parser.parse_args()


def _make_new_molecule(atoms):
    new_molecule = molecule.Molecule()
    new_molecule.add_atoms(atoms)
    atom_labels = [a.label for a in new_molecule.atoms]
    bonds = [a.bonds for a in atoms]
    flat_bonds = [bond for sublist in bonds for bond in sublist]
    keep_bonds = [bond for bond in flat_bonds if bond.atoms[0] in atoms and bond.atoms[1] in atoms]
    for bond in keep_bonds:
        a0 = new_molecule.atoms[atom_labels.index(bond.atoms[0].label)]
        a1 = new_molecule.atoms[atom_labels.index(bond.atoms[1].label)]
        bt = bond.bond_type
        try:
            new_molecule.add_bond(bt, a0, a1)
        except RuntimeError:
            continue
    new_molecule.standardise_aromatic_bonds()
    return new_molecule


def _make_new_simple_molecule(atoms):
    new_molecule = molecule.Molecule()
    new_molecule.add_atoms(atoms)
    atom_labels = [a.label for a in new_molecule.atoms]
    bonds = [a.bonds for a in atoms]
    flat_bonds = [bond for sublist in bonds for bond in sublist]
    keep_bonds = [bond for bond in flat_bonds if bond.atoms[0] in atoms and bond.atoms[1] in atoms]
    for bond in keep_bonds:
        a0 = new_molecule.atoms[atom_labels.index(bond.atoms[0].label)]
        if '_Z' in a0.label:
            bt = bond.bond_type
            a1 = new_molecule.atoms[atom_labels.index(bond.atoms[1].label)]
        else:
            bt = molecule.Bond.BondType(1)
            a1 = new_molecule.atoms[atom_labels.index(bond.atoms[1].label)]

            a0.formal_charge = 0
            a1.formal_charge = 0
        try:
            new_molecule.add_bond(bt, a0, a1)
        except RuntimeError:
            continue
    new_molecule.standardise_aromatic_bonds()
    return new_molecule


class RfLookup(object):

    def __init__(self, dbs='internal', protein_atom_type='N_pi_don'):
        self.los_home = ''
        self.structure_db = []
        self.lookup_paths = []
        for db in dbs.split(','):
            if db == 'internal':
                rf_db = 'full_p2cq_roche_oct2019_rf.gzip'
                self.structure_db.append('full_p2cq_roche_oct2019.csdsql')
            if db == 'public':
                rf_db = 'full_p2cq_pub_oct2019_rf.gzip'
                self.structure_db.append('full_p2cq_pub_oct2019.csdsql')
            self.lookup_paths.append(PurePath(self.los_home, rf_db))
        # if protein_atom_type == 'O_ali_mix' or protein_atom_type == 'O_pi_mix':
        #     protein_atom_type = 'O_mix'
        self.protein_atom_type = protein_atom_type

    def return_competitive_df_from_geometry(self):
        competitive_dfs = []
        for lookup_path in self.lookup_paths:
            competitive_df = pd.read_parquet(lookup_path, columns=['protein_atom_type', 'rf_total_error',
                                                                   'rf_total', 'identifier',
                                                                   'los_atom_index', 'ligand_atom_index',
                                                                   'ligand_smiles', 'project', 'is_cofactor',
                                                                   'is_glycol']
                                             ).sort_values(by='rf_total', ascending=False)
            competitive_df = competitive_df.loc[(competitive_df.loc[:, 'protein_atom_type'] == self.protein_atom_type) &
                                                (competitive_df.loc[:, 'rf_total'] - competitive_df.loc[:,
                                                                                     'rf_total_error']
                                                 > 1)].copy()
            competitive_df = competitive_df[(competitive_df['is_cofactor'] == False) &
                                            (competitive_df['is_glycol'] == False)]
            competitive_dfs.append(competitive_df)
        competitive_df = pd.concat(competitive_dfs, ignore_index=True)

        project_df = competitive_df[competitive_df['project'].isna()==False].drop_duplicates(
            ['ligand_smiles', 'project'])
        no_project_df = competitive_df[competitive_df['project'].isna()]
        competitive_df = pd.concat([project_df, no_project_df], ignore_index=True).sort_values(by='rf_total',
                                                                                               ascending=False)
        return competitive_df

    def _trim_reference_protein(self, reference_protein, reference_protein_index, env=3):
        ref_atoms_to_keep = set([reference_protein.atom('_RP_XX')])
        for env in range(env):
            ref_atoms_to_keep.update([n for sublist in [a.neighbours for a in ref_atoms_to_keep] for n in sublist])
        ref_atoms_to_keep.update(reference_protein.atom('_RP_XX').neighbours)
        reference_protein = _make_new_molecule(ref_atoms_to_keep)
        return reference_protein

    def _prepare_reference_protein(self, reference_protein, reference_protein_index,
                                   reference_protein_dummy_coordinates):

        if '.mol2' in reference_protein or '.sdf' in reference_protein:
            reference_protein = protein.Protein.from_entry(io.EntryReader(reference_protein)[0])

        elif len(reference_protein) == 8:
            rdr = io.EntryReader(os.path.join(self.los_home, 'full_p2cq_pub_oct2019.csdsql'))
            reference_protein = protein.Protein.from_entry(rdr.entry(reference_protein))

        elif len(reference_protein) == 9:
            rdr = io.EntryReader(os.path.join(self.los_home, 'full_p2cq_roche_oct2019.csdsql'))
            reference_protein = protein.Protein.from_entry(rdr.entry(reference_protein))

        reference_protein.atoms[reference_protein_index].label = '_RP_XX'
        reference_protein.remove_all_waters()
        reference_protein.remove_hydrogens()

        if len(reference_protein.atom('_RP_XX').neighbours) == 1:
            protein_cutoff = 2
        else:
            protein_cutoff = 1
        reference_protein = self._trim_reference_protein(reference_protein, reference_protein_index, protein_cutoff)
        dummy_atom = molecule.Atom(coordinates=reference_protein_dummy_coordinates, label='XY')
        reference_protein.add_atom(dummy_atom)

        return reference_protein

    def _fragment_is_similar_to_existing(self, fragments, new_fragment):
        for fragment in fragments:

            searcher = search.SimilaritySearch(fragment)
            similar_hits = searcher.search_molecule(new_fragment)
            similarity = round(similar_hits.similarity, 2)

            if similarity > 0.7:  # check for conformational similarity
                return True

                # structural alignment problem not yet solved
                # fragment_protein = [c for c in fragment.components if '_Z' not in c.atoms[0].label][0]
                # hit_protein_protein = [c for c in hit_protein.components if '_Z' not in c.atoms[0].label][0]
                # substructure_analyzer = descriptors.MolecularDescriptors.MaximumCommonSubstructure()
                # mcs_atoms = substructure_analyzer.search(fragment_protein, hit_protein_protein)[0]
                # try:
                #     alihit_prot, rmsd, tan_rmsd, transformation = descriptors.MolecularDescriptors.overlay_rmsds_and_transformation(
                #         fragment, hit_protein, mcs_atoms, with_symmetry=False)
                #     with io.MoleculeWriter('overlay.sdf') as w:
                #         w.write(fragment)
                #         w.write(alihit_prot)
                #     # if tan_rmsd >= 0.7:
                #     #     return True
                # except RuntimeError:
                #     print('fail')
                #     # continue
            else:
                continue
        return False

    def _get_matrices_and_for(self, hit_protein_ex_ligand):

        transformed_mol = hit_protein_ex_ligand.copy()
        point_group_analyzer = descriptors.MolecularDescriptors.point_group_analysis(transformed_mol, return_matrices=True)
        rotation_matrices = point_group_analyzer[3]
        frame_of_ref = point_group_analyzer[4]
        return rotation_matrices, frame_of_ref

    def _get_overlay_atoms(self, reference_protein, hit_protein_ex_ligand):

        # Overlay
        reference_protein_neighbours = reference_protein.atom('_RP_XX').neighbours
        hit_protein_ex_ligand_neighbours = hit_protein_ex_ligand.atom('_P_XX').neighbours
        if len(reference_protein_neighbours) >= len(hit_protein_ex_ligand_neighbours):
            overlay_atoms = [(reference_protein.atom('_RP_XX'), hit_protein_ex_ligand.atom('_P_XX'))] + \
                            [(reference_protein.atom('_RP_XX').neighbours[i], n)
                             for i, n in enumerate(hit_protein_ex_ligand_neighbours)]

        else:
            overlay_atoms = [(reference_protein.atom('_RP_XX'), hit_protein_ex_ligand.atom('_P_XX'))] + \
                            [(n, hit_protein_ex_ligand.atom('_P_XX').neighbours[i])
                             for i, n in enumerate(reference_protein_neighbours)]

        if len(reference_protein_neighbours) == len(hit_protein_ex_ligand_neighbours) == 1:
            if len(reference_protein_neighbours[0].neighbours) >= len(hit_protein_ex_ligand_neighbours[0].neighbours):
                overlay_atoms = overlay_atoms + [(reference_protein_neighbours[0].neighbours[i], n) for
                                                 i, n in enumerate(hit_protein_ex_ligand_neighbours[0].neighbours) if
                                                 '_P_XX' not in n.label]
            else:
                overlay_atoms = overlay_atoms + [(n, hit_protein_ex_ligand_neighbours[0].neighbours[i]) for
                                                 i, n in enumerate(reference_protein_neighbours[0].neighbours) if
                                                 '_RP_XX' not in n.label]

        elif 'pi' in self.protein_atom_type:
            if len(overlay_atoms) < 4:
                overlay_atoms = overlay_atoms + [(n, hit_protein_ex_ligand_neighbours[0].neighbours[i]) for
                                                 i, n in enumerate(reference_protein_neighbours[0].neighbours) if
                                                 '_RP_XX' not in n.label]
        return overlay_atoms

    def _align_ligand(self, iterrow, reference_protein, reference_ligand, ligand_atom_index, fragments):

        index, row = iterrow
        rdr = io.EntryReader([os.path.join(self.los_home, db) for db in self.structure_db])
        identifier = row['identifier']
        protein_contact_atom_index = int(row['los_atom_index'])
        ligand_contact_atom_index = int(row['ligand_atom_index'])
        hit_protein = protein.Protein.from_entry(rdr.entry(identifier))
        hit_protein.remove_all_waters()
        hit_protein = los_utilities.assign_index_to_atom_label(hit_protein)

        hit_protein.atoms[ligand_contact_atom_index].label = '_Z_XX'
        hit_protein.atoms[protein_contact_atom_index].label = '_P_XX'

        atoms_to_keep = set([at for at in hit_protein.atoms if '_Z' in at.label])

        # also keep protein atoms
        protein_atoms_to_keep = set([hit_protein.atom('_P_XX')])
        if len(hit_protein.atom('_P_XX').neighbours) == 1:
            protein_cutoff = 2
        elif 'pi' in self.protein_atom_type and len(hit_protein.atom('_P_XX').neighbours) < 3:
            protein_cutoff = 2
        else:
            protein_cutoff = 1
        for env in range(protein_cutoff):
            protein_atoms_to_keep.update(
                [n for sublist in [a.neighbours for a in protein_atoms_to_keep] for n in sublist])

        atoms_to_keep.update(protein_atoms_to_keep)
        hit_protein = _make_new_simple_molecule(atoms_to_keep)
        hit_protein_ex_ligand = [c for c in hit_protein.components if '_Z' not in c.atoms[0].label][0]

        overlay_atoms = self._get_overlay_atoms(reference_protein, hit_protein_ex_ligand)

        overlaid_hit_protein = descriptors.MolecularDescriptors.overlay(reference_protein, hit_protein, overlay_atoms,
                                                                        with_symmetry=False, invert=True)
        overlaid_hit_protein_ex_ligand = [c for c in overlaid_hit_protein.components if '_Z' not in c.atoms[0].label][0]

        rotation_matrices, frame_of_ref = self._get_matrices_and_for(overlaid_hit_protein_ex_ligand)
        distance = descriptors.MolecularDescriptors.atom_distance(reference_protein.atom('XY'),
                                                                  overlaid_hit_protein.atom('_Z_XX'))
        aligned_ligands_df = pd.DataFrame({'aligned_molecule': [overlaid_hit_protein], 'distance': [distance]})
        for rotation_matrix in rotation_matrices:
            t_hit_protein = overlaid_hit_protein.copy()
            move_to_for = t_hit_protein.Transformation.from_rotation_and_translation(
                ((1, 0, 0), (0, 1, 0), (0, 0, 1)), frame_of_ref[0]).inverse()
            move_to_for2 = t_hit_protein.Transformation.from_rotation_and_translation(frame_of_ref[1], (0, 0, 0))
            sym_op = t_hit_protein.Transformation.from_rotation_and_translation(rotation_matrix, (0, 0, 0))
            t_hit_protein.transform(move_to_for)
            t_hit_protein.transform(move_to_for2)
            t_hit_protein.transform(sym_op)
            t_hit_protein.transform(move_to_for2.inverse())
            t_hit_protein.transform(move_to_for.inverse())
            protein_atom_distance = descriptors.MolecularDescriptors.atom_distance(reference_protein.atom('_RP_XX'),
                                                                                   t_hit_protein.atom('_P_XX'))
            if protein_atom_distance > 0.5:
                continue

            t_hit_protein_ex_ligand = [c for c in t_hit_protein.components if '_Z' not in c.atoms[0].label][0]
            rmsd_atoms = self._get_overlay_atoms(reference_protein, t_hit_protein_ex_ligand)
            protein_rmsd = descriptors.MolecularDescriptors.rmsd(reference_protein, t_hit_protein_ex_ligand,
                                                                 atoms=rmsd_atoms, with_symmetry=False)
            if protein_rmsd > 2:
                continue

            ligand_atom_distance = descriptors.MolecularDescriptors.atom_distance(reference_protein.atom('XY'),
                                                                                  t_hit_protein.atom('_Z_XX'))
            if ligand_atom_distance > 2:
                continue

            aligned_ligands_df = aligned_ligands_df.append({'aligned_molecule': t_hit_protein,
                                                            'ligand_atom_distance': ligand_atom_distance,
                                                            'protein_atom_distance': protein_atom_distance},
                                                           ignore_index=True)

        hit_protein = aligned_ligands_df.loc[aligned_ligands_df['ligand_atom_distance'].idxmin(), 'aligned_molecule']

        pl_complex = entry.Entry.from_molecule(hit_protein)


        ligand = [c for c in hit_protein.components if '_Z' in c.atoms[0].label][0]
        fragment = ligand.copy()
        # Ligand fragment
        atoms_to_keep = set([fragment.atom('_Z_XX')])
        # also keep neighbours
        for env in range(3):
            atoms_to_keep.update([n for sublist in [a.neighbours for a in atoms_to_keep] for n in sublist])

        atoms_to_remove = [at for at in fragment.atoms if at not in atoms_to_keep]
        fragment.remove_atoms(atoms_to_remove)

        if self._fragment_is_similar_to_existing(fragments, fragment):
            return

        reference_ligand_atom = reference_ligand.atoms[ligand_atom_index]
        fragment_central_atom = fragment.atom('_Z_XX')
        overlay_atoms = [(reference_ligand_atom, fragment_central_atom)]
        if len(reference_ligand_atom.neighbours) <= len(fragment_central_atom.neighbours):
            for i, n in enumerate(reference_ligand_atom.neighbours):
                overlay_atoms.append((n, fragment_central_atom.neighbours[i]))
        else:
            for i, n in enumerate(fragment_central_atom.neighbours):
                overlay_atoms.append((reference_ligand_atom.neighbours[i], n))
        ligand_rmsd = descriptors.MolecularDescriptors.rmsd(reference_ligand, fragment, atoms=overlay_atoms,
                                                            with_symmetry=False)

        pl_complex.attributes = {'rf_total': row['rf_total'], 'identifier': identifier, 'fragment_rmsd': ligand_rmsd}
        return pl_complex, ligand, fragment

    def align_ligands(self, competitive_df, reference_protein, ligand_file, reference_protein_index, ligand_atom_index,
                      reference_protein_dummy_coordinates, nproc=1, search_limit=50):

        if nproc > 1:
            parallel_return_structure_df = partial(self._align_ligand, reference_protein=reference_protein,
                                                   reference_protein_index=reference_protein_index,
                                                   reference_protein_dummy_coordinates=reference_protein_dummy_coordinates
                                                   )
            pool = ProcessingPool(nproc)

        else:
            fragments = []
            ligands = []
            pl_complexes = []

            reference_ligand = io.MoleculeReader(ligand_file)[0]
            reference_protein = self._prepare_reference_protein(reference_protein, reference_protein_index,
                                                                reference_protein_dummy_coordinates)

            for iterrow in competitive_df.iterrows():
                try:
                    pl_complex, ligand, fragment = self._align_ligand(iterrow, reference_protein, reference_ligand,
                                                                      ligand_atom_index, fragments)
                    pl_complexes.append(pl_complex)
                    fragments.append(fragment)
                    ligands.append(ligand)
                except (KeyError, TypeError):
                    continue
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    print(e)
                if len(fragments) >= search_limit:
                    break
            with io.EntryWriter('ligands.sdf') as wl, io.EntryWriter('fragments.sdf') as wf, \
                    io.EntryWriter('complex.sdf') as wc:
                for f in fragments:
                    if f:
                        wf.write(f)
                for lig in ligands:
                    if lig:
                        wl.write(lig)
                for pl_com in pl_complexes:
                        wc.write(pl_com)


def main():
    args = parse_args()
    if type(args.alignment_coordinates) == str:
        alignment_coordinates = tuple([float(coord) for coord in args.alignment_coordinates.split(',')])
    else:
        alignment_coordinates = args.alignment_coordinates
    looker = RfLookup(dbs=args.database, protein_atom_type=args.protein_atom_type)
    competitive_df = looker.return_competitive_df_from_geometry()
    looker.align_ligands(competitive_df, args.protein_file, args.ligand_file, args.protein_atom_index,
                         args.ligand_atom_index, alignment_coordinates, nproc=int(args.nproc),
                         search_limit=args.search_limit)


if __name__ == '__main__':
    main()
