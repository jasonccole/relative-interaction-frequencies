#!/usr/bin/env python

import __future__
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from pathlib import Path
import argparse

########################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description='Split SDF file into N files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-N',
        '--mol_num',
        help='Number of structures per file',
        type=int,
        default=5
    )

    parser.add_argument(
        '-i',
        '--input',
        help='Input file.',
        default='file.sdf'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='Output base filename.',
        default='out'
    )

    parser.add_argument(
        '-m',
        '--minimize',
        help='Output base filename.',
        action='store_true'
    )

    parser.add_argument(
        '-es',
        '--enumerate_stereo',
        help='If passed, unassigned stereo centers will be enumerated.',
        action='store_true'
    )

    return parser.parse_args()

'''
s = Chem.SDMolSupplier('tmp_aligned_3d_sdf_sanitized/ligand_templates_for_mcs_moka.sdf', removeHs=False)
w = Chem.SDWriter('tmp_aligned_3d_sdf_sanitized/ligand_templates_for_mcs_moka_rdkit.sdf')
w.SetKekulize(False)
for m in s:
    m = Chem.AddHs(m, addCoords=True)
    w.write(m)
w.close()'''


def split_sdf(args):
    input_sdf = args.input
    output = args.output
    strucs_per_file = args.mol_num

    flag_substructures = ['[S]([F])([F])([F])([F])([F])', '[B]']
    flag_substructures = [Chem.MolFromSmarts(smarts) for smarts in flag_substructures]

    if Path(input_sdf).suffix == '.sdf':
        supplier = Chem.rdmolfiles.SDMolSupplier(input_sdf, removeHs=False)

    cnt = 0
    all_subsets = []
    subset_mols = []

    nMols = len(supplier)
    for i in range(nMols):
        try:
            m = supplier[i]
            substructure_flag = False
            for substructure in flag_substructures:
                if m.HasSubstructMatch(substructure):
                    substructure_flag = True
            if substructure_flag:
                print('Substructure Flag detected')
                continue
        except:
            print("Could not read mol.")
            continue
        cnt += 1
        if m:
            subset_mols.append(m)
        else:
            print(i)
        if cnt == strucs_per_file:
            cnt = 0
            all_subsets.append(subset_mols)
            subset_mols = []
    if subset_mols:
        all_subsets.append(subset_mols)

    # if strucs_per_file == -1:
    #     all_subsets = all_subsets[0]

    for i, subset in enumerate(all_subsets):
        out_path = Path(output)
        out_path = out_path.parent / Path(out_path.stem + f'_{i+1}.sdf')
        w = Chem.SDWriter(str(out_path))
        w.SetKekulize(False)
        for m in subset:
            try:
                remover = SaltRemover.SaltRemover()
                m2 = remover.StripMol(m)

                if 'SRN' in m2.GetPropsAsDict(includePrivate=True).keys():
                    srn = m2.GetProp('SRN')
                else:
                    srn = ''
                if args.enumerate_stereo:
                    opts = StereoEnumerationOptions(maxIsomers=9)
                    isomers = tuple(EnumerateStereoisomers(m, options=opts))
                    print('Enumerating stereoisomers...')
                else:
                    if args.minimize:

                        stereocenters = Chem.FindMolChiralCenters(m2, includeUnassigned=True, useLegacyImplementation=False)
                        unassigned_stereocenters = [s[1] for s in stereocenters if '?' in s[1]]
                        if unassigned_stereocenters:
                            print(srn, ' has unassigned stereo center.')
                            continue
                        else:
                            isomers = [m2]
                    else:
                        isomers = [m2]
                for m2 in isomers:
                    if args.minimize:
                        m2 = Chem.AddHs(m2, addCoords=True)  # add hydrogens to preserve stereo info

                        if m2.GetNumAtoms() == 0:
                            print(f'failed to remove salts for {srn}')
                            continue
                        print(m2.GetProp('_Name'))
                        AllChem.EmbedMolecule(m2, useRandomCoords=True)
                        AllChem.UFFOptimizeMolecule(m2)
                        minimized_mol = m2
                        w.write(minimized_mol)
                    else:
                        w.write(m2)
            except KeyboardInterrupt:
                sys.exit()
            except:
                continue
        w.close()


def main():
    args = parse_args()
    split_sdf(args)


if __name__ == '__main__':
    main()


