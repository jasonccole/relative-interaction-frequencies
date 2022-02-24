from openeye import oechem
import argparse


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
        default='input.oeb',
    )

    parser.add_argument(
        '-o',
        '--output',
        help='Output filename.',
        default='output.sdf'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ifs = oechem.oemolistream()
    ofs = oechem.oemolostream()

    if ifs.open(args.input):
        if ofs.open(args.output):
            for mol in ifs.GetOEGraphMols():
                oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Fatal(f"Unable to create {args.output}")
    else:
        oechem.OEThrow.Fatal(f"Unable to open {args.input}")


if __name__ == '__main__':
    main()
