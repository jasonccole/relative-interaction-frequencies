#!/usr/bin/env python

########################################################################################################################

import os
import subprocess as sp
import pandas as pd
import argparse
import time
from glob import glob
from pathlib import Path
import numpy as np

########################################################################################################################


def submit(output_files, executable):

        # check for duplicates:
    files_with_executable = list(sorted([file_with_executable for file_with_executable in output_files if executable in
                                         file_with_executable]))
    if len(files_with_executable) == 0:
        return True

    if len(files_with_executable) >= 2:
        duplicates = files_with_executable[0:-1]
        print('removing duplicates...')
        for duplicate in duplicates:
            if os.path.isfile(duplicate):
                os.remove(duplicate)

    if executable in files_with_executable[-1]:
        f = files_with_executable[-1]
        f = os.path.basename(f)
        jobid = f.split('_')[1]
        status = sp.check_output(['sacct', '-j', f'{jobid}', '-P', '-n', '-b'])
        status = status.decode('utf-8').split('\n')[0].split('|')[-2]
        if 'COMPLETED' in status or 'RUNNING' in status or 'PENDING' in status:
            return False
        else:

            for f_2 in output_files:
                if executable == 'contacts' and os.path.isfile(f_2):
                    if os.path.isfile(f_2):
                        os.remove(f_2)
                else:
                    if executable in f_2 and os.path.isfile(f_2):
                        if os.path.isfile(f_2):
                            os.remove(f_2)
            print('Resubmitting...')
            return True


def submit_to_hpc(mode, geometries, protein_atom_types=['']):
    output_files = glob(f'slurm_output/slurm_*_{mode}_*.out')
    if len(output_files) == 0 or submit(output_files, 'contacts'):
        slurm_command = ['sbatch', '--parsable', f'slurm_input_{mode}.sh']
        print(os.getcwd().split('/')[-1], mode, 'contact count...')
        contact_out = sp.check_output(slurm_command).rstrip().decode('utf-8')
        print(contact_out)

        for protein_atom_type in protein_atom_types:
            if mode == 'protein':
                file_ext = f'_{protein_atom_type}'
            else:
                file_ext = ''
            dependent_slurm_filter = ['sbatch', '--parsable', f'--dependency=afterany:{contact_out}', f'slurm_input_{mode}_filter{file_ext}.sh']
            filter_out = sp.check_output(dependent_slurm_filter).rstrip().decode('utf-8')
            for geometry in geometries:
                dependent_slurm = ['sbatch', f'--dependency=afterany:{filter_out}',
                                   f'slurm_input_{mode}_{geometry}{file_ext}.sh']
                sp.Popen(dependent_slurm)

    else:
        for protein_atom_type in protein_atom_types:
            if mode == 'protein':
                file_ext = f'_{protein_atom_type}'
                folder = protein_atom_type
            else:
                file_ext = ''
                folder = 'query_atom'

            output_files = glob(f'slurm_output/slurm_*_{mode}_*{file_ext}.out')

            stats_check = {geometry: False for geometry in geometries}

            for geometry in geometries:
                if (Path(folder) / Path(f'statistics_{geometry}.csv')).is_file():
                    stats_check[geometry] = True
                elif geometry == 'h' and 'pi' not in protein_atom_type and mode == 'protein':
                    stats_check[geometry] = True

            if np.prod(list(stats_check.values())) == 1:
                continue

            if submit(output_files, 'filter'):
                print(os.getcwd().split('/')[-1], mode, 'postprocessing...')

                slurm_filter = ['sbatch', '--parsable', f'slurm_input_{mode}_filter{file_ext}.sh']
                filter_out = sp.check_output(slurm_filter).rstrip().decode('utf-8')
                print(filter_out)
                for geometry in geometries:
                    dependent_slurm = ['sbatch', f'--dependency=afterany:{filter_out}',
                                         f'slurm_input_{mode}_{geometry}{file_ext}.sh']
                    sp.Popen(dependent_slurm)

            else:

                for geometry in geometries:
                    if submit(output_files, geometry):
                        dependent_slurm = ['sbatch', f'slurm_input_{mode}_{geometry}{file_ext}.sh']
                        sp.Popen(dependent_slurm)


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description='Submit Calculate Rf values on Basel HPC cluster.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '--input_csv',
        help='Filepath for input CSV file with SMARTS and SMARTS_index columns',
        default='ligand_atom_types.csv'
    )

    parser.add_argument(
        '-np',
        help='Number of parallel processes for multiprocessing.',
        default=24
    )

    parser.add_argument(
        '--los_home',
        help='Path to LoS home folder.',
        default='.'
    )

    parser.add_argument(
        '-db',
        '--database',
        nargs='*',
        help='Database name in LoS home folder.',
        default=['full_p2cq_pub_aug2021.csdsql', 'full_p2cq_roche_aug2021.csdsql']
    )

    return parser.parse_args()


class LoSSlurm(object):
    def __init__(self, csv_file, np, db=['full_p2cq_pub_aug2021.csdsql', 'full_p2cq_roche_aug2021.csdsql'],
                 los_home=''):
        self.df = pd.read_csv(os.path.join(los_home, csv_file), sep='\t')
        self.np = np
        self.los_home = os.path.abspath(los_home)
        self.db = db
        self.protein_atom_types_df = pd.read_csv(os.path.join(los_home, 'protein_atom_types.csv'), sep='\t')
        self.protein_atom_types = self.protein_atom_types_df['protein_atom_type'].unique()

    def write_slurm_input(self, mode, executable, smarts, smarts_index, ligand_atom_type, pi_atom):
        if mode == 'protein':
            runtime = '1-12:00'
        else:
            runtime = '1-12:00'
        '''

        :param mode: protein or ligand 
        :return: 
        '''
        slurm_string = f"""#!/bin/bash
#SBATCH --job-name {mode}
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {self.np}
#SBATCH --time {runtime}
#SBATCH --qos normal
#SBATCH --output slurm_output/slurm_%J_{mode}_{executable}.out

#LOAD ENVIRONMENT

los_hpc.py -m {mode} -exe {executable} --smarts \'{smarts}\' --smarts_index {smarts_index} --ligand_atom_type \'{ligand_atom_type}\' {pi_atom} -np {self.np} --los_home {self.los_home} -db {' '.join(self.db)}
"""
        slurm_file = open(f'slurm_input_{mode}.sh', 'w')
        slurm_file.write(slurm_string)

    def write_slurm_postprocessing_input(self, mode, executable, pi_atom):
        geometry = executable
        if executable in ['tau', 'alpha']:
            executable = 'angle'
            angle_name = f'--angle_name {geometry}'
        else:
            angle_name = ''
        if mode == 'protein':
            runtime = '3-00:00'
            protein_atom_types = self.protein_atom_types

        else:
            runtime = '3-00:00'
            protein_atom_types = ['']

        for protein_atom_type in protein_atom_types:
            if mode == 'protein':
                protein_atom_type_flag = f'--protein_atom_type {protein_atom_type}'
                filename_ext = f'_{protein_atom_type}'
                if executable == 'h' and 'pi' not in protein_atom_type:
                    continue
            else:
                protein_atom_type_flag = ''
                filename_ext = ''
            slurm_string = f"""#!/bin/bash
#SBATCH --job-name post_{mode[:3]}
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time {runtime}
#SBATCH --qos normal
#SBATCH --output slurm_output/slurm_%J_{mode}_{geometry}{filename_ext}.out

#LOAD ENVIRONMENT

los_hpc.py -m {mode} -exe {executable} {pi_atom} {angle_name} --los_home {self.los_home} {protein_atom_type_flag}
"""

            slurm_file = open(f'slurm_input_{mode}_{geometry}{filename_ext}.sh', 'w')
            slurm_file.write(slurm_string)

    def csv_iterator(self):
        for index, row in self.df.iterrows():
            if type(row['RDKit_SMARTS_index']) == str:
                smarts_indices = row['RDKit_SMARTS_index'].split(';')
            else:
                smarts_indices = [row['RDKit_SMARTS_index']]

            for smarts_index in smarts_indices:
                smarts = row['RDKit_SMARTS']
                pi_atom = row['pi_atom']
                if pi_atom:
                    pi_atom = '-pi'
                else:
                    pi_atom = ''

                ligand_atom_type = row['ligand_atom_type']
                output_folder = os.path.join('output', ligand_atom_type)

                # if ligand_atom_type not in ['nitrogen_aromatic_acceptor_n_don', 'nitrogen_aromatic_donor_n_acc']:
                #     continue

                if not os.path.exists('output'):
                    os.mkdir('output')
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                if not os.path.exists(os.path.join(output_folder, 'slurm_output')):
                    os.mkdir(os.path.join(output_folder, 'slurm_output'))

                os.chdir(output_folder)

                self.write_slurm_input('ligand', 'contacts', smarts, smarts_index, ligand_atom_type, pi_atom)
                self.write_slurm_input('protein', 'contacts', smarts, smarts_index, ligand_atom_type, pi_atom)

                ligand_geometries = ['alpha']
                protein_geometries = ['alpha', 'h']
                if pi_atom:
                    ligand_geometries.append('h')
                if 'oxygen' in output_folder and 'carboxylate' in output_folder:
                    self.write_slurm_postprocessing_input('ligand', 'tau', pi_atom)
                    ligand_geometries.append('tau')

                self.write_slurm_postprocessing_input('ligand', 'filter', pi_atom)
                self.write_slurm_postprocessing_input('protein', 'filter', pi_atom)

                for geometry in ligand_geometries:
                    self.write_slurm_postprocessing_input('ligand', geometry, pi_atom)

                for geometry in protein_geometries:
                    self.write_slurm_postprocessing_input('protein', geometry, pi_atom)

                running_jobs = sp.check_output(['squeue', '-t', 'running']).decode('utf-8').split('\n')
                submitted_jobs = sp.check_output(['squeue']).decode('utf-8').split('\n')
                running_jobs = len(running_jobs) - 2
                submitted_jobs = len(submitted_jobs) - 2
                print('Submitted jobs:', submitted_jobs)
                print('Running jobs: ', running_jobs)
                while running_jobs > 1000 or submitted_jobs > 900:
                    running_jobs = sp.check_output(['squeue', '-t', 'running']).decode('utf-8').split('\n')
                    submitted_jobs = sp.check_output(['squeue']).decode('utf-8').split('\n')
                    running_jobs = len(running_jobs) - 2
                    submitted_jobs = len(submitted_jobs) - 2
                    print('Submitted jobs:', submitted_jobs)
                    print('Running jobs: ', running_jobs)
                    print('Waiting for license...')
                    time.sleep(20)

                submit_to_hpc('ligand', ligand_geometries)
                submit_to_hpc('protein', protein_geometries, self.protein_atom_types)
                time.sleep(1)
                os.chdir('../..')


def main():
    args = parse_args()
    slurm_submittor = LoSSlurm(args.input_csv, args.np, args.database, args.los_home)
    slurm_submittor.csv_iterator()


if __name__ == "__main__":
    main()
