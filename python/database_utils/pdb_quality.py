#!/usr/bin/env python

# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2019-08-14: created by the Cambridge Crystallographic Data Centre
#
'''
Obtain quality attributes for Proasis entries from PDB and write them to a CSV.
'''

###################################################################################################################

import __future__
import argparse
import pandas as pd
import numpy as np
import os
import shelve
from pathos.multiprocessing import ProcessingPool
from pathos.helpers import freeze_support
from functools import partial
from ccdc import io
from ccdc_roche.python import rf_structure_analysis

###################################################################################################################


def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description=
        """
        Obtain quality attributes for Proasis entries from PDB and write them to a CSV.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # To display default values in help message.
    )

    parser.add_argument(
        '-db',
        '--database',
        help='Path to input database in csdsql format. The database entries must be proteins.',
        default=''
    )

    parser.add_argument(
        '-in',
        '--input',
        help='Folder for input files.',
        default=os.getcwd()
    )

    parser.add_argument(
        '-np',
        '--nproc',
        type=int,
        help='Number of processes',
        default=1
    )

    return parser.parse_args()


def _get_ligname(p):
    central_ligand_atoms = rf_structure_analysis.return_central_ligand_atoms(p)
    lignames = []
    for a in central_ligand_atoms:
        lignames.append((a.residue_label, a.chain_label))
    lignames = list(set(lignames))
    return lignames


class PdbVal(object):

    def __init__(self, dbpath='', nproc=1):
        self.dbpath = dbpath
        self.nproc = nproc

    def scrap_pdb(self, pdb_id):
        '''
        Get structure quality information from PDB.
        :param hit_protein:
        :param query_atom:
        :return: Dictionary with structure quality information.
        '''

        import requests

        def url_response(url):
            """
            Getting JSON response from URL
            :param url: String
            :return: JSON
            """
            r = requests.get(url=url, verify=False)
            # Status code 200 means 'OK'
            if r.status_code == 200:
                json_result = r.json()
                return json_result
            else:
                print(r.status_code, r.reason)
                return None

        def run_val_search(pdb_entry):
            """
            Check pdbe search api documentation for more detials
            :param pdbe_search_term: String
            :return: JSON
            """
            # This constructs the complete query URL
            base_url = r'https://www.ebi.ac.uk/pdbe/api/'
            validation_url = r'validation/summary_quality_scores/entry/'
            full_query = base_url + validation_url + pdb_entry
            val_score = url_response(full_query)
            return val_score

        def parse_xml(pdb_entry):
            import xml.etree.ElementTree as ET
            import os
            base_url = r'https://www.ebi.ac.uk/pdbe/entry-files/download/'
            validation_url = r'_validation.xml'
            full_query = base_url + pdb_entry + validation_url
            xml = requests.get(full_query, verify=False)
            with open(xml_filename, 'wb') as f:
                f.write(xml.content)
                # create element tree object
            tree = ET.parse('temp' + pdb_id + '.xml')
            os.remove('temp' + pdb_id + '.xml')
            root = tree.getroot()
            return root

        pdb_id = pdb_id.lower()
        xml_filename = 'temp' + pdb_id + '.xml'
        root = parse_xml(pdb_id)
        overall_quality = run_val_search(pdb_id)[pdb_id]['overall_quality']
        if os.path.isfile(xml_filename):
            os.remove(xml_filename)
        return root, overall_quality

    def return_df(self, pdb_id, pdb_dic, intents=5):
        from ccdc import protein
        import pandas as pd
        try:
            print(pdb_id)
            bs_ids = pdb_dic[pdb_id]
            for intent in range(intents):
                try:
                    root, overall_quality = self.scrap_pdb(pdb_id)
                    break
                except TimeoutError:
                    continue
            bs_df_list = []
            for bs in bs_ids:
                try:
                    with io.EntryReader(self.dbpath) as rdr:
                        entry = rdr.entry(bs)
                        p = protein.Protein.from_entry(entry)
                        lignames = _get_ligname(p)
                        dic = {'identifier': [bs], 'overall_quality': [overall_quality], 'ligand_rscc': [],
                               'ligand_chain_id': [], 'resolution': [], 'ligand_avgoccu': [], 'ligand_altcode': [],
                               'ligand_name': []}
                        for ligname in lignames:
                            lig_quality_dict = self.extract_ligand_quality(root, ligname)
                            dic['ligand_rscc'].append(lig_quality_dict['ligand_rscc'])
                            dic['ligand_chain_id'].append(lig_quality_dict['ligand_chain_id'])
                            dic['resolution'].append(lig_quality_dict['resolution'])
                            dic['ligand_avgoccu'].append(lig_quality_dict['ligand_avgoccu'])
                            dic['ligand_altcode'].append(lig_quality_dict['ligand_altcode'])
                            dic['ligand_name'].append(ligname[0])
                        dic['ligand_name'] = [dic['ligand_name']]
                        dic['ligand_chain_id'] = [dic['ligand_chain_id']]
                        dic['resolution'] = [dic['resolution'][0]]
                        dic['ligand_avgoccu'] = [np.median(dic['ligand_avgoccu'])]
                        dic['ligand_rscc'] = [np.median(dic['ligand_rscc'])]
                        if len(set(dic['ligand_altcode'])) == 1 and dic['ligand_altcode'][0] == ' ':
                            dic['ligand_altcode'] = [' ']
                        else:
                            dic['ligand_altcode'] = ['A']

                        row = pd.DataFrame.from_dict(dic)
                        bs_df_list.append(row)
                except Exception as ex:
                    print(ex)
                    row = pd.DataFrame()
                    return row
            bs_df = pd.concat(bs_df_list)
            return bs_df

        except Exception as ex:
            print(ex)
            row = pd.DataFrame()
            return row

    def extract_ligand_quality(self, root, ligname):

        resname = ligname[0][0:3]
        chain_label = ligname[1]
        ligand_rscc = np.nan
        ligand_avgoccu = np.nan
        ligand_altcode = np.nan
        resolution = np.nan

        for child in root:
            try:
                resolution = child.attrib['PDB-resolution']
                break
            except:
                continue

        for child in root:
            if 'resname' in child.attrib and child.attrib['resname'] == resname and 'chain' in child.attrib and \
                    child.attrib['chain'] == chain_label:
                try:
                    ligand_rscc = float(child.attrib['rscc'])
                    ligand_avgoccu = float(child.attrib['avgoccu'])
                    ligand_altcode = child.attrib['altcode']
                except Exception as ex:
                    pass
                break

        return {'ligand_rscc': ligand_rscc, 'ligand_chain_id': chain_label, 'resolution': resolution,
                'ligand_avgoccu': ligand_avgoccu, 'ligand_altcode': ligand_altcode}

    def run_pdb_val(self):

        def run_multiprocessing(pdb_dic):
            '''Multiprocessing of hits'''
            parallel_return_df = partial(self.return_df, pdb_dic=pdb_dic)
            pool = ProcessingPool(self.nproc)
            matrices = pool.map(parallel_return_df, pdb_dic)
            return matrices

        def run_single_processing(pdb_dic):
            matrices = []
            for pdb_id in pdb_dic:
                matrices.append(self.return_df(pdb_id, pdb_dic))
            return matrices

        with io.EntryReader(self.dbpath) as rdr:
            pdb_dic = {}
            # e = rdr.entry('4DJ4_010')
            for cnt, e in enumerate(rdr):
                if e.identifier.split('_')[0] not in pdb_dic:
                    pdb_dic[e.identifier.split('_')[0]] = [e.identifier]
                else:
                    pdb_dic[e.identifier.split('_')[0]].append(e.identifier)
                # if cnt == 3:
                #     break
            if e.identifier.split('_')[0] not in pdb_dic:
                pdb_dic[e.identifier.split('_')[0]] = [e.identifier]
            else:
                pdb_dic[e.identifier.split('_')[0]].append(e.identifier)

        if self.nproc > 1:
            matrices = run_multiprocessing(pdb_dic)
        else:
            matrices = run_single_processing(pdb_dic)
        print('Concatenating data frames...')
        matrix = pd.concat(matrices, ignore_index=True, sort=False)
        print('Writing out CSV file...')
        matrix.to_csv('data_quality_aug2021.csv', index=False, header=True)
        print('Output files have been written.')

    def return_water_quality_df(self, intents=10):

        def _return_pdb_water_df(pdb_id):
            good_water_df = pd.DataFrame()
            root = False

            for intent in range(intents):
                try:
                    root, overall_quality = self.scrap_pdb(pdb_id)
                    break
                except TimeoutError:
                    continue
                except:
                    break

            if root:
                for child in root:
                    if 'resname' in child.attrib and child.attrib['resname'] == 'HOH':
                        try:
                            if float(child.attrib['avgoccu']) == 1 and child.attrib['altcode'] == " " and float(child.attrib['rscc']) >= 0.9:
                                chain = np.nan
                                if 'chain' in child.attrib:
                                    chain = child.attrib['chain']
                                good_water_df = good_water_df.append(
                                    {'pdb_id': pdb_id, 'resname': child.attrib['resname'],
                                     'resnum': child.attrib['resnum'],
                                     'atomname': child.attrib['resname'] + child.attrib['resnum'],
                                     'avgoccu': child.attrib['avgoccu'],
                                     'altcode': child.attrib['altcode'],
                                     'rscc': child.attrib['rscc'],
                                     'chain': chain}, ignore_index=True)
                        except:
                            continue
            return good_water_df

        #DEPRECATED
        # with shelve.open('pdb_supplier.db') as db:
        #     pdb_ids = set([key.split('_')[0].lower() for key in db.keys()])

        def single_processing(pdb_ids):
            good_water_df_list = []
            for pdb_id in pdb_ids:
                good_water_df_list.append(_return_pdb_water_df(pdb_id))

        def multi_processing(pdb_ids):
            parallel_return_df = partial(_return_pdb_water_df)
            pool = ProcessingPool(self.nproc)
            good_water_df_list = pool.map(parallel_return_df, pdb_ids)
            return good_water_df_list
        if self.nproc > 1:
            good_water_df_list = multi_processing(pdb_ids)
        else:
            good_water_df_list = single_processing(pdb_ids)
        print('Finished scraping PDB.')
        good_water_df = pd.concat(good_water_df_list, ignore_index=True)

        good_water_df.to_parquet('good_water.gzip', compression='gzip')


def main():
    args = parse_args()
    pdbval = PdbVal(dbpath=args.database, nproc=args.nproc)
    pdbval.run_pdb_val()
    # pdbval.return_water_quality_df()


if __name__ == "__main__":
    freeze_support()
    main()
