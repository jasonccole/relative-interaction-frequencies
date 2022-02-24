from ccdc import io, protein, entry
import pandas as pd

strucids = pd.read_csv('code.txt', header=None)[0].values
public_db = '../rf_statistics_oct2020/full_p2cq_pub_oct2020.csdsql'
roche_db = '../rf_statistics_oct2020/roche.csdsql'

with io.EntryReader(public_db) as public_rdr, io.EntryReader(roche_db) as roche_rdr, \
        io.EntryWriter('central_ligands.sdf') as w:
    public_identifiers = [e.identifier for e in public_rdr]
    roche_identifiers = [e.identifier for e in roche_rdr]
    for strucid in strucids:
        print(strucid)
        try:
            if len(strucid) == 4:
                binding_site_identifiers = [i for i in public_identifiers if strucid.upper() in i]
                entries = [public_rdr.entry(binding_site_identifier) for binding_site_identifier in binding_site_identifiers]
            else:
                binding_site_identifiers = [i for i in roche_identifiers if strucid.upper() in i]
                entries = [roche_rdr.entry(binding_site_identifier) for binding_site_identifier in binding_site_identifiers]
            for e in entries:
                p = protein.Protein.from_entry(e)
                central_ligand = [c for c in p.components if '_Z' in c.atoms[0].label][0]
                central_ligand_entry = entry.Entry.from_molecule(central_ligand)
                central_ligand_entry.attributes['strucid'] = strucid
                w.write(central_ligand_entry)
        except Exception as e:
            print(e)
            continue