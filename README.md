Authors: Andreas Tosstorff, Jason Cole, Robin Taylor, Bernd Kuhn, 2019-2021

This repository contains scripts to calculate ratio of frequencies values (RF). If you use this code or derived
statistics, please cite:

- https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00858
- https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/cmdc.202100387


Use of the RF interaction analysis as described in:

Augmenting Structure-Based Design with Experimental Protein-Ligand Interaction Data: Molecular Recognition, Interactive 
Visualization, and Rescoring. 
A. Tosstorff, J.C. Cole, R. Bartelt, B. Kuhn 
ChemMedChem 2021, 16(22), 3428-3438 
https://doi.org/10.1002/cmdc.202100387

can be a very powerful tool to highlight potentially favorable and unfavorable protein-ligand interactions in a 
binding site. However, the user should be aware of the following considerations when using the MOE RF tool:

1) RF values indicate the competitiveness for a given interaction by comparing its observed frequency of occurence 
against a statistical null model based on exposed surface areas. RF values > 1 indicate interactions that occur more 
often than expected by chance ("competitive"). Conversely, RF < 1 describes an interaction that occurs less often than 
expected ("non-competitive"). Non-competitive interactions are not necessarily unfavorable, it just means that other 
interactions are more likely to occur in the PDB. From our experience, low RF values < 0.8 likely reflect a poor 
protein-ligand contact and should be addressed to improve binding. The RF sliders in MOE can be used to limit the RF 
interaction display to the desired range.

2) Due to the implicit hydrogen-atom approach in generating the underlying statistics, the statistics is less accurate 
for OH groups. This applies for ex. to the sidechain oxygen atoms of Ser, Thr, and Tyr, as well as to water.

3) Short protein-ligand contacts are not well captured by the RF interaction statistics, and the MOE clash detection 
should be turned on to detect such cases. In particular, ligand atoms with unusually high RF scores (e.g. > 3) and many 
protein contacts sometimes involve a steric clash. Their interatomic distances should be checked carefully, for ex. 
using distance labeling in the MOE RF panel.

4) MOE RF interactions that are statistically significant are highlighted as thick lines and should be higher weighted 
than those with thin lines, indicating only a trend but no statistical significance. 

5) In rare cases, a short protein-ligand contact might exist that is not detected by the RF interaction tool. This can 
arise due to a missing ligand atom type, and can be detected by checking "Ligand Type" and "Line-of-Sight" labels in the 
MOE RF panel.


---
**Dependencies**
- CSD Python API 3.0.9
- RDKit 2021
- Pathos
- Seaborn
- Pandas

---
**Running Protonate3D on all Proasis binding sites**
- database_utils/protonate3d/submit_proasis_protonate3d.sh submits jobs to the SLURM queue. Breaks are in place to limit 
the number of MOE licenses being used.  
---
**Generating Databases** 

Binding sites should be in the Proasis format. \
Proasis3, version 3.532, © 2010-2020, Desert Scientific Software, Pty Ltd, All rights reserved.


- database_utils/rdkit_library.py generates an RDKit Library for substructure search from Proasis binding sites.
- database_utils/ccdc_library.py generates .csdsql file from Proasis binding sites.
- database_utils/pdb_quality.py extracts crystal structure quality information from the PDB.

---
**Calculating new RF statistics**  \
Requires CSD API 3.0.9 or higher
- los_csv_to_lsf.py will submit jobs to the Roche LSF queue.
- Atom type definitions should be provided in CSV files. Ligand atom types should be sorted manually. 
Ligand atom types at the top of the list will be matched first.
- Input protein-ligand binding sites should be in the "Proasis" format, i.e. central ligand atoms start with _Z, 
other HET groups start with _U. Protein atoms start with _[one_letter_code].
- update_lookup_files.py writes lookup CSV files for MOE
---
**Generating PyMOL scenes with RF values**
```
# From Proasis database
rf_assignment.py -i ABCD_001
rf_assignment.py -i ABCDE_001

# From GOLD output
rf_assignment.py -i gold.sdf --gold gold.conf
```

---
**Generating CSV with RF values for an SDF of ligands and a target protein**
```
rf_values_from_pdb_and_sdf.py -p apo_protein.pdb -l ligands.sdf
```
---
