# Disorder: Extraction and Analysis of distribution of disorder in crystalline materials reported in the ICSD

This repository contains:

(1) The reader of crystallographic structures from CIFs: **Read_CIF** (CIFReader.py). The main difference of this tool compared to analogious tool from pymatgen and ASE is that it correctly labels crystallographic orbits occupied by the same element. For example, in Hg2 Na2 Se6 Sn2 (ICSD col code 013796) there are two orbits occupied by Se: occupies two orbits 16l, and 8h. Atoms on those two orbits will be assigned different orbit labels by Read_CIF in contrast to pymatgen and ASE readers. 

```
from CIFReader import Read_CIF

file=Read_CIF(file, occ_tol=1.05)

formula=file.read_formula()
ID=file.read_id()
Z=file.z()
cell=file.cell()
space_group=file.space_group()
symmetry_operations=file.symmetry()

# Reading all orbit infromation, output is a dataframe
orbits=file.orbits()

# Generating positions for all atoms, with proper orbit labels. Output is whether dataframe, or dataframe and pymatgen Structure object
positions=file.positions(orbits,symmetry_operations,pystruct=False,merge_sites=False,merge_tol=1e-2,r=3,dist_tol=1e-2)

```

(2) The class allowing to classify all orbits according to their class of disorder in disorder.py: **Disorder**. Classification outputs the dataframe describing orbits with their disorder labales as 'orbit_disorder' column.

```
from disorder import Disorder

disorder=Disorder(CIF_file, radius_file='data/all_radii.csv', cutoff=0.5, occ_tol=1.05, merge_tol=0.005, pymatgen_dist_matrix=False, dist_tol=1e-3)
# making classification
output = disorder.classify()
```
(3) The class allowing to calculate entropy in disorder.py: **Entropy**. Input is 'orbits' dataframe.
```
from disorder import Entropy

entropy=Entropy(orbits)
mixing_entropy=entropy.mixing_entropy()
configurational_entropy=entropy.configurational_entropy()
```
(4) Pyhton Notebooks used to extract data from CIFs and analyse extracted information.

