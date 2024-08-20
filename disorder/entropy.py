import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from disorder.disorder import Disorder


class Entropy:
    """
    data is a row of a dataframe with the output from disorder classification
    """
    def __init__(self, file: str, radius_file: str = 'data/all_radii.csv', 
                 cutoff: float = 0.5, occ_tol: float = 1.05, merge_tol: float = 0.01, \
                 pymatgen_dist_matrix: bool = False, dist_tol: float = 0.01):
        
        self.material=Disorder(file,radius_file=radius_file, cutoff=cutoff,occ_tol=occ_tol,merge_tol=merge_tol,\
                               pymatgen_dist_matrix=pymatgen_dist_matrix,dist_tol=dist_tol)

        self.formula=self.material.material.read_formula
        self.z=self.material.material.z
        self.occ_tol=occ_tol
    
    def get_data(self):
        self.data=self.material.classify()
        return self.data

    def calculate_entropy(self) -> float:
        '''
        Function calculates entropy for compound
        input: entropy_type is : 'mixing', 'configurational', or 'mc_configurational'
        output: float value of entropy
        '''
        
        if not hasattr(self, 'data'):
            self.data=self.material.classify()
        
        entropy=0
        p=self.material.positions   

        new_orb_list=[]
        for ind,orb in enumerate(self.data['intersect_orbit_connected'].values):
            new_orb=tuple(np.sort(orb))
            new_orb_list.append(new_orb)
        for orb in list(set(new_orb_list)):
            for sub_orb in orb:
                internal_intersection=self.data.loc[self.data['label']==sub_orb]['internal_intersection'].values[0]
                disorder_type=self.data.loc[self.data['label']==sub_orb]['orbit_disorder'].values[0]
            if(len(orb)>1 or internal_intersection==True):
                x=orb
                n_exp=100000
                orbit=p.loc[p['atom_site_label'].isin(x)]
                index=orbit.index.values
                occupancies=orbit['atom_site_occupancy'].values
                intersections=orbit['intersecting_sites'].values
                non_reject=np.ones(n_exp)
                num_conf=np.ones(n_exp)
                total_occ=np.zeros(n_exp)
                for exp in range(n_exp):
                    prob=np.random.rand(len(orbit))
                    atoms=np.array(prob<occupancies,dtype=int)
                    occup_sites=atoms*index
                    total_occ[exp]=np.sum(atoms)
                        # if(total_occ[exp]!=2):
                        #     num_conf[exp]=0
                    occupied_indices = np.where(occup_sites != 0)[0]
                    for j in occupied_indices:
                        for site in intersections[j]:
                            if(site in occup_sites):
                                non_reject[exp]=0  
                if(np.sum(non_reject)>0):    
                    entropy+=np.log(np.sum(non_reject)/np.sum(num_conf))
                else:
                    entropy=np.nan
                for sub_orb in x:
                    total_occ=0
                    mult=p.loc[p['atom_site_label']==sub_orb]['atom_site_symmetry_multiplicity'].values[0]
                    orbit_content=p.loc[p['atom_site_label']==sub_orb]['atom_site_type_symbol'].values[0]
                    for elem,occ in orbit_content.items():
                        total_occ+=occ
                        if(occ>0):
                            entropy+=-mult*(occ*np.log(occ))
                    if(total_occ<1):    
                        entropy+=-mult*((1-total_occ)*np.log(1-total_occ))
                    elif(total_occ>1.05):
                        entropy=np.nan
            elif(disorder_type in {'V','S','SV'}):
                total_occ=0
                mult=p.loc[p['atom_site_label']==orb[0]]['atom_site_symmetry_multiplicity'].values[0]
                orbit_content=p.loc[p['atom_site_label']==orb[0]]['atom_site_type_symbol'].values[0]
                for elem,occ in orbit_content.items():
                    total_occ+=occ
                    if(occ>0):
                        entropy+=-mult*(occ*np.log(occ))
                if(total_occ<1):    
                    entropy+=-mult*((1-total_occ)*np.log(1-total_occ))
                elif(total_occ>1.05):
                    entropy=np.nan
            
        comp = Composition(self.formula)
        natoms = np.sum(list(comp.as_dict().values()))
        entropy = round(entropy/(self.z * natoms),3)
        
        return entropy
