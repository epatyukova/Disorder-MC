import numpy as np
import pandas as pd
import json
import os

from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, Process

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Species

from disorder.disorder import Disorder
from disorder.entropy import Entropy
from disorder.cifreader import Read_CIF

elem_list=['Ac', 'Ag', 'Al', 'Am', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br', 'C', 'Ca',\
 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe',\
 'Fr', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Lu',\
 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa',\
 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc',\
 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V',\
 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr']

# class TimeoutError(Exception):
#     pass

# def wrapper_function(func, timeout):
#     proc = Process(target=func)
#     proc.start()
#     proc.join(timeout)
#     if proc.is_alive():
#         proc.terminate()
#         raise TimeoutError(f"Function execution exceeded the time limit of {timeout} seconds")

def process_file(file: str):
    path='/Users/elenapatyukova/Documents/GitHub/ICSDClient/cifs/'
    h_switch=0
    form_switch=0
    exc_large=[]
    errors_large=[]
    exc=[]
    exc_h=[]
    exc_form=[]
    compound={}
    
    try:
        cif=Read_CIF(file=path+file)
        composition=Composition(cif.read_formula).as_dict()
        for el in composition.keys():
            if(el not in elem_list):
                exc_h.append(file)
                h_switch=1
                break
                
        o=cif.orbits()
        form_el=composition.keys()
        struct_el=[]
        for name in o['atom_site_type_symbol'].values:
            struct_el.append(str(Species(name).element))
        struct_el=set(struct_el)
        if(struct_el!=form_el):
            form_switch=1
            exc_form.append(file)
                
        if(h_switch==0 and form_switch==0):
            ent=Entropy(file=path+file, radius_file='data/all_radii.csv')
            compound['formula']=ent.formula
            compound['ICSD_ID']=ent.material.material.read_id
            compound['group_num']=ent.material.material.space_group
            compound['Z']=ent.z

            orbits=ent.get_data()
            compound=compound|orbits.to_dict()
            compound['entropy']=ent.calculate_entropy()

            # if(len(ent.material.positions)<501):
            #     orbits=ent.get_data()
            #     compound=compound|orbits.to_dict()
            #     compound['entropy']=ent.calculate_entropy()
            # else:
            #     exc_large.append(file)
            #     errors_large.append(list(set(ent.material.return_errors())))
                                    
    except:
        exc.append(file)
    return file, compound, exc_large, errors_large, exc, exc_h, exc_form

if __name__ == '__main__':
    path='/Users/elenapatyukova/Documents/GitHub/ICSDClient/cifs/'
    list_of_files=os.listdir(path)

    limit_low=220000
    limit_high=len(list_of_files)
    
    disordered_comp={}
    exc_large=[]
    errors_large=[]
    exc=[]
    exc_h=[]
    exc_form=[]

    time1=time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, list_of_files[i]) for i in range(limit_low, limit_high)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                file, compound, e1, e2, e3, e5, e6 = result
                exc_large.append(e1)
                errors_large.append(e2)
                exc.append(e3)
                exc_h.append(e5)
                exc_form.append(e6)
                disordered_comp[file] = compound

    time2=time()
    print(time2-time1)

    with open('disorder_results_'+str(limit_low)+'_'+str(limit_high)+'.json', 'w') as fp:
        json.dump(disordered_comp, fp, indent=2)
    
    err=pd.DataFrame()
    err['id']=exc_h
    err.to_csv('H_files_'+str(limit_low)+'_'+str(limit_high)+'.csv')

    err=pd.DataFrame()
    err['id']=exc_form
    err.to_csv('incomplete_structure_'+str(limit_low)+'_'+str(limit_high)+'.csv')

    err=pd.DataFrame()
    err['id']=exc
    err.to_csv('excluded_files_'+str(limit_low)+'_'+str(limit_high)+'.csv')

    err=pd.DataFrame()
    err['id']=exc_large
    err['errors']=errors_large
    err.to_csv('files_number_of_sites_larger_500_'+str(limit_low)+'_'+str(limit_high)+'.csv')

